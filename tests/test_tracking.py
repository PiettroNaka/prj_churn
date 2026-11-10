"""O treino tem de deixar rastro: um run com params, métricas e linhagem.

Treina de verdade, mas numa amostra minúscula e contra um backend descartável
(`tmp_path`) — o que se testa aqui é a instrumentação, não a floresta.
"""
import mlflow
import pytest

from churn.config import Settings
from churn.data import coerce_total_charges
from churn.features import CategoricalEncoder, add_derived_features
from churn.model import parametros, train
from churn.serving import ChurnPipeline


@pytest.fixture
def cfg(raw_frame, tmp_path) -> Settings:
    """Config apontando para uma amostra e um MLflow que morre com o teste."""
    dataset = tmp_path / "churn.csv"
    raw_frame.to_csv(dataset, index=False)
    return Settings(
        data_path=dataset,
        artifact_path=tmp_path / "modelo.pkl",
        quarantine_dir=None,
        test_size=0.5,  # a amostra tem 4 linhas; metade para cada lado
        n_estimators=5,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-teste",
        mlflow_run_name="rf-teste",
    )


def test_parametros_nao_levam_o_endereco_do_caderno():
    """Onde anotar não é o que foi treinado — `mlflow_*` fica fora dos params."""
    params = parametros(Settings())

    assert "n_estimators" in params
    assert not [chave for chave in params if chave.startswith("mlflow_")]


def test_treino_registra_run_com_linhagem(cfg):
    _, metrics = train(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[cfg.mlflow_experiment])
    assert len(runs) == 1
    run = runs.iloc[0]

    assert run["tags.mlflow.runName"] == "rf-teste"

    # linhagem: sem ponteiro .dvc, o md5 é o do próprio arquivo
    assert len(run["tags.data_md5"]) == 32
    assert run["tags.git_commit"]

    # o que foi escolhido virou param; o que foi medido, métrica
    assert run["params.n_estimators"] == "5"
    assert float(run["metrics.roc_auc"]) == pytest.approx(metrics["roc_auc"])


def test_modelo_logado_roda_com_cliente_cru(cfg, raw_frame):
    """O teste que separa "registrado" de "utilizável".

    A entrada é a linha como sai do sistema de origem — com ``customerID``, com
    ``TotalCharges`` em texto. O estimador solto quebraria aqui; o pipeline
    logado não pode quebrar, ou o modelo não roda em lugar nenhum.
    """
    train(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    run_id = mlflow.search_runs(experiment_names=[cfg.mlflow_experiment]).iloc[0][
        "run_id"
    ]

    modelo = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    previsto = modelo.predict(raw_frame)

    assert list(previsto.columns) == ["churn", "probabilidade"]
    assert len(previsto) == len(raw_frame)
    assert set(previsto["churn"]) <= {"Yes", "No"}
    assert previsto["probabilidade"].between(0, 1).all()


def test_assinatura_nao_exige_o_alvo_nem_o_identificador(cfg):
    """Para prever o churn não se pode ser obrigado a informar o churn.

    O MLflow trata a assinatura como contrato e exige TODA coluna declarada.
    Inferindo-a do CSV cru inteiro, o modelo logado recusava qualquer chamada
    real com ``Model is missing inputs ['customerID', 'Churn']`` — um modelo
    impossível de usar, que só não aparecia porque o teste acima manda a linha
    completa. O pipeline sempre descartou as duas colunas; faltava a assinatura
    dizer o mesmo.
    """
    train(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    run_id = mlflow.search_runs(experiment_names=[cfg.mlflow_experiment]).iloc[0][
        "run_id"
    ]

    entrada = mlflow.models.get_model_info(f"runs:/{run_id}/model").signature.inputs
    declaradas = {campo.name for campo in entrada.inputs}

    assert cfg.target not in declaradas, "a assinatura pede o alvo que vai prever"
    assert cfg.id_column not in declaradas, "identificador não é feature"
    assert "tenure" in declaradas, "as features de verdade continuam no contrato"


def test_pipeline_aceita_a_linha_sem_id_e_sem_alvo(cfg, raw_frame):
    """Quem chama manda a linha que tem — com ou sem as colunas que não são feature."""
    modelo, _ = train(cfg)

    # um encoder ajustado no mesmo formato que o pipeline recebe
    limpo = raw_frame.drop(columns=["customerID", "Churn"])
    encoder = CategoricalEncoder().fit(add_derived_features(coerce_total_charges(limpo)))
    pipeline = ChurnPipeline(modelo, encoder, "customerID", "Churn", "Yes")

    com_tudo = pipeline.preparar(raw_frame)
    so_features = pipeline.preparar(limpo)

    assert list(com_tudo.columns) == list(so_features.columns)
    assert "customerID" not in com_tudo.columns
    assert "gasto_por_mes" in com_tudo.columns


def test_pipeline_aceita_colunas_fora_de_ordem(cfg, raw_frame):
    """O sklearn compara feature_names_in_ posição a posição e recusa a chamada.

    A ordem diverge com facilidade: o CSV lista `tenure` logo depois de
    `Dependents`, o contrato o declara junto das outras numéricas, no fim. Montar
    o cliente a partir do contrato — que é o que a API e o formulário fazem — já
    produz a ordem "errada".

    Isso passava despercebido porque o MLflow reordena pela assinatura antes de
    chamar o modelo; só quem instancia o pipeline direto batia no erro. Agora o
    pipeline se basta.
    """
    modelo, _ = train(cfg)

    limpo = raw_frame.drop(columns=["customerID", "Churn"])
    encoder = CategoricalEncoder().fit(add_derived_features(coerce_total_charges(limpo)))
    pipeline = ChurnPipeline(modelo, encoder, "customerID", "Churn", "Yes")

    invertido = limpo[list(reversed(limpo.columns))]

    na_ordem = pipeline.predict(None, limpo)
    fora_de_ordem = pipeline.predict(None, invertido)

    assert list(fora_de_ordem["churn"]) == list(na_ordem["churn"])
    assert fora_de_ordem["probabilidade"].tolist() == na_ordem["probabilidade"].tolist()


def test_coluna_faltando_diz_qual(cfg, raw_frame):
    """A mensagem do sklearn não diz qual coluna sumiu; esta diz."""
    modelo, _ = train(cfg)

    limpo = raw_frame.drop(columns=["customerID", "Churn"])
    encoder = CategoricalEncoder().fit(add_derived_features(coerce_total_charges(limpo)))
    pipeline = ChurnPipeline(modelo, encoder, "customerID", "Churn", "Yes")

    with pytest.raises(ValueError, match="Contract"):
        pipeline.predict(None, limpo.drop(columns=["Contract"]))
