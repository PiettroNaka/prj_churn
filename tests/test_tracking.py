"""O treino tem de deixar rastro: um run com params, métricas e linhagem.

Treina de verdade, mas numa amostra minúscula e contra um backend descartável
(`tmp_path`) — o que se testa aqui é a instrumentação, não a floresta.
"""
import mlflow
import pytest

from churn.config import Settings
from churn.model import parametros, train


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


def test_modelo_logado_pode_ser_recarregado(cfg):
    """Um modelo registrado que não carrega de volta não serve para nada."""
    train(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[cfg.mlflow_experiment])
    run_id = runs.iloc[0]["run_id"]

    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    assert hasattr(modelo, "predict")
