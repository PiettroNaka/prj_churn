"""A API tem de recusar o que o contrato recusa, e responder o que o modelo diz.

Nada aqui sobe servidor nem toca num Registry de verdade: o ``TestClient`` fala
com o app em processo, e o pipeline é um dublê. O que se testa é a fronteira —
o que entra, o que sai, e o que acontece quando não há modelo.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn import api
from churn.api import Cliente, app, estado
from churn.schema import COLUNAS_CRU

#: Um cliente válido, montado do próprio contrato gerado — se o contrato mudar,
#: esta amostra muda junto, em vez de envelhecer calada.
VALIDO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 12,
    "MonthlyCharges": 55.5,
    "TotalCharges": 666.0,
}


class PipelineFalso:
    """Responde no formato do ChurnPipeline, sem treinar nada."""

    def predict(self, entrada: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "churn": ["Yes" if t < 10 else "No" for t in entrada["tenure"]],
                "probabilidade": [
                    0.9 if t < 10 else 0.1 for t in entrada["tenure"]
                ],
            },
            index=entrada.index,
        )


@pytest.fixture(autouse=True)
def estado_limpo():
    """O modelo vive num global; sem isto um teste contamina o próximo."""
    anterior = (estado.pipeline, estado.versao, estado.execucao)
    yield
    estado.pipeline, estado.versao, estado.execucao = anterior


@pytest.fixture
def cliente(monkeypatch):
    """API no ar, com um pipeline que responde."""

    def carregar_falso(cfg=None):
        estado.pipeline = PipelineFalso()
        estado.versao = "7"
        estado.execucao = "cafe" * 8

    monkeypatch.setattr(api, "carregar", carregar_falso)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def cliente_sem_modelo(monkeypatch):
    """API no ar sem campeão promovido — o caso do primeiro deploy."""

    def explodir(cfg=None):
        raise RuntimeError("nenhum campeão promovido")

    monkeypatch.setattr(api, "carregar", explodir)
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# o contrato de entrada é gerado, não escrito
# --------------------------------------------------------------------------


def test_entrada_cobre_o_contrato_menos_id_e_alvo():
    """A junta que quebraria calada: schema ganha coluna, API não fica sabendo.

    Uma lista paralela aqui seria *training/serving skew* na forma mais
    discreta — o treino passaria a ver uma feature que a API nunca aceita.
    """
    assert set(Cliente.model_fields) == COLUNAS_CRU - {"customerID", "Churn"}


def test_dominio_das_categoricas_vem_do_contrato():
    """`Contract` é a feature nº 1 do modelo; seu domínio não pode divergir."""
    from churn.schema import ChurnRawSchema

    permitidos = (
        ChurnRawSchema.to_schema().columns["Contract"].checks[0].statistics[
            "allowed_values"
        ]
    )

    assert set(Cliente.model_fields["Contract"].annotation.__args__) == set(permitidos)


def test_total_charges_e_numero_na_api():
    """No CSV é texto (por causa dos brancos); em JSON não há motivo para isso."""
    assert Cliente.model_fields["TotalCharges"].annotation is float


# --------------------------------------------------------------------------
# previsão
# --------------------------------------------------------------------------


def test_preve_na_ordem_em_que_chegaram(cliente):
    novo, antigo = dict(VALIDO, tenure=2), dict(VALIDO, tenure=60)

    resposta = cliente.post("/prever", json=[novo, antigo])

    assert resposta.status_code == 200
    assert [p["churn"] for p in resposta.json()] == ["Yes", "No"]


def test_probabilidade_sai_entre_zero_e_um(cliente):
    (previsao,) = cliente.post("/prever", json=[VALIDO]).json()

    assert 0.0 <= previsao["probabilidade"] <= 1.0


def test_lote_grande_vira_uma_resposta_por_cliente(cliente):
    """A chamada de verdade é em lote: a carteira inteira, não um por vez."""
    resposta = cliente.post("/prever", json=[VALIDO] * 25)

    assert len(resposta.json()) == 25


# --------------------------------------------------------------------------
# o que o contrato recusa
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("Contract", "Plano Vitalicio"),  # categoria que não existe
        ("tenure", 900),                  # acima do teto de 72 meses
        ("tenure", -1),
        ("MonthlyCharges", 9999.0),       # fora da faixa do profiling
        ("gender", "Outro"),
        ("TotalCharges", -5.0),
    ],
)
def test_valor_fora_do_contrato_e_recusado(cliente, campo, valor):
    resposta = cliente.post("/prever", json=[dict(VALIDO, **{campo: valor})])

    assert resposta.status_code == 422


def test_coluna_faltando_e_recusada(cliente):
    sem_contrato = {k: v for k, v in VALIDO.items() if k != "Contract"}

    assert cliente.post("/prever", json=[sem_contrato]).status_code == 422


def test_lista_vazia_e_recusada(cliente):
    """Responder 200 com lista vazia esconderia um erro de quem chamou."""
    assert cliente.post("/prever", json=[]).status_code == 422


# --------------------------------------------------------------------------
# operação
# --------------------------------------------------------------------------


def test_saude_diz_qual_versao_responde(cliente):
    assert cliente.get("/saude").json() == {"status": "ok", "versao": "7"}


def test_modelo_mostra_o_endereco_estavel(cliente):
    corpo = cliente.get("/modelo").json()

    assert corpo["uri"] == "models:/churn@champion"
    assert corpo["carregado"] is True
    assert corpo["versao"] == "7"


def test_sem_campeao_a_api_sobe_e_explica(cliente_sem_modelo):
    """Subir quebrada e dizer o porquê é melhor que um restart loop mudo."""
    corpo = cliente_sem_modelo.get("/saude").json()

    assert corpo["status"] == "sem modelo"
    assert "campeão" in corpo["detalhe"]
    assert "churn.registry" in corpo["saida"]


def test_sem_campeao_prever_responde_503(cliente_sem_modelo):
    """503 e não 500: o serviço está de pé, o modelo é que não chegou."""
    resposta = cliente_sem_modelo.post("/prever", json=[VALIDO])

    assert resposta.status_code == 503


def test_openapi_descreve_a_rota_de_previsao(cliente):
    """O /docs é o que o consumidor lê antes de perguntar qualquer coisa."""
    rotas = cliente.get("/openapi.json").json()["paths"]

    assert "/prever" in rotas
    assert "/saude" in rotas
