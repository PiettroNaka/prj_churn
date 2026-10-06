"""O portão de promoção tem de barrar — senão é só cerimônia.

Os testes trocam a busca de runs por um dublê: a regra que importa é de negócio,
não do MLflow, e não precisa de backend de tracking para ser verificada.
"""
import mlflow
import pandas as pd
import pytest

from churn import registry
from churn.config import settings
from churn.experiments import METRICA
from churn.registry import PromocaoRecusada, promover, uri_do_campeao


def _run(score: float, run_id: str = "deadbeefcafe") -> pd.Series:
    return pd.Series({"run_id": run_id, f"metrics.{METRICA}": score})


@pytest.fixture
def registro_falso(monkeypatch):
    """Captura o que teria sido registrado, sem tocar num Registry de verdade."""
    capturado: dict[str, object] = {}

    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        lambda uri, nome: capturado.update(uri=uri, nome=nome)
        or type("MV", (), {"version": 3})(),
    )
    monkeypatch.setattr(
        registry,
        "apontar",
        lambda versao, cfg=settings, alias=None: capturado.update(
            versao=versao, alias=alias
        ),
    )
    return capturado


def test_abaixo_do_piso_nao_registra(monkeypatch, registro_falso):
    monkeypatch.setattr(registry, "candidato", lambda cfg=settings: _run(0.42))

    with pytest.raises(PromocaoRecusada, match="abaixo do piso"):
        promover()

    assert registro_falso == {}, "o portão deixou passar"


def test_no_piso_exato_promove(monkeypatch, registro_falso):
    """O piso é inclusivo: um modelo na linha não é reprovado por arredondamento."""
    monkeypatch.setattr(
        registry, "candidato", lambda cfg=settings: _run(settings.mlflow_promocao_minima)
    )

    assert promover() == 3


def test_registra_o_modelo_do_run_vencedor(monkeypatch, registro_falso):
    """A URI tem de apontar o artefato do run que ganhou, não outro qualquer."""
    monkeypatch.setattr(registry, "candidato", lambda cfg=settings: _run(0.93, "abc123"))

    promover()

    assert registro_falso["uri"] == "runs:/abc123/model"
    assert registro_falso["nome"] == settings.mlflow_registered_model


def test_sem_run_com_modelo_a_mensagem_diz_o_que_fazer(monkeypatch):
    """Quem rodou só a varredura precisa ser mandado ao treino, não a um traceback."""
    monkeypatch.setattr(registry, "_conectar", lambda cfg: None)
    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: pd.DataFrame())

    with pytest.raises(PromocaoRecusada, match="churn.model"):
        registry.candidato()


def test_busca_ignora_os_filhos_do_grid(monkeypatch):
    """Os runs do grid pontuam mas não logam modelo — registrar um deles falharia."""
    capturado = {}
    monkeypatch.setattr(registry, "_conectar", lambda cfg: None)

    def falsa_busca(**kwargs):
        capturado.update(kwargs)
        return pd.DataFrame([{"run_id": "x", f"metrics.{METRICA}": 0.9}])

    monkeypatch.setattr(mlflow, "search_runs", falsa_busca)
    registry.candidato()

    assert f"run_name = '{settings.mlflow_run_name}'" in capturado["filter_string"]


def test_uri_do_campeao_usa_alias_e_nao_versao_fixa():
    """Alias e não versão: o consumidor não deve ser reescrito a cada promoção."""
    assert uri_do_campeao() == (
        f"models:/{settings.mlflow_registered_model}@{settings.mlflow_champion_alias}"
    )


def test_carregar_campeao_fixa_o_backend_antes(monkeypatch):
    """Sem fixar a URI, o load procura o registry no backend errado.

    O sintoma é "Registered Model not found" com o modelo registrado — o mesmo
    atrito de runs espalhados do Encontro 6, agora na hora de consumir.
    """
    chamadas = []
    monkeypatch.setattr(registry, "_conectar", lambda cfg: chamadas.append("conectou"))
    monkeypatch.setattr(registry, "carregar", lambda uri: chamadas.append(uri))

    registry.carregar_campeao()

    assert chamadas == ["conectou", uri_do_campeao()]
