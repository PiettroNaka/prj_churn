"""Testes do portão de promoção.

O que importa aqui não é o MLflow — é a regra de negócio: um modelo abaixo do
limiar **não** pode virar champion. Um portão que não barra é pior que nenhum,
porque dá a impressão de que alguém está conferindo.

Os testes trocam ``best_run`` por um dublê, então não precisam de backend de
tracking nem de um grid rodado.
"""
import pandas as pd
import pytest

from churn import registry
from churn.config import settings
from churn.registry import METRICA, PromotionError, promote_best


def _run_falso(roc_auc: float) -> pd.Series:
    return pd.Series({"run_id": "deadbeef", f"metrics.{METRICA}": roc_auc})


def test_abaixo_do_limiar_nao_promove(monkeypatch):
    monkeypatch.setattr(registry, "best_run", lambda cfg=settings: _run_falso(0.5))

    def nao_deveria_registrar(*args, **kwargs):  # pragma: no cover
        raise AssertionError("register_model foi chamado apesar do portão fechado")

    monkeypatch.setattr(registry.mlflow, "register_model", nao_deveria_registrar)

    with pytest.raises(PromotionError, match="abaixo do limiar"):
        promote_best()


def test_exatamente_no_limiar_promove(monkeypatch):
    """O limiar é inclusivo: >= passa. Um modelo na linha não é reprovado."""
    registrados = {}
    monkeypatch.setattr(
        registry, "best_run", lambda cfg=settings: _run_falso(settings.promotion_min_roc_auc)
    )
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        lambda uri, nome: registrados.setdefault("mv", type("MV", (), {"version": 7})()),
    )
    monkeypatch.setattr(
        registry, "MlflowClient", lambda: type("C", (), {"set_registered_model_alias": lambda *a: None})()
    )

    assert promote_best() == 7


def test_promove_a_partir_do_run_vencedor(monkeypatch):
    """A URI registrada tem de apontar para o modelo do run que ganhou."""
    capturado = {}
    monkeypatch.setattr(registry, "best_run", lambda cfg=settings: _run_falso(0.9))
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        lambda uri, nome: capturado.update(uri=uri, nome=nome)
        or type("MV", (), {"version": 1})(),
    )
    monkeypatch.setattr(
        registry, "MlflowClient", lambda: type("C", (), {"set_registered_model_alias": lambda *a: None})()
    )

    promote_best()
    assert capturado["uri"] == "runs:/deadbeef/model"
    assert capturado["nome"] == settings.registered_model_name


def test_experimento_vazio_tem_mensagem_acionavel(monkeypatch):
    import mlflow

    monkeypatch.setattr(registry, "setup_tracking", lambda cfg=settings: None)
    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: pd.DataFrame())

    with pytest.raises(PromotionError, match="churn.experiments"):
        registry.best_run()
