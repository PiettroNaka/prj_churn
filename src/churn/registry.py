"""Do melhor run ao modelo de produção — Model Registry.

Um run é um experimento; não é um artefato de produção. Este módulo faz a
travessia: acha o melhor run por código, confere o **portão de qualidade** e só
então registra uma versão e move o alias ``champion`` para ela.

``uv run python -m churn.registry``

Por que o portão: registrar não pode ser automático. Se o melhor run de hoje é
pior que o limiar, promovê-lo seria rebaixar a produção silenciosamente — o
pipeline deve falhar alto, não publicar um modelo ruim.
"""
from __future__ import annotations

import mlflow
import pandas as pd
from mlflow import MlflowClient

from churn.config import Settings, settings
from churn.model import setup_tracking

#: Ponteiro móvel para a versão que serve. Promover = mover isto.
CHAMPION = "champion"

#: Candidata em teste, ainda não promovida.
CHALLENGER = "challenger"

#: Métrica que decide o vencedor. roc_auc por ser insensível ao desbalanceamento
#: do alvo (32,5% de churn) — accuracy premiaria um modelo que só diz "não".
METRICA = "roc_auc"


class PromotionError(RuntimeError):
    """O melhor run não alcançou o limiar — nada foi promovido."""


def best_run(cfg: Settings = settings) -> pd.Series:
    """O melhor run do experimento, pela métrica de decisão.

    ``search_runs`` devolve um DataFrame pandas, então filtrar e cruzar com
    qualquer análise é pandas normal.
    """
    setup_tracking(cfg)
    runs = mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        order_by=[f"metrics.{METRICA} DESC"],
    )
    if runs.empty:
        raise PromotionError(
            f"Nenhum run no experimento '{cfg.experiment_name}'. "
            "Rode `uv run python -m churn.experiments` antes."
        )
    return runs.iloc[0]


def promote_best(cfg: Settings = settings, alias: str = CHAMPION) -> int:
    """Registra o melhor run e aponta o alias para ele. Devolve a versão.

    Levanta :class:`PromotionError` se o melhor run não passar do limiar.
    """
    melhor = best_run(cfg)
    score = float(melhor[f"metrics.{METRICA}"])

    if score < cfg.promotion_min_roc_auc:
        raise PromotionError(
            f"Melhor {METRICA} foi {score:.4f}, abaixo do limiar "
            f"{cfg.promotion_min_roc_auc:.2f} — nada promovido."
        )

    versao = mlflow.register_model(
        f"runs:/{melhor['run_id']}/model", cfg.registered_model_name
    )
    MlflowClient().set_registered_model_alias(
        cfg.registered_model_name, alias, versao.version
    )
    return int(versao.version)


def load_champion(cfg: Settings = settings, alias: str = CHAMPION):
    """Carrega o modelo de produção pelo flavor universal (pyfunc).

    ``models:/nome@alias`` é a referência estável: quem consome não precisa
    saber qual run gerou o modelo, nem qual biblioteca o treinou. É exatamente
    esta chamada que a API do Encontro 12 vai fazer.
    """
    setup_tracking(cfg)
    return mlflow.pyfunc.load_model(f"models:/{cfg.registered_model_name}@{alias}")


def main() -> None:
    melhor = best_run()
    print(
        f"Melhor run: {melhor['tags.mlflow.runName']} — "
        f"{METRICA} {float(melhor[f'metrics.{METRICA}']):.4f}"
    )
    versao = promote_best()
    print(
        f"Registrado: {settings.registered_model_name} v{versao} @{CHAMPION}\n"
        f"Consuma com: mlflow.pyfunc.load_model"
        f'("models:/{settings.registered_model_name}@{CHAMPION}")'
    )


if __name__ == "__main__":
    main()
