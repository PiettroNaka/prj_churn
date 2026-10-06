"""Do melhor run ao modelo de produção — a tarefa do Encontro 7.

Um run é um experimento; não é um artefato de produção. Achar o melhor é fácil —
``experiments.melhores()`` já faz. O que falta é o endereço estável: semanas
depois, "qual era mesmo o run bom?" não se responde rolando uma lista.

    uv run python -m churn.registry

O Registry resolve com três peças:

    registered model   um nome ("churn") que agrupa versões
    version            um modelo registrado, imutável
    alias              um ponteiro móvel (@champion) para a versão que serve

Promover é mover o alias. Reverter é mover de volta. Nenhum deploy no meio.

**Qual run é promovível.** A varredura do Encontro 6 não loga modelo de
propósito — são nove florestas de até 500 árvores, e ela existe para escolher a
configuração, não para guardar candidatos. Quem treina e loga o pipeline é o
``churn.model``. Então o que se promove são os runs dele (``mlflow_run_name``),
não os filhos do grid: um run sem modelo não tem o que registrar.

O fluxo completo, portanto:

    1. uv run python -m churn.experiments     # acha a melhor configuração
    2. CHURN_MAX_DEPTH=8 uv run python -m churn.model   # treina e loga o vencedor
    3. uv run python -m churn.registry        # registra e promove
"""
from __future__ import annotations

import mlflow
import pandas as pd
from mlflow import MlflowClient

from churn.config import Settings, settings
from churn.experiments import METRICA
from churn.serving import NOME_DO_MODELO, carregar

#: Candidata em teste, ainda não promovida. O par de ``champion``.
CHALLENGER = "challenger"


class PromocaoRecusada(RuntimeError):
    """O candidato não passou no portão — nada foi registrado nem promovido."""


def _conectar(cfg: Settings) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)


def candidato(cfg: Settings = settings) -> pd.Series:
    """O melhor run **com modelo logado**, pela métrica de decisão.

    Filtra pelo nome de run do ``churn.model``: os filhos do grid pontuam, mas
    não guardam modelo, e registrar um deles falharia no artefato ausente.
    """
    _conectar(cfg)
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow_experiment],
        filter_string=(
            f"attributes.run_name = '{cfg.mlflow_run_name}' "
            f"and metrics.{METRICA} > 0"
        ),
        order_by=[f"metrics.{METRICA} DESC"],
    )
    if runs.empty:
        raise PromocaoRecusada(
            f"Nenhum run '{cfg.mlflow_run_name}' com {METRICA} no experimento "
            f"'{cfg.mlflow_experiment}'. A varredura escolhe a configuração, mas "
            "quem loga o modelo é o treino: rode `uv run python -m churn.model`."
        )
    return runs.iloc[0]


def promover(
    cfg: Settings = settings,
    alias: str | None = None,
) -> int:
    """Registra o candidato e aponta o alias para ele. Devolve a versão.

    Levanta :class:`PromocaoRecusada` se a métrica não alcançar o piso. Falhar
    alto é o ponto: um portão que deixa passar dá a impressão de que alguém
    conferiu.
    """
    alias = alias or cfg.mlflow_champion_alias
    melhor = candidato(cfg)
    score = float(melhor[f"metrics.{METRICA}"])

    if score < cfg.mlflow_promocao_minima:
        raise PromocaoRecusada(
            f"{METRICA} {score:.4f} abaixo do piso {cfg.mlflow_promocao_minima:.2f} "
            "— nada promovido."
        )

    versao = mlflow.register_model(
        f"runs:/{melhor['run_id']}/{NOME_DO_MODELO}", cfg.mlflow_registered_model
    )
    apontar(int(versao.version), cfg, alias)
    return int(versao.version)


def apontar(
    versao: int, cfg: Settings = settings, alias: str | None = None
) -> None:
    """Move um alias para uma versão já registrada.

    É o mecanismo de rollback: ``apontar(1)`` devolve a produção à versão 1 sem
    treinar, sem registrar e sem tocar em servidor.
    """
    _conectar(cfg)
    MlflowClient().set_registered_model_alias(
        cfg.mlflow_registered_model, alias or cfg.mlflow_champion_alias, versao
    )


def uri_do_campeao(cfg: Settings = settings, alias: str | None = None) -> str:
    """``models:/nome@alias`` — a referência estável que o consumidor usa."""
    return f"models:/{cfg.mlflow_registered_model}@{alias or cfg.mlflow_champion_alias}"


def carregar_campeao(cfg: Settings = settings, alias: str | None = None):
    """Carrega o pipeline em produção.

    Quem chama não precisa saber qual run o gerou nem qual configuração venceu —
    é esta chamada que a API do Encontro 12 vai fazer.

    O ``_conectar`` antes do load não é decoração: sem ele o MLflow procura o
    registry no backend padrão e responde "Registered Model not found" mesmo com
    o modelo registrado — o atrito nº 1 do Encontro 6, agora na porta do 7.
    """
    _conectar(cfg)
    return carregar(uri_do_campeao(cfg, alias))


def versao_do_alias(cfg: Settings = settings, alias: str | None = None):
    """A versão que o alias aponta agora, ou ``None`` se ninguém apontou ainda."""
    _conectar(cfg)
    try:
        return MlflowClient().get_model_version_by_alias(
            cfg.mlflow_registered_model, alias or cfg.mlflow_champion_alias
        )
    except mlflow.exceptions.MlflowException:
        return None


def main() -> None:
    melhor = candidato()
    print(
        f"Candidato: run {melhor['run_id'][:8]} — "
        f"{METRICA} {float(melhor[f'metrics.{METRICA}']):.4f} "
        f"(data_md5 {melhor.get('tags.data_md5', '?')})"
    )
    versao = promover()
    print(f"Registrado: {settings.mlflow_registered_model} v{versao} @{settings.mlflow_champion_alias}")
    print(f"Consuma com: churn.serving.carregar(\"{uri_do_campeao()}\")")


if __name__ == "__main__":
    main()
