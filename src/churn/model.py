"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).

Desde o Encontro 6 cada execução vira um **run** do MLflow: params (o que foi
escolhido), métricas (o que foi medido), o modelo como artefato e as tags de
linhagem — qual dado e qual código produziram aquilo. O ``print`` mostrava o
número e o esquecia; o run guarda.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.lineage import data_md5, git_commit, ponteiro_de
from churn.serving import log_pipeline


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa (reprodutível).

    Os seis hiperparâmetros são os mesmos que o ``churn.tuning`` varre. É o que
    fecha o ciclo: o que a busca encontra, a config diz, e este construtor
    treina de novo — sem ninguém reescrever uma chamada à mão.
    """
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        max_features=cfg.max_features,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )


def save_artifact(model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path) -> None:
    """Persiste modelo + encoder juntos (a inferência precisa do encoder ajustado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def parametros(cfg: Settings) -> dict[str, object]:
    """A config Pydantic vira params do run — os dois contratos se encontram aqui.

    Fora as chaves ``mlflow_*``: elas dizem *onde anotar*, não *o que foi
    treinado*. Param é entrada do experimento; endereço do caderno não é.

    Fora também as ``tuning_*``, pela mesma régua invertida: elas descrevem uma
    busca que este run não fez. Logá-las aqui faria todo treino parecer resultado
    de uma varredura de 24 candidatos, inclusive os que ninguém varreu.
    """
    return {
        chave: valor
        for chave, valor in cfg.model_dump().items()
        if not chave.startswith(("mlflow_", "tuning_"))
    }


def ajustar(
    cfg: Settings = settings,
) -> tuple[RandomForestClassifier, CategoricalEncoder, dict[str, float]]:
    """Dados -> features -> split -> encode -> treino -> avaliação. Sem MLflow.

    É o miolo do treino, separado do registro de propósito: quem só precisa de um
    modelo ajustado — o modo demonstração do :mod:`churn.demonstracao`, por
    exemplo — não deveria ter de subir um backend de tracking para consegui-lo.

    Devolve o encoder junto porque sem ele o estimador não serve: foi ajustado
    sobre categóricas já codificadas, e a inferência precisa da mesma codificação.
    """
    frame = add_derived_features(
        load_clean(cfg.data_path, cfg.id_column, cfg.target, cfg.quarantine_dir)
    )
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    encoder = CategoricalEncoder().fit(x_train)
    model = build_model(cfg).fit(encoder.transform(x_train), y_train)
    return model, encoder, evaluate(model, encoder.transform(x_test), y_test)


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """O mesmo ajuste, registrado num run do MLflow.

    O que acontece aqui e não em :func:`ajustar` é só o registro: params,
    métricas, linhagem e o pipeline logado.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        # linhagem primeiro: se o treino falhar no meio, o run já sabe de onde veio
        mlflow.set_tags(
            {
                "data_md5": data_md5(ponteiro_de(cfg.data_path)),
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(parametros(cfg))

        model, encoder, metrics = ajustar(cfg)
        mlflow.log_metrics(metrics)

        # O que é logado é o pipeline inteiro (pyfunc), não o estimador solto:
        # um modelo que não aceita a linha como ela sai do sistema de origem não
        # roda em lugar nenhum. MLflow 3 usa `name=`, não `artifact_path=`.
        # A assinatura é inferida da entrada CRUA — é o contrato de quem chama.
        log_pipeline(model, encoder, cfg, pd.read_csv(cfg.data_path, nrows=3))

        # E o pickle local continua: é a saída declarada no dvc.yaml.
        save_artifact(model, encoder, cfg.artifact_path)

    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
