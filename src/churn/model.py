"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).

A partir do Encontro 6 cada treino é um **run do MLflow**: params, métricas, o
modelo empacotado e as tags de linhagem ficam registrados. O ``print`` continua
existindo para o uso interativo, mas já não é a única memória do experimento.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import Settings, model_params, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.lineage import tags as lineage_tags


def setup_tracking(cfg: Settings = settings) -> None:
    """Aponta o MLflow para o backend do projeto e seleciona o experimento.

    Chamado por todo entrypoint que registra runs. Fixar a URI em código, e não
    só em variável de ambiente, é o que impede runs de se espalharem por
    backends diferentes conforme o shell de quem executa.
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa (reprodutível)."""
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )


def save_artifact(model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path) -> None:
    """Persiste modelo + encoder juntos (a inferência precisa do encoder ajustado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def prepare(
    cfg: Settings = settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, CategoricalEncoder]:
    """Dados -> features -> split -> encode. Devolve tudo pronto para o ``fit``.

    Extraído do ``train`` para que o grid de experimentos reuse exatamente o
    mesmo preparo: comparar runs que prepararam o dado de formas diferentes não
    compara hiperparâmetro nenhum.
    """
    frame = add_derived_features(load_clean(cfg.data_path, cfg.id_column, cfg.target))
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    encoder = CategoricalEncoder().fit(x_train)
    return encoder.transform(x_train), encoder.transform(x_test), y_train, y_test, encoder


def log_model(model: RandomForestClassifier, x_train: pd.DataFrame) -> None:
    """Empacota o modelo no formato MLflow Models, com contrato de entrada.

    ``signature`` documenta o schema e é validado no serving; ``input_example``
    dá um caso real; ``cloudpickle`` evita o atrito do skops que o MLflow 3
    escolhe por padrão e que reclama ao recarregar o modelo.
    """
    assinatura = infer_signature(x_train, model.predict(x_train))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=assinatura,
        input_example=x_train.head(3),
        serialization_format="cloudpickle",
    )


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Executa o pipeline e registra o treino como um run do MLflow."""
    setup_tracking(cfg)
    x_train, x_test, y_train, y_test, encoder = prepare(cfg)

    with mlflow.start_run(run_name="rf"):
        mlflow.set_tags(lineage_tags())
        mlflow.log_params(model_params(cfg))

        model = build_model(cfg).fit(x_train, y_train)
        metrics = evaluate(model, x_test, y_test)

        mlflow.log_metrics(metrics)
        log_model(model, x_train)

    save_artifact(model, encoder, cfg.artifact_path)
    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
