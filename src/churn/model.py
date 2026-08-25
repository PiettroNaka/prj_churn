"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).
"""
from __future__ import annotations

import pickle
from pathlib import Path

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


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa (reprodutível)."""
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators, random_state=cfg.random_state
    )


def save_artifact(model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path) -> None:
    """Persiste modelo + encoder juntos (a inferência precisa do encoder ajustado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Executa o pipeline: dados -> features -> split -> encode -> treino -> avaliação."""
    frame = add_derived_features(
        load_clean(cfg.data_path, cfg.id_column, cfg.target)
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
    metrics = evaluate(model, encoder.transform(x_test), y_test)

    save_artifact(model, encoder, cfg.artifact_path)
    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
