"""Avaliação do modelo — métricas como dicionário, sem efeito colateral."""
from __future__ import annotations

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    model: ClassifierMixin, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    """Calcula um conjunto de métricas de classificação binária."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
