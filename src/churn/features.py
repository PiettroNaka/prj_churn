"""Engenharia de features.

Transformações *stateless* são funções puras. O encoding de categóricas guarda
estado (as categorias vistas) — por isso vive numa classe com fit/transform,
que aprende SOMENTE no treino e evita vazamento de dados.

Nota de projeto: não há normalização de escala aqui de propósito. Random Forest
não é sensível à escala das features; a normalização por números mágicos do
script original era desnecessária.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas (pura, row-wise — segura antes do split)."""
    out = df.copy()
    out["gasto_por_mes"] = out["TotalCharges"] / (out["tenure"] + 1)
    return out


def split_features_target(
    df: pd.DataFrame, target: str, positive_label: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) de y (alvo binário 0/1)."""
    features = df.drop(columns=[target])
    labels = (df[target] == positive_label).astype(int)
    return features, labels


class CategoricalEncoder:
    """Codifica colunas categóricas em inteiros.

    Aprende as categorias no ``fit`` (apenas com o treino) e as aplica no
    ``transform`` de treino e teste — o encapsulamento é o que impede o
    vazamento que existia no script monolítico.
    """

    def __init__(self) -> None:
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame) -> "CategoricalEncoder":
        self._columns = features.select_dtypes(include="object").columns.tolist()
        self._encoder.fit(features[self._columns])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out[self._columns] = self._encoder.transform(out[self._columns])
        return out

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
