"""Carga, limpeza e validação dos dados.

Funções puras (recebem/retornam DataFrame, sem efeito colateral) compostas por
uma orquestração fina em ``load_clean``.

A validação escrita à mão do Encontro 2 foi substituída, no Encontro 4, pelo
contrato declarativo do pandera: os schemas vivem em :mod:`churn.schema` e são
executados por :mod:`churn.validation`. O pipeline é validado **duas vezes**: no
dado cru, antes de qualquer transformação, e no dado já limpo, antes de virar
feature.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.schema import COLUNAS_CRU
from churn.validation import validate_clean, validate_raw

# Colunas esperadas do dataset Telco Customer Churn.
# Vêm do ChurnRawSchema para não existirem duas listas de colunas divergindo.
EXPECTED_COLUMNS: frozenset[str] = COLUNAS_CRU


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV cru do disco."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Valida o dado cru contra o ``ChurnRawSchema``. Levanta ValueError.

    ``target`` é conferido contra o schema: se a configuração apontar para uma
    coluna-alvo que o contrato não declara, isso é erro de config, não de dado.
    """
    if target not in COLUNAS_CRU:
        raise ValueError(
            f"Alvo '{target}' não é uma coluna declarada no schema do dataset."
        )
    return validate_raw(df)


def drop_identifier(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Remove a coluna de identificador (não é feature)."""
    return df.drop(columns=[id_column])


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges (texto, com brancos) em número.

    Os brancos correspondem a clientes com ``tenure == 0`` (novos), cujo total
    gasto é, de fato, 0 — não um número mágico como no script original. O
    ``ChurnSchema`` exige essa relação: zero só é aceito com ``tenure == 0``.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    return out


def load_clean(data_path: Path, id_column: str, target: str) -> pd.DataFrame:
    """Pipeline de dados: carrega -> valida cru -> remove id -> coage -> valida limpo."""
    df = load_raw(data_path)
    df = validate(df, target)
    df = drop_identifier(df, id_column)
    df = coerce_total_charges(df)
    return validate_clean(df)
