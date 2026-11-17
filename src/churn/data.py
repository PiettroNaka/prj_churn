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

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from churn.quarentena import quarentenar
from churn.schema import COLUNAS_CRU
from churn.validation import DataValidationError, validate_clean, validate_raw

# Colunas esperadas do dataset Telco Customer Churn.
# Vêm do ChurnRawSchema para não existirem duas listas de colunas divergindo.
EXPECTED_COLUMNS: frozenset[str] = COLUNAS_CRU


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV cru do disco."""
    return pd.read_csv(path)


def carregar_origem(data_path: Path, database_url: str | None) -> pd.DataFrame:
    """Lê o dado cru da origem configurada: do banco quando há URL, do arquivo
    quando não.

    O resto do pipeline não sabe de qual das duas veio, e é esse o ponto. O
    contrato roda igual nos dois casos; o que muda é só de onde os bytes saem —
    a mesma régua que já valia quando o CSV passou a ser buscado por HTTP no
    deploy de vitrine.

    O import do :mod:`churn.db` é tardio de propósito: o driver do Postgres vive
    num grupo opcional de dependências, e a imagem de treino que lê de arquivo
    não deve quebrar por não tê-lo instalado.
    """
    if database_url:
        from churn.db import ler, motor

        return ler(motor(database_url))
    return load_raw(data_path)


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


def _com_quarentena(
    df: pd.DataFrame,
    validar: Callable[[pd.DataFrame], pd.DataFrame],
    pasta: Path | None,
) -> pd.DataFrame:
    """Valida e, se reprovar, separa o lote ruim antes de deixar a falha subir.

    A quarentena mora aqui, na orquestração, e não dentro de ``validate_*``: o
    validador responde se o dado serve, sem escrever nada em disco. Quem decide
    o que fazer com um lote reprovado é o pipeline.
    """
    if pasta is None:
        return validar(df)
    try:
        return validar(df)
    except DataValidationError as erro:
        destino = quarentenar(df, erro, pasta)
        raise DataValidationError(
            f"{erro}\n\nLote em quarentena: {destino}",
            failure_cases=erro.failure_cases,
        ) from erro


def load_clean(
    data_path: Path,
    id_column: str,
    target: str,
    quarantine_dir: Path | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Pipeline de dados: carrega -> valida cru -> remove id -> coage -> valida limpo.

    ``database_url`` troca a origem: com ela o dado vem do Postgres, sem ela do
    CSV. Tudo o que vem depois é idêntico — a validação, a limpeza e a segunda
    validação não sabem nem precisam saber de onde o DataFrame saiu.

    ``quarantine_dir`` liga a quarentena do Encontro 4: reprovando, o lote é
    gravado lá antes de a exceção subir. ``None`` desliga, e é o padrão — quem
    escolhe onde o lote ruim mora é quem orquestra (``model.train`` passa
    ``cfg.quarantine_dir``), não esta função.
    """
    df = carregar_origem(data_path, database_url)
    df = _com_quarentena(df, lambda frame: validate(frame, target), quarantine_dir)
    df = drop_identifier(df, id_column)
    df = coerce_total_charges(df)
    return _com_quarentena(df, validate_clean, quarantine_dir)
