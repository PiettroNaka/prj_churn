"""Validação de dados como teste — o "JUnit4Data" do Encontro 4.

A validação do Encontro 2 era uma checagem de colunas escrita à mão. Aqui ela
vira um **schema declarativo** (:mod:`churn.schema`) executado como teste: o
contrato quebra alto quando o dado desvia, em vez de deixar o treino seguir em
silêncio com lixo dentro.

Este módulo é a *execução* do contrato — o schema em si mora em
:mod:`churn.schema`. A separação é a mesma de um teste: o schema descreve o que
é um dado válido, o runner decide o que fazer quando não é.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from churn.schema import ChurnRawSchema, ChurnSchema

__all__ = [
    "DataValidationError",
    "validate_clean",
    "validate_raw",
]


class DataValidationError(ValueError):
    """Dado fora do contrato declarado.

    Herda de ``ValueError`` de propósito: quem chama não precisa conhecer o
    pandera para tratar a falha.

    Carrega ``failure_cases`` — o DataFrame de (coluna, regra, valor) que o
    pandera monta no modo ``lazy``. É o que se loga para saber *o que* quebrou,
    sem ter de reler o texto da exceção.
    """

    def __init__(self, mensagem: str, failure_cases: pd.DataFrame | None = None):
        super().__init__(mensagem)
        self.failure_cases = failure_cases


def _aplicar(modelo: type[pa.DataFrameModel], df: pd.DataFrame) -> pd.DataFrame:
    """Roda o contrato em modo ``lazy``: relata TODAS as falhas, não só a primeira.

    Um relatório parcial faz o aluno consertar em rodadas; o completo mostra o
    estrago inteiro de uma vez.
    """
    try:
        return modelo.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as erro:
        raise DataValidationError(
            f"Dataset reprovado no schema '{modelo.Config.name}':\n{erro}",
            failure_cases=getattr(erro, "failure_cases", None),
        ) from erro


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o CSV cru. Levanta :class:`DataValidationError` se desviar."""
    return _aplicar(ChurnRawSchema, df)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame já limpo. Levanta :class:`DataValidationError`."""
    return _aplicar(ChurnSchema, df)
