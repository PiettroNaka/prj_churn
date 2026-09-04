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

import logging
import warnings

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from churn.schema import ChurnRawSchema, ChurnSchema

logger = logging.getLogger(__name__)

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

    Nem toda violação é erro. As colunas de catálogo (ver
    ``CATEGORIAS_DE_CATALOGO``) declaram ``raise_warning``: uma categoria nova
    ali vira ``SchemaWarning``, o pipeline segue, e o aviso é registrado e
    reemitido — alerta que ninguém lê é validação que não valida. Alerta e erro
    convivem no mesmo passe: o relatório traz os dois.
    """
    reprovacao: SchemaError | SchemaErrors | None = None
    validado = df

    # Os alertas são registrados FORA deste bloco de propósito: reemitir um
    # aviso aqui dentro o gravaria de volta na lista sendo percorrida, e o
    # registro nunca terminaria.
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        try:
            validado = modelo.validate(df, lazy=True)
        except (SchemaError, SchemaErrors) as erro:
            reprovacao = erro
        avisos = list(capturados)

    _registrar(avisos)

    if reprovacao is not None:
        raise DataValidationError(
            f"Dataset reprovado no schema '{modelo.Config.name}':\n{reprovacao}",
            failure_cases=getattr(reprovacao, "failure_cases", None),
        ) from reprovacao

    return validado


def _registrar(avisos: list[warnings.WarningMessage]) -> None:
    """Loga cada alerta e o reemite, para quem chamou poder capturá-lo."""
    for aviso in avisos:
        logger.warning(
            "ALERTA de contrato (o pipeline segue): %s — categoria nova de "
            "catálogo é evento de negócio, não dado corrompido. Se tiver de "
            "reprovar, o que muda é o schema.",
            aviso.message,
        )
        warnings.warn(aviso.message, aviso.category, stacklevel=3)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o CSV cru. Levanta :class:`DataValidationError` se desviar."""
    return _aplicar(ChurnRawSchema, df)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame já limpo. Levanta :class:`DataValidationError`."""
    return _aplicar(ChurnSchema, df)
