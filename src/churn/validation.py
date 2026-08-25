"""Validação de dados como teste — o "JUnit4Data" do Encontro 4.

A validação do Encontro 2 era uma checagem de colunas escrita à mão. Aqui ela
vira um **schema declarativo**: o contrato do dataset fica explícito, versionado
e executável, e quebra alto quando o dado desvia — em vez de deixar o treino
seguir em silêncio com lixo dentro.

Dois schemas, espelhando o Data Pipeline (exploration → validation → wrangling):

- :data:`RAW_SCHEMA` valida o CSV **como ele sai do disco**, antes de qualquer
  transformação. É onde o ``TotalCharges`` ainda é texto e pode vir em branco.
- :data:`CLEAN_SCHEMA` valida o DataFrame **depois** de ``load_clean``, quando o
  identificador já saiu e ``TotalCharges`` virou número.

Nota de projeto — por que nada é descartado:

    As 11 linhas com ``TotalCharges`` em branco têm, todas, ``tenure == 0``.
    Não são dado corrompido: são clientes que assinaram e ainda não fecharam o
    primeiro ciclo de cobrança, cujo total acumulado é 0 de fato. Quatro delas
    são churners — descartá-las jogaria fora sinal da classe minoritária. Em vez
    de excluir, o schema **exige** essa relação (ver
    ``_branco_apenas_para_cliente_novo``): se um branco aparecer em um cliente
    com ``tenure > 0``, aí sim é corrupção, e a validação falha.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

SIM_NAO = ["Yes", "No"]
SEM_INTERNET = "No internet service"
SEM_TELEFONE = "No phone service"

#: Colunas de serviço que só existem para quem tem internet contratada.
SERVICOS_INTERNET = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

#: Contrato de 72 meses é o teto do produto — nada acima disso é plausível.
TENURE_MAX = 72


class DataValidationError(ValueError):
    """Dado fora do contrato declarado.

    Herda de ``ValueError`` de propósito: quem chama não precisa conhecer o
    pandera para tratar a falha.
    """


# --------------------------------------------------------------------------
# checagens que envolvem mais de uma coluna
# --------------------------------------------------------------------------


def _texto_numerico_ou_branco(serie: pd.Series) -> pd.Series:
    """No CSV cru, TotalCharges é texto: ou converte para número, ou é branco."""
    limpo = serie.astype("string").str.strip()
    return limpo.eq("") | pd.to_numeric(limpo, errors="coerce").notna()


def _branco_apenas_para_cliente_novo(df: pd.DataFrame) -> bool:
    """TotalCharges em branco só se justifica com tenure == 0.

    Esta é a regra que substitui o descarte das 11 linhas: o branco é aceito
    como semântica (cliente novo), nunca como sujeira genérica.
    """
    branco = df["TotalCharges"].astype("string").str.strip().eq("")
    return bool((df.loc[branco, "tenure"] == 0).all())


def _zero_apenas_para_cliente_novo(df: pd.DataFrame) -> bool:
    """Depois da coerção, o mesmo contrato: total 0 só para tenure == 0."""
    return bool((df.loc[df["TotalCharges"] == 0, "tenure"] == 0).all())


def _servicos_de_internet_coerentes(df: pd.DataFrame) -> bool:
    """Sem internet contratada, os 6 add-ons têm de dizer 'No internet service'."""
    sem_internet = df["InternetService"] == "No"
    if not sem_internet.any():
        return True
    return all(
        bool((df.loc[sem_internet, coluna] == SEM_INTERNET).all())
        for coluna in SERVICOS_INTERNET
    )


def _multiplas_linhas_coerente(df: pd.DataFrame) -> bool:
    """Sem telefone, MultipleLines tem de dizer 'No phone service'."""
    sem_telefone = df["PhoneService"] == "No"
    if not sem_telefone.any():
        return True
    return bool((df.loc[sem_telefone, "MultipleLines"] == SEM_TELEFONE).all())


def _total_compativel_com_mensalidade(df: pd.DataFrame) -> bool:
    """TotalCharges não pode ser menor que uma mensalidade se já houve cobrança.

    Pega inversão de colunas e truncamento de valores — erros que passam por
    qualquer checagem de faixa isolada.
    """
    cobrado = df["tenure"] > 1
    if not cobrado.any():
        return True
    return bool((df.loc[cobrado, "TotalCharges"] >= df.loc[cobrado, "MonthlyCharges"]).all())


_CHECAGENS_CRUZADAS = [
    pa.Check(
        _servicos_de_internet_coerentes,
        name="servicos_de_internet_coerentes",
        error="Cliente sem InternetService tem add-on de internet diferente de "
        f"'{SEM_INTERNET}'.",
    ),
    pa.Check(
        _multiplas_linhas_coerente,
        name="multiplas_linhas_coerente",
        error=f"Cliente sem PhoneService tem MultipleLines diferente de '{SEM_TELEFONE}'.",
    ),
]


# --------------------------------------------------------------------------
# colunas comuns aos dois schemas
# --------------------------------------------------------------------------


def _colunas_categoricas() -> dict[str, pa.Column]:
    colunas: dict[str, pa.Column] = {
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(SIM_NAO)),
        "Dependents": pa.Column(str, pa.Check.isin(SIM_NAO)),
        "PhoneService": pa.Column(str, pa.Check.isin(SIM_NAO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(SIM_NAO + [SEM_TELEFONE])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "Contract": pa.Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(SIM_NAO)),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
    }
    for coluna in SERVICOS_INTERNET:
        colunas[coluna] = pa.Column(str, pa.Check.isin(SIM_NAO + [SEM_INTERNET]))
    return colunas


def _colunas_numericas() -> dict[str, pa.Column]:
    return {
        "tenure": pa.Column(
            int,
            pa.Check.in_range(0, TENURE_MAX),
            description=f"Meses de contrato, 0 a {TENURE_MAX}.",
        ),
        "MonthlyCharges": pa.Column(
            float,
            pa.Check.gt(0),
            description="Mensalidade corrente; sempre positiva.",
        ),
    }


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------

#: Contrato do CSV como ele sai do disco. ``strict`` rejeita coluna a mais.
RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(
            str,
            pa.Check.str_matches(r"^\d{4,}-[A-Z]{4}$"),
            unique=True,
            description="Identificador do cliente; não é feature.",
        ),
        **_colunas_categoricas(),
        **_colunas_numericas(),
        "TotalCharges": pa.Column(
            str,
            pa.Check(
                _texto_numerico_ou_branco,
                name="texto_numerico_ou_branco",
                error="TotalCharges tem texto que não é número nem branco.",
            ),
            description="Vem como texto por causa dos brancos de cliente novo.",
        ),
        "Churn": pa.Column(str, pa.Check.isin(SIM_NAO), nullable=False),
    },
    checks=[
        *_CHECAGENS_CRUZADAS,
        pa.Check(
            _branco_apenas_para_cliente_novo,
            name="branco_apenas_para_cliente_novo",
            error="TotalCharges em branco em cliente com tenure > 0 — isso é "
            "corrupção, não cliente novo.",
        ),
    ],
    strict=True,
    name="churn_raw",
)

#: Contrato do DataFrame depois de ``load_clean``.
CLEAN_SCHEMA = pa.DataFrameSchema(
    columns={
        **_colunas_categoricas(),
        **_colunas_numericas(),
        "TotalCharges": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=False,
            description="Já coagido; 0 apenas para cliente novo.",
        ),
        "Churn": pa.Column(str, pa.Check.isin(SIM_NAO), nullable=False),
    },
    checks=[
        *_CHECAGENS_CRUZADAS,
        pa.Check(
            _zero_apenas_para_cliente_novo,
            name="zero_apenas_para_cliente_novo",
            error="TotalCharges igual a 0 em cliente com tenure > 0.",
        ),
        pa.Check(
            _total_compativel_com_mensalidade,
            name="total_compativel_com_mensalidade",
            error="TotalCharges menor que uma mensalidade em cliente com tenure > 1.",
        ),
    ],
    strict=True,
    name="churn_clean",
)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def _aplicar(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    """Roda o schema em modo ``lazy``: relata TODAS as falhas, não só a primeira.

    Um relatório parcial faz o aluno consertar em rodadas; o completo mostra o
    estrago inteiro de uma vez.
    """
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as erro:
        raise DataValidationError(
            f"Dataset reprovado no schema '{schema.name}':\n{erro}"
        ) from erro


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o CSV cru. Levanta :class:`DataValidationError` se desviar."""
    return _aplicar(RAW_SCHEMA, df)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame já limpo. Levanta :class:`DataValidationError`."""
    return _aplicar(CLEAN_SCHEMA, df)
