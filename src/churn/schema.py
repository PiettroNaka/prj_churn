"""O contrato do dataset como código — o ``ChurnSchema`` do Encontro 4.

O schema é uma **classe** (``DataFrameModel``), a mesma intuição do Pydantic do
Encontro 2: lá o contrato é da config, aqui é dos dados. Sendo código, ele
versiona, é revisado em PR e roda no CI.

São dois contratos, espelhando o Data Pipeline (exploration → validation →
wrangling):

- :class:`ChurnRawSchema` valida o CSV **como ele sai do disco**, antes de
  qualquer transformação. É onde ``TotalCharges`` ainda é texto e pode vir em
  branco.
- :class:`ChurnSchema` valida o DataFrame **depois** de ``load_clean``, quando o
  identificador já saiu e ``TotalCharges`` virou número.

Validar nos dois pontos é o que sustenta o ``coerce=True`` da Config: a coerção
é comodidade que esconde sujeira, e o segundo schema cobra o dado num tipo já
definido, depois de a limpeza do ``data.py`` ter passado.

As faixas não são chute — saem do profiling (``scripts/profiling.py``):

===============  ==================  ================================
coluna           observado           faixa no contrato
===============  ==================  ================================
tenure           0–72                0–72 (teto do produto)
MonthlyCharges   18,25–118,75        18–119 (o observado, arredondado)
TotalCharges     branco em 11 linhas ≥ 0, e branco só com tenure 0
===============  ==================  ================================

Nota de projeto — por que nada é descartado:

    As 11 linhas com ``TotalCharges`` em branco têm, todas, ``tenure == 0``.
    Não são dado corrompido: são clientes que assinaram e ainda não fecharam o
    primeiro ciclo de cobrança, cujo total acumulado é 0 de fato. Quatro delas
    são churners — descartá-las jogaria fora sinal da classe minoritária. Em vez
    de excluir, o schema **exige** essa relação (ver
    ``branco_apenas_para_cliente_novo``): se um branco aparecer em um cliente
    com ``tenure > 0``, aí sim é corrupção, e a validação falha.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

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

ADD_ONS_INTERNET = SIM_NAO + [SEM_INTERNET]

#: Contrato de 72 meses é o teto do produto — nada acima disso é plausível.
TENURE_MAX = 72

#: Tolerância da relação TotalCharges ≈ tenure × MonthlyCharges. O profiling dá
#: 1,5% de desvio mediano e 3,0% no pior caso; 10% deixa folga para reajuste no
#: meio do contrato e mês proporcional, sem abrir a porta para coluna invertida
#: ou valor truncado.
TOLERANCIA_TOTAL = 0.10

#: Faixa da mensalidade, como o encontro declara: 18–119. São o piso e o teto
#: observados no profiling (18,25 e 118,75), arredondados para fora. É uma faixa
#: colada no dado — pega 999, -1 e 1e6, e também reprova o dataset inteiro no dia
#: em que um plano novo custar 130. Apertar assim é decisão consciente: o dataset
#: do curso é fechado, e faixa larga demais não ensina nada.
MONTHLY_MIN = 18.0
MONTHLY_MAX = 119.0


# --------------------------------------------------------------------------
# checagens que envolvem mais de uma coluna
# --------------------------------------------------------------------------


def _texto_numerico_ou_branco(serie: pd.Series) -> pd.Series:
    """No CSV cru, TotalCharges é texto: ou converte para número, ou é branco."""
    limpo = serie.astype("string").str.strip()
    return limpo.eq("") | pd.to_numeric(limpo, errors="coerce").notna()


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


# --------------------------------------------------------------------------
# o que os dois contratos têm em comum
# --------------------------------------------------------------------------


class _ChurnBase(pa.DataFrameModel):
    """Colunas e regras que valem no dado cru e no dado limpo.

    Herança de ``DataFrameModel``: o cru e o limpo diferem só em ``customerID``
    e no tipo de ``TotalCharges``. Tudo o mais é declarado uma vez só.
    """

    # ---- categóricas: o domínio fechado de cada coluna ----
    gender: Series[str] = pa.Field(isin=["Female", "Male"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=SIM_NAO)
    Dependents: Series[str] = pa.Field(isin=SIM_NAO)
    PhoneService: Series[str] = pa.Field(isin=SIM_NAO)
    MultipleLines: Series[str] = pa.Field(isin=SIM_NAO + [SEM_TELEFONE])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    OnlineBackup: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    DeviceProtection: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    TechSupport: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    StreamingTV: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    StreamingMovies: Series[str] = pa.Field(isin=ADD_ONS_INTERNET)
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=SIM_NAO)
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    # ---- numéricas: faixa ----
    tenure: Series[int] = pa.Field(
        ge=0, le=TENURE_MAX, description=f"Meses de contrato, 0 a {TENURE_MAX}."
    )
    MonthlyCharges: Series[float] = pa.Field(
        ge=MONTHLY_MIN,
        le=MONTHLY_MAX,
        description=f"Mensalidade corrente, de {MONTHLY_MIN:.0f} a {MONTHLY_MAX:.0f}.",
    )

    # ---- alvo: nunca nulo, ou o rótulo do treino é inventado ----
    Churn: Series[str] = pa.Field(isin=SIM_NAO, nullable=False)

    class Config:
        # A Config do encontro, literal.
        #
        # coerce: o schema converte o tipo ao validar. Comodidade com preço — o
        # próprio encontro avisa que coerção mascara sujeira de verdade. O
        # antídoto aqui é validar em DOIS pontos (cru e limpo): o que a coerção
        # esconder no primeiro, o segundo cobra num tipo já definido.
        coerce = True
        # strict=False: coluna a mais é tolerada. Um extrator que passa a mandar
        # uma coluna nova não derruba o pipeline; o contrato cobra o que declara,
        # não a ausência do que não declara.
        strict = False

    @pa.dataframe_check(
        name="servicos_de_internet_coerentes",
        error=f"Cliente sem InternetService tem add-on diferente de '{SEM_INTERNET}'.",
    )
    def servicos_de_internet_coerentes(cls, df: pd.DataFrame) -> bool:
        return _servicos_de_internet_coerentes(df)

    @pa.dataframe_check(
        name="multiplas_linhas_coerente",
        error=f"Cliente sem PhoneService tem MultipleLines diferente de '{SEM_TELEFONE}'.",
    )
    def multiplas_linhas_coerente(cls, df: pd.DataFrame) -> bool:
        return _multiplas_linhas_coerente(df)


# --------------------------------------------------------------------------
# contrato do CSV cru
# --------------------------------------------------------------------------


class ChurnRawSchema(_ChurnBase):
    """O CSV como ele sai do disco: com identificador e TotalCharges em texto."""

    customerID: Series[str] = pa.Field(
        str_matches=r"^\d{4,}-[A-Z]{4}$",
        unique=True,
        description="Identificador do cliente; não é feature.",
    )
    TotalCharges: Series[str] = pa.Field(
        description="Vem como texto por causa dos brancos de cliente novo."
    )

    class Config:
        name = "churn_raw"

    @pa.check(
        "TotalCharges",
        name="texto_numerico_ou_branco",
        error="TotalCharges tem texto que não é número nem branco.",
    )
    def texto_numerico_ou_branco(cls, serie: Series[str]) -> Series[bool]:
        return _texto_numerico_ou_branco(serie)

    @pa.dataframe_check(
        name="branco_apenas_para_cliente_novo",
        error="TotalCharges em branco em cliente com tenure > 0 — isso é "
        "corrupção, não cliente novo.",
    )
    def branco_apenas_para_cliente_novo(cls, df: pd.DataFrame) -> bool:
        """O branco é aceito como semântica (cliente novo), nunca como sujeira."""
        branco = df["TotalCharges"].astype("string").str.strip().eq("")
        return bool((df.loc[branco, "tenure"] == 0).all())


# --------------------------------------------------------------------------
# contrato do DataFrame limpo — o que o treino consome
# --------------------------------------------------------------------------


class ChurnSchema(_ChurnBase):
    """O DataFrame depois de ``load_clean``: sem identificador, TotalCharges float."""

    TotalCharges: Series[float] = pa.Field(
        ge=0, nullable=False, description="Já coagido; 0 apenas para cliente novo."
    )

    class Config:
        name = "churn_clean"

    @pa.dataframe_check(
        name="zero_apenas_para_cliente_novo",
        error="TotalCharges igual a 0 em cliente com tenure > 0.",
    )
    def zero_apenas_para_cliente_novo(cls, df: pd.DataFrame) -> bool:
        """Depois da coerção, o mesmo contrato do branco: total 0 só com tenure 0."""
        return bool((df.loc[df["TotalCharges"] == 0, "tenure"] == 0).all())

    @pa.dataframe_check(
        name="total_coerente_com_tenure_e_mensalidade",
        error="TotalCharges longe de tenure × MonthlyCharges "
        f"(tolerância de {TOLERANCIA_TOTAL:.0%}) em cliente com tenure > 1.",
    )
    def total_coerente_com_tenure_e_mensalidade(cls, df: pd.DataFrame) -> bool:
        """A relação entre as três colunas numéricas — a regra mais forte do contrato.

        Uma faixa isolada aprova ``TotalCharges = 20`` num cliente de 70 meses
        pagando 100 por mês; a relação não. É o que pega coluna invertida, valor
        truncado e unidade errada — erros que nenhum ``ge``/``le`` enxerga.
        """
        cobrado = df["tenure"] > 1
        if not cobrado.any():
            return True
        esperado = df.loc[cobrado, "tenure"] * df.loc[cobrado, "MonthlyCharges"]
        desvio = (df.loc[cobrado, "TotalCharges"] - esperado).abs() / esperado
        return bool((desvio <= TOLERANCIA_TOTAL).all())


#: Colunas declaradas no contrato do dado cru. Fonte única — quem precisa saber
#: quais colunas o dataset tem pergunta ao schema, não a uma lista paralela.
COLUNAS_CRU: frozenset[str] = frozenset(ChurnRawSchema.to_schema().columns)
