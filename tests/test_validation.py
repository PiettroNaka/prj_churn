"""Testes do Encontro 4: quebrar de propósito e ver a validação pegar.

Cada teste parte de um frame que o schema aprova e suja **uma** coisa. Se algum
destes passar a não levantar, a validação regrediu e um dado ruim voltou a ter
caminho livre até o treino.
"""
from pathlib import Path

import pandas as pd
import pytest

from pandera.errors import SchemaWarning

from churn.validation import (
    DataValidationError,
    validate_clean,
    validate_raw,
)

DATASET = Path("data/churn.csv")


# --------------------------------------------------------------------------
# a linha de base tem de passar — sem isso, os testes de falha não provam nada
# --------------------------------------------------------------------------


def test_amostra_valida_passa_no_schema_cru(raw_frame):
    assert validate_raw(raw_frame) is not None


def test_amostra_valida_passa_no_schema_limpo(clean_frame):
    assert validate_clean(clean_frame) is not None


# --------------------------------------------------------------------------
# schema: colunas
# --------------------------------------------------------------------------


def test_coluna_faltando(raw_frame):
    with pytest.raises(DataValidationError):
        validate_raw(raw_frame.drop(columns=["Contract"]))


def test_coluna_extra_e_tolerada(raw_frame):
    """``strict=False`` (a Config do encontro): coluna a mais não reprova.

    O contrato cobra o que declara. Um extrator que passa a mandar uma coluna
    nova não derruba o pipeline — e é por isso que a coluna faltando, logo
    acima, continua sendo erro: falta é quebra de contrato, sobra não.
    """
    sujo = raw_frame.assign(coluna_intrusa=1)
    assert validate_raw(sujo) is not None


def test_identificador_duplicado(raw_frame):
    sujo = raw_frame.copy()
    sujo.loc[1, "customerID"] = sujo.loc[0, "customerID"]
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


# --------------------------------------------------------------------------
# schema: categorias
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coluna", "valor"),
    [
        ("gender", "F"),
        ("Contract", "Vitalício"),
        ("Churn", "Sim"),
        ("OnlineSecurity", "Talvez"),
        ("MultipleLines", "Três"),
    ],
)
def test_categoria_fora_do_dominio_reprova(raw_frame, coluna, valor):
    """Alvo, binários e derivados: valor novo ali é corrupção, não lançamento.

    ``Contract`` entra aqui apesar de ser catálogo: é a feature nº 1 do modelo
    (importância 0,175), e um valor novo muda o que o modelo significa.
    """
    sujo = raw_frame.copy()
    sujo.loc[0, coluna] = valor
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


@pytest.mark.parametrize(
    ("coluna", "valor"),
    [("PaymentMethod", "Pix"), ("InternetService", "5G")],
)
def test_categoria_nova_de_catalogo_e_alerta(raw_frame, coluna, valor):
    """Catálogo comercial: o negócio lançou algo, o dado está certo.

    O pipeline segue e o alerta sai — reprovar aqui seria derrubar produção por
    causa de um produto novo.
    """
    novo = raw_frame.copy()
    novo.loc[0, coluna] = valor

    with pytest.warns(SchemaWarning, match=valor):
        validado = validate_raw(novo)

    assert validado.loc[0, coluna] == valor


def test_alerta_e_erro_convivem_no_mesmo_relatorio(raw_frame):
    """Uma linha com as duas coisas: o erro reprova, o alerta é emitido junto."""
    sujo = raw_frame.copy()
    sujo.loc[0, "PaymentMethod"] = "Pix"
    sujo.loc[0, "tenure"] = 999

    with pytest.warns(SchemaWarning, match="Pix"):
        with pytest.raises(DataValidationError) as erro:
            validate_raw(sujo)

    casos = erro.value.failure_cases
    assert "tenure" in set(casos["column"])
    assert "PaymentMethod" not in set(casos["column"])


def test_alvo_nulo(raw_frame):
    sujo = raw_frame.copy()
    sujo.loc[2, "Churn"] = None
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


# --------------------------------------------------------------------------
# schema: faixas numéricas
# --------------------------------------------------------------------------


@pytest.mark.parametrize("valor", [-1, 73, 999])
def test_tenure_fora_da_faixa(raw_frame, valor):
    sujo = raw_frame.copy()
    sujo.loc[0, "tenure"] = valor
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


@pytest.mark.parametrize("valor", [0.0, -10.0, 999.0])
def test_mensalidade_fora_da_faixa(raw_frame, valor):
    """Zero, negativo e absurdo: a faixa cobre os dois lados, não só o piso."""
    sujo = raw_frame.copy()
    sujo.loc[0, "MonthlyCharges"] = valor
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


def test_coerce_converte_o_tipo_em_silencio(raw_frame):
    """O preço do ``coerce=True``: tipo errado vira tipo certo, sem reclamar.

    ``tenure = 12.7`` não é reprovado, é truncado para 12. Está aqui como teste
    para o custo ficar versionado e visível: é exatamente a "sujeira mascarada"
    de que o encontro avisa, e a razão de o pipeline validar duas vezes.
    """
    quase = raw_frame.copy()
    quase["tenure"] = quase["tenure"].astype(float)
    quase.loc[0, "tenure"] = 12.7

    validado = validate_raw(quase)

    assert validado.loc[0, "tenure"] == 12


def test_senior_citizen_fora_do_binario(raw_frame):
    sujo = raw_frame.copy()
    sujo.loc[0, "SeniorCitizen"] = 2
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


# --------------------------------------------------------------------------
# a regra que substitui o descarte das 11 linhas
# --------------------------------------------------------------------------


def test_total_em_branco_e_aceito_quando_o_cliente_e_novo(raw_frame):
    """tenure == 0 com TotalCharges em branco é semântica, não sujeira."""
    assert raw_frame.loc[1, "tenure"] == 0
    assert raw_frame.loc[1, "TotalCharges"].strip() == ""
    assert validate_raw(raw_frame) is not None


def test_total_em_branco_e_rejeitado_quando_o_cliente_e_antigo(raw_frame):
    """O mesmo branco em quem já pagou 12 meses é corrupção — tem de falhar."""
    sujo = raw_frame.copy()
    sujo.loc[0, "TotalCharges"] = " "
    assert sujo.loc[0, "tenure"] > 0
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


def test_total_zerado_em_cliente_antigo(clean_frame):
    sujo = clean_frame.copy()
    sujo.loc[0, "TotalCharges"] = 0.0
    with pytest.raises(DataValidationError):
        validate_clean(sujo)


def test_total_com_texto_lixo(raw_frame):
    sujo = raw_frame.copy()
    sujo.loc[0, "TotalCharges"] = "seiscentos"
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


def test_total_negativo(clean_frame):
    sujo = clean_frame.copy()
    sujo.loc[0, "TotalCharges"] = -1.0
    with pytest.raises(DataValidationError):
        validate_clean(sujo)


def test_total_menor_que_uma_mensalidade(clean_frame):
    """Pega colunas invertidas e valores truncados, que passam por faixa isolada."""
    sujo = clean_frame.copy()
    sujo.loc[0, "TotalCharges"] = sujo.loc[0, "MonthlyCharges"] - 1
    with pytest.raises(DataValidationError):
        validate_clean(sujo)


@pytest.mark.parametrize("fator", [0.5, 1.5])
def test_total_longe_do_produto_tenure_x_mensalidade(clean_frame, fator):
    """Metade ou uma vez e meia o esperado: fora da tolerância, reprova."""
    sujo = clean_frame.copy()
    esperado = sujo.loc[0, "tenure"] * sujo.loc[0, "MonthlyCharges"]
    sujo.loc[0, "TotalCharges"] = esperado * fator
    with pytest.raises(DataValidationError):
        validate_clean(sujo)


def test_total_dentro_da_tolerancia_passa(clean_frame):
    """Reajuste no meio do contrato não pode reprovar o dataset inteiro."""
    quase = clean_frame.copy()
    esperado = quase.loc[0, "tenure"] * quase.loc[0, "MonthlyCharges"]
    quase.loc[0, "TotalCharges"] = esperado * 1.05
    assert validate_clean(quase) is not None


# --------------------------------------------------------------------------
# coerência entre colunas
# --------------------------------------------------------------------------


def test_add_on_de_internet_sem_internet_contratada(raw_frame):
    sujo = raw_frame.copy()
    assert sujo.loc[2, "InternetService"] == "No"
    sujo.loc[2, "StreamingTV"] = "Yes"
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


def test_multiplas_linhas_sem_telefone(raw_frame):
    sujo = raw_frame.copy()
    assert sujo.loc[1, "PhoneService"] == "No"
    sujo.loc[1, "MultipleLines"] = "Yes"
    with pytest.raises(DataValidationError):
        validate_raw(sujo)


# --------------------------------------------------------------------------
# o dataset real (pulado quando o CSV não está presente — ele não vai no Git)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATASET.exists(), reason="data/churn.csv não versionado")
def test_dataset_real_passa_nos_dois_schemas():
    from churn.data import coerce_total_charges, drop_identifier

    cru = pd.read_csv(DATASET)
    validate_raw(cru)
    validate_clean(coerce_total_charges(drop_identifier(cru, "customerID")))


@pytest.mark.skipif(not DATASET.exists(), reason="data/churn.csv não versionado")
def test_dataset_real_com_linha_adulterada_reprova():
    """A tarefa do Encontro 4: quebrar de propósito e ver o contrato pegar.

    Duas quebras na MESMA linha, para provar o ``lazy=True``: o relatório traz
    as duas de uma vez, e não só a primeira que o pandera encontrou.
    """
    sujo = pd.read_csv(DATASET)
    sujo.loc[0, "tenure"] = 999
    sujo.loc[0, "Contract"] = "Vitalício"

    with pytest.raises(DataValidationError) as erro:
        validate_raw(sujo)

    casos = erro.value.failure_cases
    assert {"tenure", "Contract"} <= set(casos["column"])
    assert {999, "Vitalício"} <= set(casos["failure_case"])
