"""O contrato traduzido tem de dizer a mesma coisa que o contrato original.

Este módulo existe para que a API e o formulário não mantenham cada um a sua
cópia da lista de categorias. O risco que isso cria é o de a tradução divergir
da fonte — e é exatamente o que se testa aqui.
"""
import pytest

from churn.contrato import (
    FORA_DA_ENTRADA,
    Campo,
    campos_de_entrada,
    cliente_exemplo,
    total_esperado,
)
from churn.schema import COLUNAS_CRU, TOLERANCIA_TOTAL


def test_cobre_o_contrato_menos_id_e_alvo():
    nomes = {campo.nome for campo in campos_de_entrada()}

    assert nomes == COLUNAS_CRU - FORA_DA_ENTRADA


def test_categoricas_trazem_os_valores_do_contrato():
    por_nome = {campo.nome: campo for campo in campos_de_entrada()}

    assert por_nome["Contract"].permitidos == (
        "Month-to-month",
        "One year",
        "Two year",
    )
    assert por_nome["gender"].e_categoria


def test_numericas_trazem_a_faixa_do_contrato():
    por_nome = {campo.nome: campo for campo in campos_de_entrada()}

    assert (por_nome["tenure"].minimo, por_nome["tenure"].maximo) == (0, 72)
    assert por_nome["tenure"].inteiro
    assert not por_nome["tenure"].e_categoria


def test_total_charges_e_numero_e_nao_texto():
    """No CSV cru é texto (brancos de cliente novo); aqui já é número."""
    total = next(c for c in campos_de_entrada() if c.nome == "TotalCharges")

    assert not total.e_categoria
    assert total.minimo == 0.0


@pytest.mark.parametrize("campo", campos_de_entrada(), ids=lambda c: c.nome)
def test_padrao_esta_dentro_do_proprio_dominio(campo: Campo):
    """O formulário abre preenchido; se o padrão já fosse inválido, a primeira
    previsão falharia antes de alguém digitar qualquer coisa."""
    if campo.e_categoria:
        assert campo.padrao in campo.permitidos
    else:
        assert campo.minimo <= campo.padrao <= campo.maximo


def test_cliente_exemplo_respeita_a_relacao_do_contrato():
    """TotalCharges ≈ tenure × MonthlyCharges — a regra mais forte do schema.

    Tomar o meio da faixa de cada campo isoladamente produzia um exemplo
    incoerente (36 meses × 68,50 = 2.466, contra um TotalCharges de 4.284).
    Quem abrisse o formulário veria um aviso de violação antes de mexer em nada.
    """
    cliente = cliente_exemplo()
    esperado = total_esperado(cliente["tenure"], cliente["MonthlyCharges"])

    desvio = abs(cliente["TotalCharges"] - esperado) / esperado

    assert desvio <= TOLERANCIA_TOTAL


def test_cliente_exemplo_e_aceito_pela_api():
    """A junta entre o formulário e a API, verificada nos dois sentidos.

    O padrão do front tem de passar na validação do Pydantic da API. As duas
    pontas leem o mesmo contrato, então o teste só falha se a tradução de um
    dos lados escorregar — que é precisamente o que ele guarda.
    """
    pytest.importorskip("fastapi", reason="fastapi ausente neste ambiente")
    from churn.api import Cliente

    assert Cliente(**cliente_exemplo())
