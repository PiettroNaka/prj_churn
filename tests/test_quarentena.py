"""A terceira ação do Encontro 4: o lote ruim tem para onde ir.

Falhar cedo e logar os ``failure_cases`` impedem o estrago; quarentenar é o que
permite consertar. Estes testes cobram que o lote reprovado fique em disco — e
que o lote bom não deixe rastro nenhum.
"""
from pathlib import Path

import pandas as pd
import pytest

from churn.data import load_clean
from churn.validation import DataValidationError


def _csv(frame: pd.DataFrame, destino: Path) -> Path:
    caminho = destino / "churn.csv"
    frame.to_csv(caminho, index=False)
    return caminho


def test_lote_reprovado_vai_para_a_quarentena(raw_frame, tmp_path):
    sujo = raw_frame.copy()
    sujo.loc[0, "tenure"] = 999
    origem = _csv(sujo, tmp_path)
    pasta = tmp_path / "quarentena"

    with pytest.raises(DataValidationError) as erro:
        load_clean(origem, "customerID", "Churn", pasta)

    lotes = list(pasta.iterdir())
    assert len(lotes) == 1, "cada reprovação gera um lote datado"

    linhas = pd.read_csv(lotes[0] / "linhas.csv")
    assert 999 in linhas["tenure"].values
    assert linhas["customerID"].iloc[0] == sujo.loc[0, "customerID"]

    relatorio = (lotes[0] / "relatorio.txt").read_text(encoding="utf-8")
    assert "tenure" in relatorio
    assert "999" in relatorio

    # quem trata a exceção precisa saber onde o lote foi parar
    assert str(lotes[0]) in str(erro.value)


def test_lote_bom_nao_deixa_rastro(raw_frame, tmp_path):
    origem = _csv(raw_frame, tmp_path)
    pasta = tmp_path / "quarentena"

    load_clean(origem, "customerID", "Churn", pasta)

    assert not pasta.exists()


def test_quarentena_pode_ser_desligada(raw_frame, tmp_path):
    """``None`` mantém a validação e não escreve nada — o modo dos testes."""
    sujo = raw_frame.copy()
    sujo.loc[0, "Churn"] = "Talvez"
    origem = _csv(sujo, tmp_path)

    with pytest.raises(DataValidationError):
        load_clean(origem, "customerID", "Churn", None)

    assert list(tmp_path.iterdir()) == [origem]
