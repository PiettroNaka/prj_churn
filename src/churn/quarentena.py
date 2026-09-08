"""Quarentena: separar o lote ruim em vez de treinar com ele.

O encontro dá três coisas a fazer quando o contrato reprova — falhar cedo, logar
os ``failure_cases`` e **quarentenar o lote**. As duas primeiras impedem o
estrago; a terceira é a que permite consertar: sem guardar o dado que reprovou,
o relatório vira uma reclamação sobre um arquivo que já foi embora.

O que fica em disco, por lote:

    artifacts/quarentena/<timestamp>/linhas.csv    as linhas que violaram algo
    artifacts/quarentena/<timestamp>/relatorio.txt coluna, regra, valor, e o índice

Quarentena não substitui a falha: o pipeline continua parando. Ela existe para o
lote ruim ter para onde ir — e para a próxima pergunta ("o que exatamente veio
errado?") ter resposta sem depender de alguém ter copiado o console.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churn.validation import DataValidationError

COLUNAS_DO_RELATORIO = ["index", "column", "check", "failure_case"]


def _linhas_ofensivas(df: pd.DataFrame, casos: pd.DataFrame | None) -> pd.DataFrame:
    """As linhas apontadas pelos ``failure_cases``, quando dá para apontá-las.

    Checagens de coluna trazem o índice da linha; as de DataFrame inteiro (as
    de coerência) não têm índice — o problema é a relação, não uma linha. Nesse
    caso o relatório continua valendo e o CSV sai vazio, o que é a informação
    correta: não há uma linha culpada.
    """
    if casos is None or "index" not in casos:
        return df.iloc[:0]
    indices = casos["index"].dropna().unique()
    return df.loc[df.index.intersection(indices)]


def quarentenar(
    df: pd.DataFrame,
    erro: DataValidationError,
    pasta: Path,
    momento: datetime | None = None,
) -> Path:
    """Grava o lote reprovado e devolve a pasta onde ele ficou."""
    carimbo = (momento or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    destino = pasta / carimbo
    destino.mkdir(parents=True, exist_ok=True)

    ofensivas = _linhas_ofensivas(df, erro.failure_cases)
    ofensivas.to_csv(destino / "linhas.csv", index=True)

    casos = erro.failure_cases
    if casos is not None and not casos.empty:
        colunas = [c for c in COLUNAS_DO_RELATORIO if c in casos]
        tabela = casos[colunas].to_string(index=False)
    else:
        tabela = "(sem failure_cases estruturados)"

    (destino / "relatorio.txt").write_text(
        f"{len(ofensivas)} linha(s) em quarentena de um lote de {len(df)}.\n\n"
        f"{tabela}\n\n"
        f"--- exceção ---\n{erro}\n",
        encoding="utf-8",
    )
    return destino
