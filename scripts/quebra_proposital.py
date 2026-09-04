"""Quebrar o dado de propósito e ver o contrato pegar — tarefa do Encontro 4.

Dois atos, no dataset real:

    1. o dataset como está     -> tem de PASSAR
    2. uma linha adulterada    -> tem de REPROVAR, com as duas violações juntas

O segundo ato usa as mesmas quebras do slide: ``tenure = 999`` e
``Contract = "Vitalício"``, na MESMA linha. Com ``lazy=True`` o pandera coleta
todas as falhas antes de levantar, então o relatório mostra as duas de uma vez
— e não só a primeira que ele encontrou.

    uv run python scripts/quebra_proposital.py

Sai com 0 quando o contrato se comporta (passa no bom, reprova no sujo) e com 1
quando não — é o que o CI checa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from churn.validation import DataValidationError, validate_raw

DATASET = Path("data/churn.csv")
LINHA = 0
QUEBRAS = {"tenure": 999, "Contract": "Vitalício"}


def adulterar(df: pd.DataFrame) -> pd.DataFrame:
    """Suja UMA linha com as duas quebras do slide (pura: não toca no original)."""
    sujo = df.copy()
    for coluna, valor in QUEBRAS.items():
        sujo.loc[LINHA, coluna] = valor
    return sujo


def _legivel(valor: object) -> str:
    """numpy.int64(63) não é o que se mostra em aula — 63 é."""
    return repr(valor.item() if hasattr(valor, "item") else valor)


def _relatorio(casos: pd.DataFrame | None) -> None:
    """Imprime os failure_cases como o slide: coluna, regra, valor."""
    if casos is None or casos.empty:
        print("  (o pandera não devolveu failure_cases)")
        return
    print(f"  {'column':<12} {'check':<36} failure_case")
    print(f"  {'─' * 12} {'─' * 36} {'─' * 14}")
    for _, caso in casos.iterrows():
        coluna = caso["column"] if pd.notna(caso["column"]) else "<dataframe>"
        regra = str(caso["check"])
        if len(regra) > 36:
            regra = regra[:35] + "…"
        print(f"  {coluna:<12} {regra:<36} {caso['failure_case']}")


def main() -> int:
    if not DATASET.exists():
        print(f"ERRO: {DATASET} não está no disco. Rode ./scripts/fetch_data.sh")
        return 1

    cru = pd.read_csv(DATASET)
    print(f"$ dataset: {DATASET} — {len(cru)} linhas × {len(cru.columns)} colunas\n")

    print("[1/2] o dataset como está")
    try:
        validate_raw(cru)
    except DataValidationError as erro:
        print("  REPROVOU — o dataset de partida já não cumpre o contrato:")
        _relatorio(erro.failure_cases)
        return 1
    print("  passou no ChurnRawSchema.\n")

    print(f"[2/2] linha {LINHA} adulterada de propósito")
    for coluna, valor in QUEBRAS.items():
        print(f"  {coluna:<12} {_legivel(cru.loc[LINHA, coluna])} -> {valor!r}")
    print()

    try:
        validate_raw(adulterar(cru))
    except DataValidationError as erro:
        casos = erro.failure_cases
        print(f"  DataValidationError — {len(casos)} violação(ões) num relatório só:\n")
        _relatorio(casos)
        pegou = set(casos["column"].dropna())
        faltou = set(QUEBRAS) - pegou
        if faltou:
            print(f"\n  ATENÇÃO: o contrato não pegou {sorted(faltou)}.")
            return 1
        print("\n  o contrato pegou as duas de uma vez (lazy=True). ✓")
        return 0

    print("  FALHA DO CONTRATO: o dado adulterado passou.")
    print("  É exatamente o cenário do encontro — dado quebra em silêncio.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
