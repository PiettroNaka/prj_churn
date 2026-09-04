"""Profiling do dataset — conhecer o dado antes de escrever o contrato.

Primeira etapa do Data Pipeline (exploration): distribuições, nulos, faixas e
cardinalidade. É daqui que saem os números do ``ChurnSchema`` — faixa de
``tenure``, teto de ``MonthlyCharges``, quais categorias são válidas. Contrato
escrito sem profiling é chute com cara de rigor.

    uv run python scripts/profiling.py

Para um relatório completo (correlações, alertas, HTML), o caminho é o
``ydata-profiling``; aqui fica só o retrato que cabe no console e que justifica
cada linha do schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from churn.schema import ChurnRawSchema

DATASET = Path("data/churn.csv")
NUMERICAS = ["tenure", "MonthlyCharges"]
SIMBOLO = {
    "greater_than_or_equal_to": "≥",
    "less_than_or_equal_to": "≤",
    "greater_than": ">",
    "less_than": "<",
}


def _faixa_declarada(coluna: str) -> str:
    """O que o contrato exige daquela coluna — para comparar com o observado."""
    partes = []
    for check in ChurnRawSchema.to_schema().columns[coluna].checks:
        valor = next(iter(check.statistics.values()), "")
        partes.append(f"{SIMBOLO.get(check.name, check.name)} {valor}")
    return " e ".join(partes) or "—"


def main() -> int:
    if not DATASET.exists():
        print(f"ERRO: {DATASET} não está no disco. Rode ./scripts/fetch_data.sh")
        return 1

    df = pd.read_csv(DATASET)
    print(f"$ {DATASET} — {len(df)} linhas × {len(df.columns)} colunas\n")

    print("numéricas: observado vs. contrato")
    print(f"  {'coluna':<16} {'min':>9} {'max':>9} {'média':>9}   contrato")
    for coluna in NUMERICAS:
        serie = df[coluna]
        print(
            f"  {coluna:<16} {serie.min():>9.2f} {serie.max():>9.2f} "
            f"{serie.mean():>9.2f}   {_faixa_declarada(coluna)}"
        )

    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    brancos = int(total.isna().sum())
    print(
        f"\nTotalCharges: texto no CSV — {brancos} brancos, "
        f"{int((df.loc[total.isna(), 'tenure'] == 0).sum())} deles com tenure 0"
    )
    print(f"  numérico: {total.min():.2f} a {total.max():.2f}")

    print(f"\nnulos: {int(df.isna().sum().sum())} em todo o dataset")

    print("\ncategóricas: cardinalidade e domínio")
    for coluna in df.select_dtypes("object"):
        if coluna in {"customerID", "TotalCharges"}:
            continue
        valores = sorted(df[coluna].dropna().unique())
        amostra = ", ".join(map(str, valores[:3])) + (" …" if len(valores) > 3 else "")
        print(f"  {coluna:<18} {len(valores):>2}  {amostra}")

    print(f"\ncustomerID único: {df['customerID'].is_unique}")
    balanco = (df["Churn"].value_counts(normalize=True) * 100).round(1)
    print(f"alvo Churn: {balanco.to_dict()} — classes desbalanceadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
