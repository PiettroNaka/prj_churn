"""Profiling do dataset — conhecer o dado antes de escrever o contrato.

Primeira etapa do Data Pipeline (exploration): distribuições, nulos, faixas e
cardinalidade. É daqui que saem os números do ``ChurnSchema`` — faixa de
``tenure``, teto de ``MonthlyCharges``, quais categorias são válidas. Contrato
escrito sem profiling é chute com cara de rigor.

    uv run python scripts/profiling.py            # o retrato no console
    uv run python scripts/profiling.py --html     # + o relatório completo em HTML

O relatório completo (correlações, alertas, distribuições) é o ``ydata-profiling``
do encontro. Ele **não** entra no ``pyproject.toml`` de propósito: instalado como
dependência do projeto, ele arrasta numpy e scipy para versões mais antigas — o
lock é único, então uma ferramenta de relatório passaria a ditar a pilha numérica
do treino, justo o que os Encontros 2 e 3 travaram. Aqui ele roda em ambiente
efêmero (``uvx``), sem encostar no ambiente travado.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from churn.schema import TOLERANCIA_TOTAL, ChurnRawSchema

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


HTML_PADRAO = Path("artifacts/profiling.html")
# setuptools<81: o ydata-profiling ainda importa pkg_resources, que sumiu nas
# versões novas do setuptools.
EFEMERO = ["--with", "setuptools<81", "--from", "ydata-profiling", "ydata_profiling"]


def relatorio_html(destino: Path) -> int:
    """Gera o relatório completo — pela biblioteca, se houver; senão, efêmero."""
    destino.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        pass
    else:
        ProfileReport(pd.read_csv(DATASET), title="churn").to_file(destino)
        print(f"relatório em {destino}")
        return 0

    runner = (
        ["uvx"] if shutil.which("uvx") else ["uv", "tool", "run"] if shutil.which("uv") else None
    )
    if runner is None:
        print("ERRO: sem ydata-profiling instalado e sem uv para rodá-lo.", file=sys.stderr)
        print("      uvx --with 'setuptools<81' --from ydata-profiling \\", file=sys.stderr)
        print(f"        ydata_profiling {DATASET} {destino}", file=sys.stderr)
        return 1

    print("ydata-profiling em ambiente efêmero (o lock do projeto não muda)...")
    concluido = subprocess.run(
        [*runner, *EFEMERO, "--title", "churn", str(DATASET), str(destino)],
        check=False,
    )
    if concluido.returncode != 0:
        return concluido.returncode
    print(f"relatório em {destino}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--html",
        nargs="?",
        const=HTML_PADRAO,
        type=Path,
        metavar="ARQUIVO",
        help=f"gera o relatório completo do ydata-profiling (padrão: {HTML_PADRAO})",
    )
    args = parser.parse_args()

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

    cobrado = df["tenure"] > 1
    esperado = (df["tenure"] * df["MonthlyCharges"])[cobrado]
    desvio = ((total[cobrado] - esperado).abs() / esperado).dropna()
    print(
        f"\nrelação TotalCharges ≈ tenure × MonthlyCharges (tenure > 1):"
        f"\n  desvio mediano {desvio.median():.1%}, máximo {desvio.max():.1%}"
        f"  — contrato tolera {TOLERANCIA_TOTAL:.0%}"
    )

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

    if args.html:
        print()
        return relatorio_html(args.html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
