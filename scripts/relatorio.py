"""Ver os runs sem navegador — a UI em texto, e um HTML que abre por arquivo.

Nem sempre dá para abrir `localhost`: máquina de trabalho com proxy no caminho,
porta bloqueada, ou simplesmente nenhuma vontade de deixar um servidor de pé. O
que a UI mostra está todo no banco de tracking, e aqui ele sai em dois formatos
que não precisam de servidor:

    uv run python scripts/relatorio.py           # tabela no terminal
    uv run python scripts/relatorio.py --html    # artifacts/runs.html

O HTML é um arquivo só, sem CSS nem JS de fora: abre com duplo clique, por
``file://``, sem porta e sem rede. É o caminho para quando o navegador não
alcança o `localhost`.
"""
from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from churn.config import settings

SAIDA_PADRAO = Path("artifacts/runs.html")

#: O que vale olhar, na ordem em que se olha.
COLUNAS = [
    ("tags.mlflow.runName", "run"),
    ("metrics.roc_auc", "roc_auc"),
    ("metrics.accuracy", "accuracy"),
    ("metrics.f1", "f1"),
    ("params.n_estimators", "n_estim."),
    ("params.max_depth", "max_depth"),
    ("tags.data_md5", "dado (md5)"),
    ("tags.git_commit", "código"),
]


def carregar():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["metrics.roc_auc DESC", "attributes.start_time DESC"],
    )


def _valor(run, coluna: str) -> str:
    if coluna not in run.index:
        return "—"
    bruto = run[coluna]
    if bruto is None or (isinstance(bruto, float) and bruto != bruto):  # NaN
        return "—"
    if coluna.startswith("metrics."):
        return f"{float(bruto):.4f}"
    if coluna == "tags.data_md5":
        return str(bruto)[:10] + "…"
    return str(bruto)


def no_terminal(runs) -> None:
    """Uma tabela alinhada — larguras tiradas do conteúdo, não chutadas."""
    celulas = [[_valor(run, coluna) for coluna, _ in COLUNAS] for _, run in runs.iterrows()]
    larguras = [
        max(len(rotulo), *(len(linha[i]) for linha in celulas)) if celulas else len(rotulo)
        for i, (_, rotulo) in enumerate(COLUNAS)
    ]

    print("  ".join(r.ljust(l) for (_, r), l in zip(COLUNAS, larguras)))
    print("  ".join("─" * l for l in larguras))
    for linha in celulas:
        print("  ".join(valor.ljust(l) for valor, l in zip(linha, larguras)))


def em_html(runs, destino: Path) -> None:
    linhas = []
    for _, run in runs.iterrows():
        celulas = "".join(f"<td>{html.escape(_valor(run, c))}</td>" for c, _ in COLUNAS)
        linhas.append(f"<tr>{celulas}</tr>")

    cabecalho = "".join(f"<th>{html.escape(r)}</th>" for _, r in COLUNAS)
    agora = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        f"""<!doctype html>
<html lang="pt-BR"><meta charset="utf-8">
<title>Runs — {html.escape(settings.mlflow_experiment)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; }}
  h1 {{ font-size: 1.3rem; margin-bottom: .2rem; }}
  p  {{ color: #666; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #8884; }}
  th {{ font-weight: 600; white-space: nowrap; }}
  tr:first-of-type td {{ font-weight: 600; }}
  td:nth-child(2), td:nth-child(3), td:nth-child(4) {{ font-variant-numeric: tabular-nums; }}
  code {{ font-size: .9em; }}
</style>
<h1>Experimento <code>{html.escape(settings.mlflow_experiment)}</code></h1>
<p>{len(runs)} runs, do melhor <code>roc_auc</code> para o pior · gerado em {agora}</p>
<table><thead><tr>{cabecalho}</tr></thead>
<tbody>{"".join(linhas)}</tbody></table>
<p>Cada linha carrega a linhagem: <code>dado (md5)</code> é a versão do CSV no DVC,
<code>código</code> é o commit que treinou. Arquivo estático — não precisa de servidor.</p>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--html",
        nargs="?",
        const=SAIDA_PADRAO,
        type=Path,
        metavar="ARQUIVO",
        help=f"gera um HTML estático (padrão: {SAIDA_PADRAO})",
    )
    args = parser.parse_args()

    runs = carregar()
    if runs.empty:
        print(f"Nenhum run em '{settings.mlflow_experiment}'.")
        print("Treine uma vez: uv run python -m churn.model")
        return 1

    print(f"$ {settings.mlflow_tracking_uri} — {len(runs)} runs\n")
    no_terminal(runs)

    if args.html:
        em_html(runs, args.html)
        print(f"\nHTML em {args.html.resolve()}")
        print("Abra com duplo clique — é arquivo, não precisa de servidor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
