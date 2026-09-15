"""Rodar o modelo — carregar do MLflow e prever, sem escrever Python.

O run guarda o modelo; este script o tira de lá e o usa. A entrada é a linha
**crua**, como sai do sistema de origem: com ``customerID``, com ``TotalCharges``
em texto. Quem cuida da limpeza, das features e do encoder é o próprio pipeline
logado (``churn.serving``).

    uv run python scripts/prever.py                      # 5 clientes do dataset
    uv run python scripts/prever.py --n 20
    uv run python scripts/prever.py --csv clientes.csv   # um arquivo qualquer
    uv run python scripts/prever.py --uri models:/m-abc  # outro modelo

Sem ``--uri``, usa o modelo do run mais recente do experimento. Se a entrada
trouxer a coluna ``Churn``, o relatório compara a previsão com o que aconteceu —
é conferência, não avaliação: o número que vale é o do run, medido no teste.

Para servir como HTTP em vez de rodar em lote:

    uv run mlflow models serve -m runs:/<run_id>/model --env-manager local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

from churn.config import settings
from churn.serving import NOME_DO_MODELO, carregar

DATASET = Path("data/churn.csv")


def uri_do_ultimo_run() -> str | None:
    """O modelo do run mais recente — o padrão de quem só quer rodar."""
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        filter_string=f"tags.mlflow.runName = '{settings.mlflow_run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return f"runs:/{runs.iloc[0]['run_id']}/{NOME_DO_MODELO}"


def relatorio(entrada: pd.DataFrame, previsto: pd.DataFrame) -> pd.DataFrame:
    """Junta o que entrou com o que o modelo disse."""
    colunas = [c for c in (settings.id_column, "tenure", "Contract") if c in entrada]
    tabela = pd.concat(
        [entrada[colunas].reset_index(drop=True), previsto.reset_index(drop=True)],
        axis=1,
    )
    if settings.target in entrada:
        real = entrada[settings.target].reset_index(drop=True)
        tabela.insert(len(colunas), "real", real)
        tabela["acertou"] = tabela["churn"] == real
    return tabela


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uri", help="modelo a carregar (runs:/… ou models:/…)")
    parser.add_argument("--csv", type=Path, default=DATASET, help="clientes a prever")
    parser.add_argument("--n", type=int, default=5, help="quantas linhas (0 = todas)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERRO: {args.csv} não existe. Rode ./scripts/fetch_data.sh")
        return 1

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = args.uri or uri_do_ultimo_run()
    if uri is None:
        print(f"ERRO: nenhum run em '{settings.mlflow_experiment}'.")
        print("      Rode primeiro: uv run python -m churn.model")
        return 1

    print(f"$ modelo:  {uri}")
    print(f"$ clientes: {args.csv}")
    modelo = carregar(uri)

    entrada = pd.read_csv(args.csv)
    if args.n:
        entrada = entrada.head(args.n)

    tabela = relatorio(entrada, modelo.predict(entrada))
    print()
    print(tabela.to_string(index=False))

    if "acertou" in tabela:
        acertos = int(tabela["acertou"].sum())
        print(f"\n{acertos}/{len(tabela)} conferem com o rótulo que veio no arquivo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
