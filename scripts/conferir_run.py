"""O último run registrou a linhagem certa? — checkpoint do Encontro 6.

Treinar e gravar um run não basta: o run precisa dizer de qual dado veio, e essa
resposta precisa ser a verdadeira. Este script confere as duas coisas no
tracking configurado:

    1. existe um run no experimento
    2. a tag ``data_md5`` bate com o md5 esperado (o do ponteiro do DVC)
    3. as métricas do treino estão lá

    ESPERADO=<md5> uv run python scripts/conferir_run.py

Sem ``ESPERADO``, confere só o que dá para conferir sozinho. Sai com 1 quando o
rastro não existe ou aponta para outro dado — que é o mesmo que não ter rastro.
"""
from __future__ import annotations

import os
import sys

import mlflow

from churn.config import settings

METRICAS_ESPERADAS = {"accuracy", "roc_auc", "f1", "precision", "recall"}


def main() -> int:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    print(f"$ tracking: {settings.mlflow_tracking_uri}")

    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        print(f"ERRO: nenhum run em '{settings.mlflow_experiment}'.")
        print("      O treino rodou, mas não deixou rastro — é o print de novo.")
        return 1

    run = runs.iloc[0]
    print(f"  runs no experimento: {len(runs)}")
    print(f"  último: {run.get('tags.mlflow.runName')} ({run['run_id'][:8]})")

    faltando = {m for m in METRICAS_ESPERADAS if f"metrics.{m}" not in runs}
    if faltando:
        print(f"ERRO: o run não tem as métricas {sorted(faltando)}.")
        return 1
    print(f"  roc_auc: {float(run['metrics.roc_auc']):.4f}")

    md5 = run.get("tags.data_md5")
    commit = run.get("tags.git_commit")
    print(f"  data_md5:   {md5}")
    print(f"  git_commit: {commit}")

    if not md5 or not commit:
        print("ERRO: run sem linhagem — não dá para saber que dado gerou este modelo.")
        return 1

    esperado = os.environ.get("ESPERADO")
    if esperado and md5 != esperado:
        print(f"ERRO: o run diz {md5}, mas o ponteiro do DVC diz {esperado}.")
        print("      O modelo foi treinado com outro dado que não o versionado.")
        return 1

    print("\n  o run existe, tem métricas e aponta para o dado versionado. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
