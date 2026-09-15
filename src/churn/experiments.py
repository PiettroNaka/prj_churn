"""A varredura rastreada — a tarefa do Encontro 6.

Roda um grid de Random Forest e registra cada combinação como um run do MLflow.
A dor que isto resolve é a do encontro: quarenta execuções, a acurácia anotada
num post-it, e três dias depois ninguém sabe qual configuração venceu.

    uv run python -m churn.experiments

Os runs ficam **aninhados** sob um run-pai: a varredura inteira é uma coisa só na
UI, e as nove combinações são os filhos. É o que a seção de runs aninhados do
material recomenda justamente para varredura.

O modelo não é logado aqui, de propósito. São nove florestas de até 500 árvores;
guardar todas custa centenas de MB para comparar cinco números. A varredura
escolhe a configuração — quem treina e loga o vencedor é o ``churn.model``.
"""
from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.lineage import data_md5, git_commit, ponteiro_de

#: O grid do encontro: nove combinações.
N_ESTIMATORS = (100, 200, 500)
MAX_DEPTH = (None, 8, 16)

#: Por qual métrica o vencedor é escolhido. roc_auc e não accuracy: a base é
#: desbalanceada (~32% de churn), e acurácia premia quem chuta "não churna".
METRICA = "roc_auc"


def _dados(cfg: Settings):
    """Prepara treino e teste uma vez só — o grid varia o modelo, não o dado."""
    frame = add_derived_features(
        load_clean(cfg.data_path, cfg.id_column, cfg.target, cfg.quarantine_dir)
    )
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )
    encoder = CategoricalEncoder().fit(x_train)
    return encoder.transform(x_train), encoder.transform(x_test), y_train, y_test


def run_grid(cfg: Settings = settings) -> None:
    """Treina o grid e registra um run por combinação."""
    x_train, x_test, y_train, y_test = _dados(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    linhagem = {
        "data_md5": data_md5(ponteiro_de(cfg.data_path)),
        "git_commit": git_commit(),
    }

    with mlflow.start_run(run_name="grid"):
        mlflow.set_tags(linhagem)
        mlflow.log_params({"n_estimators": list(N_ESTIMATORS), "max_depth": list(MAX_DEPTH)})

        for n in N_ESTIMATORS:
            for depth in MAX_DEPTH:
                with mlflow.start_run(run_name=f"rf-n{n}-d{depth}", nested=True):
                    # a linhagem vai em cada filho também: um run que só existe
                    # dentro de outro ainda precisa responder de qual dado veio
                    mlflow.set_tags(linhagem)
                    mlflow.log_params({"n_estimators": n, "max_depth": depth})

                    modelo = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        random_state=cfg.random_state,
                    ).fit(x_train, y_train)

                    mlflow.log_metrics(evaluate(modelo, x_test, y_test))


def melhores(cfg: Settings = settings, limite: int = 5) -> pd.DataFrame:
    """Os melhores runs por ``METRICA`` — o que a UI mostra, em DataFrame."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow_experiment],
        filter_string=f"metrics.{METRICA} > 0",
        order_by=[f"metrics.{METRICA} DESC"],
    )
    colunas = [
        "tags.mlflow.runName",
        "params.n_estimators",
        "params.max_depth",
        f"metrics.{METRICA}",
        "metrics.accuracy",
        "tags.data_md5",
    ]
    return runs[[c for c in colunas if c in runs]].head(limite)


def main() -> None:
    run_grid()
    tabela = melhores()
    print(f"\nMelhores runs por {METRICA}:\n")
    print(tabela.to_string(index=False))
    if not tabela.empty:
        melhor = tabela.iloc[0]
        print(
            f"\nVencedor: {melhor['tags.mlflow.runName']} "
            f"({METRICA} {float(melhor[f'metrics.{METRICA}']):.4f})"
        )


if __name__ == "__main__":
    main()
