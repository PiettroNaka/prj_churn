"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou, no container, ``uv run python -m churn.model``).

Desde o Encontro 6 cada execução vira um **run** do MLflow: params (o que foi
escolhido), métricas (o que foi medido), o modelo como artefato e as tags de
linhagem — qual dado e qual código produziram aquilo. O ``print`` mostrava o
número e o esquecia; o run guarda.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
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


def build_model(cfg: Settings) -> RandomForestClassifier:
    """Instancia o classificador com semente fixa (reprodutível)."""
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )


def save_artifact(model: RandomForestClassifier, encoder: CategoricalEncoder, path: Path) -> None:
    """Persiste modelo + encoder juntos (a inferência precisa do encoder ajustado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder}, handle)


def parametros(cfg: Settings) -> dict[str, object]:
    """A config Pydantic vira params do run — os dois contratos se encontram aqui.

    Fora as chaves ``mlflow_*``: elas dizem *onde anotar*, não *o que foi
    treinado*. Param é entrada do experimento; endereço do caderno não é.
    """
    return {
        chave: valor
        for chave, valor in cfg.model_dump().items()
        if not chave.startswith("mlflow_")
    }


def train(cfg: Settings = settings) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Executa o pipeline: dados -> features -> split -> encode -> treino -> avaliação.

    Tudo o que acontece aqui é registrado num run do MLflow.
    """
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
    x_train_cod, x_test_cod = encoder.transform(x_train), encoder.transform(x_test)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        # linhagem primeiro: se o treino falhar no meio, o run já sabe de onde veio
        mlflow.set_tags(
            {
                "data_md5": data_md5(ponteiro_de(cfg.data_path)),
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(parametros(cfg))

        model = build_model(cfg).fit(x_train_cod, y_train)
        metrics = evaluate(model, x_test_cod, y_test)
        mlflow.log_metrics(metrics)

        # MLflow 3 usa `name=`; `artifact_path=` ainda funciona, com aviso de
        # depreciação. A assinatura documenta o contrato do modelo — o que ele
        # espera receber e o que devolve — e é o que o serving valida depois.
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(x_train_cod, model.predict(x_train_cod)),
            input_example=x_train_cod.head(3),
        )

        # O encoder vai junto, e pequeno: sem ele o modelo não sabe ler um
        # cliente novo — o pickle do artifact_path guarda os dois.
        save_artifact(model, encoder, cfg.artifact_path)
        mlflow.log_artifact(str(cfg.artifact_path), artifact_path="pipeline")

    return model, metrics


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
