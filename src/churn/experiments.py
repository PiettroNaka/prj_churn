"""Grid de experimentos rastreados — o fim do post-it.

``uv run python -m churn.experiments`` roda a grade inteira. Cada combinação
vira um run permanente e comparável: params, métricas, o modelo empacotado e as
tags de linhagem.

Duas decisões que tornam a comparação honesta:

- o preparo do dado vem de ``model.prepare`` **uma vez só**, fora do laço. Runs
  que prepararam o dado de formas diferentes não comparam hiperparâmetro.
- ``random_state`` fixo em todos. Sem isso, a diferença entre dois runs pode ser
  só a semente.

Cada run também loga o modelo. O slide do grid não loga, mas o passo seguinte
(``registry.promote_best``) registra o vencedor por ``runs:/<run_id>/model`` —
sem o modelo no run, não há o que promover.
"""
from __future__ import annotations

from itertools import product

import mlflow

from churn.config import Settings, model_params, settings
from churn.evaluate import evaluate
from churn.lineage import tags as lineage_tags
from churn.model import build_model, log_model, prepare, setup_tracking

#: A grade do deck: 3 × 3 = 9 runs.
N_ESTIMATORS: tuple[int, ...] = (100, 200, 500)
MAX_DEPTHS: tuple[int | None, ...] = (None, 8, 16)


def run_grid(cfg: Settings = settings) -> list[dict[str, object]]:
    """Roda a grade e devolve um resumo de cada run, na ordem de execução."""
    setup_tracking(cfg)
    x_train, x_test, y_train, y_test, _ = prepare(cfg)
    tags = lineage_tags()

    resumo: list[dict[str, object]] = []
    for n_estimators, max_depth in product(N_ESTIMATORS, MAX_DEPTHS):
        variante = cfg.model_copy(
            update={"n_estimators": n_estimators, "max_depth": max_depth}
        )
        nome = f"rf-n{n_estimators}-d{max_depth}"

        with mlflow.start_run(run_name=nome) as run:
            mlflow.set_tags(tags)
            mlflow.log_params(model_params(variante))

            model = build_model(variante).fit(x_train, y_train)
            metrics = evaluate(model, x_test, y_test)

            mlflow.log_metrics(metrics)
            log_model(model, x_train)

        resumo.append({"run_name": nome, "run_id": run.info.run_id, **metrics})
        print(f"  {nome:<16} roc_auc {metrics['roc_auc']:.4f}  accuracy {metrics['accuracy']:.4f}")

    return resumo


def main() -> None:
    print(f"Rodando {len(N_ESTIMATORS) * len(MAX_DEPTHS)} runs no experimento "
          f"'{settings.experiment_name}':")
    resumo = run_grid()
    melhor = max(resumo, key=lambda r: r["roc_auc"])
    print(f"\nMelhor: {melhor['run_name']} — roc_auc {melhor['roc_auc']:.4f}")
    print("Promova com: uv run python -m churn.registry")


if __name__ == "__main__":
    main()
