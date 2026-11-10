"""Ajuste fino de hiperparâmetros — a busca que o grid do Encontro 6 não faz.

O ``churn.experiments`` varre nove pontos: três valores de ``n_estimators``
vezes três de ``max_depth``. É didático e barato, e foi o bastante para mostrar
que a varredura rastreada vale a pena. Mas nove pontos em duas dimensões é
pouco: a floresta tem mais botões, e os que sobraram de fora — quantas amostras
por folha, quantas features por split, o peso das classes — mexem no resultado
tanto quanto a profundidade.

    uv run python -m churn.tuning

Três decisões que mudam o que o número significa:

**Busca aleatória, não exaustiva.** O espaço tem 3.150 combinações; o grid
completo levaria horas para melhorar a terceira casa decimal. Amostrar 24 pontos
encontra quase sempre a mesma região, em minutos. É o resultado clássico de
Bergstra & Bengio (2012), e vale justamente quando poucos hiperparâmetros
dominam — que é o caso aqui.

**Validação cruzada no treino, teste intocado.** Escolher configuração olhando o
teste é como estudar com o gabarito: o número sai bonito e não se repete em
produção. A busca pontua por CV dentro do treino; o teste só aparece no fim,
uma vez, para medir o vencedor. É por isso que o ``roc_auc`` da CV e o do teste
não batem — e não baterem é o esperado.

**Mesmo split do resto do projeto.** A busca parte do
``experiments.dados_codificados``, com a mesma semente. O número que sai daqui é
comparável com o do ``churn.model`` e com o do grid porque os três viram
exatamente o mesmo recorte do dado.

No fim, o vencedor é treinado e logado pelo caminho normal (``churn.model``), de
modo que o run resultante é promovível pelo ``churn.registry`` — a busca entrega
um candidato de produção, não um relatório.
"""
from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from churn.config import Settings, settings
from churn.evaluate import evaluate
from churn.experiments import METRICA, dados_codificados
from churn.lineage import data_md5, git_commit, ponteiro_de

#: O espaço de busca. Seis eixos, 3.150 combinações — a busca amostra
#: ``cfg.tuning_n_iter`` delas.
ESPACO: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 4, 6, 8, 12, 16, 24],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"],
}

#: Os campos da ``Settings`` que a busca mexe. Serve para dois lados: montar a
#: config do vencedor e imprimir as variáveis de ambiente equivalentes.
HIPERPARAMETROS = tuple(ESPACO)


def espaco_total() -> int:
    """Quantas combinações existem — o denominador do que a busca amostra."""
    total = 1
    for valores in ESPACO.values():
        total *= len(valores)
    return total


def buscar(cfg: Settings = settings) -> RandomizedSearchCV:
    """Roda a busca e devolve o objeto ajustado, com ``cv_results_`` completo.

    Sem MLflow aqui de propósito: a busca é uma função de dado e config, e dá
    para testá-la sem subir tracking nenhum. Quem registra é o :func:`rodar`.
    """
    x_train, _, y_train, _ = dados_codificados(cfg)
    busca = RandomizedSearchCV(
        RandomForestClassifier(random_state=cfg.random_state),
        param_distributions=ESPACO,
        n_iter=cfg.tuning_n_iter,
        cv=cfg.tuning_cv,
        scoring=METRICA,
        random_state=cfg.random_state,
        n_jobs=-1,
        # o vencedor sai re-treinado no treino inteiro, não só nas dobras
        refit=True,
    )
    return busca.fit(x_train, y_train)


def config_vencedora(melhores: dict[str, Any], cfg: Settings = settings) -> Settings:
    """A ``Settings`` do projeto com os hiperparâmetros que venceram.

    ``model_copy`` e não um construtor novo: tudo o que não é hiperparâmetro —
    caminho do dado, semente, endereço do MLflow — tem de continuar igual, ou o
    treino do vencedor não seria o mesmo treino.
    """
    return cfg.model_copy(update=dict(melhores))


def como_variaveis(melhores: dict[str, Any]) -> list[str]:
    """As linhas ``CHURN_X=y`` que reproduzem o vencedor noutra máquina.

    ``None`` sai como ``null``, não como vazio. Um ``CHURN_MAX_DEPTH=`` levanta
    ValidationError ("" não é int) e um ``CHURN_MAX_FEATURES=`` é pior: passa,
    vira a string vazia e treina outro modelo sem reclamar. A palavra vem do
    ``env_parse_none_str`` da config — as duas pontas têm de concordar, e é o
    que o teste desta função verifica.
    """
    return [
        f"CHURN_{nome.upper()}={'null' if valor is None else valor}"
        for nome, valor in melhores.items()
    ]


def _registrar_candidatos(busca: RandomizedSearchCV, linhagem: dict[str, str]) -> None:
    """Um run aninhado por candidato — a busca inteira fica auditável.

    O ``cv_results_`` já tem tudo; o que falta é endereço. Sem isto, a pergunta
    "por que esta configuração e não aquela?" só se responde rodando de novo.
    """
    resultados = busca.cv_results_
    for posicao in range(len(resultados["params"])):
        parametros = resultados["params"][posicao]
        rotulo = "-".join(
            f"{nome[:3]}{'X' if valor is None else valor}"
            for nome, valor in sorted(parametros.items())
        )
        with mlflow.start_run(run_name=rotulo, nested=True):
            mlflow.set_tags(linhagem)
            mlflow.log_params(parametros)
            mlflow.log_metrics(
                {
                    f"cv_{METRICA}": float(resultados["mean_test_score"][posicao]),
                    f"cv_{METRICA}_desvio": float(resultados["std_test_score"][posicao]),
                    "cv_ranking": int(resultados["rank_test_score"][posicao]),
                    "segundos_por_treino": float(resultados["mean_fit_time"][posicao]),
                }
            )


def rodar(cfg: Settings = settings) -> tuple[dict[str, Any], dict[str, float]]:
    """Busca, registra tudo no MLflow e mede o vencedor no teste.

    Devolve ``(melhores_parametros, metricas_no_teste)``. As métricas saem do
    holdout que a busca não viu — é o único número honesto para comparar com o
    do ``churn.model``.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    linhagem = {
        "data_md5": data_md5(ponteiro_de(cfg.data_path)),
        "git_commit": git_commit(),
    }

    with mlflow.start_run(run_name="tuning"):
        mlflow.set_tags(linhagem)
        mlflow.log_params(
            {
                "tuning_n_iter": cfg.tuning_n_iter,
                "tuning_cv": cfg.tuning_cv,
                "tuning_espaco_total": espaco_total(),
                "tuning_metrica": METRICA,
            }
        )

        busca = buscar(cfg)
        _registrar_candidatos(busca, linhagem)

        melhores = dict(busca.best_params_)
        # o vencedor medido onde ninguém olhou durante a busca
        _, x_test, _, y_test = dados_codificados(cfg)
        no_teste = evaluate(busca.best_estimator_, x_test, y_test)

        mlflow.log_params({f"melhor_{k}": v for k, v in melhores.items()})
        mlflow.log_metrics({f"cv_{METRICA}": float(busca.best_score_)})
        mlflow.log_metrics(no_teste)

    return melhores, no_teste


def main() -> None:
    cfg = settings
    print(
        f"Busca: {cfg.tuning_n_iter} candidatos de {espaco_total()}, "
        f"cv={cfg.tuning_cv}, métrica={METRICA}."
    )
    print("A validação cruzada roda no treino; o teste só mede o vencedor.\n")

    melhores, no_teste = rodar(cfg)

    print("Vencedor:")
    for nome, valor in melhores.items():
        print(f"  {nome:>18}: {valor}")
    print("\nNo teste (holdout que a busca não viu):")
    for nome, valor in no_teste.items():
        print(f"  {nome:>18}: {valor:.4f}")

    print("\nPara reproduzir noutra máquina:")
    for linha in como_variaveis(melhores):
        print(f"  {linha}")

    # Treinar o vencedor pelo caminho normal é o que o torna promovível: o
    # `churn.registry` só olha runs com o nome do `churn.model` e com pipeline
    # logado. Sem este passo, a busca terminaria com um relatório e nenhum
    # modelo para servir.
    print("\nTreinando o vencedor pelo caminho normal (run promovível)...")
    from churn.model import train

    _, metricas = train(config_vencedora(melhores, cfg))
    print(f"  run '{cfg.mlflow_run_name}' registrado — {METRICA} {metricas[METRICA]:.4f}")
    print("\nPara promover:  uv run python -m churn.registry")


if __name__ == "__main__":
    main()
