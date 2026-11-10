"""Configuração central do projeto (Pydantic).

Todo caminho, hiperparâmetro e constante mágica do script original vive aqui —
nunca hardcoded no meio da lógica. Valores podem ser sobrescritos por variáveis
de ambiente com prefixo ``CHURN_`` ou por um arquivo ``.env``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros do pipeline de churn."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        protected_namespaces=(),
        # Sem isto não há como dizer "None" por variável de ambiente, e metade
        # dos campos opcionais fica inatingível de fora. `CHURN_MAX_DEPTH=`
        # (vazio) não vira None: levanta ValidationError, porque "" não é int.
        # Pior, `CHURN_MAX_FEATURES=` vira a string vazia e o sklearn treina
        # outra coisa, calado. A palavra `null` dá um jeito de escrever o nulo:
        # CHURN_MAX_DEPTH=null  CHURN_CLASS_WEIGHT=null
        env_parse_none_str="null",
    )

    # dados
    data_path: Path = Path("data/churn.csv")
    id_column: str = "customerID"
    target: str = "Churn"
    positive_label: str = "Yes"

    # split / modelo
    test_size: float = 0.25
    random_state: int = 42
    n_estimators: int = 200
    # None = árvores crescem até a pureza. O grid do Encontro 6 mostra que 8 vai
    # melhor (roc_auc 0,8220 contra 0,8150), mas promover o vencedor é assunto
    # do Encontro 7 — aqui o campo existe para que a configuração vencedora seja
    # ao menos dizível: CHURN_MAX_DEPTH=8.
    max_depth: int | None = None
    # Os quatro abaixo não existiam enquanto a varredura era um grid de 3x3 em
    # cima de n_estimators e max_depth. O ``churn.tuning`` busca nos seis, e um
    # vencedor que a config não sabe dizer é um vencedor que não dá para treinar
    # de novo — o oposto do que o projeto inteiro persegue.
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    #: "sqrt", "log2" ou None (todas as features em cada split).
    max_features: str | None = "sqrt"
    #: None ou "balanced". Com ~27% de churn, "balanced" compra recall pagando
    #: em precisão; qual dos dois vale mais é decisão de negócio, não do modelo.
    class_weight: str | None = None

    # busca de hiperparâmetros (churn.tuning)
    #
    # 24 candidatos de 3.150: busca aleatória, não grid completo. O grid levaria
    # horas para ganhar a terceira casa decimal. cv=4 porque 5 dobras custam 25%
    # a mais para mover o resultado dentro do próprio desvio-padrão.
    tuning_n_iter: int = 24
    tuning_cv: int = 4

    # saída
    artifact_path: Path = Path("artifacts/churn_model.pkl")
    # None desliga a quarentena, como no ``load_clean`` — a config tem de
    # conseguir dizer o que a função aceita.
    quarantine_dir: Path | None = Path("artifacts/quarentena")

    # rastreamento de experimentos (Encontro 6)
    #
    # O atrito nº 1 do encontro é rodar de pastas diferentes e espalhar runs em
    # vários backends. Fixar aqui (sobrescrevível por CHURN_MLFLOW_TRACKING_URI)
    # é o que mantém todo run no mesmo lugar.
    #
    # SQLite e não o ./mlruns do material: o MLflow 3 recusa o file store com
    # "the filesystem tracking backend is in maintenance mode", exigindo
    # MLFLOW_ALLOW_FILE_STORE=true para abrir exceção. O banco é o caminho que
    # o próprio MLflow recomenda — e é o que o material já indica para uso em
    # time. Os artefatos ficam em ./mlartifacts.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "churn"
    mlflow_run_name: str = "rf"

    # promoção no Model Registry (Encontro 7)
    #
    # O nome agrupa as versões; o alias aponta a que serve. Aliases e não stages:
    # Staging/Production estão depreciados no MLflow 3.
    mlflow_registered_model: str = "churn"
    mlflow_champion_alias: str = "champion"
    # O portão. Promover sem conferir é publicar qualquer coisa que terminou de
    # treinar — inclusive um modelo pior que o que já está em produção. O valor
    # é baixo de propósito: é um piso de sanidade, não a meta.
    mlflow_promocao_minima: float = 0.80


settings = Settings()
