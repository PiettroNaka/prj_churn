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


settings = Settings()
