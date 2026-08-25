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

    # saída
    artifact_path: Path = Path("artifacts/churn_model.pkl")


settings = Settings()
