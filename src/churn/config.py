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
    max_depth: int | None = None

    # saída
    artifact_path: Path = Path("artifacts/churn_model.pkl")

    # MLflow — infraestrutura de rastreamento, não hiperparâmetro.
    # Fixar a URI aqui mata o atrito nº4 do deck ("runs somem"): sem isso cada
    # shell precisa exportar MLFLOW_TRACKING_URI e um esquecimento espalha os
    # runs por backends diferentes. O MLflow 3 recusa o file store ./mlruns.
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "churn"
    registered_model_name: str = "churn"
    promotion_min_roc_auc: float = 0.80


#: Campos que descrevem onde registrar, não o que foi treinado. Ficam de fora
#: dos params do run: param é o que muda o resultado.
INFRA_FIELDS: frozenset[str] = frozenset(
    {
        "tracking_uri",
        "experiment_name",
        "registered_model_name",
        "promotion_min_roc_auc",
        "artifact_path",
    }
)


def model_params(cfg: "Settings") -> dict[str, object]:
    """Os campos de ``Settings`` que de fato influenciam o modelo."""
    return cfg.model_dump(exclude=set(INFRA_FIELDS))


settings = Settings()
