"""Projeto churn — projeto-fio-condutor do curso de MLOps.

Refatoração do script monolítico ``train_churn.py`` na estrutura modular:
config (Pydantic) · data (carga+validação) · features (transformações) ·
model (treino) · evaluate (métricas).
"""

__version__ = "0.1.0"
