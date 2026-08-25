# prj_churn

Projeto-fio-condutor do curso de MLOps: previsão de **churn** (classificação
binária tabular). Este repositório é o *depois* da refatoração do Encontro 2 —
o script monolítico `train_churn.py` reorganizado na estrutura modular.

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ data.py        # carga + validação + limpeza
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ model.py       # treino (entrypoint)
│  └─ evaluate.py    # métricas
├─ tests/            # pytest
├─ data/             # dataset (versionado por DVC a partir do Enc. 5)
└─ pyproject.toml    # deps travadas (uv)
```

## Como rodar (uv)

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
uv run python -m churn.model     # treina e imprime as métricas
uv run pytest                    # roda os testes
```

Sem uv, para desenvolvimento rápido:

```bash
pip install -e ".[dev]"
python -m churn.model
pytest
```

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só imprime `accuracy` | `evaluate` retorna accuracy, precision, recall, f1, roc_auc |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Próximos encontros

- **Enc. 5** — versionar `data/` com DVC.
- **Enc. 6** — rastrear experimentos com MLflow.
- **Enc. 3** — este projeto já roda containerizado (Dockerfile com uv).
