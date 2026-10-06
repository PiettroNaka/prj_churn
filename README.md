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
│  ├─ model.py       # treino instrumentado com MLflow (entrypoint)
│  ├─ evaluate.py    # métricas
│  ├─ validation.py  # schemas pandera (contrato do dataset)
│  ├─ lineage.py     # data_md5 + git_commit (tags de linhagem)
│  ├─ experiments.py # o grid rastreado
│  └─ registry.py    # promover o melhor run a @champion
├─ tests/            # pytest
├─ data/             # dataset (no Git até o Enc. 5, depois DVC)
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

## Rodar no container (Encontro 3)

O `pyproject.toml` + `uv.lock` travam os pacotes Python; o container trava o resto
(interpretador, SO, libs de sistema). Dados e artefatos entram por volume — não
viram camada da imagem.

```bash
docker build -t churn:0.1 .
docker run -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" churn:0.1
```

Ou, sem decorar flags:

```bash
docker compose up
```

## Validação de dados (Encontro 4)

O contrato do dataset é declarativo e executável, em `src/churn/validation.py`.
O pipeline valida **duas vezes**: o CSV cru, antes de qualquer transformação, e
o DataFrame já limpo, antes de virar feature.

| Camada | Cobre |
|---|---|
| schema | as 21 colunas, tipos, nenhuma coluna a mais (`strict`) |
| domínio | as categorias válidas de cada coluna, `customerID` único |
| faixas | `tenure` 0–72, `MonthlyCharges` > 0, `TotalCharges` ≥ 0 |
| nulos | alvo `Churn` nunca nulo |
| coerência | sem internet ⇒ os 6 add-ons dizem `No internet service`; sem telefone ⇒ `MultipleLines` diz `No phone service`; `TotalCharges` ≥ uma mensalidade quando `tenure > 1` |

Falha levanta `DataValidationError` (que é um `ValueError`) em modo `lazy` — o
relatório traz **todas** as violações de uma vez, não só a primeira.

### Por que as 11 linhas em branco não foram descartadas

`TotalCharges` vem em branco em 11 das 7043 linhas. Todas têm `tenure == 0`:
são clientes que ainda não fecharam o primeiro ciclo de cobrança, cujo total
acumulado é 0 de verdade — 4 delas são churners, e jogá-las fora tiraria sinal
da classe minoritária. Em vez de excluir, o schema **exige** a relação: branco
com `tenure > 0` é corrupção e reprova o dataset.

## ML Pipeline — MLflow (Encontros 6–8)

Cada treino é um run permanente. O `print` continua, mas já não é a memória do
experimento.

```bash
uv run python -m churn.model        # um treino, registrado
uv run python -m churn.experiments  # o grid: 9 runs comparáveis
uv run python -m churn.registry     # promove o melhor a @champion
mlflow ui --backend-store-uri sqlite:///mlflow.db   # localhost:5000
```

### O grid

| run | n_estimators | max_depth | roc_auc | accuracy |
|---|---|---|---|---|
| **rf-n200-d8** | 200 | 8 | **0.8195** | 0.7683 |
| rf-n500-d8 | 500 | 8 | 0.8192 | 0.7723 |
| rf-n100-d8 | 100 | 8 | 0.8182 | 0.7655 |
| rf-n500-d16 | 500 | 16 | 0.8068 | 0.7604 |

`max_depth=8` domina; 16 já piora (overfit) e `None` fica no meio — viés-variância
visível com dado de verdade.

### Portão de promoção

`registry.promote_best` só registra e move o alias se o melhor run passar de
`promotion_min_roc_auc` (0.80). Abaixo disso levanta `PromotionError` em vez de
rebaixar a produção em silêncio.

### Consumir o modelo de produção

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/churn@champion")
model.predict(x_novo)
```

`models:/nome@alias` é a referência estável — quem consome não precisa saber qual
run gerou o modelo. É esta chamada que a API do Encontro 12 vai fazer.

### Linhagem

Todo run carrega `data_md5`, `git_commit` e `git_dirty`, fechando a cadeia
**dado → código → experimento → modelo**.

O `data_md5` lê o ponteiro `data/churn.csv.dvc` quando ele existir; enquanto o
DVC não entra (Encontro 5), calcula o md5 do próprio CSV — e o valor é o mesmo,
porque para arquivo único o md5 que o `dvc add` grava é o md5 do conteúdo.

### Atritos do MLflow 3 já resolvidos aqui

| Atrito | Como está tratado |
|---|---|
| file store `./mlruns` recusado | `tracking_uri` = `sqlite:///mlflow.db` no `config.py` |
| `log_model(artifact_path=)` removido | usamos `name=` |
| skops reclama ao recarregar sklearn | `serialization_format="cloudpickle"` |
| runs somem entre shells | URI fixada em código, não só em env var |
| stages depreciados | aliases `@champion` / `@challenger` |

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| validação ausente | schema pandera no dado cru e no limpo |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só imprime `accuracy` | `evaluate` retorna accuracy, precision, recall, f1, roc_auc |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Próximos encontros

- **Enc. 3** ✓ — containerizado: `Dockerfile`, `.dockerignore` e `compose.yml`.
- **Enc. 4** ✓ — validação de dados com pandera (`validation.py`).
- **Enc. 6–8** ✓ — MLflow: tracking, registry e empacotamento.
- **Enc. 5** — versionar `data/` com DVC (pendente; o `lineage.py` já o espera).
- **Enc. 9–11** — orquestrar com Airflow.
