# prj_churn

Projeto-fio-condutor do curso de MLOps: previsão de **churn** (classificação
binária tabular). Este repositório é o *depois* da refatoração do Encontro 2 —
o script monolítico `train_churn.py` reorganizado na estrutura modular.

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ schema.py      # ChurnSchema — o contrato dos dados (pandera)
│  ├─ validation.py  # executa o contrato e traduz a falha
│  ├─ data.py        # carga + validação + limpeza
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ model.py       # treino (entrypoint)
│  └─ evaluate.py    # métricas
├─ tests/            # pytest
├─ data/             # churn.csv.dvc (ponteiro) — os bytes vêm do DVC
├─ scripts/          # fetch_data.sh · profiling.py · quebra_proposital.py
├─ docs/             # nota de leitura do Data Pipeline
├─ .dvc/             # config do DVC: remote e cache
└─ pyproject.toml    # deps travadas (uv)
```

## Como rodar (uv)

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
./scripts/fetch_data.sh          # traz data/churn.csv (o Git não o guarda mais)
uv run python -m churn.model     # treina e imprime as métricas
uv run pytest                    # roda os testes
```

Os testes não precisam do dataset — usam amostras de `tests/conftest.py`. Quem
precisa dele é o treino.

Sem uv, para desenvolvimento rápido:

```bash
pip install -e ".[dev]"
./scripts/fetch_data.sh
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

O contrato do dataset é declarativo e executável. Ele é uma **classe** —
`DataFrameModel` do pandera, a mesma intuição do Pydantic do Encontro 2: lá o
contrato é da config, aqui é dos dados.

```
src/churn/schema.py       ChurnRawSchema · ChurnSchema   → o que é um dado válido
src/churn/validation.py   validate_raw · validate_clean  → o que fazer quando não é
```

O pipeline valida **duas vezes**: o CSV cru, antes de qualquer transformação, e o
DataFrame já limpo, antes de virar feature. A coerção de tipos acontece entre os
dois, à vista — em vez de `coerce=True` mascarando sujeira de verdade.

| Camada | Cobre |
|---|---|
| schema | as 21 colunas, tipos, nenhuma coluna a mais (`strict`) |
| domínio | as categorias válidas de cada coluna, `customerID` único |
| faixas | `tenure` 0–72, `MonthlyCharges` 0 < x ≤ 200, `TotalCharges` ≥ 0 |
| nulos | alvo `Churn` nunca nulo |
| coerência | sem internet ⇒ os 6 add-ons dizem `No internet service`; sem telefone ⇒ `MultipleLines` diz `No phone service`; `TotalCharges` ≥ uma mensalidade quando `tenure > 1` |

Falha levanta `DataValidationError` (que é um `ValueError`) em modo `lazy` — o
relatório traz **todas** as violações de uma vez, não só a primeira, e vem
também estruturado em `erro.failure_cases` (coluna, regra, valor) para logar.

### Conhecer antes de validar

As faixas saem do profiling, não de chute:

```bash
uv run python scripts/profiling.py     # observado vs. o que o contrato exige
```

```
coluna                 min       max     média   contrato
tenure                0.00     72.00     36.48   ≥ 0 e ≤ 72
MonthlyCharges       18.25    118.75     68.68   > 0 e ≤ 200.0
```

O teto de `MonthlyCharges` é 200 e não os 118,75 observados de propósito: a
faixa existe para pegar `999`, `-1` e `1e6`, não para reprovar um reajuste
legítimo. "Comece frouxo, aperte com o tempo" — o aperto que vale a pena aqui é
o `strict=True`, porque coluna nova neste dataset é sintoma de extração errada.

### Quebrar de propósito

O contrato só vale se alguém já o viu pegar alguma coisa:

```bash
uv run python scripts/quebra_proposital.py
```

Ele valida o dataset real (tem de passar), adultera **uma** linha com
`tenure = 999` e `Contract = "Vitalício"`, e mostra o relatório:

```
  column       check                                failure_case
  Contract     isin(['Month-to-month', 'One year',… Vitalício
  tenure       less_than_or_equal_to(72)            999

  o contrato pegou as duas de uma vez (lazy=True). ✓
```

As duas violações num relatório só é o `lazy=True` fazendo efeito. O script sai
com código 1 se o dado adulterado passar — por isso ele roda no CI, junto dos
testes que sujam uma coisa de cada vez (`tests/test_validation.py`).

### Por que as 11 linhas em branco não foram descartadas

`TotalCharges` vem em branco em 11 das 7043 linhas. Todas têm `tenure == 0`:
são clientes que ainda não fecharam o primeiro ciclo de cobrança, cujo total
acumulado é 0 de verdade — 4 delas são churners, e jogá-las fora tiraria sinal
da classe minoritária. Em vez de excluir, o schema **exige** a relação: branco
com `tenure > 0` é corrupção e reprova o dataset.

Onde cada etapa do Data Pipeline mora no projeto: [`docs/data-pipeline.md`](docs/data-pipeline.md).

## Versionamento de dados (Encontro 5)

O CSV saiu do Git. O que o Git versiona agora é `data/churn.csv.dvc` — md5,
tamanho e caminho — e os bytes ficam no cache do DVC e no *remote*.

| | antes (Enc. 2–4) | agora |
|---|---|---|
| quem guarda os bytes | Git | cache do DVC + remote |
| o que entra no commit | 981 KB de CSV | 3 linhas de ponteiro |
| trocar de dataset | diff binário ilegível | uma linha de md5 mudando |
| conferir integridade | à mão | `dvc status` |

```bash
uv sync --group data   # instala o dvc (fora do grupo padrão: o container não precisa)
uv run dvc pull        # traz o dado a partir do remote
uv run dvc status      # o arquivo no disco ainda bate com o ponteiro?
uv run dvc push        # manda para o remote o que você acabou de rastrear
```

Ciclo de troca de dataset: substitua `data/churn.csv`, rode `dvc add
data/churn.csv` (o `.dvc` muda), `dvc push` (os bytes sobem) e commite o `.dvc`.
Git e DVC andam juntos — `git checkout` de um commit antigo seguido de `dvc
checkout` devolve o dado daquele commit.

### O remote

O padrão é um diretório local, irmão do repositório:

```ini
['remote "storage"']
    url = ../../dvc-storage      # relativo a .dvc/config, isto é: <repo>/../dvc-storage
```

Serve para exercitar o fluxo sem credencial nenhuma, mas não atravessa máquinas.
Para um remote de verdade, uma linha resolve — o resto do fluxo não muda:

```bash
uv run dvc remote add -d storage s3://meu-bucket/churn   # ou gdrive://, azure://, ssh://
uv run dvc remote modify --local storage ...             # credenciais em .dvc/config.local (não versionado)
```

### E quem clona hoje?

Um remote local não viaja junto com o clone. Por isso existe
`scripts/fetch_data.sh`: tenta `dvc pull` e, se não houver remote alcançável,
recupera o CSV do blob que ficou no histórico do Git (o dataset esteve
versionado até o Encontro 5) e **confere o md5 contra o `.dvc`**. É como o CI
materializa o dado para rodar o treino no container.

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure=0`) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| validação ausente | `ChurnSchema` (pandera) no dado cru e no limpo |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só imprime `accuracy` | `evaluate` retorna accuracy, precision, recall, f1, roc_auc |
| tudo em nível de módulo | funções puras + orquestração em `train()` |

## Próximos encontros

- **Enc. 3** ✓ — containerizado: `Dockerfile`, `.dockerignore` e `compose.yml`.
- **Enc. 4** ✓ — contrato de dados com pandera (`schema.py`, `validation.py`).
- **Enc. 5** ✓ — `data/` versionado com DVC (`data/churn.csv.dvc`, remote, `scripts/fetch_data.sh`).
- **Enc. 6** — rastrear experimentos com MLflow.
