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
├─ scripts/          # fetch_data.sh · profiling · quebra_proposital · gerar_v2 · alternar_versao
├─ docs/             # nota de leitura do Data Pipeline
├─ dvc.yaml          # o pipeline: deps (dado + código) → outs (modelo)
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
| schema | as 21 colunas e seus tipos (`coerce=True` converte ao validar; `strict=False` tolera coluna a mais) |
| domínio | as categorias válidas de cada coluna, `customerID` único — erro fatal, exceto nas duas colunas de catálogo, onde é alerta (abaixo) |
| faixas | `tenure` 0–72, `MonthlyCharges` 18–119, `TotalCharges` ≥ 0 |
| nulos | alvo `Churn` nunca nulo |
| coerência | sem internet ⇒ os 6 add-ons dizem `No internet service`; sem telefone ⇒ `MultipleLines` diz `No phone service` |
| relações | `TotalCharges` ≈ `tenure` × `MonthlyCharges` (tolerância de 10%) quando `tenure > 1` |

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

```
relação TotalCharges ≈ tenure × MonthlyCharges (tenure > 1):
  desvio mediano 1.5%, máximo 3.0%  — contrato tolera 10%
```

A faixa de `MonthlyCharges` é a do encontro — 18–119, o observado arredondado
para fora. É uma faixa colada no dado: pega `999`, `-1` e `1e6`, e também
reprovaria o dataset no dia em que um plano novo custasse 130. A tolerância de
10% na relação é mais folgada porque o pior caso observado é 3%. E a relação é a
regra mais forte do contrato: faixa isolada aprova `TotalCharges = 20` num
cliente de 70 meses pagando 100 por mês; a relação não.

A `Config` também é a do encontro — `coerce = True`, `strict = False`. A
comodidade tem preço, e o próprio encontro avisa qual: `tenure = 12.7` não
reprova, vira `12` em silêncio (há um teste fixando isso). O que segura o
estrago é validar em dois pontos — o que a coerção esconde no CSV cru, o schema
do dado limpo cobra num tipo já definido. Coluna **faltando** continua sendo
erro; `strict=False` tolera só a coluna a mais. "Comece frouxo, aperte com o tempo" — o aperto que vale a pena aqui é
o `strict=True`, porque coluna nova neste dataset é sintoma de extração errada.

### Erro ou alerta: nem toda violação é a mesma coisa

Categoria nova em produção é o atrito clássico do encontro — o `isin` reprova um
valor que é legítimo. Só que tratar *toda* categoria como alerta desarma o
contrato. O corte aqui é por consequência, não por gosto:

| Coluna | Valor novo significa | Severidade |
|---|---|---|
| `InternetService`, `PaymentMethod` | o negócio lançou algo (5G, Pix) — o cliente é real, a linha está certa | **alerta**: passa, registra, alguém decide se o schema cresce |
| `Contract` | mudou a feature nº 1 do modelo (importância 0,175 — 3,7× `PaymentMethod`) | **erro**: seguir treinando é pior que parar |
| `Churn` | o alvo virou outra coisa | **erro** |
| binários e derivados (`gender`, add-ons, `MultipleLines`…) | um terceiro valor onde só existem dois | **erro**: é corrupção, não lançamento |

O alerta é `raise_warning=True` no `Field`, e `validation.py` o registra em log e
o reemite — alerta que ninguém lê é validação que não valida. Alerta e erro
convivem no mesmo passe: uma linha com `PaymentMethod = "Pix"` **e** `tenure =
999` reprova pelo `tenure` e avisa do `Pix`, no mesmo relatório.

### Quebrar de propósito

O contrato só vale se alguém já o viu pegar alguma coisa:

```bash
uv run python scripts/quebra_proposital.py
```

São três atos: valida o dataset real (tem de passar), adultera **uma** linha com
`tenure = 999` e `Contract = "Vitalício"` (tem de reprovar), e por fim põe
`PaymentMethod = "Pix"` (tem de passar, com alerta). O relatório do segundo ato:

```
  column       check                                failure_case
  Contract     isin(['Month-to-month', 'One year',… Vitalício
  tenure       less_than_or_equal_to(72)            999

  o contrato pegou as duas de uma vez (lazy=True). ✓
```

As duas violações num relatório só é o `lazy=True` fazendo efeito. O script sai
com código 1 se o dado adulterado passar, se a categoria de catálogo derrubar o
pipeline, ou se ela passar em silêncio — os três modos de o contrato estar
errado. Por isso ele roda no CI, junto dos testes que sujam uma coisa de cada vez
(`tests/test_validation.py`).

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

### Duas versões registradas

| | clientes | md5 | accuracy | roc_auc |
|---|---|---|---|---|
| **v1** | 7043 | `d390bd07…` | 0,7570 | 0,8053 |
| **v2** | 7143 | `f150a64a…` | 0,7609 | 0,8150 |

A v2 é a v1 mais 100 clientes, e não é um arquivo que alguém precise guardar:
`scripts/gerar_v2.py` sorteia 100 clientes existentes com semente fixa e os
recadastra com identificador novo, copiando o resto da linha byte a byte. Daí
saem duas propriedades — a v2 já nasce cumprindo o contrato do Encontro 4, e os
bytes são os mesmos em qualquer máquina.

As métricas mudam entre as duas. É exatamente a pergunta do encontro: *dois
modelos, duas acurácias — qual CSV gerou qual?* A resposta agora é o commit, que
carrega código e ponteiro juntos.

### Alternar entre elas

São dois movimentos, e essa é a parte que costuma escapar:

```bash
git checkout <commit> -- data/churn.csv.dvc   # o Git traz o PONTEIRO
uv run dvc checkout                           # o DVC traz os BYTES
```

O Git sozinho troca 100 bytes de texto: sem o segundo comando o CSV no disco
ainda é o da versão anterior, e o repositório passa a mentir sobre qual dado
está ali (`dvc status` acusa). O script faz os dois e confere o md5 no fim:

```bash
./scripts/alternar_versao.sh v1     # 7043 clientes
./scripts/alternar_versao.sh v2     # 7143 clientes
```

Ele acha o commit pelo hash do conteúdo, não por número de commit — que muda a
cada rebase. O CI roda o ciclo inteiro (v1 → v2) a cada push: é o checkpoint do
encontro virando teste.

### O pipeline em `dvc.yaml`

O `dvc add` versiona o dado; o `dvc.yaml` versiona a **relação** entre dado,
código e modelo:

```yaml
stages:
  treino:
    cmd: uv run python -m churn.model
    deps: [data/churn.csv, src/churn]
    outs: [artifacts/churn_model.pkl]
```

```bash
uv run dvc repro     # roda só o que mudou; grava o dvc.lock
uv run dvc dag       # desenha as dependências
```

O `dvc.lock` guarda o hash exato de cada entrada e da saída — é a cadeia
**dados → código → modelo** escrita por quem a executou. Trocar para a v1 muda o
hash de `data/churn.csv`, e o `dvc repro` sabe sozinho que o modelo ficou velho.
Sem mudança nenhuma ele não reexecuta nada (`Stage 'treino' didn't change`).

O `.dvcignore` exclui `__pycache__/`: sem isso o hash de `src/churn` mudaria a
cada execução e o `repro` reexecutaria o treino sem nada ter mudado.

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
recupera a v1 do blob que ficou no histórico do Git (o dataset esteve versionado
até o Encontro 5) e, se o ponteiro pedir a v2, roda `scripts/gerar_v2.py` para
chegar nela. Em qualquer caminho **confere o md5 contra o `.dvc`**. É como o CI
materializa o dado — sem DVC instalado e sem remote, só `git` e a biblioteca
padrão do Python.

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
- **Enc. 5** ✓ — `data/` versionado com DVC: ponteiro, remote, duas versões com
  alternância (`scripts/alternar_versao.sh`) e o pipeline em `dvc.yaml`.
- **Enc. 6** — rastrear experimentos com MLflow.
