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
│  ├─ quarentena.py  # separa o lote reprovado em vez de treinar com ele
│  ├─ data.py        # carga + validação + limpeza
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ model.py       # treino (entrypoint) — instrumentado com MLflow
│  ├─ experiments.py # o grid rastreado
│  ├─ lineage.py     # data_md5() e git_commit() — a costura com o DVC
│  └─ evaluate.py    # métricas
├─ tests/            # pytest
├─ data/             # churn.csv.dvc (ponteiro) — os bytes vêm do DVC
├─ scripts/          # fetch_data · profiling · quebra_proposital · gerar_v2 · alternar_versao · conferir_run
├─ docs/             # nota de leitura do Data Pipeline
├─ Dockerfile.dados  # a imagem que busca o dado (dvc pull no deploy)
├─ dvc.yaml          # o pipeline: deps (dado + código) → outs (modelo)
├─ mlflow.db         # tracking dos experimentos (gitignored)
├─ .dvc/             # config do DVC: remote e cache
└─ pyproject.toml    # deps travadas (uv)
```

## Como rodar (uv)

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
./scripts/fetch_data.sh          # traz data/churn.csv (o Git não o guarda mais)
uv run python -m churn.model     # treina, imprime as métricas e registra o run
uv run python -m churn.experiments  # o grid: 9 runs + o melhor
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # localhost:5000
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

O `compose up` sobe dois serviços: `dados` busca o CSV com `dvc pull` (imagem
`Dockerfile.dados`, que tem o DVC e o ponteiro, mas não os bytes) e, só quando
ele termina bem, `treino` roda. O dado nunca é camada de imagem — nem da imagem
de treino, que segue enxuta, sem DVC dentro.

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

São as três ações que o encontro pede diante de um lote ruim: **falhar cedo**
(antes do treino, não depois da métrica cair), **logar os `failure_cases`** e
**quarentenar o lote** — `artifacts/quarentena/<timestamp>/` guarda as linhas
ofensivas e o relatório. Quarentena não substitui a falha; ela existe porque
relatório sobre um arquivo que já foi embora não conserta nada. Quem escreve é a
orquestração (`load_clean`), não o validador: `validate_raw` e `validate_clean`
respondem se o dado serve, sem tocar em disco.

### Conhecer antes de validar

As faixas saem do profiling, não de chute:

```bash
uv run python scripts/profiling.py            # observado vs. o que o contrato exige
uv run python scripts/profiling.py --html     # + o relatório completo do ydata-profiling
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

O `--html` gera o relatório completo do encontro (correlações, alertas,
distribuições) em `artifacts/profiling.html`. O `ydata-profiling` **não** está no
`pyproject.toml` de propósito: como dependência do projeto ele puxa numpy e scipy
para versões mais antigas, e o lock é único — uma ferramenta de relatório passaria
a ditar a pilha numérica do treino, justo o que os Encontros 2 e 3 travaram. Ele
roda em ambiente efêmero (`uvx`), sem encostar no ambiente travado.

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

### O deploy: o dado vem por `dvc pull`

`Dockerfile.dados` é uma imagem com o DVC e o ponteiro — sem os bytes. Ela roda
`dvc pull`, escreve o CSV no volume compartilhado e sai; a imagem de treino,
enxuta e sem DVC dentro, encontra o dado montado. É o `docker compose up`:

```yaml
dados:   # roda primeiro, busca o dado
  build: {dockerfile: Dockerfile.dados}
  volumes: [./data:/app/data, ../dvc-storage:/dvc-storage]
  command: pull
treino:
  depends_on: {dados: {condition: service_completed_successfully}}
```

O remote é montado em `/dvc-storage` porque é para lá que o `../../dvc-storage`
da config aponta, visto de `/app/.dvc/config`. Dentro da imagem não há `.git` —
ela é o deploy, não o repositório — então o build liga `core.no_scm`.

O CI prova o caminho inteiro em vez de afirmá-lo: semeia o remote com o próprio
DVC dentro do container, **apaga** `data/churn.csv`, e um container novo tem de
trazê-lo de volta do remote com o md5 batendo com o ponteiro.

### E quem clona hoje?

Um remote local não viaja junto com o clone. Por isso existe
`scripts/fetch_data.sh`: tenta `dvc pull` e, se não houver remote alcançável,
recupera a v1 do blob que ficou no histórico do Git (o dataset esteve versionado
até o Encontro 5) e, se o ponteiro pedir a v2, roda `scripts/gerar_v2.py` para
chegar nela. Em qualquer caminho **confere o md5 contra o `.dvc`**. É como o CI
materializa o dado — sem DVC instalado e sem remote, só `git` e a biblioteca
padrão do Python.

## Rastreamento de experimentos (Encontro 6)

O `print(accuracy)` mostra o número e o esquece. Cada execução do treino agora
vira um **run**: o que foi escolhido, o que foi medido, o modelo e — a parte que
fecha o bloco anterior — de qual dado e de qual código aquilo saiu.

```bash
uv run python -m churn.model                              # um run
uv run python -m churn.experiments                        # o grid: 9 runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # localhost:5000
```

A regra que evita a confusão de sempre:

| | é | exemplo aqui |
|---|---|---|
| **param** | o que **você escolheu** | `n_estimators`, `test_size`, `max_depth` |
| **metric** | o que **você mediu** | `roc_auc`, `accuracy`, `f1` |
| **artifact** | um **arquivo** gerado | o modelo, o pickle com o encoder |
| **tag** | **metadado** do run | `data_md5`, `git_commit` |

A config Pydantic vira params numa linha (`cfg.model_dump()`) — os dois contratos
do curso se encontram aqui. Fora as chaves `mlflow_*`: elas dizem *onde anotar*,
não *o que foi treinado*.

### A linhagem: onde o DVC e o MLflow se costuram

```
data_md5   f150a64ab5537f24b16770ba76ad7b95   ← o ponteiro .dvc (Encontro 5)
git_commit d7ee6e5                            ← o commit do código
```

`src/churn/lineage.py` lê as duas respostas e o treino as grava como tags. O DVC
responde "qual dado e qual código"; o MLflow responde "qual experimento, com
quais params e métricas"; a tag é o fio entre eles. Sem ponteiro `.dvc` por perto,
`data_md5()` cai no md5 do próprio arquivo — mesma resposta, sem a garantia de
que a versão está registrada. E `git_commit()` marca `-sujo` quando há mudança
não commitada: um run treinado sobre código não commitado aponta para um commit
que não contém aquele código, e a tag precisa dizer isso.

O CI cobra esse fio a cada push (`scripts/conferir_run.py`): o run tem de existir,
ter métricas, e a tag `data_md5` tem de bater com o md5 do ponteiro. Um modelo
treinado com outro dado que não o versionado reprova o build.

### O que o grid achou

| run | n_estimators | max_depth | roc_auc | accuracy |
|---|---|---|---|---|
| **rf-n500-d8** | 500 | 8 | **0,8220** | 0,7749 |
| rf-n200-d8 | 200 | 8 | 0,8220 | 0,7727 |
| rf-n100-d8 | 100 | 8 | 0,8215 | 0,7732 |
| *rf* (o default) | 200 | — | 0,8150 | 0,7609 |

Podar a árvore em 8 níveis vale mais que triplicar o número delas — e o default
atual (sem poda) perde para as nove combinações com `max_depth` definido. O campo
`max_depth` existe na config para que o vencedor seja dizível (`CHURN_MAX_DEPTH=8`),
mas **promover** o vencedor é assunto do Encontro 7, com o Model Registry.

Os runs do grid ficam aninhados sob um run-pai: a varredura é uma coisa só na UI.
E o modelo não é logado nos nove — são florestas de até 500 árvores, e guardar
todas custa centenas de MB para comparar cinco números.

### Dois desvios do material, por causa da versão instalada

O guia foi escrito para o backend de arquivos (`./mlruns`) e para `artifact_path=`.
Com **MLflow 3.16**:

- **`./mlruns` é recusado.** O erro é explícito: *"the filesystem tracking backend
  is in maintenance mode"*, e só abre exceção com `MLFLOW_ALLOW_FILE_STORE=true`.
  O padrão aqui é `sqlite:///mlflow.db` — o caminho que o próprio MLflow
  recomenda, e que o material já indica na tabela de backends. Os artefatos ficam
  em `mlruns/`, tudo gitignored.
- **`artifact_path=` virou `name=`** em `log_model`. Não é só um rename: o modelo
  deixa de ser artefato do run e vira uma entidade própria (*logged model*).
  Carregar continua funcionando pelos dois endereços — `runs:/<run_id>/model` e
  `models:/<model_id>` —, e há um teste garantindo isso.


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
- **Enc. 6** ✓ — experimentos rastreados com MLflow: runs, grid e a linhagem
  `data_md5` + `git_commit` costurada ao DVC.
- **Enc. 7** — Model Registry: promover o vencedor do grid.
