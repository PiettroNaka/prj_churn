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
│  ├─ tuning.py      # a busca aleatória em 6 eixos — o ajuste fino
│  ├─ lineage.py     # data_md5() e git_commit() — a costura com o DVC
│  ├─ serving.py     # ChurnPipeline — o modelo que aceita cliente cru
│  ├─ registry.py    # promove o vencedor a @champion (Encontro 7)
│  ├─ contrato.py    # o contrato traduzido: alimenta a API e o formulário
│  ├─ api.py         # FastAPI — o campeão atrás de HTTP
│  └─ evaluate.py    # métricas
├─ app/              # Streamlit: Inicio.py (v1) + pages/ (v2)
├─ tests/            # pytest
├─ data/             # churn.csv.dvc (ponteiro) — os bytes vêm do DVC
├─ scripts/          # preparar · relatorio (sem navegador) · fetch_data · mlflow_ui · prever · profiling · gerar_v2 · alternar_versao · quebra_proposital · conferir_run
├─ docs/             # nota de leitura do Data Pipeline
├─ ver-runs.bat      # Windows: duplo clique → o relatório dos runs
├─ abrir-mlflow.bat  # Windows: duplo clique → a UI do MLflow
├─ atualizar.bat     # Windows: duplo clique → traz a versão nova, desenrosca o Git
├─ Dockerfile.dados  # a imagem que busca o dado (dvc pull no deploy)
├─ Dockerfile.api    # a imagem da API
├─ Dockerfile.app    # a imagem do front
├─ compose.yml       # a pilha: dados · tuning · promover · api · app · mlflow
├─ dvc.yaml          # o pipeline: deps (dado + código) → outs (modelo)
├─ mlflow.db         # tracking dos experimentos (gitignored)
├─ .dvc/             # config do DVC: remote e cache
└─ pyproject.toml    # deps travadas (uv)
```

> Primeira vez na sua máquina? O passo a passo completo, com o que instalar e o
> que fazer quando der errado, está em [`docs/rodar-local.md`](docs/rodar-local.md).

## Como rodar (uv)

```bash
uv sync                          # cria o ambiente a partir do pyproject/uv.lock
./scripts/fetch_data.sh          # traz data/churn.csv (o Git não o guarda mais)
uv run python -m churn.model     # treina, imprime as métricas e registra o run
uv run python -m churn.experiments  # o grid: 9 runs + o melhor
uv run python -m churn.tuning       # a busca fina: 24 candidatos + treina o vencedor
uv run python -m churn.registry  # promove o vencedor a @champion
uv run --group api uvicorn churn.api:app          # a API em localhost:8000
uv run --group app streamlit run app/Inicio.py    # o front em localhost:8501
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

| | clientes | md5 | accuracy | roc_auc | |
|---|---|---|---|---|---|
| **v1** | 7043 | `d390bd07…` | 0,7570 | 0,8053 | ← **versão rastreada** |
| v2 | 7143 | `f150a64a…` | 0,7609 | 0,8150 | exercício do `dvc checkout` |

A v2 é a v1 mais 100 clientes: `scripts/gerar_v2.py` sorteia 100 clientes
existentes com semente fixa e os recadastra com identificador novo, copiando o
resto da linha byte a byte. Isso dá duas propriedades boas — a v2 nasce cumprindo
o contrato do Encontro 4, e os bytes são iguais em qualquer máquina.

> **A v2 não serve para treinar.** Aquelas 100 linhas são cópias exatas de
> clientes reais, só com `customerID` trocado. O split é aleatório por linha,
> então cliente e cópia caem em lados diferentes: medimos **44 pares** com uma
> metade no treino e a outra no teste. São 44 linhas que o modelo já viu, byte a
> byte, contadas como acerto — e é daí que vem o `roc_auc` maior, não de um
> modelo melhor. A v1 reproduz exatamente os números do material
> (`roc_auc 0,8195`, `accuracy 0,7683`, `f1 0,6151`); a v2 infla em +0,0026.
>
> Por isso a **v1 é a versão rastreada**, e o `registry.candidato()` filtra por
> `tags.data_md5`: um run treinado sobre outra versão do dataset não compete por
> uma vaga em produção. A v2 continua no histórico para o `dvc checkout` do
> encontro — é exercício de versionamento, não base de treino.

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
uv run python -m churn.model         # um run
uv run python -m churn.experiments   # o grid: 9 runs
./scripts/mlflow_ui.sh               # a UI em localhost:5000
```

No Windows, onde o PowerShell não roda `.sh`, o equivalente é
`uv run python scripts/preparar.py` — mesmo caminho, só a biblioteca padrão.

E quando o navegador não alcança o `localhost` (proxy corporativo, porta
bloqueada), `uv run python scripts/relatorio.py --html` escreve a mesma tabela
num arquivo estático que abre com duplo clique.

O `mlflow_ui.sh` faz o que falta antes de abrir: materializa o dataset se não
estiver no disco, treina uma vez se ainda não houver run nenhum (UI vazia não
mostra modelo), e só então sobe o servidor no mesmo backend que o treino usa.
Quem prefere container: `docker compose up mlflow`.

> O caminho dos artefatos é gravado **absoluto** no banco. Runs criados no host
> (artefatos em `<repo>/mlruns`) e runs criados em container (`/app/mlruns`) não
> se misturam: as métricas aparecem nos dois casos, mas o link do artefato só
> resolve no mundo em que o run nasceu. Escolha um e fique nele.

**Duas coisas que confundem ao abrir a UI 3.16 pela primeira vez** — as duas
custaram tempo aqui, então ficam registradas:

- Ela **abre na aba GenAI** (traces, judges, playground), que não tem nada a ver
  com este projeto e aparece vazia. Os runs estão em **"Model training"**, no
  canto superior esquerdo, ou direto em `#/experiments/1/runs`.
- As **colunas de métrica não vêm ligadas**. A tabela mostra só nome, data e
  duração; para ver `roc_auc`, `accuracy` e as tags de linhagem, use o botão
  **Columns**. Para ordenar sem clicar em nada, o endereço aceita os parâmetros:

  ```
  #/experiments/1/runs?orderByKey=metrics.roc_auc&orderByAsc=false
  ```

A telemetria do MLflow 3 (`config.mlflow-telemetry.io`) fica desligada por
`MLFLOW_DISABLE_TELEMETRY` — num projeto de curso ela não serve para nada.

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
data_md5   d390bd07b5514a2256a2396993b8b0e3   ← o ponteiro .dvc (Encontro 5)
git_commit d7ee6e5                            ← o commit do código
```

`src/churn/lineage.py` lê as duas respostas e o treino as grava como tags. Dentro
do container não há `.git` — a imagem carrega o código, não o repositório —, então
quem sabe o commit informa: `CHURN_GIT_COMMIT` tem precedência, e o CI passa o
`GITHUB_SHA` do job. Sem isso, a metade "qual código" ficaria em branco justamente
no modelo que o CI produz. O DVC responde "qual dado e qual código"; o MLflow responde "qual experimento, com
quais params e métricas"; a tag é o fio entre eles. Sem ponteiro `.dvc` por perto,
`data_md5()` cai no md5 do próprio arquivo — mesma resposta, sem a garantia de
que a versão está registrada. E `git_commit()` marca `+alteracoes` quando há
mudança não commitada: um run treinado sobre código não commitado aponta para um
commit que não contém aquele código, e a tag precisa dizer isso. (O Git chama
esse estado de *dirty*; aqui "sujo" já é o dado fora do contrato do Encontro 4,
então o marcador usa outra palavra de propósito.)

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

### Rodar o modelo

O run guarda o modelo; para usá-lo não é preciso escrever Python:

```bash
uv run python scripts/prever.py              # 5 clientes do dataset
uv run python scripts/prever.py --n 20
uv run python scripts/prever.py --csv clientes.csv
uv run python scripts/prever.py --uri models:/m-abc123
```

```
customerID  tenure       Contract real churn  probabilidade  acertou
 7004-MAFB       8 Month-to-month  Yes   Yes          0.815     True
 7001-QKTG      16       Two year   No    No          0.035     True
```

A entrada é a linha **crua**, com `customerID` e `TotalCharges` em texto, como
sai do sistema de origem. Isso não é detalhe: o estimador sozinho recusa essa
linha —

```
ValueError: The feature names should match those that were passed during fit.
```

— porque foi treinado com features derivadas e categóricas já codificadas. Por
isso o que vai para o MLflow é o **pipeline inteiro** (`churn.serving`,
`pyfunc`): limpeza, features, encoder e estimador num objeto só, na mesma ordem
que o treino usou. Ordem diferente aqui seria *training/serving skew* — o modelo
veria em produção um dado diferente do que viu no treino.

O pacote vai dentro do artefato (`code_paths`), então o modelo carrega onde o
projeto não está instalado. Para servir como HTTP em vez de rodar em lote:

```bash
uv run mlflow models serve -m runs:/<run_id>/model --env-manager local
curl -X POST localhost:5000/invocations -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

O CI roda o modelo a cada push: carrega do MLflow e prevê a partir de linhas
cruas. Modelo que não roda reprova o build.

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

E um desvio que não é da versão, é de projeto: o material loga
`mlflow.sklearn.log_model(model, …)`, o estimador solto. Aqui o que se loga é o
`pyfunc` do pipeline inteiro, pela razão da seção anterior — o estimador solto
não aceita um cliente. O flavor sklearn continua lá dentro; o que muda é a
fronteira do que foi empacotado.


## Ajuste fino (`churn.tuning`)

O grid varre nove pontos em dois eixos. A floresta tem mais botões, e os que
ficaram de fora mexem no resultado tanto quanto a profundidade. A busca abre
seis eixos e sorteia 24 das 3.150 combinações:

```bash
uv run python -m churn.tuning
```

Ela termina treinando o vencedor pelo caminho normal, então o que sai é um run
promovível — não um relatório.

### O que mudou no modelo

Mesmo dado (v1), mesmo split, mesma semente. A diferença é só a configuração:

| | padrão | ajustado | |
|---|---|---|---|
| roc_auc | 0.8053 | **0.8206** | +0.015 |
| **recall** | 0.5602 | **0.7609** | **+0.20** |
| f1 | 0.6000 | **0.6662** | +0.066 |
| precision | 0.6459 | 0.5924 | −0.054 |
| accuracy | 0.7570 | 0.7518 | −0.005 |

A configuração vencedora: `n_estimators=200`, `max_depth=8`,
`min_samples_leaf=16`, `min_samples_split=5`, `max_features=log2`,
`class_weight=balanced`.

O ganho que importa não é o `roc_auc`. É o **recall**: de 56% para 76% dos
clientes que de fato cancelaram. Num problema de churn os dois erros não custam
o mesmo — um falso positivo gasta um desconto, um falso negativo perde o
cliente. Foi `class_weight=balanced` que comprou esses 20 pontos, pagando 5 de
precisão. A acurácia caiu, e é por isso que ela não decide nada aqui: com 27% de
churn na base, quem chuta "ninguém cancela" acerta 73%.

### Três decisões que mudam o que o número significa

**Busca aleatória, não exaustiva.** O grid completo das 3.150 combinações
levaria horas para ganhar a terceira casa decimal. Sortear 24 pontos encontra
quase sempre a mesma região, em minutos — o resultado de Bergstra & Bengio
(2012), que vale justamente quando poucos hiperparâmetros dominam.

**Validação cruzada no treino, teste intocado.** Escolher configuração olhando o
teste é estudar com o gabarito. A busca pontua por CV dentro do treino; o teste
aparece uma vez, no fim, para medir o vencedor. O `roc_auc` da CV e o do teste
não batem — e não baterem é o esperado.

**Mesmo split do resto do projeto.** A busca parte do mesmo
`dados_codificados` que o grid, com a mesma semente. É o que torna os números
das três varreduras comparáveis entre si.

### Dizer o vencedor

A busca imprime as variáveis que o reproduzem:

```
CHURN_N_ESTIMATORS=200   CHURN_MAX_DEPTH=8        CHURN_MAX_FEATURES=log2
CHURN_MIN_SAMPLES_LEAF=16  CHURN_MIN_SAMPLES_SPLIT=5  CHURN_CLASS_WEIGHT=balanced
```

Repare no `null` quando um valor é nulo (`CHURN_MAX_DEPTH=null`). Vazio não
serve: `CHURN_MAX_DEPTH=` levanta `ValidationError`, e `CHURN_MAX_FEATURES=`
é pior — passa, vira a string vazia e treina **outro modelo**, calado. Quem
resolve é `env_parse_none_str="null"` na `Settings`, e há teste amarrando as
duas pontas.

## Model Registry (Encontro 7)

A varredura acha o melhor run. Semanas depois, "qual era mesmo?" não se responde
rolando uma lista — e um run é um experimento, não um artefato de produção. O
Registry dá o endereço estável.

```bash
uv run python -m churn.tuning     # 1. busca, e já treina o vencedor (run promovível)
uv run python -m churn.registry   # 2. registra e promove
```

Pelo grid, se preferir os nove pontos do Encontro 6, o passo do meio é manual:

```bash
uv run python -m churn.experiments              # 1. acha a melhor configuração
CHURN_MAX_DEPTH=8 uv run python -m churn.model  # 2. treina e loga o vencedor
uv run python -m churn.registry                 # 3. registra e promove
```

Três peças: o **registered model** (`churn`) agrupa versões; cada **version** é
imutável; o **alias** `@champion` é um ponteiro móvel para a que serve. Promover
é mover o alias; reverter é movê-lo de volta — `registry.apontar(1)` devolve a
produção à versão 1 sem treinar, sem registrar e sem tocar em servidor.

Aliases e não stages: `Staging`/`Production` estão depreciados no MLflow 3.

### Qual run é promovível

Os filhos do grid pontuam mas **não logam modelo** — são nove florestas de até
500 árvores, e a varredura existe para escolher a configuração, não para guardar
candidatos. Quem loga o pipeline é o `churn.model`. Então `candidato()` filtra
pelo nome de run do treino; promover um filho do grid falharia no artefato
ausente, e a mensagem de erro manda rodar o treino em vez de estourar um
traceback.

### O portão

`promover()` só registra se a métrica alcançar `mlflow_promocao_minima` (0.80).
Abaixo disso levanta `PromocaoRecusada`. Um portão que deixa passar é pior que
nenhum: dá a impressão de que alguém conferiu.

### Consumir

```python
from churn.registry import carregar_campeao

modelo = carregar_campeao()           # models:/churn@champion
modelo.predict(cliente_cru)           # 21 colunas, como saem do sistema
```

Verificado de ponta a ponta: `churn v2 @champion`, treinado com `max_depth=8`
sobre a **v1 real** do dataset (`roc_auc 0.8195`, `accuracy 0.7683`, `f1 0.6151`
— os três iguais aos do material; `data_md5 d390bd07…`), carregado pelo alias e
respondendo a uma linha crua com `churn` e `probabilidade`.

A versão 1 do modelo registrado ficou para trás de propósito: ela veio de um run
treinado sobre a v2, com a métrica inflada pelas linhas duplicadas. O filtro por
`data_md5` passou a impedir que um run assim volte a ser promovido.

### Uma pegadinha que custou uma depuração

`carregar_campeao` precisa fixar a tracking URI **antes** do load. Sem isso o
MLflow procura o registry no backend padrão e responde `Registered Model with
name=churn not found` — com o modelo registrado. É o mesmo atrito de runs
espalhados do Encontro 6, reaparecendo na hora de consumir. Há um teste de
regressão travando a ordem das chamadas.


## A API (`churn.api`)

O Registry dá o endereço estável; a API dá a porta. Quem consome não instala o
projeto, não conhece o MLflow e não sabe qual run venceu.

```bash
uv run --group api uvicorn churn.api:app --reload   # localhost:8000/docs
```

| rota | o que faz |
|---|---|
| `POST /prever` | um lote de clientes em JSON → rótulo e probabilidade, na ordem |
| `GET /saude` | pronto para responder? é o que o healthcheck do compose pergunta |
| `GET /modelo` | qual versão está servindo, e de qual run ela veio |
| `GET /docs` | a documentação interativa, gerada do contrato |

### O contrato de entrada não é escrito — é gerado

Os campos que a API aceita saem do `ChurnRawSchema`, o mesmo contrato que valida
o CSV no treino. Categoria vira `Literal` com os valores reais; faixa numérica
vira `Field(ge=…, le=…)` com o piso e o teto do profiling.

Uma lista paralela aqui seria *training/serving skew* na forma mais discreta: o
dia em que o schema ganhasse uma categoria, a API continuaria recusando-a e
ninguém ligaria os dois fatos. Há um teste que falha se os dois conjuntos
divergirem. Efeito colateral melhor do que o esperado — o `/docs` sai com os
domínios reais em cada campo, porque o OpenAPI vem da mesma fonte.

### A assinatura que pedia a resposta de volta

Na primeira chamada real, o modelo recusou:

```
Model is missing inputs ['customerID', 'Churn']
```

Para prever o churn era preciso **informar o churn**. A assinatura vinha de
`infer_signature` sobre o CSV cru inteiro, e o MLflow trata assinatura como
contrato: exige toda coluna declarada. O `ChurnPipeline` sempre descartou as
duas colunas — só a assinatura não sabia. O defeito estava lá desde que o
pipeline passou a ser logado, invisível porque o único teste de ponta a ponta
mandava a linha completa, com `Churn` e tudo.

Agora a assinatura sai de `entrada_de_servico(exemplo)`, e um teste lê a
assinatura do modelo logado para garantir que o alvo nunca volte a ser exigido.

## O front em Streamlit

Duas páginas, duas versões.

```bash
uv run --group app streamlit run app/Inicio.py   # localhost:8501
```

**v1 — `app/Inicio.py`.** Título, código e dados. Não prevê nada, de propósito:
um front que calcula antes de deixar ver o dado e o código vira uma caixa-preta
bonita, e a caixa-preta é o que o curso passa doze encontros desmontando. O
seletor mostra qualquer módulo de `src/churn/`; as abas mostram a tabela, o tipo
de cada coluna e o resumo estatístico.

**v2 — `app/pages/1_Prever_churn.py`.** Os parâmetros de um cliente e o churn
calculado. O formulário é **gerado** do mesmo contrato: categoria vira
`selectbox` com as opções reais, faixa vira `slider` com o piso e o teto. Não há
como montar ali um cliente que a API recuse por domínio.

O cálculo não acontece na página — ela chama a API. É o que separa o front do
modelo: trocar de campeão é mover um alias, sem tocar numa linha do front.

### `churn.contrato` — o contrato para quem não fala pandera

A API e o formulário precisam da mesma informação (quais valores, qual faixa) e
nenhum dos dois deveria aprender a ler `Check.statistics` por dentro. O
`churn.contrato` traduz o schema em três perguntas — `permitidos`, `minimo`/
`maximo`, `inteiro` — e é o que impede as duas telas de manterem cada uma a sua
cópia da lista de categorias.

Um detalhe que ele resolve: `TotalCharges` do formulário é **derivado** de
`tenure × MonthlyCharges`. O contrato exige essa relação (10% de tolerância), e
tomar o meio da faixa de cada campo isoladamente produzia um cliente de exemplo
que violava a regra — quem abrisse a tela veria um aviso antes de digitar nada.

## Tudo em Docker

```bash
docker compose up dados       # 1. o CSV chega por dvc pull
docker compose up tuning      # 2. a busca, e o treino do vencedor
docker compose up promover    # 3. registra e aponta @champion
docker compose up app         # 4. sobe a API e o front juntos
```

| serviço | porta | o que é |
|---|---|---|
| `dados` | — | `dvc pull`; roda até o fim e sai |
| `treino` / `tuning` | — | treino simples / busca + vencedor |
| `promover` | — | registra o melhor run e move o alias |
| `api` | 8000 | o campeão atrás de HTTP |
| `app` | 8501 | as duas páginas do front |
| `mlflow` | 5000 | a UI dos experimentos |

O `app` declara `depends_on: api (service_healthy)`, e "saudável" aqui significa
**com modelo carregado**, não "a porta respondeu" — subir o front contra uma API
sem campeão só adiantaria o erro para quem está olhando.

### A pegadinha do caminho absoluto

O MLflow grava o caminho do artefato **absoluto** no banco. Um modelo treinado no
host (em `<repo>/mlruns`) não resolve dentro do container, e vice-versa: a API
subiria e diria "sem modelo" sem nenhuma pista do porquê. Por isso a pilha treina
no container — `tuning` e `api` compartilham o mesmo `/app/mlruns`.

Se você treinou no host e quer a API em Docker, escolha um mundo: ou tudo local
(`uvicorn` direto), ou tudo em compose.

### O que o CI prova

Os testes de unidade da API rodam dentro da imagem de treino — curiosamente, sem
que ela instale o grupo `api`: o próprio MLflow depende de `fastapi`, então ele
já está lá. (O que faltava era o `httpx` do `TestClient`, e por isso ele entrou
no grupo `dev`.) O `importorskip` nos testes é guarda para um ambiente sincronizado
sem os grupos padrão, não para este.

A cobertura que o teste de unidade **não** dá é a do container: `docker compose`
sobe a API, espera o healthcheck e roda `scripts/conferir_api.py` contra o
serviço pela rede. É o que prova que a imagem constrói, carrega o campeão e
serve — três coisas que nenhum `TestClient` em processo verifica.

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
