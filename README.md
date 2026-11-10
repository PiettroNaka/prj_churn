# prj_churn

Projeto-fio-condutor do curso de MLOps: previsão de **churn** (classificação
binária tabular, dataset Telco). O repositório é o caminho completo do script
monolítico `train_churn.py` até um modelo **validado, versionado, rastreado,
promovido, servido por HTTP e consumido por um front** — tudo reproduzível em
container.

O que o projeto responde, no fim: *dado um cliente, qual a probabilidade de ele
cancelar — e de qual dado e de qual código essa resposta saiu?*

---

## Sumário

| | |
|---|---|
| [O que está pronto](#o-que-está-pronto) | a pilha, encontro por encontro |
| [Como rodar](#como-rodar) | local e em Docker |
| [**Como a previsão melhorou**](#como-a-previsão-melhorou) | os números, e quanto deles é mérito do modelo |
| [Como foi implementado](#como-foi-implementado) | cada camada, e a decisão por trás |
| [**Dificuldades encontradas**](#dificuldades-encontradas) | os erros reais, e o que cada um ensinou |
| [**Ressalvas e limites**](#ressalvas-e-limites) | onde este projeto não serve |
| [Testes e CI](#testes-e-ci) | o que é provado a cada push |
| [Estrutura](#estrutura) | o mapa dos arquivos |

---

## O que está pronto

| Encontro | Entrega | Onde |
|---|---|---|
| 2 | config como contrato (Pydantic), pacote modular, ambiente travado (uv) | `config.py`, `pyproject.toml` |
| 3 | reprodutibilidade em container; dado por volume, nunca em camada | `Dockerfile`, `compose.yml` |
| 4 | contrato dos dados executável, com quarentena e alerta vs. erro | `schema.py`, `validation.py`, `quarentena.py` |
| 5 | dado versionado por ponteiro; duas versões; deploy por `dvc pull` | `data/churn.csv.dvc`, `dvc.yaml` |
| 6 | cada treino vira run, com linhagem `data_md5` + `git_commit` | `model.py`, `experiments.py`, `lineage.py` |
| 7 | promoção a `@champion` com portão de métrica e rollback por alias | `registry.py` |
| — | **ajuste fino**: busca aleatória em seis eixos | `tuning.py` |
| — | **API**: o campeão atrás de HTTP, com contrato gerado do schema | `api.py` |
| — | **front v1 e v2**: ver o projeto, e calcular o churn | `app/` |

---

## Como rodar

### Na sua máquina

```bash
uv sync                               # ambiente travado pelo uv.lock
./scripts/fetch_data.sh               # traz data/churn.csv (o Git não o guarda)

uv run python -m churn.tuning         # busca + treina o vencedor  (~2 min)
uv run python -m churn.registry       # registra e promove a @champion

uv run --group api uvicorn churn.api:app         # API  → localhost:8000/docs
uv run --group app streamlit run app/Inicio.py   # front → localhost:8501
```

Os quatro de uma vez, que é como se apresenta isto:

```bash
uv run --group api --group app python scripts/subir_front.py
```

Ele confere o dataset, promove um campeão se não houver, sobe a API em segundo
plano, espera ela responder e só então abre o Streamlit — encerrando os dois
juntos no fim. No Windows é o **`subir-front.bat`**, em duplo clique.

Outros pontos de entrada:

```bash
uv run python -m churn.model          # um treino simples
uv run python -m churn.experiments    # o grid do Encontro 6: 9 runs
./scripts/mlflow_ui.sh                # a UI em localhost:5000
uv run python scripts/relatorio.py --html   # os runs sem navegador nem servidor
uv run python scripts/prever.py --n 10      # prever em lote, pela linha de comando
uv run python scripts/conferir_api.py       # a API está servindo mesmo?
uv run pytest                         # 122 testes
```

Os testes **não precisam do dataset** — usam amostras de `tests/conftest.py`.
Quem precisa dele é o treino.

> **Windows:** o PowerShell não roda `.sh`. Há quatro arquivos de duplo clique na
> raiz — `subir-front.bat`, `ver-runs.bat`, `abrir-mlflow.bat` e
> `atualizar.bat` — e o equivalente
> em Python puro para o resto: `uv run python scripts/preparar.py`. O passo a
> passo completo está em [`docs/rodar-local.md`](docs/rodar-local.md).

### Em Docker

```bash
docker compose up dados       # 1. o CSV chega por dvc pull
docker compose up tuning      # 2. a busca, e o treino do vencedor
docker compose up promover    # 3. registra e aponta @champion
docker compose up app         # 4. sobe a API e o front juntos
```

| serviço | porta | o que é |
|---|---|---|
| `dados` | — | `dvc pull`; roda até o fim e sai |
| `treino` · `tuning` | — | treino simples · busca + vencedor |
| `promover` | — | registra o melhor run e move o alias |
| `api` | 8000 | o campeão atrás de HTTP |
| `app` | 8501 | as duas páginas do front |
| `mlflow` | 5000 | a UI dos experimentos |

O `app` declara `depends_on: api (service_healthy)`, e **saudável aqui significa
com modelo carregado**, não "a porta respondeu". Subir o front contra uma API sem
campeão só adiantaria o erro para quem está olhando.

### Publicar no Streamlit Community Cloud

| campo | valor |
|---|---|
| Repository | `PiettroNaka/prj_churn` |
| Branch | `main` |
| Main file path | `app/Inicio.py` |

Duas coisas tornam isso possível, e nenhuma delas é padrão num projeto que usa
DVC e MLflow:

**O `requirements.txt`.** A plataforma instala por pip, não por uv. A última
linha dele é `.` — é o que instala o pacote a partir do `pyproject.toml` e faz
`from churn.config import settings` funcionar. Sem ela o app quebra na primeira
linha de import, antes de desenhar qualquer coisa.

**O modo demonstração.** Na nuvem não há API, não há MLflow e não há o CSV, que
saiu do Git no Encontro 5. O front detecta isso e se vira: busca o dataset por
HTTP do commit em que ele ainda era versionado — **conferindo o md5 contra o
ponteiro `.dvc`**, transporte diferente e mesma verificação — e treina o modelo
na própria sessão, com a configuração que a busca encontrou.

A barra lateral diz qual dos dois respondeu, sempre. Um número que sai de um
modelo treinado na hora não é o mesmo que sai do campeão promovido, e quem olha
tem o direito de saber qual é qual.

> O modo demonstração é vitrine, não arquitetura. O desenho do projeto é
> front → API → Registry, e é o que roda em `docker compose up app` ou no
> `subir-front.bat`. Para apontar a vitrine para uma API hospedada de verdade,
> basta definir `CHURN_API_URL` nos *secrets* da plataforma.

---

## Como a previsão melhorou

Esta é a seção que responde "o que vocês fizeram para o modelo prever melhor".
Ela tem três partes: o que mudou, quanto disso é mérito do modelo, e o que foi
feito **no dado** para que a métrica não mentisse.

### Os três estágios, medidos na mesma base

Dataset v1 (7 043 clientes), mesmo split, mesma semente. Só a configuração muda:

| | roc_auc | **recall** | f1 | precision | accuracy |
|---|---|---|---|---|---|
| padrão (Encontro 2) | 0,8053 | 0,5602 | 0,6000 | 0,6459 | 0,7570 |
| grid (Encontro 6, `max_depth=8`) | 0,8195 | 0,5689 | 0,6151 | 0,6694 | **0,7683** |
| **ajuste fino (6 eixos)** | **0,8206** | **0,7609** | **0,6662** | 0,5924 | 0,7518 |

Configuração vencedora: `n_estimators=200`, `max_depth=8`, `min_samples_leaf=16`,
`min_samples_split=5`, `max_features=log2`, `class_weight=balanced`.

### O que foi feito, em ordem

**1. Podar a árvore (Encontro 6).** O grid de nove pontos mostrou que limitar a
profundidade em 8 vale mais que triplicar o número de árvores: `roc_auc` de
0,8053 para 0,8195. Floresta sem poda decora o treino.

**2. Abrir a busca para seis eixos (`churn.tuning`).** O grid varria
`n_estimators` e `max_depth`. A floresta tem mais botões — amostras por folha,
amostras por split, features por split, peso das classes — e eles mexem no
resultado tanto quanto a profundidade. A busca sorteia 24 das 3 150 combinações,
com validação cruzada **dentro do treino** e o teste intocado até o fim.

**3. Pesar as classes.** `class_weight=balanced` é o que move o recall de 0,57
para 0,76. Num problema de churn os dois erros não custam o mesmo: um falso
positivo gasta um desconto, um falso negativo perde o cliente. O modelo passou a
errar para o lado mais barato.

### A ressalva honesta: quanto disso é modelo?

Pouco. E vale dizer isso em vez de comemorar o recall.

Contra o **melhor anterior** (não contra o padrão), o ganho de `roc_auc` é de
**+0,0011** — ruído. `roc_auc` mede a capacidade de *ordenar* os clientes por
risco, e essa capacidade praticamente não mudou.

O que mudou foi o **ponto de operação**. E quase todo esse efeito se reproduz
sem tocar no modelo, só baixando o corte de decisão. Medido:

| no mesmo recall (0,7609) | precision | f1 | accuracy |
|---|---|---|---|
| modelo do grid, corte movido de 0,50 → **0,34** | 0,5860 | 0,6621 | 0,7473 |
| modelo ajustado, corte em 0,50 | 0,5924 | 0,6662 | 0,7518 |

A diferença real do ajuste fino, a recall igual, é **+0,006 de precisão**. O
resto era alcançável com uma linha de `if probabilidade >= 0.34`.

Isso não torna o ajuste inútil — ele entrega a mesma operação sem exigir que
quem consome saiba calibrar corte, e os seis eixos confirmaram que a
configuração do Encontro 6 já estava perto do teto deste modelo. Mas a frase
honesta é: **o ganho foi de operação, não de capacidade preditiva.** Para mexer
no `roc_auc` de verdade seria preciso trocar de família de modelo (gradient
boosting) ou criar features novas — nenhum dos dois está neste escopo.

Por isso o front mostra a **probabilidade** em destaque e diz, em texto, que o
corte em 50% é convenção: para uma campanha com orçamento limitado, o que vale é
ordenar a carteira por essa coluna e descer até o orçamento acabar.

### O que protegeu a métrica no dado

Melhorar a previsão não foi só mexer em hiperparâmetro. Três decisões no dado
evitaram que o número subisse por motivo errado:

- **As 11 linhas em branco não foram descartadas.** `TotalCharges` vem vazio em
  11 das 7 043 linhas, todas com `tenure = 0`: clientes que não fecharam o
  primeiro ciclo. Quatro são churners. Jogá-las fora tiraria sinal da classe
  minoritária; em vez disso o contrato **exige** a relação (branco só com
  `tenure = 0`).
- **A v2 foi barrada do treino.** Ela tem 100 linhas que são cópias exatas de
  clientes reais com `customerID` novo. Medimos **44 pares** com uma metade no
  treino e a outra no teste — 44 linhas que o modelo já viu, byte a byte,
  contadas como acerto. O `roc_auc` maior da v2 (0,8150 contra 0,8053) vem daí,
  não de um modelo melhor.
- **O Registry filtra por `data_md5`.** Depois desse episódio,
  `registry.candidato()` só considera runs treinados sobre a versão do dataset
  que o ponteiro `.dvc` aponta agora. Comparar métrica entre runs que viram
  dados diferentes não compara nada — e a versão inflada chegou a ser promovida
  uma vez, antes do filtro existir.

---

## Como foi implementado

### Configuração e ambiente (Encontro 2)

Todo caminho, hiperparâmetro e constante mágica vive em `config.py`, numa
`BaseSettings` do Pydantic — sobrescrevível por `CHURN_*` ou `.env`. O script
monolítico tinha `fillna(2200)` e caminhos fixos no meio da lógica.

O ambiente é travado por `pyproject.toml` + `uv.lock`. Grupos de dependência
separam o que cada imagem precisa: `data` (DVC), `api` (FastAPI, uvicorn), `app`
(Streamlit), `dev` (pytest, httpx). A imagem de treino não carrega servidor HTTP
nem Streamlit.

### Container (Encontro 3)

O `uv.lock` trava os pacotes Python; o container trava o resto — interpretador,
SO, libs de sistema. Dados e artefatos entram por volume, nunca viram camada.

```bash
docker build -t churn:0.1 .
docker run -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" churn:0.1
```

A regra de ouro das camadas está nos quatro Dockerfiles: dependências **antes**
do código, para que mudar um `.py` não reinstale o mundo.

### Validação de dados (Encontro 4)

O contrato é uma **classe** — `DataFrameModel` do pandera, a mesma intuição do
Pydantic do Encontro 2: lá o contrato é da config, aqui é dos dados. Sendo
código, ele versiona, é revisado em PR e roda no CI.

O pipeline valida **duas vezes**: o CSV cru, antes de qualquer transformação, e o
DataFrame limpo, antes de virar feature. A coerção acontece entre os dois, à
vista.

| Camada | Cobre |
|---|---|
| schema | as 21 colunas e seus tipos (`coerce=True`, `strict=False`) |
| domínio | as categorias válidas de cada coluna, `customerID` único |
| faixas | `tenure` 0–72, `MonthlyCharges` 18–119, `TotalCharges` ≥ 0 |
| nulos | o alvo `Churn` nunca nulo |
| coerência | sem internet ⇒ os 6 add-ons dizem `No internet service` |
| relações | `TotalCharges` ≈ `tenure` × `MonthlyCharges` (±10%) quando `tenure > 1` |

A relação é a regra mais forte: uma faixa isolada aprova `TotalCharges = 20` num
cliente de 70 meses pagando 100 por mês; a relação não. É o que pega coluna
invertida, valor truncado e unidade errada.

As faixas saem do profiling, não de chute:

```bash
uv run python scripts/profiling.py            # observado vs. o que o contrato exige
uv run python scripts/profiling.py --html     # o relatório do ydata-profiling
```

```
numéricas: observado vs. contrato
  coluna                 min       max     média   contrato
  tenure                0.00     72.00     36.48   ≥ 0 e ≤ 72
  MonthlyCharges       18.25    118.75     68.68   ≥ 18.0 e ≤ 119.0

TotalCharges: texto no CSV — 11 brancos, 11 deles com tenure 0
  numérico: 17.93 a 8269.27

relação TotalCharges ≈ tenure × MonthlyCharges (tenure > 1):
  desvio mediano 1.5%, máximo 3.0%  — contrato tolera 10%
```

**Erro ou alerta.** Tratar toda categoria nova como erro quebra o pipeline
quando o negócio lança um produto; tratar toda como alerta desarma o contrato. O
corte é por consequência:

| Coluna | Valor novo significa | Severidade |
|---|---|---|
| `InternetService`, `PaymentMethod` | o negócio lançou algo (5G, Pix) | **alerta** — passa e registra |
| `Contract` | mudou a feature nº 1 (importância 0,175) | **erro** |
| `Churn` | o alvo virou outra coisa | **erro** |
| binários e add-ons | um terceiro valor onde só existem dois | **erro** — é corrupção |

Falha levanta `DataValidationError` em modo `lazy`: o relatório traz **todas** as
violações de uma vez, e vem estruturado em `erro.failure_cases`. As três ações do
encontro acontecem: falhar cedo, logar os `failure_cases` e **quarentenar** o
lote em `artifacts/quarentena/<timestamp>/`.

E o contrato só vale se alguém já o viu pegar alguma coisa:

```bash
uv run python scripts/quebra_proposital.py
```

Três atos — valida o dataset real (tem de passar), adultera uma linha com
`tenure = 999` e `Contract = "Vitalício"` (tem de reprovar), e põe
`PaymentMethod = "Pix"` (tem de passar, com alerta). Roda no CI.

### Versionamento de dados (Encontro 5)

O CSV saiu do Git. O que o Git versiona é `data/churn.csv.dvc` — md5, tamanho e
caminho; os bytes ficam no cache do DVC e no remote.

| | antes | agora |
|---|---|---|
| quem guarda os bytes | Git | cache do DVC + remote |
| o que entra no commit | 981 KB de CSV | 3 linhas de ponteiro |
| trocar de dataset | diff binário ilegível | uma linha de md5 mudando |

Alternar entre versões são **dois** movimentos, e essa é a parte que escapa:

```bash
git checkout <commit> -- data/churn.csv.dvc   # o Git traz o PONTEIRO
uv run dvc checkout                           # o DVC traz os BYTES
```

Sem o segundo, o CSV no disco ainda é o da versão anterior e o repositório passa
a mentir sobre qual dado está ali. `./scripts/alternar_versao.sh v1` faz os dois
e confere o md5 no fim.

O `dvc.yaml` versiona a **relação** entre dado, código e modelo; o `dvc.lock`
guarda o hash de cada entrada e da saída. Trocar de versão muda o hash do CSV, e
`dvc repro` sabe sozinho que o modelo ficou velho.

**O deploy:** `Dockerfile.dados` é uma imagem com o DVC e o ponteiro, sem os
bytes. Ela roda `dvc pull`, escreve no volume e sai; a imagem de treino, sem DVC
dentro, encontra o dado montado. O CI prova o caminho em vez de afirmá-lo: semeia
o remote, **apaga** o CSV e exige que um container novo o traga de volta com o
md5 batendo.

### Rastreamento (Encontro 6)

Cada execução vira um **run**: o que foi escolhido (params), o que foi medido
(metrics), o modelo (artifact) e de onde aquilo veio (tags).

| | é | exemplo aqui |
|---|---|---|
| param | o que você **escolheu** | `n_estimators`, `max_depth` |
| metric | o que você **mediu** | `roc_auc`, `recall` |
| artifact | um **arquivo** gerado | o pipeline `pyfunc` |
| tag | **metadado** do run | `data_md5`, `git_commit` |

A config Pydantic vira params numa linha (`cfg.model_dump()`) — os dois contratos
do curso se encontram aí. Ficam de fora as chaves `mlflow_*` (dizem *onde
anotar*, não o que foi treinado) e as `tuning_*` (descrevem uma busca que aquele
run não fez).

**A linhagem é onde o DVC e o MLflow se costuram:**

```
data_md5   d390bd07b5514a2256a2396993b8b0e3   ← o ponteiro .dvc (Encontro 5)
git_commit d7ee6e5                            ← o commit do código
```

Dentro do container não há `.git`, então `CHURN_GIT_COMMIT` tem precedência e o
CI passa o `GITHUB_SHA`. `git_commit()` marca `+alteracoes` quando há mudança não
commitada — um run treinado sobre código não commitado aponta para um commit que
não contém aquele código.

### Promoção (Encontro 7)

Um run é um experimento, não um artefato de produção. O Registry dá o endereço
estável: o **registered model** (`churn`) agrupa versões, cada **version** é
imutável, e o **alias** `@champion` é um ponteiro móvel para a que serve.

Promover é mover o alias; reverter é movê-lo de volta — `registry.apontar(1)`
devolve a produção à versão 1 sem treinar, sem registrar e sem tocar em servidor.
Aliases e não stages: `Staging`/`Production` estão depreciados no MLflow 3.

`promover()` só registra se a métrica alcançar `mlflow_promocao_minima` (0,80).
Um portão que deixa passar é pior que nenhum: dá a impressão de que alguém
conferiu.

### A API

```bash
uv run --group api uvicorn churn.api:app   # localhost:8000/docs
```

| rota | o que faz |
|---|---|
| `POST /prever` | um lote de clientes em JSON → rótulo e probabilidade, na ordem |
| `GET /saude` | pronto para responder? é o que o healthcheck pergunta |
| `GET /modelo` | qual versão está servindo, e de qual run ela veio |
| `GET /docs` | a documentação interativa, gerada do contrato |

**O contrato de entrada não é escrito — é gerado** do `ChurnRawSchema`, o mesmo
que valida o CSV no treino. Categoria vira `Literal` com os valores reais; faixa
numérica vira `Field(ge=…, le=…)` com o piso e o teto do profiling.

Uma lista paralela aqui seria *training/serving skew* na forma mais discreta: o
dia em que o schema ganhasse uma categoria, a API continuaria recusando-a e
ninguém ligaria os dois fatos. Há teste que falha se os dois conjuntos
divergirem. Efeito colateral melhor que o esperado — o `/docs` sai com os
domínios reais em cada campo.

### O front

```bash
uv run --group app streamlit run app/Inicio.py   # localhost:8501
```

**v1 (`app/Inicio.py`)** — título, código e dados. Não prevê nada, de propósito:
um front que calcula antes de deixar ver o dado e o código vira uma caixa-preta
bonita, e a caixa-preta é o que o curso passa doze encontros desmontando. O
seletor abre qualquer módulo de `src/churn/`; as abas mostram a tabela, o tipo de
cada coluna e o resumo estatístico.

**v2 (`app/pages/1_Prever_churn.py`)** — os 19 parâmetros de um cliente e o churn
calculado. O formulário é gerado do mesmo contrato: categoria vira `selectbox`
com as opções reais, faixa vira `slider` com o piso e o teto. Não há como montar
ali um cliente que a API recuse por domínio.

O cálculo não acontece na página — ela chama a API. É o que separa o front do
modelo: trocar de campeão é mover um alias, sem tocar numa linha do front.

**`churn.contrato`** nasceu da terceira cópia. A API e o formulário precisam da
mesma informação e nenhum dos dois deveria aprender a ler `Check.statistics` por
dentro; ele traduz o schema em `permitidos` / `minimo`-`maximo` / `inteiro`.

---

## Dificuldades encontradas

Os erros que custaram tempo. Cada um está registrado porque o diagnóstico é a
parte que não aparece no código final.

### 1. O modelo exigia a resposta para dar a resposta

Na primeira chamada real da API:

```
Model is missing inputs ['customerID', 'Churn']
```

Para prever o churn era preciso **informar o churn**. A assinatura do modelo
vinha de `infer_signature` sobre o CSV cru inteiro, e o MLflow trata assinatura
como contrato: exige toda coluna declarada. O `ChurnPipeline` sempre descartou as
duas colunas — só a assinatura não sabia.

O defeito existia desde que o pipeline passou a ser logado, **invisível porque o
único teste de ponta a ponta mandava a linha completa**, com `Churn` e tudo. Foi a
API, ao mandar o que um consumidor real mandaria, que o expôs. Hoje um teste lê a
assinatura do modelo logado e falha se o alvo voltar a ser exigido.

### 2. Três formas de o nulo virar outra coisa

O `churn.tuning` imprime as variáveis que reproduzem o vencedor. Ao testar a
própria instrução que ele imprimia:

| | o que acontecia |
|---|---|
| `CHURN_MAX_DEPTH=` | `ValidationError` — `""` não é int |
| `CHURN_MAX_FEATURES=` | **passa**, vira a string vazia e treina outro modelo, calado |
| `CHURN_CLASS_WEIGHT=` | idem |

O segundo é o perigoso: nenhum erro, modelo diferente. Sem
`env_parse_none_str="null"` não havia como escrever o nulo por variável de
ambiente, e metade dos campos opcionais ficava inatingível de fora. Um teste
parametrizado agora amarra `como_variaveis` ao parser: mudar um sem o outro
quebra no pytest.

### 3. MLflow 3.16 não é o MLflow do material

- **`./mlruns` é recusado**: *"the filesystem tracking backend is in maintenance
  mode"*. O padrão virou `sqlite:///mlflow.db`.
- **`artifact_path=` virou `name=`** em `log_model`, e não é só um rename: o
  modelo deixa de ser artefato do run e vira *logged model*, com endereço
  próprio.
- **A UI abre na aba GenAI** (traces, judges, playground), vazia e sem relação com
  o projeto. Os runs estão em **"Model training"**.
- **As colunas de métrica não vêm ligadas** — a tabela mostra só nome, data e
  duração. Use **Columns**, ou vá direto a
  `#/experiments/1/runs?orderByKey=metrics.roc_auc&orderByAsc=false`.

### 4. O caminho absoluto do artefato

O MLflow grava o caminho do artefato **absoluto** no banco. Um modelo treinado no
host (`<repo>/mlruns`) não resolve dentro do container (`/app/mlruns`): as
métricas aparecem, mas o artefato não carrega — e a API sobe dizendo "sem modelo"
sem nenhuma pista do porquê.

Por isso a pilha do compose treina no container. Se você treinou no host, escolha
um mundo: ou tudo local, ou tudo em compose.

### 5. O registry que não achava o modelo registrado

`carregar_campeao` precisa fixar a tracking URI **antes** do load. Sem isso o
MLflow procura o registry no backend padrão e responde `Registered Model with
name=churn not found` — com o modelo registrado e visível na UI. É o mesmo atrito
de runs espalhados do Encontro 6 reaparecendo na hora de consumir. Há teste de
regressão travando a ordem das chamadas.

### 6. A métrica que subiu pelo motivo errado

A v2 do dataset dava `roc_auc` maior, e por um tempo isso pareceu progresso. Não
era: as 100 linhas novas são cópias de clientes reais com identificador trocado,
e 44 pares caíram com uma metade no treino e a outra no teste. O modelo acertava
linhas que já tinha visto byte a byte.

Consequência prática: a versão 1 do modelo registrado foi promovida com métrica
inflada. O filtro por `data_md5` em `registry.candidato()` existe por causa disso.

### 7. Uma ferramenta de relatório ditando a pilha numérica

`ydata-profiling` como dependência do projeto rebaixava numpy e scipy. Com lock
único, uma ferramenta de relatório passaria a decidir a versão do numpy que
treina o modelo — justo o que os Encontros 2 e 3 travaram. Ele roda em ambiente
efêmero (`uvx`), sem encostar no ambiente travado.

### 8. O `dvc repro` que rodava sem nada ter mudado

`.dvcignore` não excluía `__pycache__/`. O hash de `src/churn` mudava a cada
execução, e o `repro` retreinava o modelo por causa de bytecode.

### 9. O registro de avisos que nunca terminava

`warnings.catch_warnings(record=True)` devolve a lista **viva**. Reemitir um
aviso enquanto se percorre essa lista o grava de volta nela, e o laço não termina
— o pytest simplesmente travava. A correção é copiar (`list(capturados)`) e
registrar **fora** do bloco.

### 10. Atrito de ferramenta, que também custa tempo

- **PowerShell não roda `.sh`**, e uma janela nova abre na pasta pessoal: três
  rodadas de `not a git repository` antes de a causa ficar clara. Resolvido por
  construção, com `.bat` que fazem `cd /d "%~dp0"`.
- **`.bat` com quebra de linha do Unix quebra no `cmd`** — e quebra justamente
  nos blocos `if errorlevel`, que é onde mora o tratamento de erro. O
  `.gitattributes` marca `*.bat text eol=crlf`.
- **Reescrever a branch no GitHub trava o clone de quem a acompanha**: o `git
  pull` vira conflito sobre mudanças que não são de quem puxou. `atualizar.bat`
  é a saída em um duplo clique.
- **"sujo" já tinha dono.** O marcador de código não commitado se chamava
  `-sujo`, e num projeto cujo Encontro 4 inteiro trata de *dado sujo* isso gerou
  a pergunta "por que tem dado sujo?". Virou `+alteracoes`.

---

## Ressalvas e limites

Onde este projeto **não** serve, dito antes que alguém descubra sozinho.

### Sobre o modelo

- **O ganho do ajuste fino é de operação, não de capacidade preditiva.** Contra o
  melhor anterior, o `roc_auc` subiu 0,0011. Quase todo o salto de recall se
  reproduz movendo o corte de 0,50 para 0,34 no modelo antigo — ver
  [a seção acima](#a-ressalva-honesta-quanto-disso-é-modelo).
- **A acurácia não decide nada aqui.** Com 33% de churn na base, quem chuta que
  ninguém cancela já acerta 67% — mais do que a diferença entre qualquer duas
  linhas da tabela acima. O modelo ajustado tem acurácia *menor* que o padrão e
  é melhor para o problema.
- **O corte em 50% é convenção**, não resultado. Quem usar isto para uma campanha
  deve ordenar a carteira pela probabilidade e descer até o orçamento acabar.
- **Um modelo melhor exigiria sair do escopo**: outra família (gradient boosting)
  ou features novas. A Random Forest com estes dados está perto do seu teto.
- **Não há monitoramento de drift.** O dataset é estático e fechado; um modelo em
  produção de verdade precisaria vigiar a distribuição de entrada.

### Sobre o contrato dos dados

- **A faixa de `MonthlyCharges` (18–119) é colada neste dataset.** Ela pega 999,
  −1 e 1e6 — e também reprovaria o dataset inteiro no dia em que um plano novo
  custasse 130. É decisão consciente: faixa larga demais não ensina nada.
- **`coerce=True` esconde sujeira.** `tenure = 12.7` não reprova: vira `12` em
  silêncio (há teste fixando isso). O que segura o estrago é validar em dois
  pontos — o que a coerção esconde no cru, o schema do limpo cobra num tipo já
  definido.
- **`strict=False` tolera coluna a mais.** Coluna *faltando* continua sendo erro.
  O aperto que valeria a pena aqui é `strict=True`, porque coluna nova neste
  dataset é sintoma de extração errada.
- **A API valida coluna a coluna, não a relação entre elas.** `TotalCharges ≈
  tenure × MonthlyCharges` é regra de DataFrame, e o Pydantic da API não a
  aplica. Por isso o formulário **deriva** o total e avisa quando ele sai da
  tolerância — mas uma chamada direta à API pode enviar um cliente incoerente e
  receber uma previsão (que é extrapolação).

### Sobre a infraestrutura

- **O remote do DVC é um diretório local** (`../dvc-storage`). Exercita o fluxo
  sem credencial nenhuma, mas **não atravessa máquinas**. Para um remote de
  verdade basta uma linha (`dvc remote add -d storage s3://…`); o resto do fluxo
  não muda.
- **A API não tem autenticação** e roda com **um worker**. O modelo é carregado
  por processo; quatro workers seriam quatro cópias da floresta na memória, para
  um serviço de curso.
- **Trocar de campeão exige reiniciar a API.** O pipeline é carregado uma vez, na
  subida. É adequado aqui — num container, reiniciar *é* o deploy — mas não é
  *hot reload*.
- **Host e container não compartilham runs** (ver dificuldade nº 4).
- **O modo demonstração não passa pelo Registry.** Ele treina na sessão, então
  não há versão, não há alias e não há rollback — as três coisas que o Encontro
  7 existe para dar. Serve para a vitrine pública; não serve como deploy.
- **O portão de promoção (0,80) é um piso de sanidade, não uma meta.** Ele barra
  um modelo quebrado, não um modelo medíocre.

---

## Testes e CI

```bash
uv run pytest          # 122 testes
```

Os testes não cobrem "o modelo acerta" — cobrem as **juntas** onde as coisas
quebram caladas: o contrato traduzido diverge da fonte? o vencedor da busca
sobrevive à viagem até o próximo treino? a assinatura exige o alvo? o padrão do
formulário é aceito pela API?

O CI (`.github/workflows/docker.yml`) roda a cada push e **prova** o que o README
afirma, em vez de afirmá-lo:

| Passo | O que prova |
|---|---|
| build + pytest no container | o ambiente travado reproduz fora da sua máquina |
| `alternar_versao.sh v1 → v2` | dá para alternar entre versões, e o dado segue o ponteiro |
| apaga o CSV e faz `dvc pull` | o dado vem do remote, não da imagem |
| `quebra_proposital.py` | o contrato reprova o dado adulterado e alerta no catálogo |
| treino no container | o dataset entra por volume |
| `conferir_run.py` | o run existe e a tag `data_md5` bate com o ponteiro |
| `prever.py` | o modelo **roda** a partir do MLflow, com linha crua |
| `docker compose` + `conferir_api.py` | a imagem sobe, carrega o campeão e serve pela rede |

O último é a cobertura que o teste de unidade não dá. Saída real do CI:

```
API: http://api:8000
  ok   /saude responde 200 (200)
  ok   modelo carregado ({'status': 'ok', 'versao': '1'})
  ok   aponta para o alias, não para um run solto (models:/churn@champion)
  ok   /prever responde 200 (200)
  ok   recusa categoria fora do contrato (422)
  ok   recusa tenure acima do teto (422)
A API está servindo o modelo promovido.
```

---

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py      # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ schema.py      # o contrato dos dados (pandera) — a fonte da verdade
│  ├─ contrato.py    # o contrato traduzido: alimenta a API e o formulário
│  ├─ validation.py  # executa o contrato e traduz a falha
│  ├─ quarentena.py  # separa o lote reprovado em vez de treinar com ele
│  ├─ data.py        # carga + validação + limpeza
│  ├─ features.py    # transformações puras + encoder sem vazamento
│  ├─ model.py       # treino (entrypoint) — instrumentado com MLflow
│  ├─ experiments.py # o grid rastreado — 9 combinações
│  ├─ tuning.py      # a busca aleatória em 6 eixos — o ajuste fino
│  ├─ evaluate.py    # métricas
│  ├─ lineage.py     # data_md5() e git_commit() — a costura com o DVC
│  ├─ serving.py     # ChurnPipeline — o modelo que aceita cliente cru
│  ├─ registry.py    # promove o vencedor a @champion
│  ├─ api.py         # FastAPI — o campeão atrás de HTTP
│  └─ demonstracao.py # o modo vitrine: sem API, sem Registry, sem CSV no disco
├─ app/              # Streamlit: Inicio.py (v1) + pages/ (v2)
├─ requirements.txt  # para plataformas que instalam por pip (Streamlit Cloud)
├─ tests/            # 122 testes
├─ scripts/          # preparar · relatorio · fetch_data · mlflow_ui · prever
│                    # profiling · gerar_v2 · alternar_versao · quebra_proposital
│                    # conferir_run · conferir_api
├─ data/             # churn.csv.dvc (ponteiro) — os bytes vêm do DVC
├─ docs/             # rodar-local.md · data-pipeline.md
├─ *.bat             # Windows: subir-front · ver-runs · abrir-mlflow · atualizar
├─ Dockerfile        # treino        Dockerfile.dados  # dvc pull no deploy
├─ Dockerfile.api    # a API         Dockerfile.app    # o front
├─ compose.yml       # dados · tuning · promover · api · app · mlflow
├─ dvc.yaml          # o pipeline: deps (dado + código) → outs (modelo)
└─ pyproject.toml    # deps travadas (uv), em grupos por imagem
```

---

## Do script monolítico ao projeto

| `train_churn.py` | o que está aqui |
|---|---|
| caminho hardcoded | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo, `tenure = 0`) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| validação ausente | contrato pandera no dado cru e no limpo, com quarentena |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só imprime `accuracy` | accuracy, precision, recall, f1, roc_auc — e um run por execução |
| o dado no Git | ponteiro `.dvc`, bytes no remote |
| "qual era o modelo bom?" | `models:/churn@champion` |
| o modelo no notebook | API HTTP + front, em container |
| tudo em nível de módulo | funções puras + orquestração em `train()` |
