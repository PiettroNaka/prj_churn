# prj_churn

Projeto do curso de **MLOps** (IESB): previsão de **churn** —
classificação binária tabular sobre o dataset Telco.

O repositório é o caminho completo do script monolítico `train_churn.py` até um
modelo **validado, versionado, rastreado, promovido, servido por HTTP e
consumido por um front** — tudo reproduzível em container.

A pergunta que o projeto responde, no fim:

> *Dado um cliente, qual a probabilidade de ele cancelar — e **de qual dado e de
> qual código** essa resposta saiu?*

A segunda metade da pergunta é o que separa um notebook de um sistema. É ela que
organiza todo o resto deste README.

**No ar:** [prjchurn.streamlit.app](https://prjchurn.streamlit.app) — a pilha
completa, não só o front. A barra lateral mostra `API · modelo v1`: cada previsão
é uma chamada HTTP ao campeão promovido no MLflow Model Registry.

| | |
|---|---|
| **0,8206** | `roc_auc` no teste |
| **0,7609** | `recall` — de 0,5602 no modelo inicial |
| **136** | testes, CI verde a cada push |
| **7.043** | clientes na base, 33% de churn |

---

## Como ler este README

Ele é longo porque cobre o projeto inteiro. Não precisa ser lido inteiro de uma
vez — escolha a profundidade:

| Se você tem | Leia | O que vai entender |
|---|---|---|
| **2 minutos** | [O que é, e o que resolve](#o-que-é-e-o-que-resolve) + [A pilha, em um diagrama](#a-pilha-em-um-diagrama) | o problema, a arquitetura e por que cada peça existe |
| **10 minutos** | \+ [Como a previsão melhorou](#como-a-previsão-melhorou) e [Ressalvas e limites](#ressalvas-e-limites) | os números, e o que eles **não** provam |
| **a íntegra** | \+ [Como foi implementado](#como-foi-implementado) e [Dificuldades encontradas](#dificuldades-encontradas) | cada decisão técnica e cada erro que custou tempo |
| **quer rodar** | vá direto a [Como rodar](#como-rodar) | local, em Docker, ou publicado |

As quatro seções em negrito abaixo são as que respondem ao enunciado do
trabalho:

| | |
|---|---|
| [**Como a previsão melhorou**](#como-a-previsão-melhorou) | os números, e quanto deles é mérito do modelo |
| [**Como foi implementado**](#como-foi-implementado) | cada camada, e a decisão por trás |
| [**Dificuldades encontradas**](#dificuldades-encontradas) | os treze erros reais, e o que cada um ensinou |
| [**Ressalvas e limites**](#ressalvas-e-limites) | onde este projeto não serve |

Demais seções: [Testes e CI](#testes-e-ci) · [Como rodar](#como-rodar) ·
[Estrutura](#estrutura) ·
[Do script monolítico ao projeto](#do-script-monolítico-ao-projeto)

---

## O que é, e o que resolve

### O ponto de partida

O projeto começa num arquivo só, `train_churn.py`: lê um CSV de um caminho fixo,
preenche os brancos com um número mágico (`fillna(2200)`), codifica as categorias
no dataset inteiro, treina uma Random Forest e imprime a acurácia.

Ele funciona. E é exatamente por funcionar que ele é um bom ponto de partida: o
problema não é o modelo, é tudo em volta. Ninguém sabe qual CSV gerou aquele
número, se o número de ontem é comparável ao de hoje, nem como alguém que não
seja o autor faria o mesmo rodar.

### O fio condutor

Cada encontro do curso resolveu um problema que o anterior deixou em aberto.
Lendo esta tabela de cima a baixo você tem o projeto inteiro em resumo:

| Encontro | O problema que ficou em aberto | A entrega | Onde está |
|---|---|---|---|
| **2** | caminhos fixos no meio da lógica; números mágicos; vazamento de dados no encoder | config como contrato (Pydantic), pacote modular, ambiente travado (uv) | `config.py`, `pyproject.toml` |
| **3** | "na minha máquina funciona" | reprodutibilidade em container; dado por volume, nunca em camada | `Dockerfile`, `compose.yml` |
| **4** | dado ruim treina modelo ruim — **em silêncio** | contrato dos dados executável, com quarentena e alerta vs. erro | `schema.py`, `validation.py`, `quarentena.py` |
| **5** | qual CSV gerou qual métrica? | dado versionado por ponteiro; duas versões; deploy por `dvc pull` | `data/churn.csv.dvc`, `dvc.yaml` |
| **6** | o `print` mostra o número e o esquece | cada treino vira run, com linhagem `data_md5` + `git_commit` | `model.py`, `experiments.py`, `lineage.py` |
| **7** | "qual era mesmo o run bom?" | promoção a `@champion` com portão de métrica e rollback por alias | `registry.py` |
| **+** | o modelo continua preso no repositório | **ajuste fino**, **API** FastAPI, **front** v1 e v2, deploy público | `tuning.py`, `api.py`, `app/` |
| **+** | o dado é um arquivo na máquina de quem roda | **o dataset em Postgres** e a pilha de dois containers **em AWS EC2** | `db.py`, `compose.ec2.yml` |

A última linha é o que o enunciado pediu além do curso: melhorar o modelo
(ajuste fino), montar a API, montar o front em duas versões e pôr tudo em
Docker.

---

## A pilha, em um diagrama

Do ponteiro no Git até a tela. Cada etapa abaixo é código real deste
repositório — a coluna da direita diz o que ela garante:

```
  data/churn.csv.dvc                    o Git versiona o md5 e o tamanho;
         │                               os bytes ficam no remote do DVC
         │  dvc pull
         ▼
  churn.csv  ·  7.043 clientes, 33% de churn
         │
         │  no deploy em EC2 o CSV é carregado numa tabela do Postgres e o
         │  pipeline lê de lá. Mesmas linhas, mesma ordem, mesmo modelo — a
         │  carga valida pelo contrato antes de qualquer INSERT.
         ▼
  contrato pandera  ──── reprova ────▶  artifacts/quarentena/<timestamp>/
  21 colunas · faixas · categorias           o lote e o relatório das
  coerência · relações entre colunas         violações, todas de uma vez
         │
         │  aprova
         ▼
  ajuste fino  ·  24 das 3.150 combinações, validação cruzada no treino
         │
         ▼
  MLflow run   params · métricas · o pipeline · e as tags que dão o endereço:
         │                         data_md5 (qual CSV) + git_commit (qual código)
         ▼
  portão  roc_auc ≥ 0,80  ─── barra ───▶  o run fica registrado,
         │                                 o alias não se move
         │  promove
         ▼
  Model Registry  ·  models:/churn@champion
         │              promover é mover o alias; reverter é movê-lo de volta
         ▼
  FastAPI  ·  POST /prever          resolve o alias uma vez, na subida
         │
         ▼
  Streamlit  ·  v1 ver o projeto, v2 prever     nunca carrega modelo:
                                                fala HTTP com a API
```

Três propriedades do desenho valem mais que as caixas:

1. **O dado entra por um ponteiro, não por um arquivo.** O Git versiona o md5; os
   bytes vêm do DVC. Trocar de dataset é uma linha de hash mudando.
2. **Nada chega ao treino sem passar pelo contrato.** O que reprova vai para a
   quarentena com o relatório, em vez de virar modelo.
3. **O front nunca carrega modelo.** Ele fala HTTP com a API, e a API resolve o
   alias. Trocar de campeão é mover um ponteiro — sem tocar numa linha do front.

### A linhagem, que é a tese do projeto

Toda previsão tem endereço. É isto que torna respondível a segunda metade da
pergunta lá de cima — *de qual dado e de qual código essa resposta saiu?*

```
  Pergunta: esta previsão na tela — de onde ela veio?

  churn@champion                   o alias que a API resolve na subida
        │
        │  aponta hoje para a versão 1 do modelo registrado,
        ▼  que saiu de um run do MLflow, e esse run carrega duas tags:
  MLflow run
        │
        ├─ data_md5    d390bd07b5514a2256a2396993b8b0e3
        │      │
        │      └─ é o md5 gravado em data/churn.csv.dvc  →  o CSV exato,
        │         recuperável com dvc checkout, byte a byte
        │
        └─ git_commit  d7ee6e5
               │
               └─ o commit do código que treinou  →  recuperável com
                  git checkout; vem com +alteracoes se não estava commitado

  E o caminho de volta também funciona: dado um CSV, quais modelos saíram dele.
```

Sem isso, "o modelo está com 82% de roc_auc" é uma frase sobre nada: não se
sabe sobre qual dado foi medido, nem qual código a produziu, e portanto não se
pode nem repetir a medição nem comparar com a de amanhã.

---

## Como a previsão melhorou

Esta é a seção que responde *"o que vocês fizeram para o modelo prever melhor"*.
Ela tem quatro partes: o que mudou, como foi feito, **quanto disso é mérito do
modelo**, e o que foi feito no dado para que a métrica não mentisse.

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

Uma subseção por camada, na mesma ordem do diagrama. Cada uma diz **o que foi
feito** e **por que daquele jeito**.

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
| `Contract` | mudou a feature nº 1 (importância 0,464 no campeão) | **erro** |
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
seletor abre qualquer módulo de `src/churn/`; as abas mostram quem cancela por
categoria, a tabela, o tipo de cada coluna e o resumo estatístico.

**v2 (`app/pages/1_Prever_churn.py`)** — os 19 parâmetros de um cliente e o churn
calculado. O formulário é gerado do mesmo contrato: categoria vira `selectbox`
com as opções reais, faixa vira `slider` com o piso e o teto. Não há como montar
ali um cliente que a API recuse por domínio. Cada campo traz, abaixo, uma linha
explicando o que ele significa e como pesa no churn — e três perfis prontos
(risco baixo, médio e alto) preenchem o cliente inteiro em um clique.

O cálculo não acontece na página — ela chama a API. É o que separa o front do
modelo: trocar de campeão é mover um alias, sem tocar numa linha do front.

**`churn.contrato`** nasceu da terceira cópia. A API e o formulário precisam da
mesma informação e nenhum dos dois deveria aprender a ler `Check.statistics` por
dentro; ele traduz o schema em `permitidos` / `minimo`-`maximo` / `inteiro`.

### O dado em Postgres, e o deploy em EC2

O arranjo pedido para a AWS: **dois containers numa instância EC2** — um com a
aplicação, outro com o banco — e o dataset deixando de ser arquivo para virar
tabela.

```
  ┌─ instância EC2 ─────────────────────────────────────────────┐
  │   ┌─ web ───────────────┐          ┌─ postgres ──────────┐  │
  │   │  Streamlit  :8501   │ ───────▶ │                     │  │
  │   │  FastAPI    :8000   │ ◀─────── │   tabela clientes   │  │
  │   │  o CSV, só na carga │ ──carga─▶│   7.043 linhas      │  │
  │   └─────────────────────┘          └─────────────────────┘  │
  │          ▲                           sem porta publicada    │
  └──────────┼──────────────────────────────────────────────────┘
             │ :8501  →  o navegador
```

O passo a passo completo está em [`docs/deploy-ec2.md`](docs/deploy-ec2.md).
Aqui ficam as três decisões.

**O contrato roda antes do INSERT.** `churn.db.semear` valida o CSV pelo pandera
e só então grava; se reprovar, nada entra e a exceção sobe. Um banco sem
contrato não elimina dado ruim — só troca o lugar onde ele se esconde. E, do
outro lado, `load_clean` segue validando o que saiu do banco, exatamente como
validava o que saía do arquivo. Muda o transporte, não a verificação: é a mesma
régua que já valia quando o CSV passou a ser buscado por HTTP no deploy de
vitrine.

**Uma variável troca a origem, e nada mais.** Com `CHURN_DATABASE_URL` definida
o pipeline lê do Postgres; sem ela, do CSV. `churn.data.carregar_origem` é o
único lugar que sabe da diferença — o resto do pipeline não sabe nem precisa
saber de onde o DataFrame veio. O CI prova que são intercambiáveis comparando os
dois DataFrames inteiros com `assert_frame_equal`, e depois comparando os dois
**modelos**: mesmas métricas até a 12ª casa, mesmas importâncias com tolerância
zero.

**O front continua sem carregar modelo.** Na imagem de produção
(`Dockerfile.web`) ele e a API vão no mesmo container, mas o front segue falando
HTTP com `127.0.0.1:8000`, e a API segue resolvendo `models:/churn@champion`. A
fronteira que o projeto defende não é a de processos, é a de responsabilidades:
trocar de campeão continua sendo mover um alias.

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

### 7. A ordem das colunas, que funcionava por acidente

`ValueError: The feature names should match those that were passed during fit.`

O pipeline montava as features na ordem em que o schema as declara; o modelo foi
treinado na ordem em que o CSV as traz. As duas diferem. Nunca tinha aparecido
porque o MLflow, ao aplicar a assinatura, reordenava as colunas em silêncio — o
pipeline estava errado e **funcionava pelo conserto de outra camada**. Chamado
fora do MLflow, quebrou.

Hoje `preparar()` reordena para `feature_names_in_` e levanta erro nomeando a
coluna que falta. Dois testes de regressão: um embaralha as colunas de propósito,
o outro remove uma.

### 8. Uma ferramenta de relatório ditando a pilha numérica

`ydata-profiling` como dependência do projeto rebaixava numpy e scipy. Com lock
único, uma ferramenta de relatório passaria a decidir a versão do numpy que
treina o modelo — justo o que os Encontros 2 e 3 travaram. Ele roda em ambiente
efêmero (`uvx`), sem encostar no ambiente travado.

### 9. O `dvc repro` que rodava sem nada ter mudado

`.dvcignore` não excluía `__pycache__/`. O hash de `src/churn` mudava a cada
execução, e o `repro` retreinava o modelo por causa de bytecode.

### 10. O registro de avisos que nunca terminava

`warnings.catch_warnings(record=True)` devolve a lista **viva**. Reemitir um
aviso enquanto se percorre essa lista o grava de volta nela, e o laço não termina
— o pytest simplesmente travava. A correção é copiar (`list(capturados)`) e
registrar **fora** do bloco.

### 11. O deploy que instalava uma cópia velha do pacote

No Streamlit Cloud, o app quebrou com `ImportError: cannot import name
'subir_api_embutida'`. A plataforma atualiza o repositório a cada push, mas só
reinstala as dependências quando o `requirements.txt` muda — então `app/` ficava
novo e o pacote `churn` instalado ficava velho.

Duas correções: a última linha do `requirements.txt` é `-e .` (instalação
editável, o pacote vem de `src/`), e os dois arquivos do front inserem `src/` no
`sys.path` antes de importar. O repositório é a fonte; a cópia é um artefato do
deploy.

### 12. O banco devolvia as mesmas linhas noutra ordem

Ao levar o dataset para o Postgres, ler do banco e ler o CSV davam o mesmo
conjunto de linhas — e `assert_frame_equal` reprovava mesmo assim. A diferença
era a **ordem**: um `SELECT` sem `ORDER BY` não tem ordem garantida. O Postgres
devolve na ordem que for mais barata, e ela muda com `VACUUM`, com `UPDATE` e
com o plano escolhido.

Parece detalhe de banco; não é. O `train_test_split` escolhe linhas por
**posição**. Com a mesma semente e outra ordem de linhas, o split é outro — e
com ele o modelo e as métricas. Trocar o CSV pelo banco teria mudado o modelo
caladamente, e o "mesma semente, mesmo split, mesmos números" que este README
afirma deixaria de valer, sem nenhum erro aparecendo.

A correção é gravar a posição original como coluna (`linha`) e ler sempre com
`ORDER BY`. Ela sai do DataFrame antes de voltar: serve ao banco, não ao
modelo — se vazasse, o índice da linha no CSV viraria feature.

O teste de regressão não se contenta em ler duas vezes. Ele dá um `UPDATE` que
não altera valor nenhum, só força o Postgres a regravar a linha no fim do heap,
e então exige que a ordem continue a do CSV. Sem o `ORDER BY` ele falha de
verdade — conferi removendo a cláusula.

### 13. Atrito de ferramenta, que também custa tempo

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
- **O deploy em EC2 não tem HTTPS nem autenticação.** O tráfego vai em claro e
  quem tiver o IP usa o front e a API. Para um trabalho de curso sobre dado
  público é aceitável; qualquer outro uso precisaria de um proxy com
  certificado e de um domínio. As demais ressalvas do servidor estão em
  [`docs/deploy-ec2.md`](docs/deploy-ec2.md).
- **Uma instância, sem redundância, e sem backup do banco.** O volume sobrevive
  a `down` e a reboot, não à destruição da instância. Aqui passa porque o
  dataset é recarregável do CSV com o md5 conferido; com dado que não se
  recupera, faltaria `pg_dump` agendado.
- **O Postgres guarda o dataset, não o histórico.** Não há versionamento dentro
  do banco: quem versiona o dado continua sendo o ponteiro `.dvc`, e a carga
  confere o md5 contra ele. Um banco com `UPDATE` em produção precisaria de
  outra resposta para "qual dado gerou este modelo".

---

## Testes e CI

```bash
uv run pytest          # 136 testes
```

Os testes não cobrem "o modelo acerta" — cobrem as **juntas** onde as coisas
quebram caladas: o contrato traduzido diverge da fonte? o vencedor da busca
sobrevive à viagem até o próximo treino? a assinatura exige o alvo? a ordem das
colunas mudou? o padrão do formulário é aceito pela API?

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
uv run pytest                         # 136 testes
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

### Publicar de graça, com a API junto

Publicado em **[prjchurn.streamlit.app](https://prjchurn.streamlit.app)**, com
esta configuração:

| campo | valor |
|---|---|
| plataforma | **Streamlit Community Cloud** (gratuito, sem cartão) |
| Repository | `PiettroNaka/prj_churn` |
| Branch | `main` |
| Main file path | `app/Inicio.py` |

O deploy roda a **pilha inteira**, não só o front. A plataforma roda um processo
só; em vez de o front passar a carregar modelo — o que apagaria a fronteira que
o projeto inteiro defende —, **a API sobe ao lado dele**:

```
Streamlit (porta pública)  →  FastAPI (127.0.0.1:8000)  →  modelo @champion
             └──────────── o mesmo container ────────────┘
```

Na primeira visita à página de previsão, o app materializa o dataset, treina com
a configuração vencedora, promove a `@champion` e sobe a API — uns 40 segundos,
uma vez só. Depois a barra lateral mostra **"API · modelo v1"**, e cada previsão
é uma chamada HTTP ao campeão promovido.

Três coisas tornam isso possível, e nenhuma é padrão num projeto com DVC e
MLflow:

**O `requirements.txt`.** A plataforma instala por pip, não por uv. A última
linha é `-e .` — é o que instala o pacote a partir do `pyproject.toml` e faz
`from churn.config import settings` funcionar, apontando para `src/` em vez de
para uma cópia que envelhece (ver dificuldade nº 11).

**O dataset por HTTP.** Ele saiu do Git no Encontro 5, e o clone da plataforma é
raso, então nem o histórico vem junto. O `churn.demonstracao` busca os bytes do
commit em que ele ainda era versionado e **só os aceita se o md5 bater com o
ponteiro `.dvc`** — transporte diferente, mesma verificação.

**A rede de segurança.** Se a API não subir (memória apertada, por exemplo), o
front treina o modelo na própria sessão e **diz isso na barra lateral**: "Modo
demonstração". Um número que sai de um modelo treinado na hora não é o mesmo que
sai do campeão promovido, e quem olha tem o direito de saber qual é qual.

### Em AWS EC2, com o dado em Postgres

O arranjo de dois containers — aplicação e banco — na mesma instância:

```bash
# na instância, depois de instalar o Docker e clonar o repositório
printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -base64 24)" > .env
docker compose -f compose.ec2.yml up -d --build
```

| serviço | porta | o que é |
|---|---|---|
| `web` | 8501 | Streamlit + FastAPI no mesmo container |
| `postgres` | — | o dataset em tabela; **sem porta publicada** |

A aplicação faz a sequência de partida sozinha: materializa o CSV, carrega as
7.043 linhas no banco validando pelo contrato, treina e promove um campeão se
não houver, sobe a API, espera ela responder **com modelo carregado** e só então
abre o front.

O passo a passo — tipo de instância, grupo de segurança, swap, custos e como
desligar para não pagar à toa — está em
[`docs/deploy-ec2.md`](docs/deploy-ec2.md).

Para exercitar o mesmo caminho na sua máquina, sem AWS:

```bash
docker compose up -d postgres
export CHURN_DATABASE_URL=postgresql+psycopg://churn:churn@localhost:5432/churn
uv run --group db python -m churn.db      # carrega o CSV no banco
uv run --group db python -m churn.tuning  # treina lendo do banco
```

### E o Hugging Face Spaces?

A imagem existe em `deploy/spaces/` e é construída e testada pelo CI a cada
push — ela roda a mesma pilha num container só. Mas o Hugging Face passou a
**exigir plano pago** para Spaces com Docker; só os Static Spaces seguem livres,
e eles não rodam Python. O guia está em
[`docs/deploy-huggingface.md`](docs/deploy-huggingface.md), para quem tiver PRO.

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
│  ├─ db.py         # o dataset em Postgres — carga validada e leitura ordenada
│  └─ demonstracao.py # o modo vitrine: sem API, sem Registry, sem CSV no disco
├─ app/              # Streamlit: Inicio.py (v1) + pages/ (v2)
├─ requirements.txt  # para plataformas que instalam por pip (Streamlit Cloud)
├─ tests/            # 136 testes
├─ scripts/          # preparar · relatorio · fetch_data · mlflow_ui · prever
│                    # profiling · gerar_v2 · alternar_versao · quebra_proposital
│                    # conferir_run · conferir_api · subir_front
├─ data/             # churn.csv.dvc (ponteiro) — os bytes vêm do DVC
├─ deploy/spaces/    # Dockerfile + README do Hugging Face Spaces
├─ docs/             # rodar-local.md · data-pipeline.md · deploy-ec2.md
│                    # deploy-huggingface.md
├─ *.bat             # Windows: subir-front · ver-runs · abrir-mlflow · atualizar
├─ Dockerfile        # treino        Dockerfile.dados  # dvc pull no deploy
├─ Dockerfile.api    # a API         Dockerfile.app    # o front
├─ compose.yml       # dev: dados · tuning · promover · api · app · mlflow · postgres
├─ compose.ec2.yml   # produção: dois containers — web (front+API) e postgres
├─ Dockerfile.web    # a imagem da aplicação para o EC2
├─ dvc.yaml          # o pipeline: deps (dado + código) → outs (modelo)
└─ pyproject.toml    # deps travadas (uv), em grupos por imagem
```

---

## Do script monolítico ao projeto

O resumo de uma página: o que havia em `train_churn.py`, e o que está aqui.

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
