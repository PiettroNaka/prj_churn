# Data Pipeline — onde cada etapa mora no prj_churn

Nota de leitura do Encontro 4. A leitura em si é o texto **"Data Pipeline"** em
<https://ml-ops.org> (seção *Data Engineering* de *Three Levels of ML Software*)
— este arquivo não substitui a leitura: é o mapa de onde cada etapa dela caiu
neste repositório, para consultar enquanto se lê.

## As etapas, e o que existe aqui

| Etapa (ml-ops.org) | No prj_churn | Estado |
|---|---|---|
| **Data Ingestion** — trazer o dado para dentro, de forma repetível | `data.load_raw` lê o CSV; `scripts/fetch_data.sh` materializa o arquivo a partir do ponteiro DVC | ✓ |
| **Exploration and Validation** — profiling primeiro, contrato depois | `scripts/profiling.py` (o retrato) e `churn/schema.py` (o contrato) | ✓ |
| **Data Wrangling (Cleaning)** — transformar para o formato do treino, de forma reproduzível | `data.drop_identifier`, `data.coerce_total_charges` — funções puras, sem número mágico | ✓ |
| **Data Labeling** — produzir o rótulo | não se aplica: o Telco já vem com `Churn` rotulado | — |
| **Data Splitting** — treino/teste sem vazamento | `model.train` com `train_test_split(stratify=labels, random_state=…)`; o encoder aprende **só** no treino | ✓ |

## As três ideias que sustentam o encontro

**Validação é teste, só que de dados.** Código quebra com barulho — uma exceção,
um teste vermelho. Dado quebra em silêncio: entra sujo, o pipeline roda até o
fim, a métrica cai, e ninguém percebe até alguém olhar o painel semanas depois.
Por isso o dado ganha uma suíte própria (`tests/test_validation.py`): cada teste
suja **uma** coisa e exige que o contrato pegue.

**O contrato é código.** O `ChurnSchema` é uma classe `DataFrameModel` — a mesma
intuição do Pydantic do Encontro 2, onde o contrato era da config. Sendo código,
ele versiona junto, é revisado em PR e roda no CI. Um schema em PDF é
documentação; um schema em `.py` é uma barreira.

**Validar na fronteira, e nos dois lados da limpeza.** O contrato roda quando o
dado *entra* (`ChurnRawSchema`, no CSV cru) e outra vez depois da limpeza
(`ChurnSchema`, antes de virar feature). É o que sustenta o `coerce=True` da
Config: a coerção converte tipo em silêncio — `tenure = 12.7` vira `12` sem
reclamar — e o segundo schema cobra o dado num tipo já definido, depois de a
limpeza do `data.py` ter passado. Validar uma vez só, com coerção ligada, seria
a pior das combinações.

## Os atritos, e a posição deste repositório

| Atrito | O que fizemos |
|---|---|
| **strict demais barra dado válido** | `strict=False`, como o encontro declara: coluna a mais é tolerada, coluna faltando não. A faixa de `MonthlyCharges` é a do encontro (18–119) — colada no dado, e assumidamente apertada |
| **`coerce=True` mascara sujeira** | `coerce=True`, como o encontro declara — e o antídoto é validar duas vezes: o que a coerção esconde no cru, o schema do limpo cobra num tipo já definido |
| **categoria nova quebra o `isin`** | hoje é erro fatal, porque em curso o dataset é estável. Em produção o caminho seria alertar e quarentenar o lote, não derrubar o pipeline |
| **onde validar** | na fronteira: ao ler o CSV, e de novo antes das features |

## O DATA como artefato — a ponte para o Encontro 5

Depois de validado, o dado deixa de ser arquivo solto: tem **contrato** (o
schema versionado), tem **linhagem** (qual dado gerou qual modelo) e é
**reproduzível**. Faltava a versão — é o buraco que o DVC fecha no Encontro 5, e
que neste repositório já está fechado: `data/churn.csv.dvc`.

O ponteiro e o schema respondem perguntas diferentes: o `.dvc` diz *qual* dado é
este; o schema diz se ele *serve*.
