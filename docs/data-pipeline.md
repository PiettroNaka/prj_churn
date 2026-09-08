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
| **categoria nova quebra o `isin`** | alerta nas colunas de catálogo (`InternetService`, `PaymentMethod`), erro no resto. Um meio de pagamento novo não pode derrubar o pipeline; um `Contract` novo — a feature nº 1 do modelo — tem de parar |
| **onde validar** | na fronteira: ao ler o CSV, e de novo antes das features |
| **o que fazer com o lote ruim** | falhar cedo, logar os `failure_cases` e **quarentenar** — `artifacts/quarentena/<timestamp>/` guarda as linhas e o relatório |

## O DATA como artefato — o bloco fechado (Encontro 5)

Depois de validado, o dado deixa de ser arquivo solto: tem **contrato** (o schema
versionado) e é **reproduzível**. Faltava a versão, e é o que o DVC fecha.

O ponteiro e o schema respondem perguntas diferentes: o `.dvc` diz *qual* dado é
este; o schema diz se ele *serve*. Um dataset novo passa no schema e muda o md5;
um corrompido mantém o md5 de nada e reprova no schema.

### Por que o Git não serve para dados

| Git | é ótimo para | é péssimo para |
|---|---|---|
| formato | texto, arquivos pequenos | binário, MB/GB |
| diff | linha a linha, legível | "um blob mudou" |
| histórico | leve | cada versão incha o repositório, para sempre |

O truque do DVC é não escolher: o Git versiona um ponteiro de ~100 bytes com o
hash do conteúdo, e os bytes vivem no cache e no remote, endereçados por esse
hash (`files/md5/f1/50a64ab5537f24…`).

### A cadeia dados → código → modelo

```
DADO                  CÓDIGO                MODELO
churn.csv @ v2   →    src/churn + schema →  artifacts/churn_model.pkl
md5 f150a64a          commit 523c40e        md5 008a388f
```

Quem amarra as duas primeiras pontas é o commit: ele carrega o código **e** o
ponteiro do dado. Por isso "qual dado gerou qual modelo?" passa a ter uma
resposta — é o commit. A terceira ponta é o `dvc.lock`, que grava o hash exato
de cada entrada e da saída a cada `dvc repro`; o MLflow (Enc. 6) e o Model
Registry (Enc. 7) fazem o mesmo pelo lado do modelo.

Concretamente, neste repositório: a v1 (7043 clientes) treina um modelo com
accuracy 0,7570 e a v2 (7143) chega a 0,7609. Duas acurácias, dois modelos — e a
diferença entre eles é rastreável até o md5 do CSV que os gerou.

### Os atritos do Encontro 5

| Atrito | O que fizemos |
|---|---|
| **config de remote** | remote local (`../dvc-storage`), como o encontro recomenda para começar; trocar por S3 é uma linha de `dvc remote add` |
| **esquecer o `dvc push`** | o Git fica com o ponteiro e os bytes não sobem. `scripts/fetch_data.sh` cobre o buraco reconstruindo o dado do histórico, e um teste checa que o CSV no disco bate com o ponteiro |
| **cache vs workspace** | `scripts/alternar_versao.sh` nunca troca só o ponteiro: sincroniza o dado e confere o md5 no fim |
| **primeiro `add` é lento** | uma vez só, e neste dataset (1 MB) nem aparece |
| **o dado no deploy** | `Dockerfile.dados` faz `dvc pull` para o volume; a imagem de treino nunca vê o DVC nem carrega os bytes |
