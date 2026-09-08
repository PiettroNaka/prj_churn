# data/

`churn.csv` — Telco Customer Churn com os identificadores regenerados pelo
professor. 21 colunas, em duas versões registradas.

## Quem versiona este diretório

Desde o **Encontro 5**, o DVC. No Git entra só o ponteiro:

```
data/churn.csv.dvc   ← 3 linhas: md5, tamanho, caminho  (versionado)
data/churn.csv       ← 981 KB de dado                   (ignorado pelo Git)
data/.gitignore      ← escrito pelo `dvc add`           (versionado)
```

Trocar 981 KB por 3 linhas é o ponto do encontro: o Git passa a versionar a
*identidade* do dataset, não os bytes. Um `git diff` de troca de dataset vira
uma linha de md5 — legível numa revisão, ao contrário de um diff binário.

## Como trazer o arquivo

```bash
./scripts/fetch_data.sh   # tenta dvc pull; se não houver remote, cai no histórico
dvc pull                  # o caminho normal, quando o remote está configurado
```

O script confere o md5 do que materializou contra o `.dvc`. Se não bater, ele
falha — dado silenciosamente diferente é pior que dado ausente.

## As duas versões

| | clientes | bytes | md5 |
|---|---|---|---|
| **v1** | 7043 | 981.685 | `d390bd07b5514a2256a2396993b8b0e3` |
| **v2** | 7143 | 995.054 | `f150a64ab5537f24b16770ba76ad7b95` |

A v1 é o arquivo como veio da aula. Confere byte a byte com o distribuído pelo
professor: `sha256 1b90d77671ab4d396b1bd403eadd764d8457aec50467ad99f586eaabf9304596`.

A v2 é a v1 mais 100 clientes, produzida por `scripts/gerar_v2.py` — sorteio com
semente fixa, linhas copiadas byte a byte, identificador novo. Reproduzível:
quem tem a v1 e o script chega nos mesmos 995.054 bytes. (A v2 do professor tem
md5 `49f0efac…`; aqueles 100 clientes são o arquivo dele, e não este.)

Alternar entre elas: `./scripts/alternar_versao.sh v1` (ou `v2`).

## Bytes preservados

`.gitattributes` marca `data/*.csv` como `-text` para o Git nunca converter
LF↔CRLF: o hash tem de ser o mesmo no Windows e no Linux, senão o `dvc status`
acusaria mudança onde não houve.

## Contrato

O schema do arquivo está declarado em `src/churn/validation.py` e é conferido a
cada carga. Coluna faltando, coluna a mais, categoria fora do domínio, valor
fora de faixa ou alvo nulo fazem `load_clean` levantar `DataValidationError`
antes de o treino começar.

O `.dvc` e o schema respondem perguntas diferentes: o ponteiro diz *qual* dado é
este, o schema diz se ele *serve*. Um dataset novo passa no schema e muda o md5;
um dataset corrompido mantém o md5 de nada e reprova no schema.
