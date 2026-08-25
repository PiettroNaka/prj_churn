# data/

`churn.csv` — Telco Customer Churn com os identificadores regenerados pelo
professor. 7043 linhas × 21 colunas, 981 KB.

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

## Bytes preservados

`.gitattributes` marca `data/*.csv` como `-text` para o Git nunca converter
LF↔CRLF: o hash tem de ser o mesmo no Windows e no Linux, senão o `dvc status`
acusaria mudança onde não houve.

```
md5     d390bd07b5514a2256a2396993b8b0e3   (é o que está no .dvc)
sha256  1b90d77671ab4d396b1bd403eadd764d8457aec50467ad99f586eaabf9304596
```

O DVC usa md5; o sha256 fica registrado porque é o que foi conferido em aula
contra o arquivo distribuído pelo professor.

## Contrato

O schema do arquivo está declarado em `src/churn/validation.py` e é conferido a
cada carga. Coluna faltando, coluna a mais, categoria fora do domínio, valor
fora de faixa ou alvo nulo fazem `load_clean` levantar `DataValidationError`
antes de o treino começar.

O `.dvc` e o schema respondem perguntas diferentes: o ponteiro diz *qual* dado é
este, o schema diz se ele *serve*. Um dataset novo passa no schema e muda o md5;
um dataset corrompido mantém o md5 de nada e reprova no schema.
