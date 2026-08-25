# data/

`churn.csv` — Telco Customer Churn com os identificadores regenerados pelo
professor. 7043 linhas × 21 colunas, 981 KB.

## Por que está versionado no Git

Não é o estado final: a partir do **Encontro 5** o DVC assume o rastreamento, e
o `dvc add` vai recolocar o ignore no `.gitignore` e trocar este arquivo por um
`churn.csv.dvc`. Até lá, deixá-lo aqui é o que permite:

- clonar o repositório e rodar o treino sem procurar o dataset em outro lugar;
- o CI executar o treino **dentro do container**, com o dado entrando por
  volume — o checkpoint do Encontro 3.

## Bytes preservados

`.gitattributes` marca `data/*.csv` como `-text`, para o Git não converter
LF↔CRLF no checkout. Sem isso o arquivo teria hash diferente no Windows e no
Linux — o mesmo "na minha máquina funciona" que o curso ataca, e que quebraria
o rastreamento do DVC.

Conferência: o blob no Git bate com o arquivo distribuído em aula.

```
sha256  1b90d77671ab4d396b1bd403eadd764d8457aec50467ad99f586eaabf9304596
```

## Contrato

O schema do arquivo está declarado em `src/churn/validation.py` e é conferido a
cada carga. Coluna faltando, coluna a mais, categoria fora do domínio, valor
fora de faixa ou alvo nulo fazem `load_clean` levantar `DataValidationError`
antes de o treino começar.
