# data/

O dataset `churn.csv` (Telco Customer Churn, ~981 KB) **não é versionado no Git**
por decisão de projeto: a partir do Encontro 5 ele passa a ser rastreado por DVC.
Veja a regra `/data/*.csv` no `.gitignore` da raiz.

Para rodar o treino agora, coloque o arquivo aqui:

```
data/churn.csv
```

O CSV é o distribuído em aula (mesmo arquivo usado pelo `train_churn.py` original).
Colunas esperadas estão declaradas em `src/churn/data.py` (`EXPECTED_COLUMNS`);
se faltar alguma, `validate()` levanta `ValueError` na carga.
