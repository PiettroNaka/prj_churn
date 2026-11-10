---
title: Previsão de Churn
emoji: 📉
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Previsão de churn — projeto de MLOps

Previsão de cancelamento de clientes (dataset Telco), do contrato dos dados até
o modelo servido por HTTP.

Duas páginas:

- **Início** — o código do projeto e os dados do CSV.
- **Prever churn** — os parâmetros de um cliente e a probabilidade de ele
  cancelar.

## O que está rodando aqui

Este Space roda a pilha inteira num container só:

```
Streamlit (porta 7860, pública)  →  FastAPI (127.0.0.1:8000)  →  modelo @champion
```

O front **não carrega o modelo** — ele pergunta para a API, que carrega o
campeão promovido no MLflow Model Registry. É o mesmo desenho que roda em
`docker compose up app`; o que muda é só que aqui os dois processos dividem um
container, porque a plataforma expõe uma porta só.

O modelo é treinado e promovido na **construção da imagem**, com a configuração
que a busca de hiperparâmetros encontrou (`roc_auc` 0,8206 · `recall` 0,7609).
O dataset é buscado do commit em que ainda era versionado pelo Git e só é aceito
se o md5 bater com o ponteiro do DVC.

## O projeto

Código, decisões, dificuldades encontradas e ressalvas:
**https://github.com/PiettroNaka/prj_churn**
