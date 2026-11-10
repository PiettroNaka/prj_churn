# A imagem do Hugging Face Spaces — a pilha inteira num container só.
#
# O Spaces roda UM container e expõe UMA porta. O projeto tem dois processos (a
# API e o front), e o desenho depende disso: o front não carrega modelo, ele
# pergunta para a API. Em vez de desmontar o desenho para caber na plataforma,
# os dois sobem aqui dentro — a API em 127.0.0.1:8000, só para o front, e o
# Streamlit na porta pública.
#
# É o mesmo `scripts/subir_front.py` que roda no Windows e no Linux. Nada aqui é
# específico do Spaces além do usuário e da porta.
#
# O código vem do GitHub por clone, não por COPY: assim o repositório do Space
# tem só dois arquivos (este e o README.md) e a fonte da verdade continua sendo
# um lugar só. Para atualizar, é "Factory rebuild" no Space.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ARG REPO=https://github.com/PiettroNaka/prj_churn
ARG REF=main

ENV UV_LINK_MODE=copy \
    MLFLOW_DISABLE_TELEMETRY=true \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# O Spaces roda o container como usuário 1000, não root. Sem isto o treino não
# conseguiria escrever o mlflow.db nem o mlruns, e o Space subiria sem modelo.
RUN useradd -m -u 1000 usuario
USER usuario
ENV HOME=/home/usuario \
    PATH=/home/usuario/.local/bin:$PATH

WORKDIR /home/usuario
RUN git clone --depth 1 --branch "$REF" "$REPO" app
WORKDIR /home/usuario/app

RUN uv sync --frozen --no-dev --group api --group app

# O modelo é treinado e promovido AQUI, na construção da imagem — não na
# subida. São ~20 segundos que ninguém precisa esperar olhando uma tela branca,
# e o Space passa a nascer com `churn v1 @champion` pronto.
#
# O dataset vem por HTTP (o clone é raso, não tem o histórico) e só é aceito se
# o md5 bater com o ponteiro do DVC. Transporte diferente, mesma verificação.
ENV CHURN_MLFLOW_TRACKING_URI=sqlite:////home/usuario/app/mlflow.db \
    CHURN_N_ESTIMATORS=200 \
    CHURN_MAX_DEPTH=8 \
    CHURN_MIN_SAMPLES_LEAF=16 \
    CHURN_MIN_SAMPLES_SPLIT=5 \
    CHURN_MAX_FEATURES=log2 \
    CHURN_CLASS_WEIGHT=balanced

RUN uv run --frozen --no-dev --group api --group app \
        python scripts/preparar.py --so-dado \
    && uv run --frozen --no-dev --group api --group app \
        python -m churn.model \
    && uv run --frozen --no-dev --group api --group app \
        python -m churn.registry

# 7860 é a porta que o Spaces espera por padrão.
ENV PORT=7860
EXPOSE 7860

# 0.0.0.0 porque o Streamlit precisa aceitar conexões de fora do container.
CMD ["sh", "-c", "uv run --frozen --no-dev --group api --group app \
     python scripts/subir_front.py \
     --porta ${PORT} --endereco 0.0.0.0 --sem-navegador"]
