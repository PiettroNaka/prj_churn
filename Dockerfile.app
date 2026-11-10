# A imagem do front — as duas páginas do Streamlit.
#
# O front não carrega modelo: ele fala com a API. É o que mantém a troca de
# campeão fora daqui — promover é mover um alias, e esta imagem não muda.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_LINK_MODE=copy
# No compose o serviço se chama `api`; o DNS interno resolve o nome.
ENV CHURN_API_URL=http://api:8000

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --group app

COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --group app

# As páginas por último: elas mudam com frequência e não entram na resolução de
# dependências. Antes do sync, editar um texto da tela reinstalaria o ambiente.
#
# A v1 mostra o código-fonte e o CSV. O código vai na imagem (em src/, acima);
# o CSV NÃO — ele chega por volume, como no resto do projeto (Encontro 5).
COPY app/ ./app/

EXPOSE 8501

CMD ["uv", "run", "--frozen", "--no-dev", "--group", "app", \
     "streamlit", "run", "app/Inicio.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
