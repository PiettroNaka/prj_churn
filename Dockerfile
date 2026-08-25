# Encontro 3 — a camada de reprodutibilidade.
# Base oficial do uv: já traz Python 3.11 + uv na mesma imagem.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Evita o warning de hardlink do uv dentro do container.
ENV UV_LINK_MODE=copy

WORKDIR /app

# Regra de ouro das camadas: deps ANTES do código.
# Mudar uma linha de .py não pode reinstalar o mundo.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Agora o código. --frozen usa o uv.lock exatamente como está, sem re-resolver.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen

# Dados e artefatos NÃO entram na imagem: vêm por volume (ver compose.yml).
# É o mesmo princípio do DVC (Encontro 5): dado é artefato à parte, rastreado por
# ponteiro. A imagem fica com o código; o CSV chega montado, já materializado no
# host por `dvc pull` ou por ./scripts/fetch_data.sh.
CMD ["uv", "run", "python", "-m", "churn.model"]
