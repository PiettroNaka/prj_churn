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
# É o mesmo princípio do .gitignore — dado é artefato à parte, DVC no Encontro 5.
CMD ["uv", "run", "python", "-m", "churn.model"]
