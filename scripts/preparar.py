"""Prepara tudo e sobe a UI do MLflow — a versão que roda em qualquer terminal.

Os scripts `.sh` do projeto não rodam no PowerShell nem no CMD do Windows. Este
faz o mesmo caminho em Python puro (só a biblioteca padrão e o ``git``), então
funciona igual no Windows, no macOS e no Linux:

    uv run python scripts/preparar.py          # dado + treino + UI em localhost:5000
    uv run python scripts/preparar.py --porta 5001
    uv run python scripts/preparar.py --so-dado    # materializa o CSV e para

Passo a passo, pulando o que já estiver pronto:

    1. data/churn.csv — do remote do DVC, ou reconstruído do histórico do Git
    2. um run — treina uma vez, se o banco ainda estiver vazio
    3. a UI — no mesmo backend que o treino usa

O equivalente em bash continua existindo (``scripts/fetch_data.sh`` e
``scripts/mlflow_ui.sh``); este aqui é o mesmo caminho sem depender de bash.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "churn.csv"
PONTEIRO = RAIZ / "data" / "churn.csv.dvc"
GERADOR = RAIZ / "scripts" / "gerar_v2.py"

#: Blob de data/churn.csv no commit 049e49b — a v1, antes de o DVC assumir.
BLOB = "eb8a7d540c8e7316e3c4011211c1920d379bffac"


def md5_do_arquivo(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def md5_do_ponteiro() -> str:
    achado = re.search(r"md5:\s*([0-9a-f]{32})", PONTEIRO.read_text(encoding="utf-8"))
    if achado is None:
        raise SystemExit(f"ERRO: não consegui ler o md5 de {PONTEIRO}")
    return achado.group(1)


def confere(esperado: str) -> bool:
    return DATASET.exists() and md5_do_arquivo(DATASET) == esperado


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True)


def preparar_dado() -> None:
    """Deixa data/churn.csv no disco, conferido contra o ponteiro do DVC."""
    esperado = md5_do_ponteiro()

    if confere(esperado):
        print(f"[1/3] dataset já está no disco e bate com o ponteiro ({esperado[:10]}…)")
        return

    print("[1/3] materializando o dataset")
    # O caminho normal é o DVC; ele pode nem estar instalado (fica fora do
    # grupo padrão), e aí seguimos para a rede de segurança.
    dvc = subprocess.run(
        [sys.executable, "-m", "dvc", "pull", str(PONTEIRO)],
        cwd=RAIZ,
        capture_output=True,
    )
    if dvc.returncode == 0 and confere(esperado):
        print("      veio do remote do DVC.")
        return

    print(f"      sem remote alcançável; recuperando a v1 do Git (blob {BLOB[:7]})")
    existe = _git("cat-file", "-e", BLOB)
    if existe.returncode != 0:
        raise SystemExit(
            f"ERRO: o blob {BLOB} não está neste clone.\n"
            "      Num clone raso, rode: git fetch --unshallow"
        )

    conteudo = _git("cat-file", "blob", BLOB)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    # bytes crus: nada de texto aqui, ou o Windows troca LF por CRLF e o md5 muda
    DATASET.write_bytes(conteudo.stdout)

    if not confere(esperado):
        print("      o ponteiro não é o da v1; reconstruindo a v2")
        subprocess.run([sys.executable, str(GERADOR)], cwd=RAIZ, check=True)

    if not confere(esperado):
        raise SystemExit(
            f"ERRO: o arquivo recuperado não bate com o ponteiro.\n"
            f"      esperado: {esperado}\n"
            f"      obtido:   {md5_do_arquivo(DATASET)}"
        )
    print(f"      pronto e conferido ({esperado[:10]}…)")


def banco_de(uri: str) -> Path | None:
    """O arquivo do SQLite, quando o backend é local."""
    if uri.startswith("sqlite:////"):
        return Path("/" + uri[len("sqlite:////") :])
    if uri.startswith("sqlite:///"):
        return RAIZ / uri[len("sqlite:///") :]
    return None


def preparar_run(uri: str) -> None:
    """Garante pelo menos um run — UI vazia não mostra modelo."""
    banco = banco_de(uri)
    if banco is not None and banco.exists() and banco.stat().st_size > 0:
        print("[2/3] já existe run registrado")
        return

    print("[2/3] nenhum run ainda; treinando uma vez (leva ~15s)")
    subprocess.run([sys.executable, "-m", "churn.model"], cwd=RAIZ, check=True)


def subir_ui(uri: str, porta: int) -> int:
    print(f"\n[3/3] MLflow UI")
    print(f"      backend: {uri}")
    print(f"      abra:    http://localhost:{porta}")
    print("      (Ctrl+C encerra; deixe esta janela aberta)\n")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            uri,
            "--port",
            str(porta),
        ],
        cwd=RAIZ,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--porta", type=int, default=5000)
    parser.add_argument(
        "--so-dado", action="store_true", help="materializa o CSV e para por aí"
    )
    args = parser.parse_args()

    # a telemetria do MLflow 3 não serve para nada num projeto de curso
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

    preparar_dado()
    if args.so_dado:
        return 0

    from churn.config import settings

    uri = settings.mlflow_tracking_uri
    preparar_run(uri)
    return subir_ui(uri, args.porta)


if __name__ == "__main__":
    sys.exit(main())
