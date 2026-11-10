"""Sobe o front inteiro: dado, campeão, API e Streamlit — em um comando.

O front depende de três coisas que não são dele: o CSV no disco, um campeão
promovido no Registry e a API no ar. Subir isso à mão são quatro comandos e dois
terminais abertos ao mesmo tempo, e o que falha no meio falha calado — o
Streamlit abre e mostra "Sem resposta" na barra lateral, sem dizer qual das três
faltou.

    uv run --group api --group app python scripts/subir_front.py
    uv run --group api --group app python scripts/subir_front.py --porta 8502

Cada passo confere o anterior e diz o que está fazendo. A API sobe em segundo
plano e é encerrada junto com o Streamlit: fechar a janela não deixa processo
órfão segurando a porta — o que, na segunda tentativa, viraria um
"address already in use" sem causa aparente.

No Windows há o `subir-front.bat`, que é isto com duplo clique.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASSOS = 4


def passo(numero: int, texto: str) -> None:
    print(f"\n[{numero}/{PASSOS}] {texto}", flush=True)


def _rodar(*argumentos: str) -> int:
    """Roda um módulo do projeto mostrando a saída — nada de capturar e esconder."""
    return subprocess.run([sys.executable, *argumentos], cwd=RAIZ).returncode


def preparar_dado() -> None:
    passo(1, "dataset")
    if _rodar(str(RAIZ / "scripts" / "preparar.py"), "--so-dado") != 0:
        raise SystemExit("ERRO: não consegui materializar o dataset.")


def preparar_campeao() -> None:
    """Garante um modelo promovido. Sem ele a API sobe, mas não responde nada."""
    passo(2, "campeão no Registry")
    from churn.registry import versao_do_alias

    if (versao := versao_do_alias()) is not None:
        print(f"      já existe: versão {versao.version} @champion")
        return

    print("      nenhum campeão promovido; buscando e treinando (~2 min)")
    if _rodar("-m", "churn.tuning") != 0:
        raise SystemExit("ERRO: a busca falhou.")
    if _rodar("-m", "churn.registry") != 0:
        raise SystemExit("ERRO: a promoção falhou.")


def _responde(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def subir_api(porta: int) -> subprocess.Popen:
    passo(3, f"API em 127.0.0.1:{porta}")
    processo = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "churn.api:app",
            "--host", "127.0.0.1", "--port", str(porta),
        ],
        cwd=RAIZ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(40):
        if _responde(f"http://127.0.0.1:{porta}/saude"):
            print("      no ar.")
            return processo
        if processo.poll() is not None:
            raise SystemExit(
                f"ERRO: a API morreu ao subir. A porta {porta} pode estar "
                f"ocupada — tente --porta-api {porta + 1}."
            )
        time.sleep(1)

    processo.terminate()
    raise SystemExit("ERRO: a API não respondeu a tempo.")


def subir_front(porta: int, porta_api: int, navegador: bool, endereco: str) -> int:
    passo(4, f"Streamlit em http://localhost:{porta}")
    print("      (Ctrl+C encerra os dois; deixe esta janela aberta)")

    ambiente = dict(os.environ)
    # o front fala com a API por aqui; sem isto ele procuraria a porta padrão
    ambiente["CHURN_API_URL"] = f"http://127.0.0.1:{porta_api}"

    return subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", "app/Inicio.py",
            "--server.port", str(porta),
            "--server.address", endereco,
            "--server.headless", "false" if navegador else "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=RAIZ,
        env=ambiente,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--porta", type=int, default=8501, help="porta do Streamlit")
    parser.add_argument("--porta-api", type=int, default=8000, help="porta da API")
    parser.add_argument(
        "--sem-navegador", action="store_true", help="não abrir o navegador sozinho"
    )
    parser.add_argument(
        "--endereco",
        default="127.0.0.1",
        help="em que interface o Streamlit escuta; 0.0.0.0 para aceitar de fora "
        "(é o que um container precisa)",
    )
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

    preparar_dado()
    preparar_campeao()
    api = subir_api(args.porta_api)
    try:
        return subir_front(
            args.porta, args.porta_api, not args.sem_navegador, args.endereco
        )
    except KeyboardInterrupt:
        return 0
    finally:
        # o finally é o ponto do script: sem ele a API fica de pé depois que o
        # Streamlit sai, e a próxima execução esbarra na porta ocupada
        print("\nEncerrando a API...")
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()


if __name__ == "__main__":
    sys.exit(main())
