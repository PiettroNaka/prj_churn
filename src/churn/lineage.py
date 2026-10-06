"""Linhagem: o que amarra um modelo ao dado e ao código que o geraram.

Sem isto, um run do MLflow diz *qual configuração* produziu uma métrica, mas não
*qual dado* nem *qual versão do código*. Essas duas tags são o que fecha a cadeia
``dado → código → experimento → modelo``.

Nota sobre o ``data_md5``: o deck lê o hash do ponteiro ``data/churn.csv.dvc``,
que só existe depois do Encontro 5. Enquanto o DVC não entra, calculamos o md5
do próprio CSV — e **o número é o mesmo**: para uma saída de arquivo único, o
``md5`` que o ``dvc add`` grava no ponteiro é exatamente o md5 do conteúdo do
arquivo. Quando o ponteiro aparecer, esta função passa a lê-lo sem que nada mais
no projeto mude.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: Ponteiro que o `dvc add data/churn.csv` cria no Encontro 5.
DVC_POINTER = Path("data/churn.csv.dvc")

#: O dataset em si, usado enquanto o ponteiro não existe.
DATA_PATH = Path("data/churn.csv")

#: Valor registrado quando a origem não pode ser determinada. Nunca levantamos
#: exceção aqui: linhagem ausente não pode derrubar um treino.
UNKNOWN = "unknown"


def _md5_do_arquivo(path: Path, chunk: int = 1 << 20) -> str:
    """md5 do conteúdo, lido em blocos — o CSV não precisa caber na memória."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for bloco in iter(lambda: handle.read(chunk), b""):
            digest.update(bloco)
    return digest.hexdigest()


def data_md5(
    pointer: Path = DVC_POINTER, data_path: Path = DATA_PATH
) -> str:
    """Hash da versão exata do dado usada no treino.

    Lê o ponteiro do DVC quando ele existe; senão, calcula o md5 do CSV.
    """
    if pointer.exists():
        import yaml  # dependência do DVC; só é necessária quando há ponteiro

        conteudo = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        outs = conteudo.get("outs") or []
        if outs and "md5" in outs[0]:
            return str(outs[0]["md5"])
        return UNKNOWN

    if data_path.exists():
        return _md5_do_arquivo(data_path)

    return UNKNOWN


def git_commit() -> str:
    """Commit curto do HEAD. ``unknown`` fora de um repositório ou sem git."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return UNKNOWN
    saida = resultado.stdout.strip()
    return saida if resultado.returncode == 0 and saida else UNKNOWN


def git_sujo() -> bool:
    """True se há mudança não commitada — o ``git_commit`` não descreve o código.

    Um run marcado com um commit, mas treinado com a árvore suja, é linhagem
    falsa: o commit não reproduz o que rodou.
    """
    try:
        resultado = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return resultado.returncode == 0 and bool(resultado.stdout.strip())


def tags() -> dict[str, str]:
    """As tags de linhagem prontas para ``mlflow.set_tags``."""
    return {
        "data_md5": data_md5(),
        "git_commit": git_commit(),
        "git_dirty": str(git_sujo()).lower(),
    }
