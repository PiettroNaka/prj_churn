"""Linhagem: de qual dado e de qual código este modelo saiu.

Um run sem essa resposta não é reproduzível — é um número bonito sem endereço.
São duas perguntas e duas respostas curtas:

    data_md5()    qual dado   → o md5 que o DVC guarda no ponteiro (Encontro 5)
    git_commit()  qual código → o hash do commit

As duas viram *tags* no run do MLflow, e é o que costura as ferramentas: o DVC
versiona o dado, o Git versiona o código, o MLflow compara os experimentos — e a
tag é o fio que liga os três.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml

#: De onde o commit vem quando não há repositório por perto. É o caso do
#: container: ele carrega o código, não o .git. Quem sabe o commit ali é quem
#: construiu a imagem ou disparou o job — daí a variável.
ENV_COMMIT = "CHURN_GIT_COMMIT"

#: Valor das tags quando a resposta não existe naquele ambiente (um container
#: sem .git, por exemplo). Preenchido de propósito em vez de omitido: tag
#: ausente some da UI, "desconhecido" aparece e incomoda — que é o ponto.
DESCONHECIDO = "desconhecido"


def ponteiro_de(data_path: Path) -> Path:
    """O arquivo .dvc que rastreia esse dado."""
    return Path(f"{data_path}.dvc")


def data_md5(dvc_pointer: Path = Path("data/churn.csv.dvc")) -> str:
    """Lê o md5 do dado versionado pelo DVC — a versão exata do churn.csv.

    Se o ponteiro não existir (alguém clonou sem DVC, ou apontou a config para
    um CSV solto), cai no md5 do próprio arquivo. É a mesma resposta: o md5 do
    ponteiro *é* o md5 do conteúdo. O que se perde é só a garantia de que aquela
    versão está registrada em algum lugar.
    """
    if dvc_pointer.exists():
        doc = yaml.safe_load(dvc_pointer.read_text(encoding="utf-8"))
        return str(doc["outs"][0]["md5"])

    dado = Path(str(dvc_pointer).removesuffix(".dvc"))
    if dado.exists():
        return hashlib.md5(dado.read_bytes()).hexdigest()

    return DESCONHECIDO


def git_commit() -> str:
    """Hash curto do commit atual, com ``-sujo`` quando há mudança não commitada.

    O sufixo não é preciosismo: um run treinado em cima de código não commitado
    aponta para um commit que não contém aquele código. Melhor a tag dizer isso
    do que mentir com um hash limpo.

    ``CHURN_GIT_COMMIT`` tem precedência: dentro do container não há ``.git``, e
    sem ela a metade "qual código" da linhagem ficaria em branco justamente no
    modelo que o CI produz. O ambiente que sabe informa.
    """
    do_ambiente = os.environ.get(ENV_COMMIT, "").strip()
    if do_ambiente:
        return do_ambiente[:7]

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DESCONHECIDO

    sujo = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return f"{commit}-sujo" if sujo else commit
