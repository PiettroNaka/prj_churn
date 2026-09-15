"""A linhagem tem de responder certo, ou o run aponta para o dado errado."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from churn.lineage import DESCONHECIDO, data_md5, git_commit, ponteiro_de


def test_ponteiro_de_um_dataset():
    assert ponteiro_de(Path("data/churn.csv")) == Path("data/churn.csv.dvc")


def test_md5_sai_do_ponteiro_do_dvc(tmp_path):
    ponteiro = tmp_path / "churn.csv.dvc"
    ponteiro.write_text(
        "outs:\n- md5: f150a64ab5537f24b16770ba76ad7b95\n  size: 995054\n"
        "  hash: md5\n  path: churn.csv\n",
        encoding="utf-8",
    )
    assert data_md5(ponteiro) == "f150a64ab5537f24b16770ba76ad7b95"


def test_sem_ponteiro_cai_no_hash_do_arquivo(tmp_path):
    """O md5 do ponteiro é o md5 do conteúdo — sem ele, a conta é a mesma."""
    dado = tmp_path / "churn.csv"
    dado.write_bytes(b"customerID,Churn\n1234-ABCD,No\n")

    esperado = hashlib.md5(dado.read_bytes()).hexdigest()

    assert data_md5(ponteiro_de(dado)) == esperado


def test_sem_ponteiro_e_sem_arquivo_diz_desconhecido(tmp_path):
    """Melhor uma tag que incomoda do que uma tag ausente que ninguém nota."""
    assert data_md5(tmp_path / "nao-existe.csv.dvc") == DESCONHECIDO


def test_commit_do_ambiente_tem_precedencia(monkeypatch):
    """No container não há .git — quem sabe o commit é o CI, e informa."""
    monkeypatch.setenv("CHURN_GIT_COMMIT", "abcdef1234567890")

    assert git_commit() == "abcdef1"


def test_sem_git_a_tag_diz_desconhecido(monkeypatch):
    """Dentro do container não há git — e a tag tem de dizer isso, não estourar.

    É o caso real do deploy: a imagem de treino tem o código, não o repositório.
    """

    def sem_binario(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.delenv("CHURN_GIT_COMMIT", raising=False)
    monkeypatch.setattr(subprocess, "run", sem_binario)

    assert git_commit() == DESCONHECIDO


@pytest.mark.skipif(shutil.which("git") is None, reason="sem git neste ambiente")
def test_git_commit_marca_codigo_nao_commitado(tmp_path):
    """Run treinado sobre código não commitado não pode apontar um hash limpo."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("um", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "um"], cwd=tmp_path, check=True)

    import os

    anterior = os.getcwd()
    try:
        os.chdir(tmp_path)
        limpo = git_commit()
        assert limpo and "+" not in limpo

        (tmp_path / "a.txt").write_text("dois", encoding="utf-8")
        assert git_commit() == f"{limpo}+alteracoes"
    finally:
        os.chdir(anterior)
