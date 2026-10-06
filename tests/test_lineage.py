"""Testes da linhagem — o que amarra modelo, dado e código.

Linhagem errada é pior que linhagem ausente: um run marcado com o hash errado
afirma reprodutibilidade que não existe.
"""
from pathlib import Path

import pytest

from churn.lineage import UNKNOWN, data_md5, git_commit, tags

DATASET = Path("data/churn.csv")

#: md5 do churn.csv distribuído em aula. É o mesmo valor que o `dvc add` vai
#: gravar no ponteiro no Encontro 5 — para arquivo único, o md5 do DVC é o md5
#: do conteúdo. As notas do deck confirmam este hash no ponteiro de referência.
MD5_ESPERADO = "d390bd07b5514a2256a2396993b8b0e3"


@pytest.mark.skipif(not DATASET.exists(), reason="dataset ausente")
def test_md5_do_dataset_bate_com_o_do_professor():
    assert data_md5(pointer=Path("nao-existe.dvc"), data_path=DATASET) == MD5_ESPERADO


def test_le_o_ponteiro_do_dvc_quando_ele_existe(tmp_path):
    """Com o ponteiro presente, o hash vem dele — não do arquivo."""
    ponteiro = tmp_path / "churn.csv.dvc"
    ponteiro.write_text(
        "outs:\n- md5: abc123\n  size: 10\n  path: churn.csv\n", encoding="utf-8"
    )
    csv = tmp_path / "churn.csv"
    csv.write_text("conteudo diferente", encoding="utf-8")

    assert data_md5(pointer=ponteiro, data_path=csv) == "abc123"


def test_ponteiro_malformado_nao_quebra_o_treino(tmp_path):
    """Linhagem ausente degrada para 'unknown'; nunca derruba o treino."""
    ponteiro = tmp_path / "churn.csv.dvc"
    ponteiro.write_text("outs: []\n", encoding="utf-8")

    assert data_md5(pointer=ponteiro, data_path=tmp_path / "nao-existe.csv") == UNKNOWN


def test_sem_ponteiro_e_sem_dado(tmp_path):
    assert (
        data_md5(pointer=tmp_path / "x.dvc", data_path=tmp_path / "y.csv") == UNKNOWN
    )


def test_git_commit_tem_formato_de_hash_curto():
    commit = git_commit()
    assert commit == UNKNOWN or (7 <= len(commit) <= 12 and commit.isalnum())


def test_tags_trazem_as_tres_chaves():
    assert set(tags()) == {"data_md5", "git_commit", "git_dirty"}
