"""O dataset em Postgres — e a prova de que banco e arquivo são a mesma coisa.

Estes testes precisam de um Postgres de verdade. ``CHURN_TEST_DATABASE_URL``
diz onde; sem ela, o módulo inteiro é pulado. Não é teste opcional: o CI sobe um
serviço de Postgres e define a variável, então eles rodam a cada push. Um teste
de banco contra um banco falso provaria o dublê, não o banco.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

sa = pytest.importorskip("sqlalchemy")

from churn.data import load_clean  # noqa: E402
from churn.db import (  # noqa: E402
    COLUNA_ORDEM,
    TABELA,
    contar,
    esperar,
    ha_dados,
    ler,
    motor,
    semear,
)

URL = os.environ.get("CHURN_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not URL, reason="sem CHURN_TEST_DATABASE_URL — nenhum Postgres para testar"
)


@pytest.fixture
def maquina():
    """Um Engine ligado a um banco vazio: a tabela cai antes de cada teste."""
    mot = motor(URL)
    with mot.begin() as conexao:
        conexao.execute(sa.text(f"DROP TABLE IF EXISTS {TABELA}"))
    return mot


@pytest.fixture
def csv(tmp_path: Path, raw_frame: pd.DataFrame) -> Path:
    """A amostra do conftest gravada como CSV, que é a origem a comparar."""
    destino = tmp_path / "amostra.csv"
    raw_frame.to_csv(destino, index=False)
    return destino


def test_carga_devolve_o_numero_de_linhas(maquina, csv, raw_frame):
    assert semear(maquina, csv) == len(raw_frame)
    assert contar(maquina) == len(raw_frame)
    assert ha_dados(maquina)


def test_banco_vazio_nao_tem_dados(maquina):
    assert not ha_dados(maquina)
    assert contar(maquina) == 0


def test_ida_e_volta_devolve_o_mesmo_frame(maquina, csv):
    """O teste central: ler do banco é indistinguível de ler o CSV."""
    semear(maquina, csv)
    pd.testing.assert_frame_equal(pd.read_csv(csv), ler(maquina))


def test_a_coluna_de_ordem_nao_vaza(maquina, csv):
    """``linha`` serve ao banco, não ao modelo — ela não pode sair no DataFrame.

    Se vazasse, viraria uma feature: o índice da linha no CSV entraria no
    treino, e o modelo aprenderia a posição em vez do cliente.
    """
    semear(maquina, csv)
    assert COLUNA_ORDEM not in ler(maquina).columns


def test_a_ordem_das_linhas_sobrevive_a_um_update(maquina, csv):
    """Regressão: um SELECT sem ORDER BY não tem ordem garantida.

    No Postgres um UPDATE grava uma nova versão da linha no fim do heap, então
    a linha mexida passa a ser lida por último. Sem o ORDER BY, isto mudaria a
    ordem do DataFrame — e como o ``train_test_split`` escolhe linhas por
    posição, o split, o modelo e as métricas mudariam junto, caladamente.
    """
    semear(maquina, csv)
    esperado = ler(maquina)

    with maquina.begin() as conexao:
        # mexe na primeira linha sem alterar valor nenhum: só a move no disco
        conexao.execute(
            sa.text(f'UPDATE {TABELA} SET "gender" = "gender" WHERE {COLUNA_ORDEM} = 0')
        )

    sem_ordem = pd.read_sql_query(sa.text(f"SELECT * FROM {TABELA}"), maquina)
    assert sem_ordem[COLUNA_ORDEM].iloc[0] != 0, (
        "o UPDATE não moveu a linha; este teste deixou de exercitar o que promete"
    )
    pd.testing.assert_frame_equal(esperado, ler(maquina))


def test_a_carga_e_idempotente(maquina, csv):
    """Isto roda a cada `docker compose up`; recarregar a cada subida é desperdício."""
    semear(maquina, csv)
    with maquina.begin() as conexao:
        conexao.execute(sa.text(f"DELETE FROM {TABELA} WHERE {COLUNA_ORDEM} = 0"))

    semear(maquina, csv)  # tabela já populada: não deve recarregar
    assert contar(maquina) == 3

    semear(maquina, csv, recriar=True)  # agora sim
    assert contar(maquina) == 4


def test_dado_fora_do_contrato_nao_entra_no_banco(maquina, tmp_path, raw_frame):
    """Valida ANTES de gravar: um banco sem contrato só muda o dado ruim de lugar."""
    sujo = raw_frame.copy()
    sujo.loc[0, "Contract"] = "Vitalício"  # categoria que o contrato recusa
    origem = tmp_path / "sujo.csv"
    sujo.to_csv(origem, index=False)

    with pytest.raises(Exception):  # noqa: B017 — DataValidationError do pandera
        semear(maquina, origem)

    assert not ha_dados(maquina), "o lote reprovado chegou a ser gravado"


def test_o_pipeline_da_o_mesmo_resultado_pelas_duas_origens(maquina, csv):
    """Nada depois de ``carregar_origem`` sabe de onde o DataFrame veio."""
    semear(maquina, csv)
    por_arquivo = load_clean(csv, "customerID", "Churn", None, None)
    por_banco = load_clean(csv, "customerID", "Churn", None, URL)
    pd.testing.assert_frame_equal(por_arquivo, por_banco)


def test_ler_sem_tabela_diz_o_que_fazer(maquina):
    with pytest.raises(RuntimeError, match="não existe"):
        ler(maquina)


def test_esperar_desiste_com_mensagem_clara():
    """O banco vive noutro container: a espera é parte do contrato de partida."""
    morto = motor("postgresql+psycopg://ninguem@127.0.0.1:1/vazio")
    with pytest.raises(RuntimeError, match="não respondeu"):
        esperar(morto, tentativas=2, intervalo=0.01)
