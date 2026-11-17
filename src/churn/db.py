"""O dataset em Postgres — o dado como banco, não como arquivo.

Até aqui o dado era um CSV: o DVC guardava os bytes, o ponteiro ia no Git e o
pipeline lia do disco. O deploy em EC2 pede outro arranjo — um container para a
aplicação e outro para o banco —, e com isso o dataset passa a morar numa
tabela.

O que **não** muda é o contrato. O pandera segue sendo a fonte da verdade, e
segue rodando duas vezes: aqui, antes de qualquer linha entrar no banco, e
depois em :func:`churn.data.load_clean`, sobre o que saiu dele. É a mesma regra
que já valia quando o CSV passou a ser buscado por HTTP no deploy de vitrine —
muda o transporte, não a verificação. Um banco sem contrato não elimina dado
ruim: só troca o lugar onde ele se esconde.

A ordem das colunas é tratada com cuidado de propósito. O schema declara as
colunas numa ordem e o CSV traz noutra; o modelo foi treinado na ordem do CSV, e
já houve um defeito por causa disso (ver "a ordem das colunas" no README). Por
isso :func:`semear` grava na ordem em que o CSV chega e :func:`ler` devolve na
ordem em que a tabela foi criada — o Postgres preserva a ordem das colunas, e um
teste trava isso.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from churn.validation import validate_raw

#: A tabela com o dataset cru. Uma só: este é um dataset, não um modelo de
#: domínio — normalizar em cinco tabelas aqui seria inventar complexidade que o
#: projeto não usa, e o contrato valida o formato largo.
TABELA = "clientes"

#: A posição que a linha ocupava no CSV, gravada como coluna.
#:
#: Não é enfeite, é o que mantém o projeto reprodutível. Um SELECT sem ORDER BY
#: não tem ordem garantida — o Postgres devolve na ordem que for mais barata, e
#: ela muda com VACUUM, com UPDATE, com o plano escolhido. O ``train_test_split``
#: escolhe linhas por POSIÇÃO: com a mesma semente e outra ordem de linhas, o
#: split é outro, o modelo é outro e as métricas são outras.
#:
#: Sem esta coluna, trocar o CSV pelo banco mudaria o modelo caladamente, e o
#: "mesma semente, mesmo split, mesmos números" que o README afirma deixaria de
#: valer. Ela sai do DataFrame em :func:`ler`: serve ao banco, não ao modelo.
COLUNA_ORDEM = "linha"


def motor(url: str) -> sa.Engine:
    """Cria o Engine do SQLAlchemy.

    ``pool_pre_ping`` porque o banco vive noutro container: ele pode reiniciar
    enquanto a aplicação segue de pé, e uma conexão morta no pool viraria um
    erro na cara de quem está usando o front, não um reconnect.
    """
    return sa.create_engine(url, pool_pre_ping=True, future=True)


def esperar(maquina: sa.Engine, tentativas: int = 30, intervalo: float = 1.0) -> None:
    """Espera o banco aceitar conexão, ou levanta.

    O ``depends_on`` do compose garante a ordem de partida, não a prontidão: o
    container do Postgres sobe em um segundo e o servidor leva vários para
    aceitar conexão. Sem esta espera, a aplicação morre no primeiro ``connect``
    e o compose reinicia tudo num laço.
    """
    ultimo: Exception | None = None
    for _ in range(tentativas):
        try:
            with maquina.connect() as conexao:
                conexao.execute(sa.text("SELECT 1"))
            return
        except sa.exc.OperationalError as erro:  # noqa: PERF203 — é um retry
            ultimo = erro
            time.sleep(intervalo)
    raise RuntimeError(
        f"O banco não respondeu depois de {tentativas} tentativas "
        f"({tentativas * intervalo:.0f}s). Última falha: {ultimo}"
    )


def ha_dados(maquina: sa.Engine) -> bool:
    """A tabela existe e tem pelo menos uma linha?"""
    inspetor = sa.inspect(maquina)
    if not inspetor.has_table(TABELA):
        return False
    with maquina.connect() as conexao:
        total = conexao.execute(sa.text(f"SELECT count(*) FROM {TABELA}")).scalar_one()
    return bool(total)


def contar(maquina: sa.Engine) -> int:
    """Quantas linhas há na tabela. Zero se ela não existir."""
    if not sa.inspect(maquina).has_table(TABELA):
        return 0
    with maquina.connect() as conexao:
        return int(conexao.execute(sa.text(f"SELECT count(*) FROM {TABELA}")).scalar_one())


def semear(maquina: sa.Engine, csv: Path, *, recriar: bool = False) -> int:
    """Carrega o CSV para dentro do Postgres e devolve quantas linhas entraram.

    **Valida antes de gravar.** Se o CSV reprovar no contrato, nada é escrito e
    a exceção sobe — o banco não chega a existir num estado que o pipeline
    depois teria de desconfiar.

    É idempotente: com a tabela já populada não faz nada, porque isto roda a
    cada ``docker compose up`` e recarregar 7 mil linhas a cada subida não
    serviria a ninguém. ``recriar=True`` força a recarga.
    """
    if ha_dados(maquina) and not recriar:
        return contar(maquina)

    bruto = pd.read_csv(csv)
    # o contrato, antes do INSERT — esta é a linha que separa um banco de um
    # depósito
    validate_raw(bruto)

    # a ordem original vira dado, porque o banco não a guarda sozinho
    bruto = bruto.assign(**{COLUNA_ORDEM: range(len(bruto))})

    bruto.to_sql(
        TABELA,
        maquina,
        if_exists="replace",
        index=False,
        # 7 mil linhas num INSERT só estoura o limite de parâmetros do
        # protocolo; em lotes, entra em menos de um segundo
        chunksize=1000,
        method="multi",
    )
    return len(bruto)


def ler(maquina: sa.Engine) -> pd.DataFrame:
    """Lê o dataset cru do Postgres, na ordem de colunas em que ele foi gravado.

    O resultado é equivalente a ``pd.read_csv(data/churn.csv)``: mesmas colunas,
    mesma ordem de colunas, mesmos tipos **e mesma ordem de linhas**. É o que
    permite a :mod:`churn.data` tratar banco e arquivo como a mesma coisa daí
    para a frente — e um teste compara os dois DataFrames inteiros.

    O ``ORDER BY`` não é opcional: ver :data:`COLUNA_ORDEM`.
    """
    if not sa.inspect(maquina).has_table(TABELA):
        raise RuntimeError(
            f"A tabela '{TABELA}' não existe. Rode a carga antes: "
            "`uv run python -m churn.db` (ou `docker compose up dados`)."
        )
    frame = pd.read_sql_query(
        sa.text(f"SELECT * FROM {TABELA} ORDER BY {COLUNA_ORDEM}"), maquina
    )
    return frame.drop(columns=[COLUNA_ORDEM])


def principal() -> None:
    """Ponto de entrada: carrega o CSV no banco. ``python -m churn.db``."""
    from churn.config import settings

    if not settings.database_url:
        raise SystemExit(
            "CHURN_DATABASE_URL não está definida — sem ela não há banco para "
            "semear. Exemplo:\n"
            "  CHURN_DATABASE_URL=postgresql+psycopg://churn:churn@localhost:5432/churn"
        )

    maquina = motor(settings.database_url)
    print(f"esperando o banco em {_sem_senha(settings.database_url)} …")
    esperar(maquina)

    antes = contar(maquina)
    total = semear(maquina, settings.data_path)
    if antes and antes == total:
        print(f"a tabela '{TABELA}' já tinha {total} linhas — nada a fazer")
    else:
        print(f"{total} linhas carregadas em '{TABELA}', validadas pelo contrato")


def _sem_senha(url: str) -> str:
    """Esconde a senha da URL antes de imprimi-la — log não é lugar de senha."""
    try:
        return sa.engine.make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 — imprimir não pode derrubar o processo
        return "<url do banco>"


if __name__ == "__main__":
    principal()
