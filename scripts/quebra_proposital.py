"""Quebrar o dado de propósito e ver o contrato pegar — tarefa do Encontro 4.

Três atos, no dataset real:

    1. o dataset como está      -> tem de PASSAR
    2. uma linha adulterada     -> tem de REPROVAR, com as duas violações juntas
    3. categoria nova de catálogo -> tem de PASSAR, mas com ALERTA

O segundo ato usa as mesmas quebras do slide: ``tenure = 999`` e
``Contract = "Vitalício"``, na MESMA linha. Com ``lazy=True`` o pandera coleta
todas as falhas antes de levantar, então o relatório mostra as duas de uma vez
— e não só a primeira que ele encontrou.

O terceiro mostra o outro lado da mesma moeda: ``PaymentMethod = "Pix"`` também
está fora do domínio declarado, mas é catálogo comercial — o negócio lançou um
meio de pagamento, o cliente é real e a linha está certa. Ali o contrato avisa
em vez de reprovar. Derrubar produção por causa de um produto novo é o oposto de
qualidade de dados.

    uv run python scripts/quebra_proposital.py

Sai com 0 quando o contrato se comporta (passa no bom, reprova no sujo) e com 1
quando não — é o que o CI checa.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

from churn.validation import DataValidationError, validate_raw

DATASET = Path("data/churn.csv")
LINHA = 0
#: Quebras que têm de reprovar: faixa impossível e categoria estrutural.
QUEBRAS = {"tenure": 999, "Contract": "Vitalício"}
#: Categoria nova numa coluna de catálogo: tem de passar, com alerta.
CATALOGO = ("PaymentMethod", "Pix")


def adulterar(df: pd.DataFrame) -> pd.DataFrame:
    """Suja UMA linha com as duas quebras do slide (pura: não toca no original)."""
    sujo = df.copy()
    for coluna, valor in QUEBRAS.items():
        sujo.loc[LINHA, coluna] = valor
    return sujo


def _legivel(valor: object) -> str:
    """numpy.int64(63) não é o que se mostra em aula — 63 é."""
    return repr(valor.item() if hasattr(valor, "item") else valor)


def _relatorio(casos: pd.DataFrame | None) -> None:
    """Imprime os failure_cases como o slide: coluna, regra, valor."""
    if casos is None or casos.empty:
        print("  (o pandera não devolveu failure_cases)")
        return
    print(f"  {'column':<12} {'check':<36} failure_case")
    print(f"  {'─' * 12} {'─' * 36} {'─' * 14}")
    for _, caso in casos.iterrows():
        coluna = caso["column"] if pd.notna(caso["column"]) else "<dataframe>"
        regra = str(caso["check"])
        if len(regra) > 36:
            regra = regra[:35] + "…"
        print(f"  {coluna:<12} {regra:<36} {caso['failure_case']}")


def main() -> int:
    if not DATASET.exists():
        print(f"ERRO: {DATASET} não está no disco. Rode ./scripts/fetch_data.sh")
        return 1

    cru = pd.read_csv(DATASET)
    print(f"$ dataset: {DATASET} — {len(cru)} linhas × {len(cru.columns)} colunas\n")

    print("[1/3] o dataset como está")
    try:
        validate_raw(cru)
    except DataValidationError as erro:
        print("  REPROVOU — o dataset de partida já não cumpre o contrato:")
        _relatorio(erro.failure_cases)
        return 1
    print("  passou no ChurnRawSchema.\n")

    print(f"[2/3] linha {LINHA} adulterada de propósito")
    for coluna, valor in QUEBRAS.items():
        print(f"  {coluna:<12} {_legivel(cru.loc[LINHA, coluna])} -> {valor!r}")
    print()

    try:
        validate_raw(adulterar(cru))
    except DataValidationError as erro:
        casos = erro.failure_cases
        print(f"  DataValidationError — {len(casos)} violação(ões) num relatório só:\n")
        _relatorio(casos)
        pegou = set(casos["column"].dropna())
        faltou = set(QUEBRAS) - pegou
        if faltou:
            print(f"\n  ATENÇÃO: o contrato não pegou {sorted(faltou)}.")
            return 1
        print("\n  o contrato pegou as duas de uma vez (lazy=True). ✓\n")
    else:
        print("  FALHA DO CONTRATO: o dado adulterado passou.")
        print("  É exatamente o cenário do encontro — dado quebra em silêncio.")
        return 1

    return _ato_do_alerta(cru)


def _ato_do_alerta(cru: pd.DataFrame) -> int:
    """Categoria nova numa coluna de catálogo: passa, mas não em silêncio."""
    coluna, valor = CATALOGO
    print(f"[3/3] categoria nova de catálogo ({coluna} = {valor!r})")

    novo = cru.copy()
    novo.loc[LINHA, coluna] = valor

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        try:
            validate_raw(novo)
        except DataValidationError:
            print(f"  FALHA: {valor!r} derrubou o pipeline.")
            print("  Meio de pagamento novo é evento de negócio, não dado ruim.")
            return 1

    if not avisos:
        print("  FALHA: passou em silêncio, sem alerta nenhum.")
        print("  Alerta que ninguém emite é igual a não ter validação.")
        return 1

    for aviso in avisos:
        texto = str(aviso.message).replace("\n", " ")
        achado = re.search(r"Column '([^']+)'", texto)
        print(
            f"  {type(aviso.message).__name__}: {achado.group(1) if achado else coluna}"
            f" recebeu {valor!r}, fora do domínio declarado"
        )

    print("\n  passou com alerta — o contrato distingue lançamento de corrupção. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
