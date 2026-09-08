"""Gera a v2 do dataset: v1 + 100 clientes novos — tarefa do Encontro 5.

O encontro pede duas versões do dado para exercitar o ``dvc checkout``. A v2 do
professor tem md5 ``49f0efac…``; aqueles 100 clientes são o arquivo dele e não
estão neste repositório, então a v2 daqui é **reproduzível a partir da v1**, e é
essa a propriedade que importa: quem tiver a v1 e este script chega aos mesmos
bytes, sempre.

Como as 100 linhas são feitas: sorteia 100 clientes existentes (semente fixa) e
os recadastra com um ``customerID`` novo (``90001-NOVO`` … ``90100-NOVO``). O
resto da linha é copiado byte a byte, o que garante duas coisas de graça — o
arquivo continua cumprindo o contrato do Encontro 4 (faixas, categorias e a
relação ``TotalCharges ≈ tenure × MonthlyCharges`` vêm junto) e o resultado é
idêntico em qualquer máquina.

    python3 scripts/gerar_v2.py          # só a biblioteca padrão, roda em qualquer lugar

É idempotente: rodar de novo num arquivo que já é v2 não faz nada. Por isso o
``scripts/fetch_data.sh`` pode usá-lo para reconstruir a v2 quando não há remote
alcançável — a v1 vem do histórico do Git e este script termina o serviço.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "churn.csv"
SEMENTE = 5  # o número do encontro; qualquer um serve, desde que não mude
NOVOS = 100
MARCA = "90001-NOVO"  # a primeira linha nova; é o que identifica um arquivo já v2


def novas_linhas(corpo: list[str]) -> list[str]:
    """Sorteia clientes e os recadastra com identificador novo."""
    sorteados = random.Random(SEMENTE).sample(range(len(corpo)), NOVOS)
    linhas = []
    for numero, indice in enumerate(sorteados, start=1):
        _, resto = corpo[indice].split(",", 1)
        linhas.append(f"9{numero:04d}-NOVO,{resto}")
    return linhas


def main() -> int:
    if not DATASET.exists():
        print(f"ERRO: {DATASET} não está no disco. Rode ./scripts/fetch_data.sh")
        return 1

    conteudo = DATASET.read_text(encoding="utf-8")
    if MARCA in conteudo:
        print(f"{DATASET.name} já é a v2 ({len(conteudo.splitlines()) - 1} clientes).")
        return 0

    linhas = conteudo.splitlines(keepends=True)
    if not linhas[-1].endswith("\n"):
        linhas[-1] += "\n"

    cabecalho, corpo = linhas[0], linhas[1:]
    saida = "".join([cabecalho, *corpo, *novas_linhas(corpo)])
    DATASET.write_text(saida, encoding="utf-8", newline="")

    md5 = hashlib.md5(DATASET.read_bytes()).hexdigest()
    print(f"v2 gerada: {len(corpo)} + {NOVOS} = {len(corpo) + NOVOS} clientes")
    print(f"  md5 {md5}")
    print("  próximo passo: dvc add data/churn.csv && git commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
