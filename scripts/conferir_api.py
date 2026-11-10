"""Confere que a API no ar responde de verdade — o teste de fumaça do deploy.

Testes de unidade falam com o app em processo; isto fala com o serviço pela
rede, como um consumidor faria. É a diferença entre "o código está certo" e "o
container subiu, carregou o campeão e responde na porta".

    uv run python scripts/conferir_api.py
    CHURN_API_URL=http://api:8000 uv run python scripts/conferir_api.py

Só biblioteca padrão para o HTTP: assim ele roda em qualquer uma das imagens do
projeto, não só na que tem ``requests``.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from churn.contrato import cliente_exemplo

API = os.environ.get("CHURN_API_URL", "http://localhost:8000").rstrip("/")
ESPERA = 10


def _pegar(caminho: str, corpo: object | None = None) -> tuple[int, object]:
    dados = None if corpo is None else json.dumps(corpo).encode()
    pedido = urllib.request.Request(
        f"{API}{caminho}",
        data=dados,
        headers={"Content-Type": "application/json"} if dados else {},
    )
    try:
        with urllib.request.urlopen(pedido, timeout=ESPERA) as resposta:
            return resposta.status, json.load(resposta)
    except urllib.error.HTTPError as erro:
        return erro.code, json.loads(erro.read() or b"null")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as erro:
        # O caso mais comum em CI: o container não subiu. Um traceback aqui
        # esconderia o diagnóstico no meio da pilha; 0 e a causa dizem na hora.
        return 0, {"erro": f"sem resposta de {API}: {erro}"}


def main() -> int:
    print(f"API: {API}\n")
    falhas: list[str] = []

    def conferir(descricao: str, condicao: bool, detalhe: str = "") -> None:
        print(f"  {'ok  ' if condicao else 'FALHA'} {descricao}{detalhe}")
        if not condicao:
            falhas.append(descricao)

    codigo, saude = _pegar("/saude")
    conferir("/saude responde 200", codigo == 200, f" ({codigo})")
    conferir(
        "modelo carregado",
        isinstance(saude, dict) and saude.get("status") == "ok",
        f" ({saude})",
    )

    codigo, modelo = _pegar("/modelo")
    conferir("/modelo responde 200", codigo == 200, f" ({codigo})")
    if isinstance(modelo, dict):
        conferir(
            "aponta para o alias, não para um run solto",
            str(modelo.get("uri", "")).startswith("models:/"),
            f" ({modelo.get('uri')})",
        )

    # O cliente sai do contrato: se o schema mudar, esta chamada muda junto.
    exemplo = cliente_exemplo()
    codigo, previsao = _pegar("/prever", [exemplo])
    conferir("/prever responde 200", codigo == 200, f" ({codigo})")

    if codigo == 200 and isinstance(previsao, list) and previsao:
        (primeira,) = previsao
        conferir("traz rótulo e probabilidade", set(primeira) == {"churn", "probabilidade"})
        conferir(
            "probabilidade entre 0 e 1",
            0.0 <= primeira["probabilidade"] <= 1.0,
            f" ({primeira['probabilidade']:.3f})",
        )
        print(f"\n  cliente de exemplo -> {primeira['churn']} "
              f"({primeira['probabilidade']:.1%})")
    else:
        falhas.append("/prever não devolveu previsão")
        print(f"    corpo: {previsao}")

    # O contrato tem de recusar o que o schema recusa, também pela rede.
    codigo, _ = _pegar("/prever", [{**exemplo, "Contract": "Plano Vitalicio"}])
    conferir("recusa categoria fora do contrato", codigo == 422, f" ({codigo})")

    codigo, _ = _pegar("/prever", [{**exemplo, "tenure": 900}])
    conferir("recusa tenure acima do teto", codigo == 422, f" ({codigo})")

    print()
    if falhas:
        print(f"FALHOU — {len(falhas)} verificação(ões): {', '.join(falhas)}")
        return 1
    print("A API está servindo o modelo promovido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
