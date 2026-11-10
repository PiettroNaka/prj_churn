"""A API de previsão — o modelo promovido, atrás de HTTP.

O Registry dá o endereço estável (``models:/churn@champion``); isto dá a porta.
Quem consome não instala o projeto, não conhece o MLflow e não sabe qual run
venceu: manda um cliente em JSON e recebe a probabilidade de churn.

    uv run --group api uvicorn churn.api:app --reload

A documentação interativa sobe junto, em ``/docs``.

**O contrato de entrada não é escrito aqui.** Ele é *gerado* do
:mod:`churn.schema`, o mesmo ``ChurnRawSchema`` que valida o CSV no treino. Os
valores permitidos de cada categórica, a faixa de ``tenure``, o piso e o teto de
``MonthlyCharges`` — tudo sai do contrato em tempo de import. Uma lista paralela
aqui seria a forma mais discreta de criar *training/serving skew*: o dia em que
o schema ganhasse uma categoria, a API continuaria recusando-a, e ninguém
ligaria os dois fatos.

O efeito colateral é melhor do que parecia: o ``/docs`` sai com os domínios
reais em cada campo, porque o OpenAPI é construído do mesmo contrato.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from churn.config import Settings, settings
from churn.registry import carregar_campeao, versao_do_alias
from churn.schema import ChurnRawSchema

#: Colunas que o cliente NÃO manda: o identificador não é feature, e o alvo é
#: justamente o que se quer descobrir.
FORA_DA_ENTRADA = frozenset({"customerID", "Churn"})


def _tipo_e_campo(coluna) -> tuple[Any, Any]:
    """Traduz uma coluna do contrato pandera para ``(tipo, Field)`` do Pydantic.

    Só três formas interessam, porque são as três que o contrato usa: domínio
    fechado (``isin``) vira ``Literal``, faixa (``ge``/``le``) vira ``Field`` com
    limites, e o resto vira o tipo nu.
    """
    limites: dict[str, Any] = {}
    for checagem in coluna.checks:
        estatisticas = getattr(checagem, "statistics", None) or {}
        if checagem.name == "isin":
            permitidos = list(estatisticas["allowed_values"])
            return Literal[tuple(permitidos)], Field(description=coluna.description)
        if checagem.name == "greater_than_or_equal_to":
            limites["ge"] = estatisticas["min_value"]
        if checagem.name == "less_than_or_equal_to":
            limites["le"] = estatisticas["max_value"]

    tipo = float if "float" in str(coluna.dtype) else int
    return tipo, Field(description=coluna.description, **limites)


def montar_modelo_de_entrada() -> type[BaseModel]:
    """Constrói o ``Cliente`` do Pydantic a partir do ``ChurnRawSchema``."""
    contrato = ChurnRawSchema.to_schema()
    campos = {
        nome: _tipo_e_campo(coluna)
        for nome, coluna in contrato.columns.items()
        if nome not in FORA_DA_ENTRADA
    }
    # TotalCharges é texto no contrato do CSV cru (por causa dos brancos de
    # cliente novo). Em JSON isso não se justifica: quem chama manda número, e o
    # pipeline já sabe coagir. O limite inferior vem do contrato do dado limpo.
    campos["TotalCharges"] = (
        float,
        Field(ge=0, description="Total acumulado; 0 para cliente no primeiro ciclo."),
    )
    return create_model("Cliente", **campos)


Cliente = montar_modelo_de_entrada()


class Previsao(BaseModel):
    """O que a API responde para cada cliente enviado."""

    churn: str = Field(description="Rótulo previsto, no vocabulário do dataset.")
    probabilidade: float = Field(
        ge=0, le=1, description="Probabilidade de o cliente cancelar."
    )


class Modelo(BaseModel):
    """Quem está respondendo — a pergunta que todo consumidor faz na segunda semana."""

    nome: str
    versao: str | None
    alias: str
    uri: str
    carregado: bool
    execucao: str | None = Field(
        default=None, description="Run do MLflow que gerou esta versão."
    )


class _Estado:
    """O modelo vive aqui, carregado uma vez.

    Carregar por requisição custaria segundos e leria o disco à toa; o pipeline
    é imutável depois de promovido. Trocar de campeão é reiniciar o processo —
    o que, num container, é o próprio deploy.
    """

    pipeline: Any = None
    versao: str | None = None
    execucao: str | None = None


estado = _Estado()


def carregar(cfg: Settings = settings) -> None:
    """Carrega o campeão. Falhar aqui é melhor do que falhar na primeira chamada."""
    estado.pipeline = carregar_campeao(cfg)
    if (versao := versao_do_alias(cfg)) is not None:
        # str() porque o MLflow devolve a versão ora como int, ora como texto,
        # conforme o backend. Normalizar na entrada evita que a resposta da API
        # mude de tipo quando o tracking mudar de lugar.
        estado.versao = str(versao.version)
        estado.execucao = versao.run_id


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega na subida; se não houver campeão, a API sobe mesmo assim.

    Subir quebrada e dizer o porquê em ``/saude`` é melhor do que não subir: o
    container entra no ar, o healthcheck acusa, e quem está depurando vê a
    mensagem em vez de um restart loop sem explicação.
    """
    try:
        carregar()
    except Exception as erro:  # noqa: BLE001 — a causa vai para /saude
        estado.pipeline = None
        app.state.falha = str(erro)
    yield


app = FastAPI(
    title="Churn API",
    description=__doc__,
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


@app.get("/saude", tags=["operação"])
def saude() -> dict[str, Any]:
    """Pronto para responder? É o que o healthcheck do compose pergunta."""
    if estado.pipeline is None:
        return {
            "status": "sem modelo",
            "detalhe": getattr(app.state, "falha", "nenhum campeão promovido"),
            "saida": "uv run python -m churn.tuning && uv run python -m churn.registry",
        }
    return {"status": "ok", "versao": estado.versao}


@app.get("/modelo", response_model=Modelo, tags=["operação"])
def modelo() -> Modelo:
    """Qual versão está servindo — a rastreabilidade que o Registry guarda."""
    from churn.registry import uri_do_campeao

    return Modelo(
        nome=settings.mlflow_registered_model,
        versao=estado.versao,
        alias=settings.mlflow_champion_alias,
        uri=uri_do_campeao(),
        carregado=estado.pipeline is not None,
        execucao=estado.execucao,
    )


@app.post("/prever", response_model=list[Previsao], tags=["previsão"])
def prever(clientes: list[Cliente]) -> list[Previsao]:  # type: ignore[valid-type]
    """Um ou muitos clientes, uma previsão para cada — na ordem em que chegaram.

    Lista e não objeto solto porque a chamada de verdade é em lote: a área de
    retenção quer a carteira inteira, não um cliente por requisição.
    """
    if estado.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "falha", "nenhum campeão promovido"),
        )
    if not clientes:
        raise HTTPException(status_code=422, detail="Nenhum cliente na requisição.")

    entrada = pd.DataFrame([c.model_dump() for c in clientes])
    saida = estado.pipeline.predict(entrada)
    return [
        Previsao(churn=linha["churn"], probabilidade=float(linha["probabilidade"]))
        for _, linha in saida.iterrows()
    ]
