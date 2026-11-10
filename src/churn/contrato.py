"""O contrato de entrada em forma neutra — para quem não fala pandera.

O :mod:`churn.schema` é a fonte da verdade sobre o que é um cliente válido, mas
ele fala pandera: ``Field(isin=...)``, ``Check``, ``statistics``. Quem precisa
dessa informação para *construir alguma coisa* — o modelo Pydantic da API, o
formulário do Streamlit — não deveria precisar aprender esse vocabulário, nem
reabrir o schema por dentro.

Aqui ele sai traduzido para três perguntas que qualquer consumidor faz:

    é um domínio fechado?   -> ``permitidos``
    tem faixa?              -> ``minimo`` / ``maximo``
    é inteiro ou decimal?   -> ``inteiro``

Dois consumidores já usam isto, e é por isso que o módulo existe: sem ele, a
API e o formulário teriam cada um a sua cópia da lista de categorias, e as duas
envelheceriam em ritmos diferentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from churn.schema import MONTHLY_MAX, TENURE_MAX, ChurnRawSchema

#: Colunas que o cliente NÃO manda: o identificador não é feature, e o alvo é
#: justamente o que se quer descobrir.
FORA_DA_ENTRADA = frozenset({"customerID", "Churn"})

#: ``TotalCharges`` é texto no contrato do CSV cru — os brancos de cliente novo
#: obrigam. Em JSON ou num formulário isso não se justifica: quem chama manda
#: número. O piso vem do contrato do dado limpo, onde a coluna já é float.
TOTAL_CHARGES = "TotalCharges"


@dataclass(frozen=True)
class Campo:
    """Uma coluna de entrada, como um formulário ou um JSON a enxerga."""

    nome: str
    descricao: str
    permitidos: tuple[Any, ...] | None = None
    minimo: float | None = None
    maximo: float | None = None
    inteiro: bool = False

    @property
    def e_categoria(self) -> bool:
        """Domínio fechado — vira ``Literal`` na API e ``selectbox`` no front."""
        return self.permitidos is not None

    @property
    def padrao(self) -> Any:
        """Um valor plausível para abrir o formulário já preenchido.

        Primeira opção para categoria, meio da faixa para número. Não é o
        "cliente médio" — é só um ponto de partida que não precisa ser
        corrigido campo a campo antes da primeira previsão.
        """
        if self.permitidos is not None:
            return self.permitidos[0]
        meio = ((self.minimo or 0) + (self.maximo or 100)) / 2
        return int(meio) if self.inteiro else round(meio, 2)


def _campo(nome: str, coluna) -> Campo:
    """Traduz uma coluna do contrato pandera para um :class:`Campo`."""
    limites: dict[str, float] = {}
    for checagem in coluna.checks:
        estatisticas = getattr(checagem, "statistics", None) or {}
        if checagem.name == "isin":
            return Campo(
                nome=nome,
                descricao=coluna.description or "",
                permitidos=tuple(estatisticas["allowed_values"]),
            )
        if checagem.name == "greater_than_or_equal_to":
            limites["minimo"] = estatisticas["min_value"]
        if checagem.name == "less_than_or_equal_to":
            limites["maximo"] = estatisticas["max_value"]

    return Campo(
        nome=nome,
        descricao=coluna.description or "",
        inteiro="int" in str(coluna.dtype),
        **limites,
    )


def campos_de_entrada() -> tuple[Campo, ...]:
    """Todos os campos que compõem um cliente, na ordem do contrato."""
    contrato = ChurnRawSchema.to_schema()
    campos = [
        _campo(nome, coluna)
        for nome, coluna in contrato.columns.items()
        if nome not in FORA_DA_ENTRADA and nome != TOTAL_CHARGES
    ]
    campos.append(
        Campo(
            nome=TOTAL_CHARGES,
            descricao="Total acumulado; 0 para cliente no primeiro ciclo.",
            minimo=0.0,
            # o teto é o do próprio contrato, multiplicado: o maior total
            # possível é o cliente mais antigo pagando a mensalidade mais cara
            maximo=TENURE_MAX * MONTHLY_MAX,
        )
    )
    return tuple(campos)


def cliente_exemplo() -> dict[str, Any]:
    """Um cliente preenchido com os padrões — o estado inicial do formulário.

    ``TotalCharges`` não usa o meio da sua faixa: ele é *derivado* de
    ``tenure × MonthlyCharges``. O contrato exige essa relação (com 10% de
    tolerância), e um exemplo que já nasce violando a regra que o projeto
    inteiro defende seria um mau começo — além de confundir quem abre o
    formulário e vê um aviso antes de digitar qualquer coisa.
    """
    cliente = {campo.nome: campo.padrao for campo in campos_de_entrada()}
    cliente[TOTAL_CHARGES] = round(
        cliente["tenure"] * cliente["MonthlyCharges"], 2
    )
    return cliente


def total_esperado(tenure: int, mensalidade: float) -> float:
    """O ``TotalCharges`` coerente com os outros dois — a regra do contrato."""
    return round(tenure * mensalidade, 2)
