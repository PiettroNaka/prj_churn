"""Front v2 — os parâmetros de um cliente, e o churn calculado.

O formulário não é escrito campo a campo: é **gerado** do
:mod:`churn.contrato`, o mesmo contrato que valida o CSV no treino e que gera o
modelo de entrada da API. Categoria vira ``selectbox`` com as opções reais;
faixa numérica vira ``slider`` com o piso e o teto do contrato. Não há como
montar aqui um cliente que a API recuse por domínio — os dois leem a mesma
fonte.

O cálculo não acontece nesta página. Ela chama a API (:mod:`churn.api`), que
carrega o campeão do Registry. É o que separa o front do modelo: trocar de
campeão é mover um alias, sem tocar numa linha daqui.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from churn.contrato import campos_de_entrada, cliente_exemplo, total_esperado
from churn.schema import TOLERANCIA_TOTAL

#: A porta em que a API sobe quando ela vive no mesmo container que o front.
PORTA_LOCAL = 8000

#: Onde a API responde. No compose é ``http://api:8000``; na sua máquina, local.
API = os.environ.get("CHURN_API_URL", f"http://127.0.0.1:{PORTA_LOCAL}").rstrip("/")

#: Como agrupar os 19 campos. O contrato não tem noção de seção — ele descreve
#: colunas, não telas —, então a ordem de leitura é decisão do front.
SECOES: dict[str, tuple[str, ...]] = {
    "Perfil": ("gender", "SeniorCitizen", "Partner", "Dependents"),
    "Telefonia": ("PhoneService", "MultipleLines"),
    "Internet": (
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ),
    "Contrato e cobrança": ("Contract", "PaperlessBilling", "PaymentMethod"),
}

ROTULOS = {
    "gender": "Gênero",
    "SeniorCitizen": "Idoso (65+)",
    "Partner": "Tem cônjuge",
    "Dependents": "Tem dependentes",
    "PhoneService": "Serviço de telefone",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Tipo de internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de equipamento",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Fatura digital",
    "PaymentMethod": "Forma de pagamento",
    "tenure": "Meses de contrato (tenure)",
    "MonthlyCharges": "Mensalidade (R$)",
    "TotalCharges": "Total acumulado (R$)",
}

st.set_page_config(page_title="Prever churn", page_icon="🔮", layout="wide")

st.title("🔮 Prever churn")
st.caption(
    "Os parâmetros de um cliente, e a probabilidade de ele cancelar. O "
    "formulário é gerado do contrato do dataset; o cálculo vem da API."
)


def _widget(campo, valor_atual):
    """Um campo do contrato vira o controle que corresponde ao seu domínio."""
    rotulo = ROTULOS.get(campo.nome, campo.nome)

    if campo.e_categoria:
        opcoes = list(campo.permitidos)
        indice = opcoes.index(valor_atual) if valor_atual in opcoes else 0
        if campo.nome == "SeniorCitizen":
            return st.selectbox(
                rotulo, opcoes, index=indice, format_func=lambda v: "Sim" if v else "Não"
            )
        return st.selectbox(rotulo, opcoes, index=indice, help=campo.descricao or None)

    if campo.inteiro:
        return st.slider(
            rotulo,
            int(campo.minimo),
            int(campo.maximo),
            int(valor_atual),
            help=campo.descricao or None,
        )
    return st.number_input(
        rotulo,
        min_value=float(campo.minimo),
        max_value=float(campo.maximo),
        value=float(valor_atual),
        step=1.0,
        help=campo.descricao or None,
    )


campos = {campo.nome: campo for campo in campos_de_entrada()}
exemplo = cliente_exemplo()
cliente: dict = {}

# --------------------------------------------------------------------------
# o formulário
# --------------------------------------------------------------------------

for titulo, nomes in SECOES.items():
    st.subheader(titulo)
    colunas = st.columns(min(4, len(nomes)))
    for posicao, nome in enumerate(nomes):
        with colunas[posicao % len(colunas)]:
            cliente[nome] = _widget(campos[nome], exemplo[nome])

st.subheader("Valores")
tempo, mensal, total = st.columns(3)

with tempo:
    cliente["tenure"] = _widget(campos["tenure"], exemplo["tenure"])
with mensal:
    cliente["MonthlyCharges"] = _widget(
        campos["MonthlyCharges"], exemplo["MonthlyCharges"]
    )

esperado = total_esperado(cliente["tenure"], cliente["MonthlyCharges"])

with total:
    # O contrato exige TotalCharges ≈ tenure × MonthlyCharges (10% de
    # tolerância). Deixar o campo solto convidaria a montar um cliente que o
    # schema reprovaria no treino — e a previsão sairia assim mesmo, porque a
    # API valida coluna a coluna, não a relação entre elas.
    derivar = st.checkbox("Calcular pelo tenure × mensalidade", value=True)
    if derivar:
        cliente["TotalCharges"] = esperado
        st.number_input(
            ROTULOS["TotalCharges"], value=float(esperado), disabled=True, step=1.0
        )
    else:
        cliente["TotalCharges"] = _widget(campos["TotalCharges"], esperado)

if esperado > 0:
    desvio = abs(cliente["TotalCharges"] - esperado) / esperado
    if desvio > TOLERANCIA_TOTAL:
        st.warning(
            f"O total acumulado está {desvio:.0%} longe de "
            f"`tenure × mensalidade` (R$ {esperado:,.2f}). O contrato do dataset "
            f"tolera {TOLERANCIA_TOTAL:.0%} — um cliente assim seria reprovado "
            "na validação do treino. A previsão sai, mas é extrapolação."
        )

st.divider()

# --------------------------------------------------------------------------
# o cálculo
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Treinando o modelo local (primeira vez, ~20s)…")
def _pipeline_local():
    """O modo demonstração — ver :mod:`churn.demonstracao`. Treina uma vez só."""
    from churn.demonstracao import pipeline_local

    return pipeline_local()


def _saude_da_api() -> dict | None:
    try:
        return requests.get(f"{API}/saude", timeout=3).json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_resource(
    show_spinner="Subindo a API e promovendo o modelo (primeira vez, ~40s)…"
)
def _api_embutida():
    """Sobe a API ao lado do front, para plataformas que rodam um processo só.

    O Streamlit Community Cloud roda **um** processo. A alternativa seria o
    front carregar o modelo, apagando a fronteira que o projeto inteiro defende.
    Um processo a mais no mesmo container é bem menos concessão: continua
    havendo API, Registry e alias, e o front continua só perguntando.

    `cache_resource` porque isto tem de acontecer uma vez por processo, não a
    cada clique — e o processo da API precisa sobreviver ao fim do script.
    """
    from churn.demonstracao import subir_api_embutida

    return subir_api_embutida(porta=PORTA_LOCAL)


saude = _saude_da_api()

# Só tentamos subir a API quando ela deveria estar aqui dentro. Se o endereço
# aponta para outra máquina e ela não responde, subir uma local responderia a
# pergunta errada — e esconderia que o serviço de verdade está fora do ar.
if saude is None and API in {f"http://127.0.0.1:{PORTA_LOCAL}", f"http://localhost:{PORTA_LOCAL}"}:
    if _api_embutida() is not None:
        saude = _saude_da_api()

com_api = bool(saude and saude.get("status") == "ok")

with st.sidebar:
    st.subheader("De onde vem a previsão")
    if com_api:
        st.success(f"API · modelo v{saude['versao']}")
        st.code(API, language=None)
        st.caption(
            "O campeão promovido no Registry, servido por HTTP. É o desenho do "
            "projeto: trocar de modelo é mover um alias."
        )
    else:
        st.info("Modo demonstração")
        st.caption(
            "A API não subiu, então o modelo é treinado aqui mesmo — mesmo "
            "dado, mesma semente e mesmos hiperparâmetros, mas **sem** passar "
            "pelo Registry. Para o caminho de verdade: `subir-front.bat` no "
            "Windows, ou `docker compose up app`."
        )
        if saude is not None:
            st.caption(f"A API respondeu, mas sem modelo: {saude.get('detalhe', '')}")

if st.button("Calcular churn", type="primary", use_container_width=True):
    if com_api:
        try:
            resposta = requests.post(f"{API}/prever", json=[cliente], timeout=30)
        except requests.RequestException as erro:
            st.error(f"Não consegui falar com a API em `{API}`.")
            st.caption(str(erro))
            st.stop()

        if resposta.status_code != 200:
            st.error(f"A API recusou a requisição ({resposta.status_code}).")
            st.json(resposta.json())
            st.stop()

        (previsao,) = resposta.json()
    else:
        try:
            pipeline, metricas = _pipeline_local()
        except Exception as erro:  # noqa: BLE001 — a causa vai para a tela
            st.error("Não consegui preparar o modelo local.")
            st.caption(str(erro))
            st.stop()

        saida = pipeline.predict(None, pd.DataFrame([cliente]))
        previsao = saida.iloc[0].to_dict()
        st.caption(
            f"Modelo treinado nesta sessão · roc_auc {metricas['roc_auc']:.4f} · "
            f"recall {metricas['recall']:.4f}"
        )

    probabilidade = float(previsao["probabilidade"])

    rotulo, risco = st.columns([1, 2])
    with rotulo:
        if previsao["churn"] == "Yes":
            st.error("### Vai cancelar")
        else:
            st.success("### Deve ficar")
        st.metric("Probabilidade de churn", f"{probabilidade:.1%}")
    with risco:
        st.progress(min(probabilidade, 1.0))
        st.caption(
            "A probabilidade é o que vale para decidir, não o rótulo: o corte em "
            "50% é convenção. Para uma campanha de retenção com orçamento "
            "limitado, ordene a carteira por esta coluna e vá descendo."
        )

    with st.expander("O que foi enviado"):
        st.dataframe(
            pd.DataFrame([cliente]).T.rename(columns={0: "valor"}),
            use_container_width=True,
        )
