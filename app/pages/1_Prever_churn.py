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
import sys
from pathlib import Path

# O pacote vem de src/, não de uma cópia instalada — ver a nota no app/Inicio.py.
# Uma página pode ser a primeira coisa que a plataforma carrega (basta abrir a
# URL dela direto), então o caminho tem de ser montado aqui também.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from churn.contrato import campos_de_entrada, cliente_exemplo, total_esperado  # noqa: E402
from churn.schema import TOLERANCIA_TOTAL  # noqa: E402

#: A cor dos gráficos. Escolhida por contraste medido, não por gosto: 4,75:1
#: sobre o fundo escuro e 3,98:1 sobre o claro, então a mesma cor serve nos dois
#: temas. É a mesma do `primaryColor` em .streamlit/config.toml.
ACENTO = "#148f82"

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


def pct(valor: float, casas: int = 1) -> str:
    """Percentual no padrão brasileiro — vírgula decimal, não ponto.

    O resto da tela está em português; `84.6%` no meio dela é um descuido que
    aparece justamente no número que todo mundo olha.
    """
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


def _chave(nome: str) -> str:
    """A chave do widget no ``session_state``.

    Sem chave, o Streamlit guarda o valor pela posição e ignora um `value=` novo
    no rerun — trocar o perfil não mexeria em nada na tela. Com ela, escrever em
    ``st.session_state[chave]`` é o que de fato move o controle.
    """
    return f"campo_{nome}"


def _widget(campo, valor_atual):
    """Um campo do contrato vira o controle que corresponde ao seu domínio.

    A descrição aparece **sob** o campo, não só no ícone de ajuda: num
    formulário de 19 campos em vocabulário de telecom, uma explicação que exige
    passar o mouse é uma explicação que ninguém lê — e menos ainda quem está
    assistindo a uma apresentação projetada.
    """
    rotulo = ROTULOS.get(campo.nome, campo.nome)
    chave = _chave(campo.nome)

    if campo.e_categoria:
        opcoes = list(campo.permitidos)
        indice = opcoes.index(valor_atual) if valor_atual in opcoes else 0
        if campo.nome == "SeniorCitizen":
            valor = st.selectbox(
                rotulo,
                opcoes,
                index=indice,
                key=chave,
                format_func=lambda v: "Sim" if v else "Não",
            )
        else:
            valor = st.selectbox(rotulo, opcoes, index=indice, key=chave)
    elif campo.inteiro:
        valor = st.slider(
            rotulo, int(campo.minimo), int(campo.maximo), int(valor_atual), key=chave
        )
    else:
        valor = st.number_input(
            rotulo,
            min_value=float(campo.minimo),
            max_value=float(campo.maximo),
            value=float(valor_atual),
            step=1.0,
            key=chave,
        )

    if campo.descricao:
        st.caption(campo.descricao)
    return valor


#: Clientes prontos, para a demonstração não depender de acertar 19 campos ao
#: vivo. Os valores saem dos padrões de churn medidos na v1 — não são clientes
#: reais do dataset, são as combinações que o dado mostra como extremas.
PERFIS: dict[str, dict] = {
    "Risco alto": {
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 2,
        "MonthlyCharges": 95.0,
    },
    "Risco médio": {
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "tenure": 24,
        "MonthlyCharges": 60.0,
    },
    "Risco baixo": {
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "tenure": 64,
        "MonthlyCharges": 45.0,
    },
}

#: O que explica o risco de cada perfil, para a tela dizer por que mudou. Sai
#: das taxas medidas na v1, não de intuição.
PORQUE: dict[str, str] = {
    "Risco alto": "contrato mensal (50% cancelam), fibra (45%), cheque "
    "eletrônico (40%), sem suporte (40%) e só 2 meses de casa",
    "Risco médio": "contrato anual (14%), DSL (27%), sem suporte técnico e "
    "2 anos de casa",
    "Risco baixo": "contrato bienal (7%), débito automático, todos os add-ons "
    "contratados e 64 meses de casa",
}


campos = {campo.nome: campo for campo in campos_de_entrada()}

# --------------------------------------------------------------------------
# perfis prontos
# --------------------------------------------------------------------------

# Os widgets leem daqui, então trocar o perfil troca o formulário. Guardar no
# session_state e não numa variável é o que faz o valor sobreviver ao rerun que
# o próprio clique dispara.
def aplicar(ajustes: dict, nome: str = "") -> None:
    """Move os controles para um cliente inteiro, e recarrega a página."""
    st.session_state.perfil = nome
    novo = {**cliente_exemplo(), **ajustes}
    # o total acompanha os outros dois, ou o cliente nasce violando o contrato
    novo["TotalCharges"] = total_esperado(novo["tenure"], novo["MonthlyCharges"])
    for nome, valor in novo.items():
        st.session_state[_chave(nome)] = valor
    st.rerun()


@st.cache_data(show_spinner=False)
def _curva_por_tenure(cliente: dict) -> pd.DataFrame | None:
    """A probabilidade do mesmo cliente ao longo dos 72 meses possíveis.

    Um lote só, não 37 requisições: a API aceita vários clientes por chamada, e
    é para isso que ela foi feita. `cache_data` porque a curva é função pura do
    cliente — mexer no corte não precisa recalcular nada.

    `TotalCharges` acompanha o tenure, senão a curva passearia por clientes que
    o contrato do dataset reprovaria.

    Funciona nos dois modos, como o resto da página: tenta a API e, sem ela,
    usa o pipeline local. Uma curva vazia no modo demonstração seria uma
    ausência sem explicação.
    """
    meses = list(range(0, 73, 2))
    lote = []
    for m in meses:
        linha = dict(cliente)
        linha["tenure"] = m
        linha["TotalCharges"] = total_esperado(m, linha["MonthlyCharges"])
        lote.append(linha)

    try:
        resposta = requests.post(f"{API}/prever", json=lote, timeout=60)
        if resposta.status_code == 200:
            valores = [p["probabilidade"] for p in resposta.json()]
            return pd.DataFrame({"tenure": meses, "probabilidade": valores})
    except requests.RequestException:
        pass

    try:
        pipeline, _ = _pipeline_local()
        saida = pipeline.predict(None, pd.DataFrame(lote))
        return pd.DataFrame(
            {"tenure": meses, "probabilidade": saida["probabilidade"].tolist()}
        )
    except Exception:  # noqa: BLE001 — sem curva a página segue sem ela
        return None


st.caption("Comece de um cliente pronto, ou monte o seu campo a campo:")
botoes = st.columns(len(PERFIS) + 1)
for coluna, (nome_perfil, ajustes) in zip(botoes, PERFIS.items()):
    with coluna:
        ativo = st.session_state.get("perfil") == nome_perfil
        if st.button(
            nome_perfil,
            use_container_width=True,
            type="primary" if ativo else "secondary",
        ):
            aplicar(ajustes, nome_perfil)
with botoes[-1]:
    if st.button("Padrão", use_container_width=True):
        aplicar({})

# Os perfis mexem no cliente inteiro, e boa parte da mudança acontece abaixo da
# dobra. Sem esta linha, quem clica olha o topo da tela, não vê nada se mexer e
# conclui que o botão não funcionou.
if st.session_state.get("perfil"):
    st.success(
        f"**{st.session_state.perfil}** aplicado — "
        f"{PORQUE[st.session_state.perfil]}. "
        "Os campos abaixo mudaram; role para ver."
    )

exemplo = cliente_exemplo()
cliente: dict = {}

st.divider()

# --------------------------------------------------------------------------
# o formulário
# --------------------------------------------------------------------------

for titulo, nomes in SECOES.items():
    st.subheader(titulo)
    # Uma `st.columns` NOVA por linha, em vez de distribuir os campos em
    # rodízio por colunas fixas. Colunas são pilhas verticais: com legendas de
    # alturas diferentes sob cada campo, o rodízio faz os controles perderem o
    # alinhamento entre linhas — um campo sobe e fica órfão no topo da seguinte.
    por_linha = min(4, len(nomes))
    for inicio in range(0, len(nomes), por_linha):
        faixa = nomes[inicio : inicio + por_linha]
        for coluna, nome in zip(st.columns(por_linha), faixa):
            with coluna:
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

    Engole qualquer falha de propósito. Esta é a tentativa *otimista*; quem
    garante a tela é o modo demonstração logo abaixo. Deixar uma exceção subir
    daqui derrubava a página inteira com um traceback vermelho — foi o que
    aconteceu num deploy real, quando a cópia instalada do pacote estava velha
    e não tinha esta função.
    """
    try:
        from churn.demonstracao import subir_api_embutida

        return subir_api_embutida(porta=PORTA_LOCAL)
    except Exception as erro:  # noqa: BLE001 — o modo demonstração assume daqui
        from churn.demonstracao import TentativaDeApi

        return TentativaDeApi(motivo=f"{type(erro).__name__}: {erro}")


saude = _saude_da_api()

# Só tentamos subir a API quando ela deveria estar aqui dentro. Se o endereço
# aponta para outra máquina e ela não responde, subir uma local responderia a
# pergunta errada — e esconderia que o serviço de verdade está fora do ar.
tentativa = None
if saude is None and API in {
    f"http://127.0.0.1:{PORTA_LOCAL}",
    f"http://localhost:{PORTA_LOCAL}",
}:
    tentativa = _api_embutida()
    if tentativa.no_ar:
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
        # Num deploy remoto a tela é o único canal de diagnóstico que existe.
        if tentativa is not None and tentativa.motivo:
            with st.expander("Por que a API não subiu"):
                st.code(tentativa.motivo, language=None)

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
    st.session_state.ultima = {"cliente": dict(cliente), "p": probabilidade}

# --------------------------------------------------------------------------
# o resultado
# --------------------------------------------------------------------------

if "ultima" in st.session_state:
    probabilidade = st.session_state.ultima["p"]

    st.subheader("Resultado")

    # O corte é controle, não constante. A ressalva central do projeto é que o
    # ganho do ajuste fino foi de ponto de operação, não de capacidade
    # preditiva — e esse argumento se demonstra movendo este slider, não
    # explicando. Com o corte em 0,34 o modelo antigo alcança o recall do novo.
    corte = st.slider(
        "Corte de decisão",
        0.05, 0.95, 0.50, 0.01,
        help="Acima deste valor, o cliente é tratado como quem vai cancelar.",
    )
    vai_cancelar = probabilidade >= corte

    marca, medida = st.columns([1, 2])
    with marca:
        if vai_cancelar:
            st.error("### Vai cancelar")
        else:
            st.success("### Deve ficar")
    with medida:
        st.metric("Probabilidade de churn", pct(probabilidade))
        st.progress(min(probabilidade, 1.0))
        distancia = probabilidade - corte
        st.caption(
            f"{pct(abs(distancia))} {'acima' if distancia >= 0 else 'abaixo'} do "
            f"corte de {pct(corte, 0)}. Mova o corte e veja o rótulo virar sem o "
            "modelo mudar — é por isso que a probabilidade é o que vale para "
            "decidir, não o rótulo."
        )

    # --- como o risco muda ao longo do tempo de casa ---
    st.subheader("Se este cliente ficasse mais tempo")
    st.caption(
        "A mesma linha de cliente, variando só os meses de contrato. "
        "O resto permanece como está no formulário."
    )

    curva = _curva_por_tenure(st.session_state.ultima["cliente"])
    if curva is None:
        st.info("A curva precisa do modelo carregado; recarregue a página.")
    else:
        # Grade discreta e poucos rótulos: a curva tem 37 pontos, e uma linha
        # de grade por ponto seria mais tinta na estrutura do que no dado.
        base = alt.Chart(curva).encode(
            x=alt.X(
                "tenure:Q",
                title="meses de contrato",
                scale=alt.Scale(domain=[0, 72], nice=False),
                axis=alt.Axis(values=[0, 12, 24, 36, 48, 60, 72], grid=False),
            ),
            y=alt.Y(
                "probabilidade:Q",
                title="probabilidade de churn",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, gridColor="#2b363c"),
            ),
        )
        linha = base.mark_line(color=ACENTO, strokeWidth=2)
        # alvo de hover maior que a marca, e invisível: quem passa o mouse quer
        # o valor, não mais um ponto desenhado em cima da linha
        pontos = base.mark_point(size=160, opacity=0).encode(
            tooltip=[
                alt.Tooltip("tenure:Q", title="meses"),
                alt.Tooltip("probabilidade:Q", title="churn", format=".1%"),
            ]
        )
        regua = (
            alt.Chart(pd.DataFrame({"y": [corte]}))
            .mark_rule(strokeDash=[4, 4], color="#8b9a97")
            .encode(y="y:Q")
        )
        aqui = (
            alt.Chart(pd.DataFrame({
                "tenure": [st.session_state.ultima["cliente"]["tenure"]],
                "probabilidade": [probabilidade],
            }))
            .mark_point(color=ACENTO, size=120, filled=True)
            .encode(x="tenure:Q", y="probabilidade:Q")
        )
        st.altair_chart(
            (linha + pontos + regua + aqui).properties(height=280),
            use_container_width=True,
        )
        inicio = float(curva["probabilidade"].iloc[0])
        fim = float(curva["probabilidade"].iloc[-1])
        st.caption(
            f"O ponto cheio é o cliente do formulário; a linha tracejada é o "
            f"corte de {pct(corte, 0)}. De 0 a 72 meses o risco vai de "
            f"{pct(inicio, 0)} a {pct(fim, 0)} — "
            # pontos percentuais: a diferença de duas frações, em pontos
            f"**{abs(inicio - fim) * 100:.0f} pontos**. "
            "Se a queda for pequena, é sinal de que outro campo (quase sempre o "
            "tipo de contrato) está dominando a decisão."
        )

    with st.expander("O que foi enviado à API"):
        st.dataframe(
            pd.DataFrame([st.session_state.ultima["cliente"]]).T.rename(
                columns={0: "valor"}
            ),
            use_container_width=True,
        )
