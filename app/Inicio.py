"""Front do projeto de churn, v1 — título, código e dados.

A primeira versão não prevê nada: mostra o que existe. É de propósito. Um front
que calcula antes de deixar ver o dado e o código vira uma caixa-preta bonita,
e a caixa-preta é exatamente o que este curso passa doze encontros desmontando.

    uv run --group app streamlit run app/Inicio.py

A previsão é a v2, na página ao lado.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from churn.config import settings

RAIZ = Path(__file__).resolve().parent.parent
CODIGO = RAIZ / "src" / "churn"

st.set_page_config(
    page_title="Churn — projeto fio-condutor",
    page_icon="📉",
    layout="wide",
)


@st.cache_data
def carregar_dados(caminho: str) -> pd.DataFrame:
    """Lê o CSV uma vez por sessão — 7 mil linhas não precisam recarregar a cada clique."""
    return pd.read_csv(caminho)


# --------------------------------------------------------------------------
# título
# --------------------------------------------------------------------------

st.title("📉 Previsão de churn")
st.caption(
    "Projeto fio-condutor do curso de MLOps — do script monolítico ao modelo "
    "servido, versionado e rastreável."
)

st.markdown(
    """
Esta é a **v1** do front: ela mostra o projeto. O código que treina, o contrato
que valida e o dataset que alimenta os dois.

Para calcular o churn de um cliente, vá em **Prever churn**, na barra lateral —
é a v2.
"""
)

st.divider()

# --------------------------------------------------------------------------
# código
# --------------------------------------------------------------------------

st.header("Código")

modulos = sorted(p for p in CODIGO.glob("*.py") if p.name != "__init__.py")

if not modulos:
    st.warning(f"Nenhum módulo encontrado em `{CODIGO}`.")
else:
    descricoes = {
        "config.py": "Settings (Pydantic) — todo caminho e hiperparâmetro num lugar só",
        "schema.py": "o contrato dos dados, em pandera — a fonte da verdade",
        "contrato.py": "o contrato traduzido para quem não fala pandera",
        "validation.py": "executa o contrato e traduz a falha",
        "quarentena.py": "separa o lote reprovado em vez de treinar com ele",
        "data.py": "carga + validação + limpeza",
        "features.py": "transformações puras e encoder sem vazamento",
        "model.py": "o treino, instrumentado com MLflow",
        "experiments.py": "o grid rastreado — 9 combinações",
        "tuning.py": "a busca aleatória em 6 eixos — o ajuste fino",
        "evaluate.py": "as métricas",
        "lineage.py": "de qual dado e de qual código este modelo saiu",
        "serving.py": "ChurnPipeline — o modelo que aceita um cliente cru",
        "registry.py": "promove o vencedor a @champion",
        "api.py": "o campeão atrás de HTTP",
    }

    escolhido = st.selectbox(
        "Módulo",
        modulos,
        format_func=lambda p: f"{p.name}  —  {descricoes.get(p.name, '')}",
    )
    fonte = escolhido.read_text(encoding="utf-8")
    st.caption(f"`src/churn/{escolhido.name}` · {len(fonte.splitlines())} linhas")
    st.code(fonte, language="python", line_numbers=True)

st.divider()

# --------------------------------------------------------------------------
# dados
# --------------------------------------------------------------------------

st.header("Dados")

@st.cache_resource(show_spinner="Materializando o dataset…")
def materializar() -> str:
    """O CSV não vive no Git — quem o versiona é o DVC (Encontro 5).

    Num deploy de vitrine não há remote do DVC nem histórico para recuperar o
    blob, então o módulo de demonstração busca os bytes por HTTP e **confere o
    md5 contra o ponteiro** antes de aceitá-los. Transporte diferente, mesma
    verificação.
    """
    from churn.demonstracao import garantir_dataset

    # caminho absoluto: o Streamlit pode ser lançado de qualquer pasta, e um
    # caminho relativo apontaria para fora do projeto
    daqui = settings.model_copy(update={"data_path": RAIZ / settings.data_path})
    return str(garantir_dataset(daqui))


try:
    caminho = materializar()
except Exception as erro:  # noqa: BLE001 — a causa vai para a tela
    st.error(f"Não consegui materializar `{settings.data_path}`.")
    st.caption(str(erro))
    st.code("uv run python scripts/preparar.py --so-dado", language="bash")
    st.stop()

dados = carregar_dados(caminho)
churners = (dados[settings.target] == settings.positive_label).mean()

esquerda, meio, direita = st.columns(3)
esquerda.metric("Clientes", f"{len(dados):,}".replace(",", "."))
meio.metric("Colunas", len(dados.columns))
direita.metric("Taxa de churn", f"{churners:.1%}")

st.caption(
    f"Base desbalanceada: {churners:.0%} de churn. É por isso que a acurácia não "
    "decide nada aqui — quem chuta que ninguém cancela já acerta "
    f"{1 - churners:.0%}."
)

aba_tabela, aba_colunas, aba_resumo = st.tabs(["Tabela", "Colunas", "Resumo"])

with aba_tabela:
    st.dataframe(dados, use_container_width=True, height=420)

with aba_colunas:
    st.dataframe(
        pd.DataFrame(
            {
                "coluna": dados.columns,
                "tipo": [str(t) for t in dados.dtypes],
                "nulos": dados.isna().sum().values,
                "valores distintos": dados.nunique().values,
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.caption(
        "`TotalCharges` aparece como texto: 11 linhas vêm em branco, todas de "
        "clientes com `tenure = 0`. Não é corrupção — é cliente que ainda não "
        "fechou o primeiro ciclo. O contrato exige essa relação em vez de "
        "descartar as linhas."
    )

with aba_resumo:
    st.dataframe(dados.describe(), use_container_width=True)

st.caption(
    f"Dataset versionado pelo DVC · ponteiro `{settings.data_path}.dvc` · "
    "os bytes não estão no Git."
)
