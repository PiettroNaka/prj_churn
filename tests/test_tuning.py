"""A busca tem de ser reprodutível e dizível — senão o vencedor não volta.

O que se testa aqui não é se a floresta acerta: é se o resultado da busca
sobrevive à viagem até o próximo treino. Um vencedor que a config não consegue
expressar, ou que volta como outra coisa, é um vencedor perdido.

A busca em si roda sobre um dublê de dados: o que importa é a fiação (quantos
candidatos, qual métrica, qual semente), e isso não precisa das 7.000 linhas.
"""
import numpy as np
import pandas as pd
import pytest

from churn import tuning
from churn.config import Settings
from churn.tuning import (
    ESPACO,
    HIPERPARAMETROS,
    como_variaveis,
    config_vencedora,
    espaco_total,
)


@pytest.fixture
def dados_minimos(monkeypatch):
    """Duas features, duas classes, linhas suficientes para as dobras da CV."""
    gerador = np.random.default_rng(0)
    x = pd.DataFrame(
        {"a": gerador.normal(size=80), "b": gerador.normal(size=80)}
    )
    y = pd.Series((x["a"] + gerador.normal(scale=0.3, size=80) > 0).astype(int))

    monkeypatch.setattr(
        tuning, "dados_codificados", lambda cfg: (x[:60], x[60:], y[:60], y[60:])
    )
    return x, y


def test_espaco_total_e_o_produto_dos_eixos():
    esperado = 1
    for valores in ESPACO.values():
        esperado *= len(valores)

    assert espaco_total() == esperado


def test_a_busca_cobre_uma_fracao_pequena_do_espaco():
    """24 de 3.150. Se um dia o espaço encolher a ponto de caber inteiro, o
    sorteio deixa de fazer sentido e alguém tem de reparar."""
    assert espaco_total() > Settings().tuning_n_iter * 10


def test_todo_hiperparametro_buscado_existe_na_config():
    """O elo que quebra calado: buscar num eixo que a Settings não sabe dizer.

    A busca acharia o melhor `min_samples_leaf`, imprimiria a variável, e o
    `model_copy` a descartaria — treinando outra coisa, sem erro nenhum.
    """
    campos = set(Settings().model_dump())

    assert set(HIPERPARAMETROS) <= campos


def test_config_vencedora_aplica_os_hiperparametros():
    base = Settings()
    melhores = {"n_estimators": 500, "max_depth": 8, "class_weight": "balanced"}

    vencedora = config_vencedora(melhores, base)

    assert vencedora.n_estimators == 500
    assert vencedora.max_depth == 8
    assert vencedora.class_weight == "balanced"


def test_config_vencedora_preserva_o_resto():
    """Semente, caminho do dado e endereço do MLflow não são achados da busca.

    Se mudassem, o treino do vencedor seria outro treino, e a métrica que a
    busca prometeu não se repetiria.
    """
    base = Settings()

    vencedora = config_vencedora({"n_estimators": 500}, base)

    assert vencedora.random_state == base.random_state
    assert vencedora.data_path == base.data_path
    assert vencedora.mlflow_tracking_uri == base.mlflow_tracking_uri
    assert vencedora.test_size == base.test_size


@pytest.mark.parametrize("campo", ["max_depth", "max_features", "class_weight"])
def test_nulo_volta_como_nulo_pela_variavel_de_ambiente(campo, monkeypatch):
    """A ida e a volta do None — onde estavam três bugs de uma vez.

    `CHURN_MAX_DEPTH=` levanta ValidationError; `CHURN_MAX_FEATURES=` passa e
    vira a string vazia, que o sklearn rejeita só na hora do fit. Este teste
    amarra `como_variaveis` ao `env_parse_none_str` da config: mudar um sem o
    outro quebra aqui, não em produção.
    """
    (linha,) = como_variaveis({campo: None})
    nome, _, valor = linha.partition("=")
    monkeypatch.setenv(nome, valor)

    assert getattr(Settings(), campo) is None


def test_variaveis_saem_com_o_prefixo_do_projeto():
    linhas = como_variaveis({"n_estimators": 500, "max_depth": 8})

    assert linhas == ["CHURN_N_ESTIMATORS=500", "CHURN_MAX_DEPTH=8"]


def test_busca_respeita_o_numero_de_candidatos(dados_minimos):
    cfg = Settings(tuning_n_iter=3, tuning_cv=2)

    busca = tuning.buscar(cfg)

    assert len(busca.cv_results_["params"]) == 3
    assert set(busca.best_params_) == set(HIPERPARAMETROS)


def test_busca_e_reprodutivel(dados_minimos):
    """Mesma semente, mesmo vencedor. Sem isto, 'o melhor modelo' é o do dia."""
    cfg = Settings(tuning_n_iter=3, tuning_cv=2)

    assert tuning.buscar(cfg).best_params_ == tuning.buscar(cfg).best_params_


def test_o_vencedor_da_busca_e_treinavel_pela_config(dados_minimos):
    """O ciclo inteiro em miniatura: buscar -> dizer -> construir.

    É a junta onde a busca encontra o treino. Se o sklearn recusar a config que
    saiu da busca, o vencedor morre entre os dois módulos.
    """
    from churn.model import build_model

    melhores = tuning.buscar(Settings(tuning_n_iter=2, tuning_cv=2)).best_params_

    modelo = build_model(config_vencedora(melhores, Settings()))

    for nome, valor in melhores.items():
        assert getattr(modelo, nome) == valor
