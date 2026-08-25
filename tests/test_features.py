import pandas as pd

from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)


def test_add_derived_features_calcula_gasto_por_mes(raw_frame):
    frame = raw_frame.assign(TotalCharges=[666.0, 0.0, 680.0, 226.5])
    out = add_derived_features(frame)
    esperado = 666.0 / (12 + 1)
    assert out["gasto_por_mes"].iloc[0] == esperado


def test_split_features_target_mapeia_alvo(raw_frame):
    features, labels = split_features_target(raw_frame, "Churn", "Yes")
    assert "Churn" not in features.columns
    assert labels.tolist() == [0, 1, 0, 1]


def test_encoder_nao_vaza_categoria_nova():
    treino = pd.DataFrame({"cor": ["azul", "verde"], "n": [1, 2]})
    teste = pd.DataFrame({"cor": ["amarelo"], "n": [3]})  # categoria inédita
    encoder = CategoricalEncoder().fit(treino)
    transformado = encoder.transform(teste)
    assert transformado["cor"].iloc[0] == -1  # desconhecida vira -1, não quebra
