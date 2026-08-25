import pandas as pd
import pytest

from churn.data import coerce_total_charges, drop_identifier, validate


def test_coerce_total_charges_converte_branco_em_zero(raw_frame):
    out = coerce_total_charges(raw_frame)
    assert out["TotalCharges"].dtype.kind == "f"
    # a linha com tenure 0 tinha TotalCharges em branco -> 0.0
    assert out.loc[raw_frame["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_coerce_e_pura(raw_frame):
    _ = coerce_total_charges(raw_frame)
    assert raw_frame["TotalCharges"].dtype == object  # original intacto


def test_validate_falha_sem_coluna(raw_frame):
    with pytest.raises(ValueError):
        validate(raw_frame.drop(columns=["Contract"]), target="Churn")


def test_drop_identifier(raw_frame):
    out = drop_identifier(raw_frame, "customerID")
    assert "customerID" not in out.columns
