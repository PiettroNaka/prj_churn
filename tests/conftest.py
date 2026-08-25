import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Mini-amostra com o schema do Telco, incluindo um TotalCharges em branco."""
    return pd.DataFrame(
        {
            "customerID": ["7590-VHVE", "5575-GNVD", "3668-QPYB", "7795-CFOC"],
            "gender": ["Male", "Female", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [12, 0, 34, 5],
            "PhoneService": ["Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"],
            "MonthlyCharges": [55.5, 89.1, 20.0, 45.3],
            "TotalCharges": ["666.0", " ", "680.0", "226.5"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


@pytest.fixture
def clean_frame(raw_frame) -> pd.DataFrame:
    """A mesma amostra, já passada por drop_identifier + coerce_total_charges.

    É a linha de base dos testes do Encontro 4: parte de um frame que o schema
    aprova, e cada teste suja UMA coisa para ver a validação pegar.
    """
    from churn.data import coerce_total_charges, drop_identifier

    return coerce_total_charges(drop_identifier(raw_frame, "customerID"))
