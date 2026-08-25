import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Mini-amostra com o schema do Telco, incluindo um TotalCharges em branco."""
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4"],
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
