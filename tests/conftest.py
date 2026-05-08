"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw() -> pd.DataFrame:
    """A tiny raw-format DataFrame mimicking IBM's CSV (TotalCharges as object)."""
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 1, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 24, 0, 60],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "Yes", "No", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "Yes"],
            "TechSupport": ["No", "Yes", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "Two year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check", "Bank transfer (automatic)",
                "Mailed check", "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 99.65, 75.30, 24.40],
            # Row 3 (index 2) has the whitespace quirk we must clean.
            "TotalCharges": ["29.85", "2391.6", " ", "1464.0"],
            "Churn": ["No", "No", "Yes", "No"],
        }
    )


@pytest.fixture
def sample_clean(sample_raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned version of the sample (used by feature/training tests)."""
    from churn.data import clean
    return clean(sample_raw)


@pytest.fixture
def valid_payload() -> dict:
    """A valid prediction payload (high-risk customer profile)."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 75.5,
    }
