"""Centralized configuration — paths, columns, hyperparameters.

A single source of truth so notebooks, scripts, tests, and the API stay aligned.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

DATASET_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_FILE: Path = RAW_DIR / "telco_churn.csv"
PROCESSED_FILE: Path = PROCESSED_DIR / "telco_churn.parquet"

TARGET: str = "Churn"
ID_COL: str = "customerID"

NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20

MLFLOW_EXPERIMENT: str = "churn-prediction"
MLFLOW_TRACKING_URI: str = f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"
