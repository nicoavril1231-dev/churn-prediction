"""Tests for churn.data — cleaning rules must be deterministic and explicit."""

from __future__ import annotations

import pandas as pd

from churn.config import TARGET
from churn.data import clean


def test_clean_drops_whitespace_total_charges(sample_raw: pd.DataFrame) -> None:
    """Row with TotalCharges == ' ' (the IBM quirk) must be dropped."""
    out = clean(sample_raw)
    assert len(out) == 3, "Should drop 1 row out of 4"
    assert out["TotalCharges"].dtype == float


def test_clean_casts_target_to_int(sample_raw: pd.DataFrame) -> None:
    """`Churn` must become {0, 1} integers."""
    out = clean(sample_raw)
    assert out[TARGET].dtype.kind == "i"
    assert set(out[TARGET].unique()).issubset({0, 1})


def test_clean_is_idempotent(sample_raw: pd.DataFrame) -> None:
    """Running clean twice must give the same result as once
    (no double-coercion or cumulative drops)."""
    once = clean(sample_raw)
    twice = clean(sample_raw.assign(
        TotalCharges=sample_raw["TotalCharges"]
    ))
    pd.testing.assert_frame_equal(once.reset_index(drop=True),
                                  twice.reset_index(drop=True))


def test_clean_does_not_mutate_input(sample_raw: pd.DataFrame) -> None:
    """`clean` should return a new DataFrame, not mutate its argument."""
    before = sample_raw.copy()
    clean(sample_raw)
    pd.testing.assert_frame_equal(sample_raw, before)


def test_clean_preserves_row_count_when_no_quirks() -> None:
    """If TotalCharges is fully numeric, no row should be dropped."""
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [50.0, 99.0],
        "TotalCharges": ["600.0", "2376.0"],
        "Churn": ["No", "Yes"],
    })
    out = clean(df)
    assert len(out) == 2
