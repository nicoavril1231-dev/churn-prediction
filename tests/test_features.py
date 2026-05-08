"""Tests for churn.features — preprocessing must be deterministic and robust."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import ALL_FEATURES
from churn.features import add_engineered_features, build_preprocessor


def test_preprocessor_is_callable() -> None:
    """ColumnTransformer must be constructible without args."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_preprocessor_fits_on_clean_data(sample_clean: pd.DataFrame) -> None:
    """Fitting on a 3-row sample must succeed and produce a 2-D array."""
    pre = build_preprocessor()
    transformed = pre.fit_transform(sample_clean[ALL_FEATURES])
    assert transformed.ndim == 2
    assert transformed.shape[0] == len(sample_clean)


def test_preprocessor_handles_unseen_categories(sample_clean: pd.DataFrame) -> None:
    """OneHotEncoder must be configured with handle_unknown='ignore'."""
    pre = build_preprocessor()
    pre.fit(sample_clean[ALL_FEATURES])

    # Inject a never-seen value at inference
    new_row = sample_clean.iloc[[0]].copy()
    new_row.loc[:, "PaymentMethod"] = "Cryptocurrency"
    transformed = pre.transform(new_row[ALL_FEATURES])
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any(), "Unknown category should encode to all-zeros, not NaN"


def test_engineered_features_added(sample_clean: pd.DataFrame) -> None:
    """add_engineered_features should add charges_ratio and has_streaming."""
    out = add_engineered_features(sample_clean)
    assert "charges_ratio" in out.columns
    assert "has_streaming" in out.columns
    assert (out["has_streaming"].isin([0, 1])).all()


def test_charges_ratio_handles_tenure_zero(sample_clean: pd.DataFrame) -> None:
    """tenure can be 0; the (tenure + 1) denominator must prevent ZeroDivisionError."""
    df = sample_clean.copy()
    df.loc[0, "tenure"] = 0
    out = add_engineered_features(df)
    assert np.isfinite(out["charges_ratio"]).all()
