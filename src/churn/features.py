"""Feature engineering and preprocessing pipeline.

Returns a sklearn ColumnTransformer ready to plug into a Pipeline.
Strategy:
  - Numeric: median imputation + StandardScaler (XGBoost doesn't strictly need
    scaling, but it makes Logistic Regression baselines comparable).
  - Categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore")
    so unseen categories at inference don't crash.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return the full preprocessing ColumnTransformer."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that help discriminate churners.

    Inspired by EDA findings:
      - tenure_bucket: month-to-month customers churn most in months 0-12
      - charges_ratio: TotalCharges / (tenure + 1) ≈ effective monthly billing
      - has_streaming: bundled streaming reduces churn (loyalty signal)
    """
    out = df.copy()
    out["charges_ratio"] = out["TotalCharges"] / (out["tenure"] + 1)
    out["has_streaming"] = (
        (out["StreamingTV"] == "Yes") | (out["StreamingMovies"] == "Yes")
    ).astype(int)
    return out
