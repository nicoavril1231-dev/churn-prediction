"""Inference helpers — used by the CLI, tests, and the FastAPI service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn.config import ALL_FEATURES, MODELS_DIR


@lru_cache(maxsize=1)
def load_artifact(path: Path = MODELS_DIR / "best_model.joblib") -> dict[str, Any]:
    """Load the trained pipeline once per process."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m churn.train` first."
        )
    return joblib.load(path)


def predict_one(payload: dict[str, Any]) -> dict[str, float | int | str]:
    """Predict churn for a single customer record (dict)."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    df = pd.DataFrame([payload])[ALL_FEATURES]
    proba = float(pipeline.predict_proba(df)[0, 1])
    label = int(proba >= 0.5)
    return {
        "churn_probability": round(proba, 4),
        "churn_label": label,
        "model": artifact["model_name"],
    }


def predict_batch(records: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    """Predict churn for a batch."""
    if not records:
        return []
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    df = pd.DataFrame(records)[ALL_FEATURES]
    probas = pipeline.predict_proba(df)[:, 1]
    return [
        {
            "churn_probability": round(float(p), 4),
            "churn_label": int(p >= 0.5),
            "model": artifact["model_name"],
        }
        for p in probas
    ]
