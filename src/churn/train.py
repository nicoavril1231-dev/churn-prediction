"""Training pipeline with MLflow tracking.

Trains 3 candidates (Logistic Regression baseline, XGBoost, LightGBM),
logs metrics + artifacts to MLflow, and persists the best model to
`models/best_model.joblib`.

Run:
    python -m churn.train               # train all
    python -m churn.train --model xgb   # train one
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn.config import (
    ALL_FEATURES,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from churn.data import load_processed
from churn.features import build_preprocessor

logger = logging.getLogger(__name__)


def get_models() -> dict[str, Any]:
    """Candidate models with reasonable defaults.

    Hyper-parameters here are sane defaults — proper HPO comes in notebook 02.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=2.77,  # ≈ neg/pos ratio in Telco
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute the metrics we care about for an imbalanced binary problem."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def train_and_log(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit pipeline, evaluate on test, log to MLflow, return the pipeline + metrics."""
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=name):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test.to_numpy(), y_pred, y_proba)
        metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc.std())

        mlflow.log_params({"model": name, **estimator.get_params(deep=False)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "pipeline")

        logger.info(
            "%-7s — ROC-AUC: %.4f | F1: %.4f | Recall: %.4f | CV-AUC: %.4f ± %.4f",
            name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["recall"],
            metrics["cv_roc_auc_mean"],
            metrics["cv_roc_auc_std"],
        )

    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "logreg", "xgb", "lgbm"],
        default="all",
        help="Which model(s) to train.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_processed()
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d | Test: %d | Churn rate: %.2f%%",
                len(X_train), len(X_test), 100 * y_train.mean())

    models = get_models()
    if args.model != "all":
        models = {args.model: models[args.model]}

    results: dict[str, tuple[Pipeline, dict[str, float]]] = {}
    for name, estimator in models.items():
        pipe, metrics = train_and_log(name, estimator, X_train, y_train, X_test, y_test)
        results[name] = (pipe, metrics)

    best_name = max(results, key=lambda n: results[n][1]["roc_auc"])
    best_pipe, best_metrics = results[best_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": best_pipe,
            "model_name": best_name,
            "metrics": best_metrics,
            "feature_columns": ALL_FEATURES,
        },
        artifact_path,
    )

    logger.info("=" * 60)
    logger.info("BEST MODEL: %s — saved to %s", best_name, artifact_path)
    logger.info("ROC-AUC: %.4f | F1: %.4f", best_metrics["roc_auc"], best_metrics["f1"])


if __name__ == "__main__":
    main()
