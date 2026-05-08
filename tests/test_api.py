"""End-to-end tests for the FastAPI service.

These run only when a trained model is available (CI installs deps,
trains a quick model, then runs the suite).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="No trained model — run `python -m churn.train` first.",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from api.main import app
    return TestClient(app)


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "churn-prediction"


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_high_risk(client: TestClient, valid_payload: dict) -> None:
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_label"] in {0, 1}
    # High-risk profile (1 month, fiber, electronic check) should score >0.5
    assert body["churn_probability"] > 0.5


def test_predict_low_risk(client: TestClient, valid_payload: dict) -> None:
    payload = {**valid_payload, "tenure": 60, "Contract": "Two year",
               "PaymentMethod": "Bank transfer (automatic)"}
    r = client.post("/predict", json=payload)
    body = r.json()
    assert body["churn_probability"] < 0.3


def test_predict_rejects_extra_field(client: TestClient, valid_payload: dict) -> None:
    bad = {**valid_payload, "unexpected_field": 42}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_invalid_enum(client: TestClient, valid_payload: dict) -> None:
    bad = {**valid_payload, "Contract": "Three year"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_tenure(client: TestClient, valid_payload: dict) -> None:
    bad = {**valid_payload, "tenure": -1}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch(client: TestClient, valid_payload: dict) -> None:
    r = client.post("/predict/batch", json={"customers": [valid_payload, valid_payload]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_batch_empty_rejected(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"customers": []})
    assert r.status_code == 422
