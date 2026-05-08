"""FastAPI service — churn prediction inference endpoints.

Endpoints:
    GET  /         → service banner
    GET  /health   → liveness + model-load probe
    POST /predict  → single-customer prediction
    POST /predict/batch → batch prediction (≤1000)
    GET  /docs     → Swagger UI (FastAPI default)

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn import __version__
from churn.predict import load_artifact, predict_batch, predict_one

logger = logging.getLogger("churn.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model on startup so the first request isn't slow."""
    try:
        load_artifact()
        logger.info("Model loaded successfully on startup.")
    except FileNotFoundError as exc:
        logger.warning("Model not found at startup: %s", exc)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability for telecom customers.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_model_or_503():
    """Dependency: 503 if no trained model is available."""
    try:
        return load_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "churn-prediction", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        load_artifact()
        return HealthResponse(status="ok", model_loaded=True, version=__version__)
    except FileNotFoundError:
        return HealthResponse(status="no_model", model_loaded=False, version=__version__)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: CustomerFeatures,
    _: Annotated[dict, Depends(get_model_or_503)],
) -> PredictionResponse:
    result = predict_one(payload.model_dump())
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch_endpoint(
    payload: BatchRequest,
    _: Annotated[dict, Depends(get_model_or_503)],
) -> BatchResponse:
    records = [c.model_dump() for c in payload.customers]
    results = predict_batch(records)
    return BatchResponse(
        predictions=[PredictionResponse(**r) for r in results],
        count=len(results),
    )
