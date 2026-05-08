"""Pydantic schemas — request/response validation for the FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Single-customer payload. Field names match the Telco dataset exactly."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Months as a customer.")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Single-customer prediction response."""

    churn_probability: float = Field(..., ge=0, le=1)
    churn_label: Literal[0, 1]
    model: str


class BatchRequest(BaseModel):
    """Batch request — up to 1000 customers per call to bound resource usage."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    """Batch response."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health probe response."""

    status: Literal["ok", "no_model"]
    model_loaded: bool
    version: str
