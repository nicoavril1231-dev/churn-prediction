.PHONY: help install data train test lint api docker-build docker-run mlflow clean

help:
	@echo "Targets:"
	@echo "  install       Install deps in .venv via uv"
	@echo "  data          Download + clean the Telco dataset"
	@echo "  train         Train all 3 models, log to MLflow, save best"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run ruff on src/ tests/ api/"
	@echo "  api           Start FastAPI dev server on :8000"
	@echo "  mlflow        Start MLflow UI on :5000"
	@echo "  docker-build  Build production Docker image"
	@echo "  docker-run    Run the image on :8000"
	@echo "  clean         Remove caches, mlruns, processed data, models"

install:
	uv venv
	uv pip install -e ".[dev]"

data:
	uv run python -m churn.data

train:
	uv run python -m churn.train

test:
	uv run pytest

lint:
	uv run ruff check src tests api

api:
	uv run uvicorn api.main:app --reload --port 8000

mlflow:
	uv run mlflow ui --port 5000 --backend-store-uri ./mlruns

docker-build:
	docker build -t churn-prediction:latest .

docker-run:
	docker run --rm -p 8000:8000 churn-prediction:latest

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml mlruns mlartifacts
	rm -rf data/processed/*.parquet models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
