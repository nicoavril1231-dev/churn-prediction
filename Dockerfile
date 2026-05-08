# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   1. builder — install deps with uv, train the model, then strip dev deps
#   2. runtime — slim image with the model + production deps only
# Final image size: ~700 MB (sklearn + xgboost + lightgbm + fastapi)
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only what's needed for dep resolution first → maximises Docker cache
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install runtime deps (no dev) into a known venv
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python -e .

# Now bring in the rest needed for training
COPY data/raw/.gitkeep ./data/raw/.gitkeep

# Train the model at build time so the image ships a ready-to-serve artifact.
# In a real production pipeline, the model would be pulled from a registry
# (MLflow / S3) instead of trained inline — we do it here for portability.
RUN /opt/venv/bin/python -m churn.data \
    && /opt/venv/bin/python -m churn.train --model logreg

# ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

# Run as non-root for security
RUN groupadd --system app && useradd --system --gid app --no-create-home app

# Copy the virtual environment, source, model from the builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models
COPY api/ ./api/

RUN chown -R app:app /app
USER app

EXPOSE 8000

# HEALTHCHECK uses $PORT so it works whether running locally (8000) or on
# Render/Fly/etc. which inject their own port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health').read()" || exit 1

# Shell form so $PORT is expanded at runtime — required by Render which
# injects its own PORT env var (typically 10000) and expects the app to bind it.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
