"""Data acquisition and cleaning.

Loads the IBM Telco Customer Churn dataset (7043 rows × 21 columns),
handles the well-known TotalCharges quirk (11 rows with whitespace),
and writes a cleaned parquet to data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from churn.config import (
    DATASET_URL,
    PROCESSED_FILE,
    RAW_DIR,
    RAW_FILE,
    TARGET,
)

logger = logging.getLogger(__name__)


def download(force: bool = False, url: str = DATASET_URL, dest: Path = RAW_FILE) -> Path:
    """Download the raw Telco dataset.

    Idempotent: skips if the file exists unless `force=True`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("Dataset already present at %s — skip download.", dest)
        return dest

    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Wrote %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as IBM ships it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m churn.data` or `churn-download` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning rules.

    1. `TotalCharges` ships as object: 11 rows have a whitespace string for
       brand-new customers (tenure == 0). We coerce to float; those rows
       become NaN and are dropped (~0.16% of data, no signal lost).
    2. Target `Churn` is "Yes"/"No" → cast to int {0, 1}.
    3. `customerID` kept as identifier but excluded from features.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    n_before = len(out)
    out = out.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    dropped = n_before - len(out)
    if dropped:
        logger.info("Dropped %d rows with invalid TotalCharges (%.2f%%).",
                    dropped, 100 * dropped / n_before)

    out[TARGET] = (out[TARGET].str.strip().str.lower() == "yes").astype(int)

    return out


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_FILE) -> Path:
    """Persist as parquet for downstream notebooks and training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows × %d cols)", path, len(df), df.shape[1])
    return path


def load_processed(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """Read the processed parquet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m churn.data` to build it."
        )
    return pd.read_parquet(path)


def main() -> None:
    """CLI entrypoint: download + clean + save."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download()
    df = load_raw()
    df = clean(df)
    save_processed(df)
    print(f"OK — processed dataset: {len(df)} rows × {df.shape[1]} cols")


if __name__ == "__main__":
    main()
