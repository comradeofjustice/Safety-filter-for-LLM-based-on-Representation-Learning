"""I/O utilities for LLM Safety Classifier."""

import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet(path: str) -> pd.DataFrame:
    """Load parquet file."""
    logger.info(f"Loading {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str):
    """Save DataFrame to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
