"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.utils.config import get_settings


def discover_latest_dataset() -> Path:
    """Find the most recent CSV or parquet dataset in the raw data directory."""
    settings = get_settings()
    candidates = sorted(
        list(settings.raw_data_dir.glob("*.csv")) + list(settings.raw_data_dir.glob("*.parquet")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No dataset found in data/raw. Add your historical data as CSV or parquet."
        )
    return candidates[0]


def load_training_dataset(dataset_path: str | None = None) -> pd.DataFrame:
    """Load the dataset used for training and evaluation."""
    path = Path(dataset_path) if dataset_path else discover_latest_dataset()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_shelter_dataset() -> pd.DataFrame:
    """Load shelter metadata used for route optimization."""
    settings = get_settings()
    if not settings.shelter_dataset_path.exists():
        raise FileNotFoundError(
            "Shelter dataset not found. Add a CSV file at "
            f"{settings.shelter_dataset_path}."
        )
    return pd.read_csv(settings.shelter_dataset_path)
