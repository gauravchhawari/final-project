"""Preprocessing helpers for training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_training_matrices(
    dataset: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a feature frame into X/y matrices."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataset.")

    filtered = dataset.dropna(subset=[target_column]).copy()
    y = filtered[target_column].astype(str)
    X = filtered.drop(columns=[target_column])

    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_frame = X[numeric_columns].copy().fillna(0.0)

    if numeric_frame.empty:
        raise ValueError("No numeric features available after preprocessing.")

    return numeric_frame, y, numeric_columns


def split_train_test(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split data chronologically, keeping the newest rows for testing."""
    if len(X) < 2:
        raise ValueError("Need at least two rows to create a train/test split.")

    split_index = max(1, min(len(X) - 1, int(len(X) * (1 - test_size))))
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test
