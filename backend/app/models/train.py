"""Model training entry points."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.services.data_loader import load_training_dataset
from app.services.feature_engineering import build_feature_frame
from app.services.preprocessing import build_training_matrices, split_train_test
from app.utils.config import get_settings
from app.utils.constants import MODEL_METADATA_PATH, MODEL_OUTPUTS
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TrainingError(RuntimeError):
    """Raised when model training cannot proceed."""


@dataclass
class TrainingArtifact:
    """Saved metadata for a trained model."""

    model_name: str
    artifact_path: str
    metrics: dict[str, float]
    feature_columns: list[str]
    target_classes: list[str]
    model_type: str


def _build_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _build_xgb_pipeline(num_classes: int) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise TrainingError("xgboost is not installed in the current environment.") from exc

    if XGBClassifier is None:
        raise TrainingError("xgboost is not installed in the current environment.")

    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if num_classes > 2 else "logloss"
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective=objective,
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric=eval_metric,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_lstm_model(input_dim: int, num_classes: int) -> Any:
    try:
        from tensorflow import keras
    except Exception as exc:  # pragma: no cover
        raise TrainingError("tensorflow is not installed in the current environment.") from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, input_dim)),
            keras.layers.LSTM(32),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(
                num_classes if num_classes > 2 else 1,
                activation="softmax" if num_classes > 2 else "sigmoid",
            ),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def train_and_save_models(
    dataset_path: str | None = None,
    target_column: str = "flood_label",
    timestamp_column: str = "datetime",
) -> dict[str, object]:
    """Train all supported models and save their artifacts."""
    settings = get_settings()
    dataset = load_training_dataset(dataset_path=dataset_path)
    featured = build_feature_frame(
        dataset=dataset,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )
    X, y, feature_columns = build_training_matrices(
        dataset=featured,
        target_column=target_column,
    )

    if len(X) < settings.minimum_training_rows:
        raise TrainingError(
            f"Need at least {settings.minimum_training_rows} rows for training; found {len(X)}."
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y_encoded,
        test_size=settings.test_size,
    )

    trained_models: list[TrainingArtifact] = []

    rf_pipeline = _build_rf_pipeline()
    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)
    joblib.dump(rf_pipeline, MODEL_OUTPUTS["rf"])
    trained_models.append(
        TrainingArtifact(
            model_name="rf",
            artifact_path=str(MODEL_OUTPUTS["rf"]),
            metrics=_compute_metrics(y_test, rf_predictions),
            feature_columns=feature_columns,
            target_classes=[str(item) for item in encoder.classes_],
            model_type="sklearn",
        )
    )

    try:
        xgb_pipeline = _build_xgb_pipeline(num_classes=num_classes)
        xgb_pipeline.fit(X_train, y_train)
        xgb_predictions = xgb_pipeline.predict(X_test)
        joblib.dump(xgb_pipeline, MODEL_OUTPUTS["xgb"])
        trained_models.append(
            TrainingArtifact(
                model_name="xgb",
                artifact_path=str(MODEL_OUTPUTS["xgb"]),
                metrics=_compute_metrics(y_test, xgb_predictions),
                feature_columns=feature_columns,
                target_classes=[str(item) for item in encoder.classes_],
                model_type="sklearn",
            )
        )
    except TrainingError as exc:
        logger.warning("Skipping XGBoost training: %s", exc)

    try:
        lstm_model = _build_lstm_model(input_dim=len(feature_columns), num_classes=num_classes)
        lstm_train = np.asarray(X_train, dtype=np.float32).reshape((len(X_train), 1, len(feature_columns)))
        lstm_test = np.asarray(X_test, dtype=np.float32).reshape((len(X_test), 1, len(feature_columns)))
        lstm_model.fit(
            lstm_train,
            np.asarray(y_train, dtype=np.float32),
            epochs=settings.lstm_epochs,
            batch_size=settings.lstm_batch_size,
            verbose=0,
        )
        raw_output = lstm_model.predict(lstm_test, verbose=0)
        predictions = (
            np.argmax(raw_output, axis=1)
            if num_classes > 2
            else (raw_output.reshape(-1) >= 0.5).astype(int)
        )
        lstm_model.save(MODEL_OUTPUTS["lstm"])
        trained_models.append(
            TrainingArtifact(
                model_name="lstm",
                artifact_path=str(MODEL_OUTPUTS["lstm"]),
                metrics=_compute_metrics(y_test, predictions),
                feature_columns=feature_columns,
                target_classes=[str(item) for item in encoder.classes_],
                model_type="keras",
            )
        )
    except TrainingError as exc:
        logger.warning("Skipping LSTM training: %s", exc)

    metadata: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).resolve()) if dataset_path else "auto-discovered",
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "split_strategy": "chronological",
        "models": [asdict(model) for model in trained_models],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
