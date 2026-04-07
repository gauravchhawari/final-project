"""Model evaluation utilities."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from app.models.predict import PredictionError
from app.models.reporting import save_evaluation_reports
from app.models.train import TrainingError
from app.services.data_loader import load_training_dataset
from app.services.feature_engineering import build_feature_frame
from app.services.preprocessing import build_training_matrices, split_train_test
from app.utils.constants import MODEL_METADATA_PATH


def evaluate_saved_models(
    dataset_path: str | None = None,
    target_column: str = "flood_label",
    timestamp_column: str = "datetime",
) -> dict[str, Any]:
    """Evaluate all persisted models against a dataset."""
    if not MODEL_METADATA_PATH.exists():
        raise TrainingError("Model metadata not found. Train models before evaluation.")

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    dataset = load_training_dataset(dataset_path=dataset_path)
    featured = build_feature_frame(
        dataset=dataset,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )
    X, y, _ = build_training_matrices(dataset=featured, target_column=target_column)
    _, X_test, _, y_test = split_train_test(
        X,
        np.asarray(y),
        test_size=0.2,
    )

    results: dict[str, Any] = {"models": []}
    for model_entry in metadata.get("models", []):
        aligned = X_test.reindex(columns=model_entry["feature_columns"], fill_value=0.0)
        if model_entry["model_type"] == "keras":
            try:
                from tensorflow import keras
            except Exception as exc:  # pragma: no cover
                raise PredictionError("tensorflow is not installed in the current environment.") from exc
            model = keras.models.load_model(model_entry["artifact_path"])
            raw = model.predict(
                np.asarray(aligned.values, dtype=np.float32).reshape((len(aligned), 1, aligned.shape[1])),
                verbose=0,
            )
            predictions = (
                np.argmax(raw, axis=1)
                if len(model_entry["target_classes"]) > 2
                else (raw.reshape(-1) >= 0.5).astype(int)
            )
        else:
            model = joblib.load(model_entry["artifact_path"])
            predictions = model.predict(aligned)

        target_map = {label: idx for idx, label in enumerate(model_entry["target_classes"])}
        y_encoded = np.asarray([target_map[str(item)] for item in y_test])
        results["models"].append(
            {
                "model_name": model_entry["model_name"],
                "accuracy": float(accuracy_score(y_encoded, predictions)),
                "f1_weighted": float(f1_score(y_encoded, predictions, average="weighted")),
                "confusion_matrix": confusion_matrix(y_encoded, predictions).tolist(),
                "classification_report": classification_report(
                    y_encoded,
                    predictions,
                    target_names=model_entry["target_classes"],
                    zero_division=0,
                    output_dict=True,
                ),
            }
        )

    results["report_files"] = save_evaluation_reports(results)
    return results
