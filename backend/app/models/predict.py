"""Prediction helpers for saved models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app.services.altitude_service import AltitudeService
from app.services.feature_engineering import align_feature_frame, build_live_feature_row
from app.services.weather_service import WeatherService, WeatherServiceError
from app.utils.constants import MODEL_METADATA_PATH, MODEL_OUTPUTS


class PredictionError(RuntimeError):
    """Raised when model inference cannot proceed."""


def _build_fallback_weather_payload() -> dict[str, object]:
    """Return a zeroed weather payload when the live provider is unavailable."""
    observed_at = datetime.now(timezone.utc)
    return {
        "provider": "fallback",
        "current": {
            "temperature_c": 0.0,
            "humidity_pct": 0.0,
            "pressure_hpa": 0.0,
            "wind_speed_mps": 0.0,
            "rainfall_mm_hr": 0.0,
            "condition": "unavailable",
            "observation_hour": observed_at.hour,
            "observation_day": observed_at.day,
            "observation_month": observed_at.month,
        },
        "raw": {},
    }


def _build_fallback_forecast_payload() -> dict[str, object]:
    """Return an empty forecast payload when the live provider is unavailable."""
    return {
        "provider": "fallback",
        "forecast": [],
        "city": {},
    }


def list_available_models() -> list[dict[str, Any]]:
    """List trained models from saved metadata."""
    metadata = _load_metadata()
    return metadata.get("models", [])


def _load_metadata() -> dict[str, Any]:
    if not MODEL_METADATA_PATH.exists():
        raise PredictionError(
            "Model metadata was not found. Train the models first via /api/risk/train."
        )
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def _find_model_entry(model_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    for model in metadata.get("models", []):
        if model["model_name"] == model_name:
            return model
    raise PredictionError(f"Model '{model_name}' is not available. Train it first.")


def _load_model(model_name: str, model_type: str) -> Any:
    artifact_path = MODEL_OUTPUTS[model_name]
    if not Path(artifact_path).exists():
        raise PredictionError(f"Model artifact not found at {artifact_path}.")
    if model_type == "keras":
        try:
            from tensorflow import keras
        except Exception as exc:  # pragma: no cover
            raise PredictionError("tensorflow is not installed in the current environment.") from exc
        return keras.models.load_model(artifact_path)
    return joblib.load(artifact_path)


def predict_risk(
    latitude: float,
    longitude: float,
    live_rainfall_mm_hr: float | None,
    live_water_level_m: float | None,
    model_name: str,
) -> dict[str, object]:
    """Predict flood risk from live inputs and external services."""
    metadata = _load_metadata()
    model_entry = _find_model_entry(model_name=model_name, metadata=metadata)
    model = _load_model(model_name=model_name, model_type=model_entry["model_type"])

    warnings: list[str] = []
    weather_service = WeatherService()
    try:
        weather = weather_service.get_current_weather(latitude=latitude, longitude=longitude)
        forecast = weather_service.get_forecast(latitude=latitude, longitude=longitude)
    except WeatherServiceError as exc:
        weather = _build_fallback_weather_payload()
        forecast = _build_fallback_forecast_payload()
        warnings.append(f"Live weather unavailable: {exc}")

    altitude = AltitudeService().get_point_elevation(latitude=latitude, longitude=longitude)
    live_row = build_live_feature_row(
        latitude=latitude,
        longitude=longitude,
        weather_payload=weather,
        forecast_payload=forecast,
        elevation_m=altitude["elevation_m"],
        water_level_m=live_water_level_m,
        rainfall_override_mm_hr=live_rainfall_mm_hr,
    )
    aligned = align_feature_frame(
        dataset=pd.DataFrame([live_row]),
        required_columns=model_entry["feature_columns"],
    )

    if model_entry["model_type"] == "keras":
        raw_output = model.predict(
            np.asarray(aligned.values, dtype=np.float32).reshape((len(aligned), 1, aligned.shape[1])),
            verbose=0,
        )
        if len(model_entry["target_classes"]) > 2:
            predicted_index = int(np.argmax(raw_output[0]))
            confidence = float(np.max(raw_output[0]))
        else:
            confidence = float(raw_output.reshape(-1)[0])
            predicted_index = int(confidence >= 0.5)
    else:
        predicted_index = int(model.predict(aligned)[0])
        probabilities = model.predict_proba(aligned)[0]
        confidence = float(np.max(probabilities))

    label = model_entry["target_classes"][predicted_index]
    return {
        "model_name": model_name,
        "prediction": label,
        "confidence": round(confidence, 4),
        "warnings": warnings,
        "features_used": live_row,
        "weather": weather,
        "forecast": forecast,
        "altitude": altitude,
    }
