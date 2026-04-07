"""Feature engineering helpers."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _to_datetime_if_present(dataset: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    transformed = dataset.copy()
    if timestamp_column in transformed.columns:
        transformed[timestamp_column] = pd.to_datetime(
            transformed[timestamp_column],
            errors="coerce",
        )
        transformed = transformed.sort_values(timestamp_column)
        transformed["hour"] = transformed[timestamp_column].dt.hour.fillna(0)
        transformed["day"] = transformed[timestamp_column].dt.day.fillna(0)
        transformed["month"] = transformed[timestamp_column].dt.month.fillna(0)
    return transformed


def build_feature_frame(
    dataset: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Generate model-ready features from historical tabular data."""
    frame = _to_datetime_if_present(dataset=dataset, timestamp_column=timestamp_column)

    for source_column in (
        "rainfall_mm_hr",
        "rainfall_mm",
        "water_level_m",
        "temperature_c",
        "humidity_pct",
        "elevation_m",
        "drainage_capacity",
        "distance_to_water_km",
        "latitude",
        "longitude",
    ):
        if source_column in frame.columns:
            frame[source_column] = pd.to_numeric(frame[source_column], errors="coerce")

    if "rainfall_mm_hr" in frame.columns:
        frame["rainfall_prev_1"] = frame["rainfall_mm_hr"].shift(1)
        frame["rainfall_prev_3_avg"] = frame["rainfall_mm_hr"].rolling(window=3, min_periods=1).mean()
        frame["rainfall_prev_24_sum"] = frame["rainfall_mm_hr"].rolling(window=24, min_periods=1).sum()

    if "water_level_m" in frame.columns:
        frame["water_level_prev_1"] = frame["water_level_m"].shift(1)
        frame["water_level_delta"] = frame["water_level_m"].diff()

    if "elevation_m" in frame.columns and "water_level_m" in frame.columns:
        frame["elevation_water_gap"] = frame["elevation_m"] - frame["water_level_m"]

    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' is required in the historical dataset."
        )

    return frame.bfill().fillna(0.0)


def build_live_feature_row(
    latitude: float,
    longitude: float,
    weather_payload: dict[str, Any],
    forecast_payload: dict[str, Any] | None,
    elevation_m: float,
    water_level_m: float | None,
    rainfall_override_mm_hr: float | None,
) -> dict[str, float]:
    """Create a single model input row from live weather and elevation."""
    current = weather_payload.get("current", {})
    forecast_items = (forecast_payload or {}).get("forecast", [])
    rainfall_mm_hr = (
        rainfall_override_mm_hr
        if rainfall_override_mm_hr is not None
        else float(current.get("rainfall_mm_hr", 0.0))
    )
    rain_24h = float(sum(item.get("rainfall_mm", 0.0) for item in forecast_items[:8]))
    rain_48h = float(sum(item.get("rainfall_mm", 0.0) for item in forecast_items[:16]))
    water_level = float(water_level_m) if water_level_m is not None else 0.0
    humidity = float(current.get("humidity_pct", 0.0))
    temperature = float(current.get("temperature_c", 0.0))
    wind_speed = float(current.get("wind_speed_mps", 0.0))

    return {
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm": rainfall_mm_hr,
        "rainfall_mm_hr": rainfall_mm_hr,
        "rain_24h": rain_24h,
        "rain_48h": rain_48h,
        "water_level_m": water_level,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "wind_speed_mps": wind_speed,
        "elevation_m": elevation_m,
        "elevation_water_gap": elevation_m - water_level,
        "rainfall_prev_1": rainfall_mm_hr,
        "rainfall_prev_3_avg": rainfall_mm_hr,
        "rainfall_prev_24_sum": rainfall_mm_hr * 24,
        "water_level_prev_1": water_level,
        "water_level_delta": 0.0,
        "distance_to_water_km": float(current.get("distance_to_water_km", 0.0)),
        "drainage_capacity": float(current.get("drainage_capacity", 0.0)),
        "hour": float(current.get("observation_hour", 0.0)),
        "day": float(current.get("observation_day", 0.0)),
        "month": float(current.get("observation_month", 0.0)),
    }


def align_feature_frame(dataset: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Align a dataframe to the feature columns expected by a model."""
    return dataset.reindex(columns=required_columns, fill_value=0.0).fillna(0.0)


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate spherical distance between two coordinate pairs."""
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
