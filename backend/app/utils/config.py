"""Application configuration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from app.utils.constants import BASE_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openweather_api_key: str | None
    openrouteservice_api_key: str | None
    ors_directions_url: str
    ors_elevation_point_url: str
    ors_elevation_line_url: str
    ors_geocode_url: str
    nominatim_search_url: str
    openweather_current_url: str
    openweather_forecast_url: str
    shelter_dataset_path: Path
    http_timeout_seconds: int
    route_altitude_samples: int
    test_size: float
    minimum_training_rows: int
    lstm_epochs: int
    lstm_batch_size: int
    route_distance_weight: float
    route_risk_weight: float
    route_altitude_weight: float
    raw_data_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load application settings from the environment."""
    shelter_value = os.getenv(
        "SHELTER_DATASET_PATH",
        str(EXTERNAL_DATA_DIR / "shelters.csv"),
    )
    shelter_path = Path(shelter_value)
    if not shelter_path.is_absolute():
        shelter_path = BASE_DIR / shelter_path

    return Settings(
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        openrouteservice_api_key=os.getenv("OPENROUTESERVICE_API_KEY"),
        ors_directions_url=os.getenv(
            "ORS_DIRECTIONS_URL",
            "https://api.openrouteservice.org/v2/directions/driving-car/json",
        ),
        ors_elevation_point_url=os.getenv(
            "ORS_ELEVATION_POINT_URL",
            "https://api.openrouteservice.org/elevation/point",
        ),
        ors_elevation_line_url=os.getenv(
            "ORS_ELEVATION_LINE_URL",
            "https://api.openrouteservice.org/elevation/line",
        ),
        ors_geocode_url=os.getenv(
            "ORS_GEOCODE_URL",
            "https://api.openrouteservice.org/geocode/search",
        ),
        nominatim_search_url=os.getenv(
            "NOMINATIM_SEARCH_URL",
            "https://nominatim.openstreetmap.org/search",
        ),
        openweather_current_url=os.getenv(
            "OPENWEATHER_CURRENT_URL",
            "https://api.openweathermap.org/data/2.5/weather",
        ),
        openweather_forecast_url=os.getenv(
            "OPENWEATHER_FORECAST_URL",
            "https://api.openweathermap.org/data/2.5/forecast",
        ),
        shelter_dataset_path=shelter_path,
        http_timeout_seconds=int(os.getenv("HTTP_TIMEOUT_SECONDS", "15")),
        route_altitude_samples=int(os.getenv("ROUTE_ALTITUDE_SAMPLES", "20")),
        test_size=float(os.getenv("MODEL_TEST_SIZE", "0.2")),
        minimum_training_rows=int(os.getenv("MINIMUM_TRAINING_ROWS", "50")),
        lstm_epochs=int(os.getenv("LSTM_EPOCHS", "10")),
        lstm_batch_size=int(os.getenv("LSTM_BATCH_SIZE", "16")),
        route_distance_weight=float(os.getenv("ROUTE_DISTANCE_WEIGHT", "0.25")),
        route_risk_weight=float(os.getenv("ROUTE_RISK_WEIGHT", "0.5")),
        route_altitude_weight=float(os.getenv("ROUTE_ALTITUDE_WEIGHT", "0.25")),
        raw_data_dir=RAW_DATA_DIR,
    )
