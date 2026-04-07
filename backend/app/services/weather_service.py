"""Weather provider integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

from app.utils.config import get_settings


class WeatherServiceError(RuntimeError):
    """Raised when the weather provider request fails."""


class WeatherService:
    """Client for the configured weather provider."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _request(self, url: str, latitude: float, longitude: float) -> dict[str, Any]:
        if not self.settings.openweather_api_key:
            raise WeatherServiceError("OPENWEATHER_API_KEY is not configured.")

        response = requests.get(
            url,
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": self.settings.openweather_api_key,
                "units": "metric",
            },
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_current_weather(self, latitude: float, longitude: float) -> dict[str, object]:
        """Return normalized current weather."""
        payload = self._request(
            url=self.settings.openweather_current_url,
            latitude=latitude,
            longitude=longitude,
        )
        rainfall = 0.0
        if "rain" in payload:
            rainfall = float(payload["rain"].get("1h") or payload["rain"].get("3h", 0.0))
        observed_at = datetime.utcfromtimestamp(int(payload["dt"]))

        return {
            "provider": "openweather",
            "current": {
                "temperature_c": float(payload["main"]["temp"]),
                "humidity_pct": float(payload["main"]["humidity"]),
                "pressure_hpa": float(payload["main"]["pressure"]),
                "wind_speed_mps": float(payload.get("wind", {}).get("speed", 0.0)),
                "rainfall_mm_hr": rainfall,
                "condition": str(payload["weather"][0]["description"]),
                "observation_hour": observed_at.hour,
                "observation_day": observed_at.day,
                "observation_month": observed_at.month,
            },
            "raw": payload,
        }

    def get_forecast(self, latitude: float, longitude: float) -> dict[str, object]:
        """Return normalized forecast entries."""
        payload = self._request(
            url=self.settings.openweather_forecast_url,
            latitude=latitude,
            longitude=longitude,
        )
        forecast_items = []
        for item in payload.get("list", []):
            rainfall = 0.0
            if "rain" in item:
                rainfall = float(item["rain"].get("3h", 0.0))
            forecast_items.append(
                {
                    "timestamp": item["dt_txt"],
                    "temperature_c": float(item["main"]["temp"]),
                    "humidity_pct": float(item["main"]["humidity"]),
                    "rainfall_mm": rainfall,
                    "condition": str(item["weather"][0]["description"]),
                }
            )
        return {
            "provider": "openweather",
            "forecast": forecast_items,
            "city": payload.get("city", {}),
        }
