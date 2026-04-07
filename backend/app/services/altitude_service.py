"""Elevation lookup service integration."""

from __future__ import annotations

from typing import Any

import requests

from app.utils.config import get_settings


class AltitudeServiceError(RuntimeError):
    """Raised when the elevation provider cannot return a result."""


class AltitudeService:
    """Wrapper around the configured elevation provider."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_point_elevation(self, latitude: float, longitude: float) -> dict[str, float]:
        """Return elevation in meters for a single coordinate."""
        if not self.settings.openrouteservice_api_key:
            raise AltitudeServiceError("OPENROUTESERVICE_API_KEY is not configured.")

        response = requests.post(
            self.settings.ors_elevation_point_url,
            headers={
                "Authorization": self.settings.openrouteservice_api_key,
                "Content-Type": "application/json",
            },
            json={
                "format_in": "point",
                "geometry": [longitude, latitude],
            },
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        coordinates = payload.get("geometry", {}).get("coordinates", [])
        if len(coordinates) < 3:
            raise AltitudeServiceError("Elevation API returned no usable point geometry.")
        return {"elevation_m": float(coordinates[2])}

    def get_path_elevations(
        self,
        path: list[tuple[float, float]],
        samples: int,
    ) -> list[dict[str, float]]:
        """Return elevation samples along a route polyline path."""
        if not self.settings.openrouteservice_api_key:
            raise AltitudeServiceError("OPENROUTESERVICE_API_KEY is not configured.")
        if not path:
            return []

        if samples > 1 and len(path) >= 2:
            start_lat, start_lng = path[0]
            end_lat, end_lng = path[-1]
            interpolated = []
            for idx in range(samples):
                fraction = idx / (samples - 1)
                interpolated.append(
                    [
                        start_lng + (end_lng - start_lng) * fraction,
                        start_lat + (end_lat - start_lat) * fraction,
                    ]
                )
        else:
            interpolated = [[lng, lat] for lat, lng in path]

        response = requests.post(
            self.settings.ors_elevation_line_url,
            headers={
                "Authorization": self.settings.openrouteservice_api_key,
                "Content-Type": "application/json",
            },
            json={
                "format_in": "polyline",
                "geometry": interpolated,
            },
            timeout=self.settings.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        coordinates = payload.get("geometry", {}).get("coordinates", [])
        return [
            {
                "latitude": float(item[1]),
                "longitude": float(item[0]),
                "elevation_m": float(item[2]),
            }
            for item in coordinates
            if len(item) >= 3
        ]
