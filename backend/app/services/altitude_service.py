"""Elevation lookup service integration."""

from __future__ import annotations

import json
from typing import Any

import requests

from app.utils.config import get_settings


class AltitudeServiceError(RuntimeError):
    """Raised when the elevation provider cannot return a result."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class AltitudeService:
    """Wrapper around the configured elevation provider."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_point_elevation(self, latitude: float, longitude: float) -> dict[str, float]:
        """Return elevation in meters for a single coordinate."""
        if not self.settings.openrouteservice_api_key:
            raise AltitudeServiceError("OPENROUTESERVICE_API_KEY is not configured.", status_code=500)

        try:
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
        except requests.Timeout as exc:
            raise AltitudeServiceError("Elevation service timed out. Please try again.", status_code=504) from exc
        except requests.HTTPError as exc:
            raise self._build_http_error(exc) from exc
        except requests.RequestException as exc:
            raise AltitudeServiceError("Elevation service is unavailable right now.", status_code=502) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise AltitudeServiceError("Elevation service returned an invalid response.", status_code=502) from exc

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
            raise AltitudeServiceError("OPENROUTESERVICE_API_KEY is not configured.", status_code=500)
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

        try:
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
        except requests.Timeout as exc:
            raise AltitudeServiceError("Elevation service timed out. Please try again.", status_code=504) from exc
        except requests.HTTPError as exc:
            raise self._build_http_error(exc) from exc
        except requests.RequestException as exc:
            raise AltitudeServiceError("Elevation service is unavailable right now.", status_code=502) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise AltitudeServiceError("Elevation service returned an invalid response.", status_code=502) from exc

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

    def _build_http_error(self, exc: requests.HTTPError) -> AltitudeServiceError:
        """Convert upstream HTTP errors into API-safe elevation errors."""
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = "Elevation provider request failed."
        try:
            error_payload = exc.response.json() if exc.response is not None else {}
            detail = str(error_payload.get("error") or error_payload.get("message") or detail)
        except (ValueError, json.JSONDecodeError, AttributeError):
            if exc.response is not None and exc.response.text:
                detail = exc.response.text.strip() or detail
        return AltitudeServiceError(
            f"Elevation lookup failed: {detail}",
            status_code=502 if status_code >= 500 else 400,
        )
