"""Geocoding provider integration."""

from __future__ import annotations

import json
from typing import Any

import requests

from app.utils.config import get_settings


class GeocodingServiceError(RuntimeError):
    """Raised when place search cannot return a result."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeocodingService:
    """Client for typed place search using openrouteservice geocoding."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, size: int = 5) -> dict[str, object]:
        """Search a place name and return candidate coordinates."""
        if not self.settings.openrouteservice_api_key:
            raise GeocodingServiceError("OPENROUTESERVICE_API_KEY is not configured.", status_code=500)
        if not query.strip():
            raise GeocodingServiceError("Search query is required.")

        try:
            response = requests.get(
                self.settings.ors_geocode_url,
                headers={"Authorization": self.settings.openrouteservice_api_key},
                params={
                    "text": query,
                    "size": size,
                    "boundary.country": "IN",
                },
                timeout=self.settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except requests.Timeout as exc:
            raise GeocodingServiceError(
                "Location search provider timed out. Please try again.",
                status_code=504,
            ) from exc
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 502
            detail = "Location search provider request failed."
            try:
                error_payload = exc.response.json() if exc.response is not None else {}
                detail = str(
                    error_payload.get("error")
                    or error_payload.get("message")
                    or detail
                )
            except (ValueError, json.JSONDecodeError, AttributeError):
                if exc.response is not None and exc.response.text:
                    detail = exc.response.text.strip() or detail
            raise GeocodingServiceError(
                f"Location search failed: {detail}",
                status_code=502 if status_code >= 500 else 400,
            ) from exc
        except requests.RequestException as exc:
            raise GeocodingServiceError(
                "Location search provider is unavailable right now.",
                status_code=502,
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise GeocodingServiceError(
                "Location search provider returned an invalid response.",
                status_code=502,
            ) from exc

        features = []
        for item in payload.get("features", []):
            coords = item.get("geometry", {}).get("coordinates", [])
            props = item.get("properties", {})
            if len(coords) < 2:
                continue
            features.append(
                {
                    "label": props.get("label") or props.get("name") or query,
                    "latitude": float(coords[1]),
                    "longitude": float(coords[0]),
                    "region": props.get("region") or props.get("county"),
                    "locality": props.get("locality"),
                }
            )

        if not features:
            raise GeocodingServiceError(f"No matching location found for '{query}'.", status_code=404)

        return {"query": query, "results": features}
