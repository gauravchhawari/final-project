"""Geocoding provider integration."""

from __future__ import annotations

from typing import Any

import requests

from app.utils.config import get_settings


class GeocodingServiceError(RuntimeError):
    """Raised when place search cannot return a result."""


class GeocodingService:
    """Client for typed place search using openrouteservice geocoding."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, size: int = 5) -> dict[str, object]:
        """Search a place name and return candidate coordinates."""
        if not self.settings.openrouteservice_api_key:
            raise GeocodingServiceError("OPENROUTESERVICE_API_KEY is not configured.")
        if not query.strip():
            raise GeocodingServiceError("Search query is required.")

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
            raise GeocodingServiceError(f"No matching location found for '{query}'.")

        return {"query": query, "results": features}
