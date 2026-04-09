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
    """Client for typed place search with provider fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, size: int = 5) -> dict[str, object]:
        """Search a place name and return candidate coordinates."""
        if not query.strip():
            raise GeocodingServiceError("Search query is required.")

        ors_error: GeocodingServiceError | None = None
        if self.settings.openrouteservice_api_key:
            try:
                ors_payload = self._search_openrouteservice(query=query, size=size)
                ors_features = self._normalize_ors_features(payload=ors_payload, query=query)
                if ors_features:
                    return {"query": query, "results": ors_features}
            except GeocodingServiceError as exc:
                ors_error = exc

        fallback_payload = self._search_nominatim(query=query, size=size)
        fallback_features = self._normalize_nominatim_features(payload=fallback_payload)
        if fallback_features:
            return {"query": query, "results": fallback_features}

        if ors_error is not None:
            raise ors_error
        raise GeocodingServiceError(f"No matching location found for '{query}'.", status_code=404)

    def _search_openrouteservice(self, query: str, size: int) -> dict[str, Any]:
        """Query OpenRouteService geocoding."""
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
            payload = response.json()
            if not isinstance(payload, dict):
                raise GeocodingServiceError(
                    "Location search provider returned an invalid response.",
                    status_code=502,
                )
            return payload
        except requests.Timeout as exc:
            raise GeocodingServiceError(
                "Location search provider timed out. Please try again.",
                status_code=504,
            ) from exc
        except requests.HTTPError as exc:
            raise self._build_http_error(exc, prefix="Location search failed")
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

    def _search_nominatim(self, query: str, size: int) -> list[dict[str, Any]]:
        """Query Nominatim as a fallback geocoder."""
        try:
            response = requests.get(
                self.settings.nominatim_search_url,
                headers={"User-Agent": "flood-risk-system/1.0"},
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": size,
                    "countrycodes": "in",
                    "addressdetails": 1,
                },
                timeout=self.settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise GeocodingServiceError(
                    "Fallback location search provider returned an invalid response.",
                    status_code=502,
                )
            return payload
        except requests.Timeout as exc:
            raise GeocodingServiceError(
                "Fallback location search provider timed out. Please try again.",
                status_code=504,
            ) from exc
        except requests.HTTPError as exc:
            raise self._build_http_error(exc, prefix="Fallback location search failed")
        except requests.RequestException as exc:
            raise GeocodingServiceError(
                "Fallback location search provider is unavailable right now.",
                status_code=502,
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise GeocodingServiceError(
                "Fallback location search provider returned an invalid response.",
                status_code=502,
            ) from exc

    def _normalize_ors_features(self, payload: dict[str, Any], query: str) -> list[dict[str, object]]:
        """Normalize OpenRouteService features."""
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
        return features

    def _normalize_nominatim_features(self, payload: list[dict[str, Any]]) -> list[dict[str, object]]:
        """Normalize Nominatim features."""
        features = []
        for item in payload:
            address = item.get("address", {})
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is None or lon is None:
                continue
            features.append(
                {
                    "label": item.get("display_name") or address.get("road") or "Unknown location",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "region": address.get("state_district") or address.get("state"),
                    "locality": address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("suburb"),
                }
            )
        return features

    def _build_http_error(
        self,
        exc: requests.HTTPError,
        prefix: str,
    ) -> GeocodingServiceError:
        """Convert an upstream HTTP error into an API-safe geocoding error."""
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

        mapped_status = 502 if status_code >= 500 else 400
        if status_code == 404:
            mapped_status = 404
        return GeocodingServiceError(f"{prefix}: {detail}", status_code=mapped_status)
