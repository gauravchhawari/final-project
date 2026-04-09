"""Safe evacuation routing based on shelters, directions, altitude, and risk."""

from __future__ import annotations

import html
import json
import re
from typing import Any

import requests

from app.models.predict import PredictionError, predict_risk
from app.services.altitude_service import AltitudeService, AltitudeServiceError
from app.services.shelter_service import ShelterService, ShelterServiceError
from app.utils.config import get_settings


class RoutingServiceError(RuntimeError):
    """Raised when route optimization cannot proceed."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class RoutingService:
    """Generate safe routes to shelters using live routing services."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.shelter_service = ShelterService()
        self.altitude_service = AltitudeService()

    @staticmethod
    def _strip_html(raw_text: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", raw_text))

    def _fetch_route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> dict[str, Any]:
        if not self.settings.openrouteservice_api_key:
            raise RoutingServiceError("OPENROUTESERVICE_API_KEY is not configured.", status_code=500)

        try:
            response = requests.post(
                self.settings.ors_directions_url,
                headers={
                    "Authorization": self.settings.openrouteservice_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "coordinates": [
                        [origin[1], origin[0]],
                        [destination[1], destination[0]],
                    ],
                    "instructions": True,
                },
                timeout=self.settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise RoutingServiceError("Directions service timed out. Please try again.", status_code=504) from exc
        except requests.HTTPError as exc:
            raise self._build_http_error(exc) from exc
        except requests.RequestException as exc:
            raise RoutingServiceError("Directions service is unavailable right now.", status_code=502) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise RoutingServiceError("Directions service returned an invalid response.", status_code=502) from exc

        routes = payload.get("routes", [])
        if not routes:
            raise RoutingServiceError("Directions API returned no route options.")
        return routes[0]

    def _score_route(
        self,
        route: dict[str, Any],
        altitude_samples: list[dict[str, float]],
        risk_confidence: float,
    ) -> float:
        summary = route.get("summary", {})
        distance_km = float(summary.get("distance", 0.0)) / 1000.0
        min_altitude = min((sample["elevation_m"] for sample in altitude_samples), default=0.0)
        altitude_penalty = 0.0 if min_altitude <= 0 else 1.0 / max(min_altitude, 1.0)
        return (
            risk_confidence * self.settings.route_risk_weight
            + distance_km * self.settings.route_distance_weight
            + altitude_penalty * self.settings.route_altitude_weight
        )

    def get_best_route(
        self,
        latitude: float,
        longitude: float,
        region: str | None,
        max_candidates: int,
        live_water_level_m: float | None,
        live_rainfall_mm_hr: float | None,
        model_name: str | None,
    ) -> dict[str, Any]:
        try:
            shelters = self.shelter_service.find_nearest_shelters(
                latitude=latitude,
                longitude=longitude,
                region=region,
                max_candidates=max_candidates,
            )
        except ShelterServiceError as exc:
            raise RoutingServiceError(str(exc)) from exc

        try:
            risk = predict_risk(
                latitude=latitude,
                longitude=longitude,
                live_rainfall_mm_hr=live_rainfall_mm_hr,
                live_water_level_m=live_water_level_m,
                model_name=model_name or "xgb",
            )
        except PredictionError as exc:
            raise RoutingServiceError(str(exc)) from exc
        ranked_routes = []
        for shelter in shelters:
            route = self._fetch_route(
                origin=(latitude, longitude),
                destination=(float(shelter["latitude"]), float(shelter["longitude"])),
            )
            try:
                altitude_samples = self.altitude_service.get_path_elevations(
                    path=[
                        (latitude, longitude),
                        (float(shelter["latitude"]), float(shelter["longitude"])),
                    ],
                    samples=self.settings.route_altitude_samples,
                )
            except AltitudeServiceError as exc:
                raise RoutingServiceError(str(exc), status_code=exc.status_code) from exc

            score = self._score_route(
                route=route,
                altitude_samples=altitude_samples,
                risk_confidence=float(risk["confidence"]),
            )
            summary = route.get("summary", {})
            segments = route.get("segments", [])
            steps = segments[0].get("steps", []) if segments else []
            ranked_routes.append(
                {
                    "shelter": shelter,
                    "route_score": round(score, 4),
                    "distance_km": round(float(summary.get("distance", 0.0)) / 1000.0, 3),
                    "duration_minutes": round(float(summary.get("duration", 0.0)) / 60.0, 1),
                    "directions": [
                        self._strip_html(step.get("instruction", ""))
                        for step in steps
                        if step.get("instruction")
                    ],
                    "start_address": shelter.get("address"),
                    "end_address": shelter.get("address"),
                    "polyline": route.get("geometry"),
                    "altitude_profile": altitude_samples,
                    "risk_assessment": risk,
                }
            )

        if not ranked_routes:
            raise RoutingServiceError("No viable evacuation routes were found.")

        best = min(ranked_routes, key=lambda item: item["route_score"])
        return {"best_route": best, "candidates": ranked_routes}

    def _build_http_error(self, exc: requests.HTTPError) -> RoutingServiceError:
        """Convert upstream HTTP errors into API-safe routing errors."""
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = "Directions provider request failed."
        try:
            error_payload = exc.response.json() if exc.response is not None else {}
            detail = str(error_payload.get("error") or error_payload.get("message") or detail)
        except (ValueError, json.JSONDecodeError, AttributeError):
            if exc.response is not None and exc.response.text:
                detail = exc.response.text.strip() or detail
        return RoutingServiceError(
            f"Route lookup failed: {detail}",
            status_code=502 if status_code >= 500 else 400,
        )
