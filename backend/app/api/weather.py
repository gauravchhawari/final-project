"""Weather API routes."""

from fastapi import APIRouter, HTTPException, Query

from app.services.geocoding_service import GeocodingService, GeocodingServiceError
from app.services.weather_service import WeatherService, WeatherServiceError

router = APIRouter(prefix="/api/weather", tags=["weather"])


@router.get("/current")
def get_current_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, object]:
    """Return current weather from the configured provider."""
    service = WeatherService()
    try:
        return service.get_current_weather(latitude=latitude, longitude=longitude)
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/forecast")
def get_forecast(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, object]:
    """Return short-term forecast from the configured provider."""
    service = WeatherService()
    try:
        return service.get_forecast(latitude=latitude, longitude=longitude)
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/search-location")
def search_location(
    query: str = Query(..., min_length=2),
    size: int = Query(5, ge=1, le=10),
) -> dict[str, object]:
    """Search a typed location name and return coordinate candidates."""
    service = GeocodingService()
    try:
        return service.search(query=query, size=size)
    except GeocodingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
