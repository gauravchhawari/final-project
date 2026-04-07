"""Evacuation routing API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.routing_service import RoutingService, RoutingServiceError

router = APIRouter(prefix="/api/routing", tags=["routing"])


class EvacuationRequest(BaseModel):
    """Request payload for route optimization."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    region: str | None = None
    max_candidates: int = Field(default=5, ge=1, le=20)
    live_water_level_m: float | None = Field(default=None, ge=0)
    live_rainfall_mm_hr: float | None = Field(default=None, ge=0)
    model_name: str | None = None


@router.post("/evacuate")
def evacuate(payload: EvacuationRequest) -> dict[str, object]:
    """Return the best shelter and route for the given origin."""
    service = RoutingService()
    try:
        return service.get_best_route(
            latitude=payload.latitude,
            longitude=payload.longitude,
            region=payload.region,
            max_candidates=payload.max_candidates,
            live_water_level_m=payload.live_water_level_m,
            live_rainfall_mm_hr=payload.live_rainfall_mm_hr,
            model_name=payload.model_name,
        )
    except RoutingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
