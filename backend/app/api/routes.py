"""Aggregate API router for the flood risk system."""

import os
from pathlib import Path

from fastapi import APIRouter

from app.api.risk import router as risk_router
from app.api.routing import router as routing_router
from app.api.weather import router as weather_router
from app.utils.config import get_settings
from app.utils.constants import BASE_DIR, MODEL_METADATA_PATH

router = APIRouter()
router.include_router(weather_router)
router.include_router(risk_router)
router.include_router(routing_router)


@router.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Service health endpoint."""
    return {"status": "OK"}


@router.get("/api/config-check", tags=["system"])
def config_check() -> dict[str, object]:
    """Report missing keys and files required by the app."""
    settings = get_settings()
    env_path = BASE_DIR / ".env"
    env_file_present = env_path.exists()
    has_runtime_env = bool(
        os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENROUTESERVICE_API_KEY")
    )

    checks = {
        "env_file": {
            "ok": env_file_present or has_runtime_env,
            "path": str(env_path.resolve()),
            "message": (
                "Loaded from .env file."
                if env_file_present
                else "Runtime environment variables detected."
                if has_runtime_env
                else "Create a .env file from .env.example or configure environment variables."
            ),
        },
        "openweather_api_key": {
            "ok": bool(settings.openweather_api_key),
            "message": "Required for live weather and forecast." if not settings.openweather_api_key else "Configured.",
        },
        "openrouteservice_api_key": {
            "ok": bool(settings.openrouteservice_api_key),
            "message": "Required for routing and elevation." if not settings.openrouteservice_api_key else "Configured.",
        },
        "shelter_dataset": {
            "ok": settings.shelter_dataset_path.exists(),
            "path": str(settings.shelter_dataset_path.resolve()),
            "message": "Shelter CSV not found." if not settings.shelter_dataset_path.exists() else "Found.",
        },
        "model_metadata": {
            "ok": MODEL_METADATA_PATH.exists(),
            "path": str(MODEL_METADATA_PATH.resolve()),
            "message": "Train models first to create metadata." if not MODEL_METADATA_PATH.exists() else "Found.",
        },
    }

    overall_ok = all(item["ok"] for item in checks.values())
    return {"ok": overall_ok, "checks": checks}
