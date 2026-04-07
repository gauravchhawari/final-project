"""Flood risk API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.evaluate import evaluate_saved_models
from app.models.predict import PredictionError, list_available_models, predict_risk
from app.models.train import TrainingError, train_and_save_models

router = APIRouter(prefix="/api/risk", tags=["risk"])


class RiskPredictionRequest(BaseModel):
    """Payload for flood risk prediction."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    live_rainfall_mm_hr: float | None = Field(default=None, ge=0)
    live_water_level_m: float | None = Field(default=None, ge=0)
    model_name: str = "xgb"


class TrainRequest(BaseModel):
    """Payload for model training."""

    dataset_path: str | None = None
    target_column: str = "flood_label"
    timestamp_column: str = "datetime"


class EvaluateRequest(BaseModel):
    """Payload for model evaluation."""

    dataset_path: str | None = None
    target_column: str = "flood_label"
    timestamp_column: str = "datetime"


@router.post("/predict")
def predict(payload: RiskPredictionRequest) -> dict[str, object]:
    """Predict flood risk using a trained model and live context."""
    try:
        return predict_risk(
            latitude=payload.latitude,
            longitude=payload.longitude,
            live_rainfall_mm_hr=payload.live_rainfall_mm_hr,
            live_water_level_m=payload.live_water_level_m,
            model_name=payload.model_name,
        )
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models")
def models() -> dict[str, object]:
    """List saved models and their metadata."""
    try:
        return {"models": list_available_models()}
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train")
def train(payload: TrainRequest) -> dict[str, object]:
    """Train all configured models and persist them."""
    try:
        return train_and_save_models(
            dataset_path=payload.dataset_path,
            target_column=payload.target_column,
            timestamp_column=payload.timestamp_column,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate")
def evaluate(payload: EvaluateRequest) -> dict[str, object]:
    """Evaluate all saved models on the supplied dataset."""
    try:
        return evaluate_saved_models(
            dataset_path=payload.dataset_path,
            target_column=payload.target_column,
            timestamp_column=payload.timestamp_column,
        )
    except (PredictionError, TrainingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
