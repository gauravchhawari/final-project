"""Project constants."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODEL_DIR = BASE_DIR / "backend" / "app" / "models" / "saved"
REPORTS_DIR = BASE_DIR / "backend" / "app" / "models" / "reports"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_OUTPUTS = {
    "rf": MODEL_DIR / "rf_model.pkl",
    "xgb": MODEL_DIR / "xgb_model.pkl",
    "lstm": MODEL_DIR / "lstm_model.h5",
}
