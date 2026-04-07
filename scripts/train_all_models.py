"""CLI helper to train all flood-risk models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.models.train import train_and_save_models


if __name__ == "__main__":
    result = train_and_save_models()
    print(result)
