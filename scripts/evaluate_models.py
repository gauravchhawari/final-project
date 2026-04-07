"""CLI helper to evaluate trained flood-risk models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.models.evaluate import evaluate_saved_models


if __name__ == "__main__":
    result = evaluate_saved_models()
    print(result)
