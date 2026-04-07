"""Saved reporting artifacts for model evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[3] / ".mplconfig")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.utils.constants import REPORTS_DIR


def save_evaluation_reports(results: dict[str, Any]) -> dict[str, str]:
    """Persist comparison metrics and confusion matrix figures."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "evaluation_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics_rows = [
        {
            "model_name": item["model_name"],
            "accuracy": item["accuracy"],
            "f1_weighted": item["f1_weighted"],
        }
        for item in results.get("models", [])
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    csv_path = REPORTS_DIR / "model_comparison.csv"
    metrics_df.to_csv(csv_path, index=False)

    comparison_chart_path = REPORTS_DIR / "model_comparison.png"
    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        chart_df = metrics_df.set_index("model_name")[["accuracy", "f1_weighted"]]
        chart_df.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Comparison")
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(comparison_chart_path, dpi=150)
        plt.close(fig)

    saved_files = {
        "evaluation_json": str(json_path),
        "comparison_csv": str(csv_path),
        "comparison_chart": str(comparison_chart_path),
    }

    for item in results.get("models", []):
        matrix = item.get("confusion_matrix", [])
        if not matrix:
            continue
        matrix_path = REPORTS_DIR / f"confusion_matrix_{item['model_name']}.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {item['model_name']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(matrix_path, dpi=150)
        plt.close(fig)
        saved_files[f"confusion_matrix_{item['model_name']}"] = str(matrix_path)

    return saved_files
