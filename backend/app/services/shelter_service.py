"""Shelter lookup and filtering."""

from __future__ import annotations

from typing import Any

from app.services.data_loader import load_shelter_dataset
from app.services.feature_engineering import haversine_distance_km


class ShelterServiceError(RuntimeError):
    """Raised when shelters cannot be loaded or filtered."""


class ShelterService:
    """Load shelters from a dataset and rank them by proximity."""

    required_columns = {"name", "latitude", "longitude"}

    def find_nearest_shelters(
        self,
        latitude: float,
        longitude: float,
        region: str | None,
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        dataset = load_shelter_dataset()
        missing = self.required_columns - set(dataset.columns)
        if missing:
            raise ShelterServiceError(
                f"Shelter dataset is missing required columns: {sorted(missing)}."
            )

        filtered = dataset.copy()
        if region and "region" in filtered.columns:
            region_filtered = filtered[
                filtered["region"].astype(str).str.lower() == region.lower()
            ]
            if not region_filtered.empty:
                filtered = region_filtered
        if "status" in filtered.columns:
            filtered = filtered[
                filtered["status"].fillna("active").astype(str).str.lower() == "active"
            ]
        if filtered.empty:
            raise ShelterServiceError("No active shelters found for the requested region.")

        filtered["distance_km"] = filtered.apply(
            lambda row: haversine_distance_km(
                latitude,
                longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )
        ranked = filtered.sort_values("distance_km").head(max_candidates)
        return ranked.to_dict(orient="records")
