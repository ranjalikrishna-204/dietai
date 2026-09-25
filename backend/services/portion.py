"""
Portion-size (gram) estimation.

This is a heuristic, not a scale: we do not have depth or a size reference (like a plate/coin)
in the photo, so grams are estimated from how much of the frame the food occupies, calibrated
around config.REF_AREA_FRAC ("a normal serving covers about this fraction of a typical food
photo"). Treat the resulting grams as an editable starting point - the accuracy improves a lot
if you later add a reference object (e.g. ask the user to include a standard-size plate) or a
depth-aware model such as the "Food Portion Benchmark" approach from the literature review.
"""
from __future__ import annotations

import config
from . import nutrition

MIN_MULTIPLIER = 0.3
MAX_MULTIPLIER = 3.0


def estimate_from_bbox_area(food: str, bbox_area_frac: float) -> float:
    """bbox_area_frac: (box width * box height) / (image width * image height), both normalised 0-1."""
    base_g = nutrition.get_food_row(food)["serving_g"]
    multiplier = bbox_area_frac / config.REF_AREA_FRAC if config.REF_AREA_FRAC else 1.0
    multiplier = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, multiplier))
    return round(base_g * multiplier, 1)


def default_portion(food: str) -> float:
    """Used when we only have a classification (no bounding box) - assume one standard serving."""
    return round(nutrition.get_food_row(food)["serving_g"], 1)
