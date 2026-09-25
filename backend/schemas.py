"""Pydantic request/response models shared by the FastAPI routes."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

CONDITIONS = ["diabetes", "hypertension", "obesity", "heart_disease", "kidney_disease", "pcos"]


class UserProfileIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    age: int = Field(..., ge=1, le=120)
    sex: str = Field(..., pattern="^(male|female|other)$")
    height_cm: float = Field(..., gt=50, le=250)
    weight_kg: float = Field(..., gt=10, le=400)
    conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    diet_preference: str = Field("none", pattern="^(none|vegetarian|vegan|eggetarian)$")
    activity_level: str = Field("moderate", pattern="^(sedentary|light|moderate|active|very_active)$")


class UserProfileOut(UserProfileIn):
    user_id: int
    bmi: float
    bmi_category: str
    daily_kcal_target: float


class DetectedItem(BaseModel):
    food: str
    display_name: str
    confidence: float
    portion_g: float
    bbox: Optional[list[float]] = None   # [x1, y1, x2, y2] normalised 0-1, only when the detector ran


class NutritionBreakdown(BaseModel):
    food: str
    display_name: str
    portion_g: float
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    sat_fat_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float


class SuitabilityFlag(BaseModel):
    condition: str
    verdict: str          # "ok" | "caution" | "avoid"
    reasons: list[str]


class AlternativeSuggestion(BaseModel):
    food: str
    display_name: str
    why: str


class AnalyzeResult(BaseModel):
    detections: list[DetectedItem]
    nutrition: list[NutritionBreakdown]
    meal_totals: dict[str, float]
    suitability: dict[str, list[SuitabilityFlag]]     # food -> flags per condition
    alternatives: dict[str, list[AlternativeSuggestion]]  # food -> suggested swaps
    cooking_tips: dict[str, list[str]]                # food -> tips
    general_advice: list[str]


class MealLogIn(BaseModel):
    user_id: int
    foods: list[str]
    portions_g: list[float]


class DailySummary(BaseModel):
    date: str
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    sodium_mg: float
    sugar_g: float
    meals_logged: int
