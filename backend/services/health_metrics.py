"""BMI and daily-calorie-target helpers (Mifflin-St Jeor equation)."""
from __future__ import annotations

ACTIVITY_MULTIPLIER = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    h_m = height_cm / 100
    return round(weight_kg / (h_m * h_m), 1)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def daily_kcal_target(age: int, sex: str, height_cm: float, weight_kg: float, activity_level: str) -> float:
    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # female / other -> use the female coefficient as a reasonable default
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return round(bmr * ACTIVITY_MULTIPLIER.get(activity_level, 1.55))
