"""Loads data/knowledge/nutrition_db.csv and computes scaled nutrition for a given portion."""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
NUTRITION_CSV = ROOT / "data" / "knowledge" / "nutrition_db.csv"

_NUMERIC_COLS = ["serving_g", "kcal", "protein_g", "carb_g", "fat_g", "sat_fat_g", "fiber_g",
                  "sugar_g", "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vitamin_c_mg"]


@functools.lru_cache(maxsize=1)
def load_table() -> pd.DataFrame:
    df = pd.read_csv(NUTRITION_CSV)
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("food")


def get_food_row(food: str) -> dict:
    df = load_table()
    if food not in df.index:
        raise KeyError(f"'{food}' not found in nutrition_db.csv - add it there first.")
    return df.loc[food].to_dict()


def scale_to_portion(food: str, portion_g: float) -> dict:
    """Return per-portion nutrition values, linearly scaled from the reference serving size."""
    row = get_food_row(food)
    ref_g = row["serving_g"]
    factor = portion_g / ref_g if ref_g else 1.0
    scaled = {"food": food, "display_name": row["display_name"], "portion_g": round(portion_g, 1)}
    for col in ["kcal", "protein_g", "carb_g", "fat_g", "sat_fat_g", "fiber_g", "sugar_g",
                "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vitamin_c_mg"]:
        scaled[col] = round(row[col] * factor, 2)
    return scaled


def sum_meal(portions: list[dict]) -> dict:
    """Sum a list of scale_to_portion() dicts into meal totals."""
    keys = ["kcal", "protein_g", "carb_g", "fat_g", "sat_fat_g", "fiber_g", "sugar_g", "sodium_mg"]
    totals = {k: 0.0 for k in keys}
    for p in portions:
        for k in keys:
            totals[k] += p.get(k, 0.0)
    return {k: round(v, 2) for k, v in totals.items()}


def food_tags(food: str) -> set[str]:
    row = get_food_row(food)
    tags = str(row.get("tags") or "")
    return {t.strip() for t in tags.split(";") if t.strip()}


def food_gi(food: str) -> str:
    return str(get_food_row(food).get("gi") or "").strip().lower()


def food_allergens(food: str) -> set[str]:
    row = get_food_row(food)
    allergens = str(row.get("allergens") or "")
    return {a.strip() for a in allergens.split(";") if a.strip()}


def food_is_veg(food: str) -> bool:
    return bool(int(get_food_row(food).get("veg", 1)))


def all_foods() -> list[str]:
    return list(load_table().index)
