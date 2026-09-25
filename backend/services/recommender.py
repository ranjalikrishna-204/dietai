"""Alternative-food suggestions (content-based + a small hand-built swap graph) and
disease/cooking-method guidance, filtered by rules_engine so a suggested swap is verified
to be at least as suitable as the original for the user's conditions before being shown."""
from __future__ import annotations

import functools
import json
from pathlib import Path

from . import nutrition, rules_engine

ROOT = Path(__file__).resolve().parent.parent.parent
SWAPS_JSON = ROOT / "data" / "knowledge" / "healthy_swaps.json"
TIPS_JSON = ROOT / "data" / "knowledge" / "cooking_tips.json"

_VERDICT_RANK = {"ok": 0, "caution": 1, "avoid": 2}


@functools.lru_cache(maxsize=1)
def load_swaps() -> dict:
    return json.loads(SWAPS_JSON.read_text())


@functools.lru_cache(maxsize=1)
def load_tips() -> dict:
    return json.loads(TIPS_JSON.read_text())


def _worst_verdict(flags: dict[str, dict]) -> str:
    if not flags:
        return "ok"
    return max((f["verdict"] for f in flags.values()), key=lambda v: _VERDICT_RANK[v])


def suggest_alternatives(food: str, conditions: list[str], diet_preference: str,
                          allergies: list[str], portion_g: float, max_results: int = 3) -> list[dict]:
    swaps = load_swaps()
    candidates = swaps.get(food, [])
    if not candidates:
        return []

    original_portion = nutrition.scale_to_portion(food, portion_g)
    original_flags = rules_engine.evaluate_food(food, original_portion, conditions)
    original_worst = _worst_verdict(original_flags)

    results = []
    for cand in candidates:
        try:
            row = nutrition.get_food_row(cand)
        except KeyError:
            continue
        if diet_preference in ("vegetarian", "vegan") and not nutrition.food_is_veg(cand):
            continue
        if allergies and nutrition.food_allergens(cand) & set(a.lower() for a in allergies):
            continue

        cand_portion = nutrition.scale_to_portion(cand, row["serving_g"])
        cand_flags = rules_engine.evaluate_food(cand, cand_portion, conditions)
        cand_worst = _worst_verdict(cand_flags)

        # Only recommend it if it's not worse than the original for this user.
        if _VERDICT_RANK[cand_worst] > _VERDICT_RANK[original_worst]:
            continue

        why_bits = []
        if cand_portion["kcal"] < original_portion["kcal"] * 0.85:
            why_bits.append("fewer calories")
        if cand_portion["fiber_g"] > original_portion["fiber_g"] * 1.2:
            why_bits.append("more fibre")
        if cand_portion["sat_fat_g"] < original_portion["sat_fat_g"] * 0.7:
            why_bits.append("less saturated fat")
        if cand_portion["sodium_mg"] < original_portion["sodium_mg"] * 0.7:
            why_bits.append("less sodium")
        if not why_bits:
            why_bits.append("better fit for your health profile")

        results.append({
            "food": cand,
            "display_name": row["display_name"],
            "why": ", ".join(why_bits),
        })
        if len(results) >= max_results:
            break
    return results


def cooking_tips_for(food: str, conditions: list[str]) -> list[str]:
    tips_db = load_tips()
    tips: list[str] = []
    tips.extend(tips_db.get("by_food", {}).get(food, []))
    for tag in nutrition.food_tags(food):
        tips.extend(tips_db.get("by_tag", {}).get(tag, []))
    for cond in conditions:
        tips.extend(tips_db.get("by_condition", {}).get(cond, []))
    # de-duplicate while preserving order
    seen, unique = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique
