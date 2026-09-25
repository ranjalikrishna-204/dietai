"""
Disease-condition-aware suitability engine.

Loads data/knowledge/disease_rules.json (editable, dietitian-reviewable thresholds) and
scores one portion of a food against each of the user's declared health conditions.

IMPORTANT: this is a rule-based decision-support tool built from general public dietary
guidance (ADA / AHA / WHO / ICMR-NIN style thresholds), not a substitute for a doctor or
registered dietitian - see disease_rules.json's "_note" field and the app's disclaimer.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from . import nutrition

ROOT = Path(__file__).resolve().parent.parent.parent
RULES_JSON = ROOT / "data" / "knowledge" / "disease_rules.json"


@functools.lru_cache(maxsize=1)
def load_rules() -> dict:
    return json.loads(RULES_JSON.read_text())


def evaluate_food(food: str, portion: dict, conditions: list[str]) -> dict[str, dict]:
    """
    portion: the dict returned by nutrition.scale_to_portion(food, grams)
    Returns {condition: {"verdict": "ok"|"caution"|"avoid", "reasons": [...]}}
    """
    rules = load_rules()
    tags = nutrition.food_tags(food)
    gi = nutrition.food_gi(food)
    out = {}
    conditions_to_check = conditions or ["general"]
    for cond in conditions_to_check:
        rule = rules.get(cond)
        if rule is None:
            continue
        reasons = []
        verdict = "ok"

        for nutrient, limit in rule.get("item_limits", {}).items():
            value = portion.get(nutrient)
            if value is not None and value > limit:
                verdict = "avoid" if value > limit * 1.5 else "caution"
                reasons.append(f"{nutrient.replace('_', ' ')} is {value:g} (limit for this condition: {limit:g} per serving)")

        if gi in rule.get("avoid_gi", []):
            verdict = "caution" if verdict == "ok" else verdict
            reasons.append("high glycaemic-index food")

        bad_tags = tags & set(rule.get("avoid_tags", []))
        if bad_tags:
            verdict = "caution" if verdict == "ok" else verdict
            reasons.append(f"contains: {', '.join(sorted(bad_tags))}")

        good_tags = tags & set(rule.get("prefer_tags", []))
        if not reasons and good_tags:
            reasons.append(f"good source of: {', '.join(sorted(good_tags))}")
        if not reasons:
            reasons.append("within recommended limits for this condition")

        out[cond] = {"verdict": verdict, "reasons": reasons, "label": rule.get("label", cond)}
    return out


def evaluate_meal(totals: dict, conditions: list[str]) -> dict[str, dict]:
    """Same idea as evaluate_food but against the whole-meal limits."""
    rules = load_rules()
    out = {}
    for cond in (conditions or ["general"]):
        rule = rules.get(cond)
        if rule is None:
            continue
        reasons, verdict = [], "ok"
        for nutrient, limit in rule.get("meal_limits", {}).items():
            value = totals.get(nutrient)
            if value is not None and value > limit:
                verdict = "avoid" if value > limit * 1.5 else "caution"
                reasons.append(f"meal {nutrient.replace('_',' ')} {value:g} exceeds the {limit:g} guideline")
        if not reasons:
            reasons.append("meal is within the recommended limits for this condition")
        out[cond] = {"verdict": verdict, "reasons": reasons, "label": rule.get("label", cond)}
    return out


def general_advice(conditions: list[str]) -> list[str]:
    rules = load_rules()
    advice = list(rules.get("general", {}).get("advice", []))
    for cond in conditions:
        advice.extend(rules.get(cond, {}).get("advice", []))
    return advice
