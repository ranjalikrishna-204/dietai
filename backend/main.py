"""
FastAPI backend for the AI-Based Health Condition-Aware Dietary Recommendation System.

Run with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root `dietai/` so the `config` and `backend` imports resolve).

Endpoints:
    GET  /health                      liveness probe
    POST /users                       create a user profile -> returns user_id, BMI, kcal target
    GET  /users/{user_id}             fetch a stored profile
    POST /analyze                     upload a food photo (+ user_id) -> full recommendation result
    POST /log                         save analyzed foods to today's meal log
    GET  /analytics/{user_id}/daily   today's (or ?date=YYYY-MM-DD) totals
    GET  /analytics/{user_id}/history last N days of logged meals (for charts)
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `python -m uvicorn backend.main:app` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from backend import schemas
from backend.services import (database, health_metrics, nutrition, portion,
                               preprocessing, recognizer, recommender, rules_engine)

database.init_db()  # idempotent (CREATE TABLE IF NOT EXISTS) - safe to call at import time too,
                     # which keeps the DB ready for both `uvicorn backend.main:app` and TestClient(app)

app = FastAPI(title="AI-Based Health-Aware Dietary Recommendation System", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/users", response_model=schemas.UserProfileOut)
def create_user(profile: schemas.UserProfileIn):
    user_id = database.create_user(profile.model_dump())
    bmi = health_metrics.compute_bmi(profile.weight_kg, profile.height_cm)
    target = health_metrics.daily_kcal_target(profile.age, profile.sex, profile.height_cm,
                                               profile.weight_kg, profile.activity_level)
    return schemas.UserProfileOut(
        **profile.model_dump(), user_id=user_id, bmi=bmi,
        bmi_category=health_metrics.bmi_category(bmi), daily_kcal_target=target,
    )


@app.get("/users/{user_id}", response_model=schemas.UserProfileOut)
def read_user(user_id: int):
    row = database.get_user(user_id)
    if row is None:
        raise HTTPException(404, "user not found")
    bmi = health_metrics.compute_bmi(row["weight_kg"], row["height_cm"])
    target = health_metrics.daily_kcal_target(row["age"], row["sex"], row["height_cm"],
                                               row["weight_kg"], row["activity_level"])
    return schemas.UserProfileOut(
        user_id=row["user_id"], name=row["name"], age=row["age"], sex=row["sex"],
        height_cm=row["height_cm"], weight_kg=row["weight_kg"], conditions=row["conditions"],
        allergies=row["allergies"], diet_preference=row["diet_preference"],
        activity_level=row["activity_level"], bmi=bmi, bmi_category=health_metrics.bmi_category(bmi),
        daily_kcal_target=target,
    )


def _build_analysis(detections: list[dict], conditions: list[str], diet_preference: str,
                     allergies: list[str]) -> schemas.AnalyzeResult:
    nutrition_rows, suitability, alternatives, cooking_tips = [], {}, {}, {}

    for det in detections:
        food = det["food"]
        try:
            n_row = nutrition.scale_to_portion(food, det["portion_g"])
        except KeyError:
            continue  # detector/classifier predicted a class with no nutrition entry yet
        nutrition_rows.append(n_row)

        flags = rules_engine.evaluate_food(food, n_row, conditions)
        suitability[food] = [
            schemas.SuitabilityFlag(condition=c, verdict=f["verdict"], reasons=f["reasons"])
            for c, f in flags.items()
        ]
        alternatives[food] = [
            schemas.AlternativeSuggestion(**alt)
            for alt in recommender.suggest_alternatives(food, conditions, diet_preference, allergies, det["portion_g"])
        ]
        cooking_tips[food] = recommender.cooking_tips_for(food, conditions)

    meal_totals = nutrition.sum_meal(nutrition_rows)

    return schemas.AnalyzeResult(
        detections=[schemas.DetectedItem(**d) for d in detections],
        nutrition=[schemas.NutritionBreakdown(**n) for n in nutrition_rows],
        meal_totals=meal_totals,
        suitability=suitability,
        alternatives=alternatives,
        cooking_tips=cooking_tips,
        general_advice=rules_engine.general_advice(conditions),
    )


@app.post("/analyze", response_model=schemas.AnalyzeResult)
async def analyze(
    user_id: int = Form(...),
    image: UploadFile = File(...),
    manual_food: str | None = Form(None),
    manual_portion_g: float | None = Form(None),
):
    """
    Analyze one uploaded food photo. If the auto-detected food is wrong, the frontend can
    resend with `manual_food` (one of the canonical class names) to override the recognizer
    without re-uploading, and/or `manual_portion_g` to correct the estimated serving size.
    """
    user = database.get_user(user_id)
    if user is None:
        raise HTTPException(404, "user not found")

    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "please upload a JPEG or PNG image")
    raw = await image.read()
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"image too large (max {config.MAX_UPLOAD_MB} MB)")

    if manual_food:
        try:
            portion_g = manual_portion_g or portion.default_portion(manual_food)
            detections = [{
                "food": manual_food, "display_name": manual_food.replace("_", " ").title(),
                "confidence": 1.0, "portion_g": portion_g, "bbox": None,
            }]
        except KeyError:
            raise HTTPException(400, f"unknown food '{manual_food}'")
    else:
        try:
            img = preprocessing.preprocess(raw)
        except ValueError as e:
            raise HTTPException(400, str(e))
        try:
            detections = recognizer.recognize(img)
        except recognizer.ClassifierUnavailable as e:
            raise HTTPException(503, str(e))
        if not detections:
            raise HTTPException(422, "could not recognise any food in this image - try a clearer, closer photo")

    return _build_analysis(detections, user["conditions"], user["diet_preference"], user["allergies"])


@app.post("/log")
def log_meal(payload: schemas.MealLogIn):
    if database.get_user(payload.user_id) is None:
        raise HTTPException(404, "user not found")
    if len(payload.foods) != len(payload.portions_g):
        raise HTTPException(400, "foods and portions_g must be the same length")
    for food, grams in zip(payload.foods, payload.portions_g):
        try:
            row = nutrition.scale_to_portion(food, grams)
        except KeyError:
            raise HTTPException(400, f"unknown food '{food}'")
        database.log_meal_item(payload.user_id, row)
    return {"status": "logged", "items": len(payload.foods)}


@app.get("/analytics/{user_id}/daily", response_model=schemas.DailySummary)
def analytics_daily(user_id: int, date: str | None = None):
    if database.get_user(user_id) is None:
        raise HTTPException(404, "user not found")
    return database.daily_summary(user_id, date)


@app.get("/analytics/{user_id}/history")
def analytics_history(user_id: int, days: int = 14):
    if database.get_user(user_id) is None:
        raise HTTPException(404, "user not found")
    return database.history(user_id, days)


@app.get("/foods")
def list_foods():
    """All canonical foods known to the nutrition DB (used by the frontend's manual-override picker)."""
    return sorted(nutrition.all_foods())
