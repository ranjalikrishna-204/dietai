"""End-to-end API tests using FastAPI's TestClient (requires models/classifier.onnx +
models/classifier.json to exist - see README "Quick start without training" for a dummy model)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
import config

client = TestClient(app)


@pytest.fixture
def user_id():
    payload = {
        "name": "Test User", "age": 30, "sex": "female", "height_cm": 165, "weight_kg": 65,
        "conditions": ["diabetes"], "allergies": [], "diet_preference": "none", "activity_level": "moderate",
    }
    r = client.post("/users", json=payload)
    assert r.status_code == 200
    return r.json()["user_id"]


def _fake_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color=(200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_read_user(user_id):
    r = client.get(f"/users/{user_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["bmi"] > 0
    assert body["daily_kcal_target"] > 0


def test_analyze_with_manual_override(user_id):
    files = {"image": ("food.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    data = {"user_id": user_id, "manual_food": "jalebi"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert body["nutrition"][0]["food"] == "jalebi"
    assert body["suitability"]["jalebi"][0]["condition"] == "diabetes"


@pytest.mark.skipif(not config.MODELS_DIR.joinpath("classifier.onnx").exists(),
                     reason="no trained/dummy classifier model present")
def test_analyze_with_auto_recognition(user_id):
    files = {"image": ("food.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    data = {"user_id": user_id}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    assert len(r.json()["detections"]) >= 1


def test_log_and_daily_summary(user_id):
    r = client.post("/log", json={"user_id": user_id, "foods": ["idli", "sambar"], "portions_g": [90, 150]})
    assert r.status_code == 200
    r2 = client.get(f"/analytics/{user_id}/daily")
    assert r2.status_code == 200
    assert r2.json()["meals_logged"] == 2


def test_analyze_unknown_user_returns_404():
    files = {"image": ("food.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    r = client.post("/analyze", files=files, data={"user_id": 999999, "manual_food": "idli"})
    assert r.status_code == 404


def test_list_foods():
    r = client.get("/foods")
    assert r.status_code == 200
    assert "idli" in r.json()
