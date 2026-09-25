"""
Food recognition: an ONNX image classifier is the primary recognizer (one dominant dish per
photo -> label + confidence). If an optional fine-tuned YOLO detector checkpoint is present
(models/detector_best.pt, produced by ml/06_train_detector.py), it is used instead for photos
with multiple items (a thali / mixed plate), giving per-item boxes that also drive portion
estimation (see services/portion.py).

Both paths are lazy-loaded singletons so the backend starts even if only one model is trained.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

import config
from . import portion

ROOT = Path(__file__).resolve().parent.parent.parent
ONNX_PATH = ROOT / "models" / "classifier.onnx"
ONNX_META_PATH = ROOT / "models" / "classifier.json"
DETECTOR_PATH = ROOT / "models" / "detector_best.pt"

TOP_K = 3


class ClassifierUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _load_classifier():
    import onnxruntime as ort

    if not ONNX_PATH.exists():
        raise ClassifierUnavailable(
            f"{ONNX_PATH} not found. Train a model (ml/02_train_classifier.py) and export it "
            f"(ml/04_export_model.py) before running the backend."
        )
    meta = json.loads(ONNX_META_PATH.read_text())
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return session, meta


@functools.lru_cache(maxsize=1)
def _load_detector():
    if not DETECTOR_PATH.exists():
        return None
    from ultralytics import YOLO  # heavy import, done lazily so it's only paid for if used
    return YOLO(str(DETECTOR_PATH))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _classify(img_rgb: np.ndarray) -> list[dict]:
    session, meta = _load_classifier()
    size = meta["img_size"]

    import cv2
    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    mean, std = np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

    logits = session.run(["logits"], {"input": x})[0][0]
    probs = _softmax(logits)
    top_idx = np.argsort(probs)[::-1][:TOP_K]

    classes = meta["classes"]
    results = []
    for i in top_idx:
        food = classes[int(i)]
        results.append({
            "food": food,
            "display_name": food.replace("_", " ").title(),
            "confidence": round(float(probs[i]), 4),
            "portion_g": portion.default_portion(food),
            "bbox": None,
        })
    return results


def _detect(img_rgb: np.ndarray) -> list[dict]:
    model = _load_detector()
    h, w = img_rgb.shape[:2]
    preds = model.predict(img_rgb, conf=config.CONF_THRES, imgsz=config.IMG_SIZE, verbose=False)[0]

    results = []
    for box in preds.boxes:
        cls_id = int(box.cls.item())
        food = model.names[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalised
        area_frac = (x2 - x1) * (y2 - y1)
        results.append({
            "food": food,
            "display_name": food.replace("_", " ").title(),
            "confidence": round(conf, 4),
            "portion_g": portion.estimate_from_bbox_area(food, area_frac),
            "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
        })
    return results


def recognize(img_rgb: np.ndarray) -> list[dict]:
    """Returns a list of detections: [{food, display_name, confidence, portion_g, bbox}, ...]

    Uses the multi-item detector when it is trained and available; otherwise falls back to the
    single-dish classifier's top prediction (its runner-up guesses are kept out of the result so
    downstream nutrition/rules logic only ever sees one confident food from the classifier path).
    """
    detector = _load_detector()
    if detector is not None:
        dets = _detect(img_rgb)
        if dets:
            return dets
        # detector found nothing (e.g. plain single dish) -> fall back to classifier
    top = _classify(img_rgb)
    return top[:1] if top else []


def classify_with_alternatives(img_rgb: np.ndarray) -> list[dict]:
    """Exposes the classifier's top-K guesses (used by the frontend's 'not quite right?' picker)."""
    return _classify(img_rgb)
