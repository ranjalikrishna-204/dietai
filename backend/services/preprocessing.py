"""Image pre-processing shared by both the classifier and the optional detector.

Keep this module's defaults OFF (see config.PREPROCESS_*) unless you applied the exact same
step at training time - mismatched preprocessing between train and serve is one of the most
common causes of a model that "worked in the notebook" but performs badly in the app.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import config


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes -> RGB numpy array (H, W, 3)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image - is it a valid JPEG/PNG?")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_max_side(img: np.ndarray, max_side: int = config.MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def enhance_contrast_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(raw_bytes: bytes) -> np.ndarray:
    """Full pipeline used before every inference call."""
    img = load_image_bytes(raw_bytes)
    img = resize_max_side(img)
    if config.PREPROCESS_DENOISE:
        img = denoise(img)
    if config.PREPROCESS_CLAHE:
        img = enhance_contrast_clahe(img)
    return img


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)
