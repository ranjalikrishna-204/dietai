"""Central configuration. Override any value with an environment variable or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
KNOW_DIR = DATA_DIR / "knowledge"
MODELS_DIR = ROOT / "models"

# --- model / inference -------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best.pt"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))     # detection confidence threshold
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))

# --- image pre-processing (only enable if the SAME step was used during training) ---
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))
PREPROCESS_DENOISE = os.getenv("PREPROCESS_DENOISE", "0") == "1"
PREPROCESS_CLAHE = os.getenv("PREPROCESS_CLAHE", "0") == "1"

# --- portion estimation ------------------------------------------------------
# fraction of the photo a "normal serving" is assumed to cover (heuristic, tune on your data)
REF_AREA_FRAC = float(os.getenv("REF_AREA_FRAC", "0.25"))

# --- backend / database ------------------------------------------------------
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "dietai.db"))
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "7"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
