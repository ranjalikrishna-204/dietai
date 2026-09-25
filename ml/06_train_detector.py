"""
Step 6 (optional) - fine-tune a YOLO detector for multi-item plate photos.

Model choice: YOLO26 (Ultralytics, released Jan 2026) is used here instead of the YOLOv12
mentioned in the original proposal, because YOLO26 is natively NMS-free (lower, more
consistent latency - important when this runs per photo on a phone/web upload), removes
Distribution Focal Loss for cleaner ONNX/TensorRT export, and is the current Ultralytics
flagship with the best small-object accuracy of the family. If `yolo26n.pt` is not available
in your installed Ultralytics version, fall back to `yolo11n.pt` (2024 release, very well
supported) - both are drop-in via the --weights flag.

Usage:
    python ml/06_train_detector.py --weights yolo26n.pt --epochs 80
    python ml/06_train_detector.py --weights yolo11n.pt --epochs 80   # fallback
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "processed" / "detection" / "data.yaml"
MODELS_DIR = ROOT / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt", help="starting checkpoint (nano = fastest to train)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found - run ml/05_prepare_detection_dataset.py first.")

    model = YOLO(args.weights)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs" / "detect"),
        name="food_detector",
        patience=20,
    )
    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test-set metrics:", metrics.results_dict)

    best = ROOT / "runs" / "detect" / "food_detector" / "weights" / "best.pt"
    dest = MODELS_DIR / "detector_best.pt"
    if best.exists():
        dest.write_bytes(best.read_bytes())
        print(f"Copied best weights -> {dest}")


if __name__ == "__main__":
    main()
