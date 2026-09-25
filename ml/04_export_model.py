"""
Step 4 (optional but recommended) - export the trained classifier to ONNX for fast,
dependency-light inference in the backend (no torch/torchvision needed at serve time).

Usage:
    python ml/04_export_model.py --checkpoint models/classifier_best.pt --out models/classifier.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from importlib import import_module
build_model = import_module("02_train_classifier").build_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "classifier_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "classifier.onnx"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_size = ckpt["img_size"]
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    # Save the class list + preprocessing config next to the model so the backend
    # never has to guess it.
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps({
        "classes": ckpt["classes"],
        "img_size": img_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }, indent=2))
    print(f"Exported ONNX model -> {args.out}")
    print(f"Exported metadata   -> {meta_path}")


if __name__ == "__main__":
    main()
