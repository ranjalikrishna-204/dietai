"""
Step 3 - evaluate the trained classifier on the held-out test split.

Reports: overall accuracy, top-5 accuracy, per-class precision/recall/F1, and a confusion
matrix image - the same kind of table shown in the project's literature-review slides
(Precision / F1-Score / mAP-style metrics), so results can be compared directly.

Usage:
    python ml/03_evaluate_classifier.py --checkpoint models/classifier_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from importlib import import_module

_train_mod = import_module("02_train_classifier")
build_model = _train_mod.build_model
IMAGENET_MEAN = _train_mod.IMAGENET_MEAN
IMAGENET_STD = _train_mod.IMAGENET_STD

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "classification"
REPORTS_DIR = ROOT / "models" / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "classifier_best.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)
    assert test_ds.classes == classes, "test-set class order does not match the trained model's classes"
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in test_ld:
            out = model(x.to(device))
            all_logits.append(out.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = logits.argmax(1)

    top1 = (preds == labels).mean()
    k5 = min(5, len(classes))
    top5 = top_k_accuracy_score(labels, logits, k=k5, labels=list(range(len(classes))))

    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    (REPORTS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.4), max(6, len(classes) * 0.4)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-{k5} accuracy: {top5:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Full report -> {REPORTS_DIR/'classification_report.json'}")
    print(f"Confusion matrix -> {REPORTS_DIR/'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
