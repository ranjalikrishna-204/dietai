"""
Compare several architectures on the SAME dataset split with the SAME training recipe
(reviewer request: "comparative evaluations of multiple models using the same dataset").

For each architecture it
  1. trains with ml/02_train_classifier.py (identical epochs / batch size / seed / augmentation),
  2. evaluates the best checkpoint on the held-out test split:
     top-1, top-5, macro-F1, parameters, checkpoint size, CPU latency (batch = 1),
  3. writes models/reports/model_comparison.csv and model_comparison_table.tex.

Fairness rules (state these in the paper): one fixed split (data/processed/classification, built
with a fixed seed), same epochs, same optimiser/schedule/augmentation, same seed, ImageNet-pretrained
weights for every model, best-validation checkpoint used for testing.

Kaggle tip: a full run of all models can exceed one session. Train one model per session:
    python ml/08_compare_models.py --archs resnet50
    python ml/08_compare_models.py --archs densenet121
    ...
Each session leaves a small models/reports/cmp_<arch>.json. Put all of them in one models/reports/
folder (on your laptop is fine, no GPU/torch needed) and merge them with:
    python ml/08_compare_models.py --collect
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path


def _heavy():
    """Import torch & friends lazily so `--collect` works on a machine without torch."""
    global np, torch, f1_score, top_k_accuracy_score, DataLoader, datasets, transforms, train_mod
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, top_k_accuracy_score
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    train_mod = import_module("02_train_classifier")


ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
MODELS = ROOT / "models"
REPORTS = MODELS / "reports"
DATA = ROOT / "data" / "processed" / "classification"

PRETTY = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "mobilenet_v3_large": "MobileNetV3-Large",
          "efficientnet_v2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-Tiny"}


def ckpt_name(arch: str) -> str:
    return f"cmp_{arch}.pt"


def train_one(arch, epochs, batch, img_size, seed):
    cmd = [sys.executable, str(ML / "02_train_classifier.py"), "--arch", arch, "--epochs", str(epochs),
           "--batch-size", str(batch), "--img-size", str(img_size), "--seed", str(seed),
           "--ckpt-name", ckpt_name(arch)]
    print("\n>>>", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    minutes = (time.time() - t0) / 60
    (MODELS / f"cmp_{arch}_time.json").write_text(json.dumps({"train_minutes": round(minutes, 1)}))
    return minutes


def cpu_latency_ms(model, img_size, n=30):
    model = model.cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000


def evaluate(arch, batch):
    path = MODELS / ckpt_name(arch)
    ckpt = torch.load(path, map_location="cpu")
    classes, img_size = ckpt["classes"], ckpt["img_size"]
    model = train_mod.build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tf = transforms.Compose([transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize(train_mod.IMAGENET_MEAN, train_mod.IMAGENET_STD)])
    ds = datasets.ImageFolder(DATA / "test", transform=tf)
    assert ds.classes == classes, "class order mismatch between test split and checkpoint"
    logits, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
            logits.append(model(x.to(device)).cpu().numpy()); labels.append(y.numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    preds = logits.argmax(1)

    hist_path = MODELS / ckpt_name(arch).replace(".pt", "_history.json")
    best_val = max(h["val_acc"] for h in json.loads(hist_path.read_text())) if hist_path.exists() else float("nan")
    tmin = MODELS / f"cmp_{arch}_time.json"
    return {
        "arch": arch,
        "model": PRETTY.get(arch, arch),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "size_MB": round(path.stat().st_size / 1e6, 1),
        "val_acc": round(100 * best_val, 2),
        "top1": round(100 * float((preds == labels).mean()), 2),
        "top5": round(100 * float(top_k_accuracy_score(labels, logits, k=min(5, len(classes)),
                                                       labels=list(range(len(classes))))), 2),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "cpu_ms_per_img": round(cpu_latency_ms(model, img_size), 1),
        "train_minutes": json.loads(tmin.read_text())["train_minutes"] if tmin.exists() else "",
    }


def write_outputs(rows):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    best = max(r["top1"] for r in rows)
    lines = [r"\begin{table}[htbp]", r"\caption{Comparison of architectures on the same 90-class split (test set)}",
             r"\label{tab:model_comparison}", r"\centering", r"\small",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Params (M)} & \textbf{Top-1 (\%)} & \textbf{Top-5 (\%)} & \textbf{Macro-F1} & \textbf{CPU ms/img} \\",
             r"\midrule"]
    for r in rows:
        t1 = f"\\textbf{{{r['top1']:.2f}}}" if r["top1"] == best else f"{r['top1']:.2f}"
        lines.append(f"{r['model']} & {r['params_M']} & {t1} & {r['top5']:.2f} & {r['macro_f1']:.4f} & {r['cpu_ms_per_img']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORTS / "model_comparison_table.tex").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved: {REPORTS/'model_comparison.csv'} and model_comparison_table.tex")


ORDER = ["resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_v2_s", "convnext_tiny"]


def collect():
    """Merge models/reports/cmp_<arch>.json files (possibly produced in different sessions)."""
    rows = [json.loads(p.read_text()) for p in REPORTS.glob("cmp_*.json")]
    if not rows:
        raise SystemExit(f"No cmp_<arch>.json files found in {REPORTS}. Copy them there first.")
    rows.sort(key=lambda r: ORDER.index(r["arch"]) if r["arch"] in ORDER else 99)
    write_outputs(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_v2_s"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-training", action="store_true", help="only evaluate existing cmp_<arch>.pt checkpoints")
    ap.add_argument("--collect", action="store_true",
                    help="no training/evaluation: merge cmp_<arch>.json results from models/reports/ into the table")
    args = ap.parse_args()

    if args.collect:
        return collect()
    _heavy()
    if not (DATA / "test").exists():
        raise SystemExit("Run ml/01_prepare_classification_dataset.py first.")
    rows = []
    for arch in args.archs:
        if not args.skip_training:
            train_one(arch, args.epochs, args.batch_size, args.img_size, args.seed)
        if not (MODELS / ckpt_name(arch)).exists():
            print(f"[skip] {MODELS/ckpt_name(arch)} not found"); continue
        print(f"\nEvaluating {arch} ...")
        row = evaluate(arch, args.batch_size)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"cmp_{arch}.json").write_text(json.dumps(row, indent=2))   # survives across sessions
        rows.append(row)
    if rows:
        write_outputs(rows)


if __name__ == "__main__":
    main()
