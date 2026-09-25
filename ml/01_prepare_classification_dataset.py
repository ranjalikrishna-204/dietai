"""
Step 1 - build a clean, split image-classification dataset for food recognition.

WHERE TO GET IMAGES (pick one or combine several; more images = better accuracy):
  1. Kaggle: "Indian Food Images Dataset" (iamsouravbanerjee/indian-food-images-dataset), ~4000 images, 80 classes.
     kaggle datasets download -d iamsouravbanerjee/indian-food-images-dataset
  2. Kaggle: "Food-101" (dansbecker/food-101 or kmader/food41), 101,000 images, 101 classes (global dishes -
     useful extra classes such as pizza / burger / fries / ice_cream).
     kaggle datasets download -d kmader/food41
  3. Hugging Face: rajistics/indian_food_images (20 classes) - good small starter set.
       from datasets import load_dataset
       ds = load_dataset("rajistics/indian_food_images")

Whatever you download, unzip it under data/raw/<source_name>/<class_name>/*.jpg
(one sub-folder per raw label - that is the standard layout all three sources above use).

This script then:
  1. Walks every folder under data/raw/*, maps each raw folder name to our canonical class
     (data/knowledge/class_map.json) - unmapped folders are skipped with a warning so you
     can add them to class_map.json and re-run.
  2. Filters out corrupt / too-small images.
  3. Splits each class 80/10/10 into train/val/test (stratified, seeded).
  4. Writes data/processed/classification/{train,val,test}/<class>/*.jpg  and classes.json.

Usage:
    python ml/01_prepare_classification_dataset.py
    python ml/01_prepare_classification_dataset.py --list-labels   # just show raw folder names found
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from class_map import load_alias_table, normalise

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "classification"
MIN_SIDE = 64          # discard images smaller than this (likely icons/corrupt)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def find_raw_images() -> dict[str, list[Path]]:
    """raw_label -> list of image paths, gathered from every dataset dropped under data/raw/*/<label>/*.*"""
    by_label: dict[str, list[Path]] = defaultdict(list)
    if not RAW_DIR.exists():
        return by_label
    for source_dir in RAW_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for label_dir in source_dir.rglob("*"):
            if not label_dir.is_dir():
                continue
            imgs = [p for p in label_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if imgs:
                by_label[label_dir.name].extend(imgs)
    return by_label


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            return w >= MIN_SIDE and h >= MIN_SIDE
    except Exception:
        return False


def main(list_labels: bool = False) -> None:
    by_label = find_raw_images()
    if not by_label:
        print(f"No images found under {RAW_DIR}. Download a dataset first (see the docstring at the top of "
              f"this file) and unzip it as data/raw/<source>/<class_name>/*.jpg")
        return

    if list_labels:
        for label, imgs in sorted(by_label.items()):
            print(f"{label:35s} {len(imgs):6d} images")
        return

    alias_table = load_alias_table()
    canonical_imgs: dict[str, list[Path]] = defaultdict(list)
    unknown = defaultdict(int)
    for raw_label, imgs in by_label.items():
        canonical = normalise(raw_label, alias_table)
        if canonical is None:
            unknown[raw_label] += len(imgs)
            continue
        canonical_imgs[canonical].extend(imgs)

    if unknown:
        print("Skipped raw labels with no entry in class_map.json (add aliases there to include them):")
        for label, n in sorted(unknown.items(), key=lambda x: -x[1]):
            print(f"  {label:35s} {n:6d} images")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in SPLITS:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    kept_counts: dict[str, dict[str, int]] = {}
    for cls, paths in sorted(canonical_imgs.items()):
        valid = [p for p in paths if is_valid_image(p)]
        rng.shuffle(valid)
        n = len(valid)
        if n < 10:
            print(f"WARNING: class '{cls}' has only {n} valid images - consider adding more before training.")
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        split_slices = {
            "train": valid[:n_train],
            "val": valid[n_train:n_train + n_val],
            "test": valid[n_train + n_val:],
        }
        kept_counts[cls] = {}
        for split, files in split_slices.items():
            dest_dir = OUT_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(files):
                dest = dest_dir / f"{cls}_{i:05d}{src.suffix.lower()}"
                shutil.copy2(src, dest)
            kept_counts[cls][split] = len(files)

    classes = sorted(canonical_imgs.keys())
    (OUT_DIR / "classes.json").write_text(json.dumps(classes, indent=2))
    (OUT_DIR / "dataset_report.json").write_text(json.dumps(kept_counts, indent=2))

    total = sum(sum(v.values()) for v in kept_counts.values())
    print(f"\nDone. {len(classes)} classes, {total} images -> {OUT_DIR}")
    print("Per-class counts written to data/processed/classification/dataset_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-labels", action="store_true", help="print raw folder names found under data/raw/ and exit")
    args = ap.parse_args()
    main(list_labels=args.list_labels)
