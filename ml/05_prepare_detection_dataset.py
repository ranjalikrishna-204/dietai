"""
Step 5 (optional) - prepare a YOLO-format dataset for MULTI-ITEM detection (thali / mixed-plate
photos with several dishes in one frame). The single-dish classifier from steps 1-4 is the
primary recognizer; this detector is only needed when you want per-item bounding boxes and
counts on a crowded plate.

WHERE TO GET AN ANNOTATED DATASET (bounding boxes, not just folder labels):
  - Hugging Face: SohlHealth/sohl-multidish-yolo-dataset (~377 images, 16 classes, already in
    YOLO format - images/ + labels/*.txt). Good starting point; extend with Roboflow Universe
    "indian food detection" projects (search roboflow.com/universe) which export directly to
    YOLO format with a data.yaml.
  - Or annotate your own photos with Roboflow / makesense.ai / CVAT (free, browser-based) -
    draw boxes around each item on the plate and export as "YOLOv8/YOLO11 PyTorch" format.

Expected input layout (whatever you download, arrange it like this under data/raw/detection/):
    data/raw/detection/
        images/*.jpg
        labels/*.txt        # YOLO format: class_id cx cy w h  (normalised 0-1)
        classes.txt         # one class name per line, in class_id order

This script remaps `classes.txt` entries to our canonical class names (class_map.json),
rewrites the label files with the new class ids, and performs an 80/10/10 train/val/test
split, writing everything (plus the required data.yaml) to data/processed/detection/.

Usage:
    python ml/05_prepare_detection_dataset.py
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from class_map import load_alias_table, normalise

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "detection"
OUT = ROOT / "data" / "processed" / "detection"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def main():
    images_dir, labels_dir, classes_txt = RAW / "images", RAW / "labels", RAW / "classes.txt"
    if not (images_dir.exists() and labels_dir.exists() and classes_txt.exists()):
        print(f"Expected {images_dir}, {labels_dir}, {classes_txt} - see this file's docstring "
              f"for where to download an annotated multi-dish dataset.")
        return

    raw_classes = [c.strip() for c in classes_txt.read_text().splitlines() if c.strip()]
    alias_table = load_alias_table()
    id_remap: dict[int, str] = {}
    for i, raw_name in enumerate(raw_classes):
        canonical = normalise(raw_name, alias_table)
        if canonical is None:
            print(f"WARNING: raw class '{raw_name}' has no entry in class_map.json - its boxes will be dropped.")
            continue
        id_remap[i] = canonical

    new_classes = sorted(set(id_remap.values()))
    new_id_of = {c: i for i, c in enumerate(new_classes)}

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(SEED)
    rng.shuffle(stems)
    n = len(stems)
    n_train, n_val = int(n * SPLITS["train"]), int(n * SPLITS["val"])
    split_of = {}
    for i, stem in enumerate(stems):
        split_of[stem] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    kept, dropped = 0, 0
    for stem in stems:
        split = split_of[stem]
        img_src = next(images_dir.glob(f"{stem}.*"))
        label_src = labels_dir / f"{stem}.txt"
        new_lines = []
        if label_src.exists():
            for line in label_src.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                canonical = id_remap.get(old_id)
                if canonical is None:
                    dropped += 1
                    continue
                parts[0] = str(new_id_of[canonical])
                new_lines.append(" ".join(parts))
                kept += 1
        shutil.copy2(img_src, OUT / split / "images" / img_src.name)
        (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(new_lines))

    data_yaml = {
        "path": str(OUT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: c for i, c in enumerate(new_classes)},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f"{len(stems)} images -> {OUT}  ({kept} boxes kept, {dropped} boxes dropped - unmapped classes)")
    print(f"Classes ({len(new_classes)}): {new_classes}")
    print(f"data.yaml written -> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
