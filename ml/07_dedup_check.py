"""
Duplicate / leakage audit for the merged food dataset (camera-ready item: "document the
duplicate-prevention mechanism used when merging datasets").

Two independent detectors:
  1. EXACT duplicates   - SHA-256 of the file bytes.
  2. NEAR duplicates    - 64-bit difference hash (dHash) with Hamming distance <= --max-dist
                          (catches re-encoded, resized or slightly cropped copies).

Two audit modes
  --mode raw        scans data/raw/<source>/**/<class>/*.jpg BEFORE the split. Reports duplicates
                    inside a source and ACROSS sources (the merge risk).
  --mode processed  scans data/processed/classification/{train,val,test}/<class>/*.jpg.
                    Reports duplicates that straddle splits (train/test LEAKAGE), which is what
                    would inflate reported accuracy.

Outputs (in models/reports/): dedup_report_<mode>.json and remove_list_<mode>.txt.
Nothing is deleted unless you pass --apply (raw mode only). After removing files, rebuild the
dataset with ml/01_prepare_classification_dataset.py and retrain.

Usage:
    python ml/07_dedup_check.py --mode raw
    python ml/07_dedup_check.py --mode processed
    python ml/07_dedup_check.py --mode raw --apply        # delete the files in the remove list
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed" / "classification"
REPORTS = ROOT / "models" / "reports"
EXT = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return int("".join("1" if b else "0" for b in bits), 2)
    except Exception:
        return None


def collect(mode: str) -> list[dict]:
    """Return [{path, group, cls}] where group = source (raw mode) or split (processed mode)."""
    items = []
    if mode == "raw":
        for src in sorted(p for p in RAW.iterdir() if p.is_dir()):
            for f in src.rglob("*"):
                if f.suffix.lower() in EXT:
                    items.append({"path": f, "group": src.name, "cls": f.parent.name})
    else:
        for split in ("train", "val", "test"):
            for f in (PROC / split).rglob("*"):
                if f.suffix.lower() in EXT:
                    items.append({"path": f, "group": split, "cls": f.parent.name})
    return items


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def find_groups(items: list[dict], max_dist: int):
    """Union-find over exact (sha) and near (dhash) matches."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    exact_pairs = set()
    by_sha = defaultdict(list)
    for i, it in enumerate(items):
        it["sha"] = sha256(it["path"])
        by_sha[it["sha"]].append(i)
    for idx in by_sha.values():
        for j in idx[1:]:
            union(idx[0], j)
            exact_pairs.add(frozenset((idx[0], j)))

    hashes = []
    valid = []
    for i, it in enumerate(items):
        h = dhash(it["path"])
        it["dhash"] = h
        if h is not None:
            hashes.append(h)
            valid.append(i)
    arr = np.array(hashes, dtype=np.uint64)
    for a in range(len(arr) - 1):
        d = popcount64(arr[a + 1:] ^ arr[a])
        for off in np.nonzero(d <= max_dist)[0]:
            union(valid[a], valid[a + 1 + int(off)])

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1], by_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "processed"], default="raw")
    ap.add_argument("--max-dist", type=int, default=4, help="max Hamming distance of dHash for 'near duplicate'")
    ap.add_argument("--apply", action="store_true", help="delete files in the remove list (raw mode only)")
    args = ap.parse_args()

    items = collect(args.mode)
    if not items:
        raise SystemExit("No images found - check data/raw or data/processed.")
    print(f"Scanning {len(items)} images in '{args.mode}' mode ...")
    dup_groups, _ = find_groups(items, args.max_dist)

    # keep priority: raw mode keeps the file from the first source alphabetically ('indian' < 'massive_clean')
    report_groups, remove, cross_group, cross_class = [], [], 0, 0
    for g in dup_groups:
        members = sorted(g, key=lambda i: (items[i]["group"], str(items[i]["path"])))
        groups_in = sorted({items[i]["group"] for i in members})
        classes_in = sorted({items[i]["cls"] for i in members})
        exact = len({items[i]["sha"] for i in members}) < len(members)
        if len(groups_in) > 1:
            cross_group += 1
        if len(classes_in) > 1:
            cross_class += 1
        for i in members[1:]:
            remove.append(str(items[i]["path"]))
        report_groups.append({
            "kind": "exact+near" if exact else "near",
            "n_files": len(members),
            "groups": groups_in,
            "classes": classes_in,
            "files": [str(items[i]["path"].relative_to(ROOT)) for i in members],
        })

    label = "sources" if args.mode == "raw" else "splits"
    summary = {
        "mode": args.mode,
        "images_scanned": len(items),
        "max_hamming_distance": args.max_dist,
        "duplicate_groups": len(dup_groups),
        f"groups_spanning_multiple_{label}": cross_group,
        "groups_spanning_multiple_classes (label noise)": cross_class,
        "files_to_remove": len(remove),
        "pct_of_dataset": round(100 * len(remove) / len(items), 3),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"dedup_report_{args.mode}.json").write_text(json.dumps({"summary": summary, "groups": report_groups}, indent=2))
    (REPORTS / f"remove_list_{args.mode}.txt").write_text("\n".join(remove))
    print(json.dumps(summary, indent=2))
    print(f"Report -> {REPORTS / f'dedup_report_{args.mode}.json'}")

    if args.apply:
        if args.mode != "raw":
            raise SystemExit("--apply is only allowed in raw mode (delete from raw, then rebuild the split).")
        for p in remove:
            Path(p).unlink(missing_ok=True)
        print(f"Deleted {len(remove)} files. Now re-run ml/01_prepare_classification_dataset.py and retrain.")


if __name__ == "__main__":
    main()
