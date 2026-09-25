"""
Step 2 - train the food-recognition classifier.

Model: EfficientNetV2-S (torchvision), ImageNet-pretrained, fine-tuned on our food classes.
EfficientNetV2 was chosen over training a YOLO backbone from scratch because our task is
"one dominant dish per photo -> label" (classification), which trains faster, needs far less
labelled data, and reaches higher top-1 accuracy than a from-scratch detector for this use case.
(See ml/04_prepare_detection_dataset.py + ml/05_train_detector.py for the optional YOLO26
multi-item detector used for thali/mixed-plate photos.)

Run this on a GPU (Google Colab free tier or Kaggle Notebooks free tier both work well - see
README.md "Which platform should I use?").

Usage:
    python ml/02_train_classifier.py --epochs 15 --batch-size 32 --arch efficientnet_v2_s
    python ml/02_train_classifier.py --epochs 8  --arch mobilenet_v3_small   # smaller/faster, better for edge/mobile
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "classification"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m


def get_loaders(img_size: int, batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ld, val_ld, train_ds.classes


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_v2_s",
                     choices=["efficientnet_v2_s", "mobilenet_v3_small", "mobilenet_v3_large"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2,
                     help="train only the new classifier head for this many epochs first")
    args = ap.parse_args()

    if not (DATA_DIR / "train").exists():
        raise SystemExit(f"{DATA_DIR/'train'} not found - run ml/01_prepare_classification_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ld, val_ld, classes = get_loaders(args.img_size, args.batch_size, args.workers)
    print(f"{len(classes)} classes: {classes}")

    model = build_model(args.arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        # Freeze the pretrained backbone for the first few epochs so the new head
        # stabilises before we fine-tune the whole network (avoids destroying pretrained
        # ImageNet features early on with a randomly-initialised head's large gradients).
        for p in model.parameters():
            p.requires_grad = True
        if epoch < args.freeze_backbone_epochs:
            for name, p in model.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_ld, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_ld, device, criterion, optimizer=None)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  ({dt:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "arch": args.arch,
                        "classes": classes, "img_size": args.img_size},
                       MODELS_DIR / "classifier_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.3f})")

    (MODELS_DIR / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val accuracy: {best_acc:.3f}. Model saved to models/classifier_best.pt")


if __name__ == "__main__":
    main()
