"""
Step 2 - train the food-recognition classifier.

Model: EfficientNetV2-S (torchvision), ImageNet-pretrained, fine-tuned on our food classes.
EfficientNetV2 was chosen over training a YOLO backbone from scratch because our task is
"one dominant dish per photo -> label" (classification), which trains faster, needs far less
labelled data, and reaches higher top-1 accuracy than a from-scratch detector for this use case.

v2 additions (for small per-class datasets, ~40-50 images/class):
  - mixup augmentation (--mixup-alpha, default on) - blends pairs of training images/labels,
    one of the most effective regularisers when data is scarce.
  - cosine-annealing LR schedule instead of a flat LR - smoother convergence over more epochs.
  - --freeze-backbone-epochs now defaults to 1 (was 2) - with little data, letting the
    pretrained backbone start adapting earlier tends to help more than it hurts.
  - --img-size now configurable above 224 (try 288/320) - finer textures (fried vs steamed,
    gravy vs dry) often need more resolution than ImageNet's default to tell apart.
  - optional timm backbones (--arch convnext_tiny, needs `pip install timm`) - stronger
    ImageNet-22k pretraining than torchvision's weights, often a few points better on small data.

Run this on a GPU (Google Colab free tier or Kaggle Notebooks free tier both work well).

Usage:
    python ml/02_train_classifier.py --epochs 30 --batch-size 32 --arch efficientnet_v2_s
    python ml/02_train_classifier.py --epochs 30 --arch convnext_tiny --img-size 288   # needs timm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
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

TORCHVISION_ARCHES = {"efficientnet_v2_s", "mobilenet_v3_small", "mobilenet_v3_large",
                      "resnet50", "densenet121", "convnext_tiny"}


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
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        # fall back to timm for stronger/modern backbones (e.g. convnext_tiny, eva02_small_patch14_224.mim_in22k)
        try:
            import timm
        except ImportError as e:
            raise ValueError(
                f"Unsupported arch: {arch}. For non-torchvision architectures, `pip install timm` first."
            ) from e
        m = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return m


def head_param_ids(model: nn.Module) -> set:
    """ids of the parameters belonging to the classification head (trained during the frozen-backbone epoch)."""
    if hasattr(model, "get_classifier"):          # timm models
        head = model.get_classifier()
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):   # ResNet family
        head = model.fc
    else:                                          # EfficientNet / MobileNet / DenseNet / ConvNeXt
        head = model.classifier
    return {id(p) for p in head.parameters()}


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

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                           pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ld, val_ld, train_ds.classes


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Blend pairs of images/labels together (Zhang et al. 2017). Returns mixed inputs plus
    both label sets and the blend weight, so the loss can be computed as a weighted mix of the
    two cross-entropy terms. One of the most effective regularisers for small datasets - it
    manufactures new, harder-to-memorise training examples for free."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def run_epoch(model, loader, device, criterion, optimizer=None, mixup_alpha: float = 0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if is_train and mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                # accuracy against the dominant label only (lam >= 0.5 half of the time by symmetry)
                correct += (lam * (out.argmax(1) == y_a).float()
                            + (1 - lam) * (out.argmax(1) == y_b).float()).sum().item()
            else:
                out = model(x)
                loss = criterion(out, y)
                correct += (out.argmax(1) == y).sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_v2_s",
                     help="torchvision: efficientnet_v2_s | mobilenet_v3_small | mobilenet_v3_large. "
                          "Or any timm model name (needs `pip install timm`), e.g. convnext_tiny.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224, help="try 288 or 320 for finer food textures")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1,
                     help="train only the new classifier head for this many epochs first")
    ap.add_argument("--ckpt-name", default="classifier_best.pt",
                     help="checkpoint file name inside models/ (use a different name per architecture)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixup-alpha", type=float, default=0.2,
                     help="mixup strength; 0 disables mixup entirely")
    args = ap.parse_args()

    if not (DATA_DIR / "train").exists():
        raise SystemExit(f"{DATA_DIR/'train'} not found - run ml/01_prepare_classification_dataset.py first.")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ld, val_ld, classes = get_loaders(args.img_size, args.batch_size, args.workers)
    print(f"{len(classes)} classes: {classes}")

    model = build_model(args.arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # single optimizer + cosine schedule for the whole run, so the LR curve is smooth even
    # though which parameters are trainable changes after the freeze period ends below.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        for p in model.parameters():
            p.requires_grad = True
        if epoch < args.freeze_backbone_epochs:
            head_ids = head_param_ids(model)
            for p in model.parameters():
                if id(p) not in head_ids:
                    p.requires_grad = False

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_ld, device, criterion, optimizer,
                                           mixup_alpha=args.mixup_alpha)
        val_loss, val_acc = run_epoch(model, val_ld, device, criterion, optimizer=None, mixup_alpha=0.0)
        scheduler.step()
        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  lr={lr_now:.2e}  ({dt:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "arch": args.arch,
                        "classes": classes, "img_size": args.img_size},
                       MODELS_DIR / args.ckpt_name)
            print(f"  -> saved new best model (val_acc={val_acc:.3f})")

    (MODELS_DIR / args.ckpt_name.replace(".pt", "_history.json")).write_text(json.dumps(history, indent=2))
    print(f"\nBest val accuracy: {best_acc:.3f}. Model saved to models/{args.ckpt_name}")


if __name__ == "__main__":
    main()
