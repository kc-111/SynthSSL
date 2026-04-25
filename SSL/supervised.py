"""Supervised classification baseline on SynthSSL probe tasks.

Sanity check: does a ResNet-18 trained end-to-end on
``probe/<task>/train/`` hit high test accuracy on
``probe/<task>/test/``? If yes, the data is tractable under supervision,
so low SSL probe numbers reflect SSL's aug recipe rather than an
intractable task.

Expected numbers
----------------
Run on the **default Small recipe** (canvas 256, anchor-stratified
probe sets with per_anchor=5 → 15.8k train / 3.9k test per task).
Results below are after ~30 epochs with cosine LR + no color jitter
(color = identity for many emojis). Top-1 test accuracy.

``group`` (9 classes, majority baseline 61%):

    --pretrained                87% top-1
    (from scratch)              79%

``subgroup`` (98 classes, majority baseline ~13%):

    --pretrained                ~65% top-1 (estimate — run to confirm)
    (from scratch)              ~50%

``base_leaf`` (~1,943 classes, majority baseline <1%):

    --pretrained                ~25-40% (per-class support is thin)
    (from scratch)              ~15-25%

``style`` (5 classes, balanced by construction):

    --pretrained                near 100% (emoji styles are visually
                                distinctive; trivial with good features)
    (from scratch)              >95%

Interpretation
~~~~~~~~~~~~~~
- **Supervised pretrained** gives you the practical upper bound on
  what any SSL method can hope to match on these probes. A strong
  SSL recipe should approach it but typically not exceed it.
- **Supervised from-scratch** measures task difficulty without prior
  visual knowledge. The gap to pretrained (~8 pts on ``group``) tells
  you how much of the ceiling comes from ImageNet features vs from
  the task's own signal.
- **Majority baseline** (always predict the dominant class) is the
  floor — any genuine learning must beat it. For ``group`` this is
  61%, so a model at 65% has only ~4 pts of real signal.

Usage
-----
::

    python SSL/supervised.py --task group --epochs 30 --pretrained
    python SSL/supervised.py --task subgroup --epochs 30
    python SSL/supervised.py --task group --no-aug  # memorization check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO_ROOT / "data" / "SyntheticSSL" / "probe"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, task_dir: Path, split: str, transform,
                 label_to_idx: dict[str, int] | None = None):
        self.split_dir = task_dir / split
        with (self.split_dir / "metadata.jsonl").open() as f:
            meta = [json.loads(line) for line in f]
        pairs = [(r["image"], str(r["label"])) for r in meta]
        if label_to_idx is None:
            labels = sorted({l for _, l in pairs})
            label_to_idx = {l: i for i, l in enumerate(labels)}
        self.samples = [(n, label_to_idx[l]) for n, l in pairs
                        if l in label_to_idx]
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, label = self.samples[i]
        img = Image.open(self.split_dir / name).convert("RGB")
        return self.transform(img), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="group")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--pretrained", action="store_true",
                   help="Start from ImageNet-pretrained ResNet-18 weights.")
    ap.add_argument("--no-aug", action="store_true",
                   help="Disable random crop + flip. Same resize for train and test — "
                        "useful to verify the model can memorize the training set.")
    args = ap.parse_args()

    # Same crop regime as SSL — random-resized-crop + horizontal flip.
    # No color jitter / grayscale (emoji identity often carries color).
    if args.no_aug:
        train_tf = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(args.image_size, scale=(0.3, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    test_tf = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    task_dir = PROBE_ROOT / args.task
    train_ds = ProbeDataset(task_dir, "train", train_tf)
    test_ds = ProbeDataset(task_dir, "test", test_tf,
                           label_to_idx=train_ds.label_to_idx)
    n_classes = len(train_ds.label_to_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    if args.pretrained:
        from torchvision.models import ResNet18_Weights
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        model = torchvision.models.resnet18(num_classes=n_classes)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Linear warmup → cosine anneal.
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0,
        total_iters=max(1, args.warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[args.warmup_epochs])

    print(f"[supervised] task={args.task}  classes={n_classes}  "
          f"train={len(train_ds)}  test={len(test_ds)}  device={device}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for imgs, labels in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / max(total, 1)
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]

        print(f"  epoch {epoch:3d}  lr={lr_now:.4g}  loss={train_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}  ({dt:.1f}s)")
        sched.step()


if __name__ == "__main__":
    main()
