"""Visualize recon inputs (augmented views) and targets (blurred +
downscaled) used by train_equiv.py.

    python SSL/show_recon.py --n 6 --out SSL/logs/recon_samples.png

Each row is one sample:
    [view v1 (128x128)] [target v1 (16x16, upsampled NN)]
    [view v2 (128x128)] [target v2 (16x16, upsampled NN)]

Targets are upsampled with nearest-neighbor so their 16x16 pixels are
visible at the same scale as the views.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from train_equiv import (
    BLUR_ITER_RANGE,
    BLUR_SIGMA_RANGE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_AUG_SIGMA_BINS,
    PRETRAIN_DIRS,
    RECON_BLUR_ITER_RANGE,
    RECON_BLUR_KERNEL,
    RECON_BLUR_SIGMA,
    RECON_SIZE,
    EquivariantPairDataset,
    PretrainDictDataset,
    _apply_aug_blur_gpu,
    _build_aug_blur_table_1d,
    _build_iter_blur_table_1d,
    _cached_separable_blur_downsample,
)


def denorm(t: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalize, clamp to [0, 1] for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def to_hwc(t: torch.Tensor):
    return denorm(t).permute(1, 2, 0).numpy()


def upsample_nn(t: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(t.unsqueeze(0), size=size, mode="nearest").squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default="small", choices=list(PRETRAIN_DIRS))
    p.add_argument("--n", type=int, default=6, help="Number of samples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "logs"
                   / "recon_samples.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)

    base = PretrainDictDataset(PRETRAIN_DIRS[args.recipe], probe_tasks=[])
    ds = EquivariantPairDataset(base, image_size=IMAGE_SIZE, recon=True)

    fig, axes = plt.subplots(args.n, 4, figsize=(9, 2.2 * args.n))
    if args.n == 1:
        axes = axes.reshape(1, -1)
    titles = ["view v1", f"target v1 ({RECON_SIZE}x{RECON_SIZE})",
              "view v2", f"target v2 ({RECON_SIZE}x{RECON_SIZE})"]

    n_hi = RECON_BLUR_ITER_RANGE[1]
    # Same cached kernel tables that train forward() builds on the GPU,
    # built once here on CPU for the demo.
    blur_table = _build_iter_blur_table_1d(
        n_hi, RECON_BLUR_SIGMA,
        device=torch.device("cpu"), dtype=torch.float32)
    aug_blur_table = _build_aug_blur_table_1d(
        BLUR_ITER_RANGE[1], N_AUG_SIGMA_BINS, BLUR_SIGMA_RANGE,
        device=torch.device("cpu"), dtype=torch.float32)
    for row in range(args.n):
        idx = random.randrange(len(ds))
        s = ds[idx]
        v1 = s["image_v1"].unsqueeze(0)
        v2 = s["image_v2"].unsqueeze(0)
        n1 = s["recon_n_v1"].view(1)
        n2 = s["recon_n_v2"].view(1)
        # The dataloader emits *pre-aug-blur* views now (blur is deferred
        # to forward). Apply the same cached aug blur here so what we
        # display matches what the encoder consumes.
        ab1 = s["aug_blur_v1"].view(1, 3)
        ab2 = s["aug_blur_v2"].view(1, 3)
        v1 = _apply_aug_blur_gpu(
            v1, ab1[:, 0].bool(), ab1[:, 1], ab1[:, 2], aug_blur_table)
        v2 = _apply_aug_blur_gpu(
            v2, ab2[:, 0].bool(), ab2[:, 1], ab2[:, 2], aug_blur_table)
        target_v1 = _cached_separable_blur_downsample(
            v1, n1, blur_table, RECON_SIZE)[0]
        target_v2 = _cached_separable_blur_downsample(
            v2, n2, blur_table, RECON_SIZE)[0]
        tensors = [
            v1[0],
            upsample_nn(target_v1, IMAGE_SIZE),
            v2[0],
            upsample_nn(target_v2, IMAGE_SIZE),
        ]
        per_row_n = (int(n1.item()), int(n2.item()))
        ab_for = (ab1[0].tolist(), ab2[0].tolist())
        for col, t in enumerate(tensors):
            ax = axes[row, col]
            ax.imshow(to_hwc(t))
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col in (1, 3):
                ax.set_xlabel(f"N={per_row_n[col // 2]}", fontsize=8)
            if col in (0, 2):
                ap, sb, ni = ab_for[col // 2]
                tag = (f"aug-blur σ_bin={sb}, n={ni}" if ap
                       else "aug-blur off")
                ax.set_xlabel(tag, fontsize=8)

    fig.suptitle(
        f"Recon inputs vs targets - blur sigma={RECON_BLUR_SIGMA} "
        f"(kernel {RECON_BLUR_KERNEL}, "
        f"{RECON_BLUR_ITER_RANGE[0]}-{RECON_BLUR_ITER_RANGE[1]} iters per view), "
        f"downscale {IMAGE_SIZE} -> {RECON_SIZE}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[saved] {args.out}  ({args.n} samples, recipe={args.recipe})")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
