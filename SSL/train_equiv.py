"""Equivariant-predictor SSL pretraining on SynthSSL.

Same skeleton as ``train.py`` but with an *augmentation-conditioned
predictor*. For each sample we draw two random views ``v1, v2`` and
record the **full augmentation chain** that produced each — every
random decision lands in a per-view action vector::

    t_i = (
        crop_cx, crop_cy, crop_w, crop_h,         #  4: random resized crop
        flip,                                     #  1: horizontal flip
        color_applied, brightness, contrast,
        saturation, hue,                          #  5: color jitter
        gray_applied,                             #  1: random grayscale
        blur_applied, blur_sigma,                 #  2: gaussian blur
        solarize_applied,                         #  1: solarize
        K × (applied, cx, cy, w, h),              # 5K: hard random-erase
    )

(14 + 5·K numbers per view; with K=4 → 34/view → 68 concatenated as
``t = [t_v1, t_v2]``). The predictor is a small MLP::

    g : (z_1, t)  ↦  ẑ_2

trained to predict the second view's projection from the first view's
projection plus the action. Loss is::

    L = λ · reg((V, B, D))  +  (1 − λ) · ( inv_view + inv_pred )

where

    inv_view = mean[ max(0, ‖z_i − z̄‖² − margin_view) ] / D    # anchors z1, z2
    inv_pred = mean[ ‖ẑ_2 − z_2‖² ] / D                         # predictor (strict)

The view-pair margin (``--inv-tol``) lets z₁ and z₂ differ by up to a
fraction of the N(0, I) prior floor — without this, they'd be pulled
identical and the predictor would just learn the identity map. The
prediction term is **strict** (no margin): once the encoder has
produced anchored, action-coupled features, the predictor must match
``z_2`` exactly.

Why one-way (predict z_2 from z_1, not symmetric)
-------------------------------------------------
We want the *encoder* to internalize how augmentations affect features
— it has to produce a z_1 that the predictor can map *forward* with
only a small action vector to summarize what changed. Going both
directions makes the loss easier to satisfy without forcing the
encoder to do this work.

Why concatenated action (not delta)
-----------------------------------
``t_v1`` tells the predictor what frame ``z_1`` is in, ``t_v2`` tells
it what frame to land in. The MLP figures out the relation. Delta
form (``t_v2 − t_v1``) bakes in a group structure that the photometric
ops don't satisfy (e.g. solarize isn't invertible).

Why every aug — including content-dependent ones
------------------------------------------------
Color jitter / grayscale / blur / solarize are not invertible from the
action alone (they depend on image content), but knowing they were
applied lets the predictor *expect* the corresponding distribution
shift in z_2 instead of treating it as noise. The encoder then has an
incentive to retain enough content info in z_1 for the predictor to
forecast that shift — i.e. *not* collapse to invariance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import stable_pretraining as spt

from losses import make_regularizer
from online_probe import OnlineProbe


# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "SyntheticSSL"
PROBE_ROOT = DATASET_ROOT / "probe"

PRETRAIN_DIRS = {
    "small": DATASET_ROOT / "pretrain_small",
    "large": DATASET_ROOT / "pretrain_large",
}

IMAGE_SIZE = 128
EMB_DIM = 512

# Per-view action vector layout. Every aug contributes to ``t`` so the
# predictor sees the full transformation that produced its target view.
# Continuous params are normalized so each entry sits in roughly the same
# range; identity values (no-change) are filled when an op is gated off.
#
#   idx  field             range          identity
#   0    crop_cx           [0, 1]         (n/a — always applied)
#   1    crop_cy           [0, 1]
#   2    crop_w            (0, 1]
#   3    crop_h            (0, 1]
#   4    flip              {0, 1}         0
#   5    color_applied     {0, 1}         0
#   6    brightness        [-1, 1]        0   (factor 1.0 → 0 after norm)
#   7    contrast          [-1, 1]        0
#   8    saturation        [-1, 1]        0
#   9    hue               [-1, 1]        0   (factor 0.0 → 0)
#   10   gray_applied      {0, 1}         0
#   11   blur_applied      {0, 1}         0
#   12   blur_sigma        [0, 1]         0   (sigma 0 → 0)
#   13   solarize_applied  {0, 1}         0
#   14..18   mask_0: (applied, cx, cy, w, h)  {0,1} / [0,1]
#   19..23   mask_1: ...
#   24..28   mask_2: ...
#   29..33   mask_3: ...
#
# Mask slots are FIXED LENGTH so the action vector size is constant. An
# unused slot is all zeros; an active slot carries its bbox like the
# crop bbox does. Slots are independent — no canonical ordering is
# imposed, the predictor handles the small permutation-invariance burden.

# ColorJitter ranges (must match the original train.py recipe).
BRIGHTNESS_RANGE = 0.4
CONTRAST_RANGE = 0.4
SATURATION_RANGE = 0.2
HUE_RANGE = 0.1
COLOR_APPLY_P = 0.8
GRAY_APPLY_P = 0.2
BLUR_APPLY_P = 0.5
BLUR_SIGMA_RANGE = (0.1, 2.0)
SOLARIZE_APPLY_P = 0.2
SOLARIZE_THRESHOLD = 128   # uint8 PIL — applied before ToTensor

# Hard random-erase masking. Several small patches per view rather than
# one large one — the predictor should anticipate localized info loss in
# z_2, but a single big mask wipes too much for a small encoder to recover.
MASK_K = 4                      # max masks per view (fixed action-vector length)
MASK_APPLY_P = 0.5              # per-slot independent apply prob (E[N] = K·p)
MASK_SCALE = (0.02, 0.10)       # area fraction per mask — capped small
MASK_RATIO = (0.5, 2.0)         # aspect bounds (less extreme than torchvision)
T_DIM_PER_VIEW = 14 + 5 * MASK_K
T_DIM = 2 * T_DIM_PER_VIEW

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_PROBE_TASKS = ["group"]#, "subgroup"]#, "base_leaf"]


# ---------------------------------------------------------------------------
# Datasets — emit dicts directly so probes can read label_<task>
# ---------------------------------------------------------------------------

def _build_label_map(task_dir: Path) -> dict[str, int]:
    """Label string → int idx, built from a probe task's train+test."""
    labels: set[str] = set()
    for split in ("train", "test"):
        with (task_dir / split / "metadata.jsonl").open() as f:
            for line in f:
                labels.add(str(json.loads(line)["label"]))
    return {lbl: i for i, lbl in enumerate(sorted(labels))}


class _BaseDictDataset(Dataset):
    """Base. Each sample is a dict with raw PIL image + every probe task's
    label (set or -1)."""

    def __init__(self, probe_tasks: list[str]):
        self.probe_tasks = probe_tasks

    def _pack(self, img: Image.Image, labels: dict[str, int]) -> dict:
        out: dict = {"_pil": img}
        for task in self.probe_tasks:
            out[f"label_{task}"] = labels.get(task, -1)
        return out


class PretrainDictDataset(_BaseDictDataset):
    """Pretrain images: SSL only, every label_<task> = -1."""

    def __init__(self, root: Path, probe_tasks: list[str]):
        super().__init__(probe_tasks)
        self.paths = sorted(Path(root).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self._pack(img, {})


class ProbeDictDataset(_BaseDictDataset):
    """One task's probe/<task>/train split. Only that task's label is set."""

    def __init__(self, task_dir: Path, label_to_idx: dict[str, int],
                 probe_tasks: list[str], this_task: str):
        super().__init__(probe_tasks)
        self.split_dir = task_dir / "train"
        self.this_task = this_task
        with (self.split_dir / "metadata.jsonl").open() as f:
            meta = [json.loads(line) for line in f]
        self.samples = [
            (r["image"], label_to_idx[str(r["label"])])
            for r in meta if str(r["label"]) in label_to_idx
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, lbl = self.samples[i]
        img = Image.open(self.split_dir / name).convert("RGB")
        return self._pack(img, {self.this_task: lbl})


# ---------------------------------------------------------------------------
# Equivariant pair wrapper — produces (image_v1, image_v2, t_v1, t_v2)
# ---------------------------------------------------------------------------

class EquivariantPairDataset(Dataset):
    """Wrap a ``_BaseDictDataset``-style source. For each sample, draw two
    independent views and record an *action vector* fully describing
    every random choice (geometric + photometric) made for that view.

    Each photometric op is implemented manually with the torchvision
    functional API so we can record both an "applied" bit and the
    continuous parameters that were sampled. ``T.Compose`` /
    ``T.ColorJitter`` would hide those parameters behind their internal
    sampling, so we don't use them here.
    """

    def __init__(self, base: Dataset, image_size: int,
                 scale: tuple[float, float] = (0.5, 1.0),
                 ratio: tuple[float, float] = (3/4, 4/3),
                 photo_aug: bool = True):
        self.base = base
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.photo_aug = photo_aug

    def __len__(self):
        return len(self.base)

    def _view(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one full augmentation chain on ``img`` and return
        ``(tensor view, T_DIM_PER_VIEW action vector)``.

        The action records *every* random decision so the predictor can
        see exactly what produced its target view. Non-applied gated ops
        leave their continuous fields at the identity value (= 0 after
        normalization).
        """
        W, H = img.size

        # 1. Geometric: random resized crop + horizontal flip. Always
        #    applied (crop is the cheapest source of view diversity).
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=list(self.scale), ratio=list(self.ratio))
        flip = random.random() < 0.5
        v = TF.resized_crop(img, i, j, h, w,
                            [self.image_size, self.image_size])
        if flip:
            v = TF.hflip(v)

        # Defaults = "identity" for non-applied photometric ops.
        color_applied = 0
        b_factor, c_factor, s_factor, h_factor = 1.0, 1.0, 1.0, 0.0
        gray_applied = 0
        blur_applied = 0
        sigma = 0.0
        solar_applied = 0

        if self.photo_aug:
            # 2. Color jitter: gated by COLOR_APPLY_P; if on, sample each
            #    factor uniformly within its range. Apply in fixed order
            #    (b → c → s → h) so the action ↔ pixels map is consistent;
            #    T.ColorJitter would shuffle the order each call.
            if random.random() < COLOR_APPLY_P:
                color_applied = 1
                b_factor = random.uniform(1.0 - BRIGHTNESS_RANGE,
                                          1.0 + BRIGHTNESS_RANGE)
                c_factor = random.uniform(1.0 - CONTRAST_RANGE,
                                          1.0 + CONTRAST_RANGE)
                s_factor = random.uniform(1.0 - SATURATION_RANGE,
                                          1.0 + SATURATION_RANGE)
                h_factor = random.uniform(-HUE_RANGE, HUE_RANGE)
                v = TF.adjust_brightness(v, b_factor)
                v = TF.adjust_contrast(v, c_factor)
                v = TF.adjust_saturation(v, s_factor)
                v = TF.adjust_hue(v, h_factor)

            # 3. Random grayscale (binary; 3-channel output preserves
            #    downstream tensor shape).
            if random.random() < GRAY_APPLY_P:
                gray_applied = 1
                v = TF.rgb_to_grayscale(v, num_output_channels=3)

            # 4. Gaussian blur: gated by BLUR_APPLY_P; sigma sampled
            #    uniformly in BLUR_SIGMA_RANGE when on.
            if random.random() < BLUR_APPLY_P:
                blur_applied = 1
                sigma = random.uniform(*BLUR_SIGMA_RANGE)
                v = TF.gaussian_blur(v, kernel_size=23,
                                     sigma=[sigma, sigma])

            # 5. Solarize: gated by SOLARIZE_APPLY_P (operates on uint8
            #    PIL — keep it before ToTensor below).
            if random.random() < SOLARIZE_APPLY_P:
                solar_applied = 1
                v = TF.solarize(v, threshold=SOLARIZE_THRESHOLD)

        # 6. ToTensor + ImageNet normalize.
        v = TF.to_tensor(v)
        v = TF.normalize(v, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # 7. Hard random-erase masking. K independent slots per view; each
        #    decides independently whether to mask (Bernoulli MASK_APPLY_P).
        #    Applied AFTER normalize: filling with 0 = roughly the
        #    ImageNet channel mean. Mask boxes are sampled by area
        #    fraction × aspect ratio (log-uniform) and recorded into the
        #    action vector so the predictor can anticipate the localized
        #    info loss in this view.
        H_, W_ = self.image_size, self.image_size
        mask_actions: list[float] = []
        if self.photo_aug:
            for _ in range(MASK_K):
                if random.random() < MASK_APPLY_P:
                    area_frac = random.uniform(*MASK_SCALE)
                    log_lo, log_hi = math.log(MASK_RATIO[0]), math.log(MASK_RATIO[1])
                    aspect = math.exp(random.uniform(log_lo, log_hi))   # h / w
                    area_pixels = area_frac * H_ * W_
                    h_mask = max(1, min(H_, int(round(math.sqrt(area_pixels * aspect)))))
                    w_mask = max(1, min(W_, int(round(math.sqrt(area_pixels / aspect)))))
                    i_m = random.randint(0, H_ - h_mask)
                    j_m = random.randint(0, W_ - w_mask)
                    v[:, i_m:i_m + h_mask, j_m:j_m + w_mask] = 0.0
                    mask_actions.extend([
                        1.0,
                        (j_m + w_mask / 2) / W_,
                        (i_m + h_mask / 2) / H_,
                        w_mask / W_,
                        h_mask / H_,
                    ])
                else:
                    mask_actions.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            mask_actions = [0.0] * (5 * MASK_K)

        # Action vector. See module-level layout doc for index meanings.
        # Normalize continuous params to roughly [-1, 1] (or [0, 1]) so
        # the predictor's first linear layer sees comparable scales.
        t = torch.tensor([
            (j + w / 2) / W,
            (i + h / 2) / H,
            w / W,
            h / H,
            float(flip),
            float(color_applied),
            (b_factor - 1.0) / BRIGHTNESS_RANGE,
            (c_factor - 1.0) / CONTRAST_RANGE,
            (s_factor - 1.0) / SATURATION_RANGE,
            h_factor / HUE_RANGE,
            float(gray_applied),
            float(blur_applied),
            sigma / BLUR_SIGMA_RANGE[1],
            float(solar_applied),
            *mask_actions,
        ], dtype=torch.float32)
        return v, t

    def __getitem__(self, idx):
        sample = self.base[idx]
        img = sample.pop("_pil")
        v1, t1 = self._view(img)
        v2, t2 = self._view(img)
        sample["image_v1"] = v1
        sample["image_v2"] = v2
        sample["t_v1"] = t1
        sample["t_v2"] = t2
        return sample


# ---------------------------------------------------------------------------
# Model bits
# ---------------------------------------------------------------------------

def make_projector(in_dim: int, out_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def make_predictor(z_dim: int, t_dim: int, hidden: int) -> nn.Module:
    """Two-hidden-layer MLP, input = [z_1 ‖ t], output = ẑ_2 (size z_dim)."""
    return nn.Sequential(
        nn.Linear(z_dim + t_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, z_dim),
    )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def make_forward(probe_tasks: list[str]):
    label_keys = [f"label_{t}" for t in probe_tasks]

    def forward(self, batch, stage):
        out: dict = {}

        # Inference path (probe test eval): batch has just ``image``.
        if "image_v1" not in batch:
            emb = self.backbone(batch["image"])
            out["embedding"] = emb
            out["projection"] = self.projector(emb)
            self._last_output = out
            return out

        v1, v2 = batch["image_v1"], batch["image_v2"]
        t1, t2 = batch["t_v1"], batch["t_v2"]
        t_cat = torch.cat([t1, t2], dim=-1)              # (B, T_DIM)

        h1 = self.backbone(v1)
        h2 = self.backbone(v2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Predictor input = [z_1 ‖ t]. One-way: target is z_2 (no stop-grad
        # — we want both encoders to learn from the loss).
        pred = self.predictor(torch.cat([z1, t_cat], dim=-1))

        # Distributional regularizer on both views' projections.
        z_stack = torch.stack([z1, z2], dim=0)            # (V=2, B, D)
        V, _, D = z_stack.shape
        reg = self.regularizer(z_stack)
        prior_floor = D * (V - 1) / V                     # = D/2 here

        # 1. View-pair invariance with margin. Anchors z_1 and z_2 within
        #    a tolerance of the N(0, I) prior floor — keeps them in the
        #    same neighborhood so the predictor has a sensible target,
        #    while still leaving room for action-conditional variation.
        mean_z = z_stack.mean(dim=0, keepdim=True)        # (1, B, D)
        per_view_sq = (z_stack - mean_z).square().sum(dim=-1)  # (V, B)
        margin_view = self.inv_tol * prior_floor
        inv_view = torch.clamp(per_view_sq - margin_view,
                               min=0.0).mean() / D

        # Diagnostic: fraction of (view, sample) entries whose squared
        # deviation exceeded the margin, i.e. where the clamp was actually
        # *active* this batch. Averaged over the epoch this tells you
        # whether the margin is loose (≈ 0 — never triggers, identical
        # to no margin at all) or tight (≈ 1 — always triggers, behaves
        # like strict invariance with a constant offset). The useful
        # range is somewhere in between.
        inv_view_active = (per_view_sq > margin_view).float().mean()

        # 2. Prediction loss between ẑ_2 and z_2 — strict, no margin.
        #    Once features are anchored by inv_view, the predictor must
        #    fit exactly: that's what forces the encoder to produce
        #    action-coupled (rather than invariant) representations.
        inv_pred = (pred - z2).square().sum(dim=-1).mean() / D

        loss = (self.lambd * reg
                + (1.0 - self.lambd) * (inv_view + inv_pred))
        out["loss"] = loss

        # Features for online probes: concat both views' embeddings/projs +
        # duplicate per-task labels.
        out["embedding"] = torch.cat([h1, h2], dim=0)
        out["projection"] = torch.cat([z1, z2], dim=0)
        for k in label_keys:
            if k in batch:
                out[k] = torch.cat([batch[k], batch[k]], dim=0)

        self.log(f"{stage}/loss",     loss,     on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/reg",      reg,      on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/inv_view", inv_view, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/inv_pred", inv_pred, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/inv_view_active", inv_view_active,
                 on_step=True, on_epoch=True, sync_dist=True)
        # Sum logged as ``inv`` so OnlineProbe's epoch-end print picks it
        # up the same way as train.py.
        self.log(f"{stage}/inv", inv_view + inv_pred,
                 on_step=True, on_epoch=True, sync_dist=True)

        self._last_output = out
        return out

    return forward


# ---------------------------------------------------------------------------
# Run dir
# ---------------------------------------------------------------------------

def allocate_run(log_dir: Path, recipe: str, args_dict: dict) -> tuple[Path, str]:
    """Pick the next free ``equiv_<recipe>_<id>/`` under log_dir and dump args."""
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"equiv_{recipe}_"
    ids = []
    for d in log_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                ids.append(int(d.name[len(prefix):]))
            except ValueError:
                continue
    run_id = max(ids, default=0) + 1
    run_name = f"{prefix}{run_id}"
    run_dir = log_dir / run_name
    run_dir.mkdir()
    (run_dir / "args.json").write_text(
        json.dumps(args_dict, indent=2, sort_keys=True))
    return run_dir, run_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _probe_sources(choice: str, proj_dim: int) -> list[tuple[str, str, int]]:
    if choice == "backbone":
        return [("backbone", "embedding", EMB_DIM)]
    if choice == "projector":
        return [("projector", "projection", proj_dim)]
    if choice == "both":
        return [("backbone", "embedding", EMB_DIM),
                ("projector", "projection", proj_dim)]
    raise ValueError(f"--probe unknown: {choice!r}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default="small", choices=["small", "large"])
    p.add_argument("--regularizer", default="w1",
                   choices=["sigreg", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.95,
                   help="Weight on reg vs prediction loss.")
    p.add_argument("--inv-tol", type=float, default=0.5,
                   help="Margin on the VIEW-pair invariance loss as a "
                        "fraction of the N(0,I) prior floor D·(V−1)/V "
                        "(V=2 → D/2). Anchors z_1 and z_2 within a "
                        "tolerance — small but non-zero is what lets the "
                        "encoder leave action-conditional variation for "
                        "the predictor to model. The prediction loss is "
                        "always strict (no margin).")
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--pred-hidden", type=int, default=2048,
                   help="Predictor MLP hidden width.")
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--probe", default="both",
                   choices=["backbone", "projector", "both"])
    p.add_argument("--probe-tasks", nargs="*", default=DEFAULT_PROBE_TASKS)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--photo-aug", choices=["none", "full"], default="full",
                   help="full: keep color jitter / grayscale / blur / "
                        "solarize on top of the geometric crop+flip. "
                        "none: only crop+flip (the actions encoded in t).")
    p.add_argument("--scale-min", type=float, default=0.3)
    p.add_argument("--scale-max", type=float, default=1.0)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    run_dir, run_name = allocate_run(log_dir, args.recipe, vars(args))
    pl.seed_everything(args.seed, workers=True)
    print(f"[run] {run_name}  seed={args.seed}  dir={run_dir}")

    probe_tasks = list(args.probe_tasks)
    label_maps = {t: _build_label_map(PROBE_ROOT / t) for t in probe_tasks}
    for t in probe_tasks:
        print(f"  {t:<20s} {len(label_maps[t]):>4d} classes")

    # --- Data: pretrain + per-probe-task train, all wrapped as pair views ---
    pretrain_raw = PretrainDictDataset(PRETRAIN_DIRS[args.recipe], probe_tasks)
    probe_raws = [
        ProbeDictDataset(
            task_dir=PROBE_ROOT / task,
            label_to_idx=label_maps[task],
            probe_tasks=probe_tasks,
            this_task=task,
        )
        for task in probe_tasks
    ]
    concat = ConcatDataset([pretrain_raw, *probe_raws])

    train_ds = EquivariantPairDataset(
        concat, image_size=IMAGE_SIZE,
        scale=(args.scale_min, args.scale_max),
        photo_aug=args.photo_aug == "full",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory=True,
    )
    print(f"[data] pretrain={len(pretrain_raw)}  "
          f"+ probe_train={[len(d) for d in probe_raws]}  "
          f"=> total={len(concat)}  T_DIM={T_DIM}")

    # --- Model ---
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=False)
    projector = make_projector(EMB_DIM, args.proj_dim, args.proj_hidden)
    predictor = make_predictor(args.proj_dim, T_DIM, args.pred_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        forward=make_forward(probe_tasks),
        regularizer=regularizer,
        lambd=args.lambd,
        inv_tol=args.inv_tol,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    sources = _probe_sources(args.probe, args.proj_dim)
    probe_cb = OnlineProbe(
        tasks=probe_tasks,
        sources=sources,
        label_maps=label_maps,
        probe_root=PROBE_ROOT,
        image_size=IMAGE_SIZE,
        eval_every_n_epochs=args.eval_every,
        batch_size=args.batch_size,
        num_workers=max(1, args.num_workers // 4),
    )
    print(f"[probes] {len(probe_tasks)} task(s) × {len(sources)} source(s) = "
          f"{len(probe_tasks) * len(sources)} head(s), "
          f"test_eval_every={args.eval_every}")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_last=True, save_top_k=0,
    )
    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[ckpt_cb, probe_cb],
        precision=args.precision,
        logger=logger,
        default_root_dir=str(run_dir),
    )
    trainer.fit(module, train_dataloaders=train_dl)


if __name__ == "__main__":
    main()
