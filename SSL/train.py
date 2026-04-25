"""LeJEPA-style SSL pretraining on SynthSSL.

Loss::

    L = λ · reg((V, B, D))  +  (1 − λ) · inv((V, B, D))

- ``reg``: sigreg / w1 / w2 (``losses.make_regularizer``).
- ``inv``: mean view-axis variance of the projector output
  (= ½ · ‖z₁ − z₂‖² when V = 2).

Online probing
--------------
The train dataloader is a **ConcatDataset**:

    pretrain/                       → (img, −1, −1, −1, …)   SSL only
    probe/group/train/              → (img, group_idx, −1, −1, …)
    probe/subgroup/train/           → (img, −1, subgroup_idx, −1, …)
    probe/base_leaf/train/          → (img, −1, −1, base_leaf_idx, …)

Every sample contributes to the SSL loss (backbone + projector + reg + inv).
Each per-task ``LinearOnlineProbe`` callback reads its task's label off
the shared forward output and trains only on the samples whose label
is non-negative — so each probe trains on its own eval-distribution
split, not on pretrain. No extra backbone forwards.

Usage::

    python SSL/train.py --recipe small --regularizer sigreg --epochs 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_list

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

DEFAULT_PROBE_TASKS = ["group", "subgroup", "base_leaf"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _build_label_map(task_dir: Path) -> dict[str, int]:
    """Label → int index from a probe task's train+test metadata."""
    labels: set[str] = set()
    for split in ("train", "test"):
        with (task_dir / split / "metadata.jsonl").open() as f:
            for line in f:
                labels.add(str(json.loads(line)["label"]))
    return {lbl: i for i, lbl in enumerate(sorted(labels))}


class _MultiLabelImageDataset(torch.utils.data.Dataset):
    """Base: returns ``(PIL image, *N label ints)`` where unset labels are −1."""

    N_LABEL_FIELDS: int

    def _pack(self, img: Image.Image, labels: dict[int, int]) -> tuple:
        """Build the ``(img, *labels_in_order)`` tuple."""
        out = [img] + [labels.get(i, -1) for i in range(self.N_LABEL_FIELDS)]
        return tuple(out)


class PretrainDataset(_MultiLabelImageDataset):
    """Pretrain samples: SSL only, all labels −1 so probes ignore them."""

    def __init__(self, root: Path, n_label_fields: int):
        self.N_LABEL_FIELDS = n_label_fields
        self.paths = sorted(Path(root).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg files under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self._pack(img, {})


class ProbeTaskTrainDataset(_MultiLabelImageDataset):
    """One task's train split. Only this task's label field is populated."""

    def __init__(self, task_dir: Path, label_to_idx: dict[str, int],
                 n_label_fields: int, task_field_idx: int):
        self.N_LABEL_FIELDS = n_label_fields
        self.split_dir = task_dir / "train"
        self.task_field_idx = task_field_idx
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
        return self._pack(img, {self.task_field_idx: lbl})


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

def _make_train_tf():
    one = transforms.Compose(
        transforms.RGB(),
        # transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0)),
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=0.5),
        transforms.RandomSolarize(p=0.2, threshold=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )
    return transforms.MultiViewTransform([one, one])


def make_data(
    recipe: str,
    batch_size: int,
    num_workers: int,
    probe_tasks: list[str],
    label_maps: dict[str, dict[str, int]],
) -> spt.data.DataModule:
    n_fields = len(probe_tasks)

    pretrain_raw = PretrainDataset(PRETRAIN_DIRS[recipe], n_label_fields=n_fields)

    probe_raws = [
        ProbeTaskTrainDataset(
            task_dir=PROBE_ROOT / task,
            label_to_idx=label_maps[task],
            n_label_fields=n_fields,
            task_field_idx=i,
        )
        for i, task in enumerate(probe_tasks)
    ]

    concat = torch.utils.data.ConcatDataset([pretrain_raw, *probe_raws])
    names = ["image"] + [f"label_{t}" for t in probe_tasks]
    train_ds = spt.data.FromTorchDataset(
        concat, names=names, transform=_make_train_tf())

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=True,
    )

    # No val dataloader — probes report training top-1; offline test
    # eval is eval.py. Keeps the training loop plain single-val-free.
    return spt.data.DataModule(train=train_dl)


# ---------------------------------------------------------------------------
# Model
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


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def make_forward(probe_tasks: list[str]):
    label_keys = [f"label_{t}" for t in probe_tasks]

    def forward(self, batch, stage):
        out: dict = {}
        views = _get_views_list(batch)

        if views is None:
            emb = self.backbone(batch["image"])
            out["embedding"] = emb
            out["projection"] = self.projector(emb)
            self._last_output = out
            return out

        live_emb = [self.backbone(v["image"]) for v in views]
        live_z = [self.projector(e) for e in live_emb]

        z_stack = torch.stack(live_z, dim=0)        # (V, B, D)
        V, _, D = z_stack.shape
        reg = self.regularizer(z_stack)
        mean_z = z_stack.mean(dim=0, keepdim=True)  # (1, B, D)

        # Margin-hinge invariance. Under iid N(0, I) prior,
        #   E[ ‖z_i − z_bar‖² ]  =  D · (V − 1) / V
        # so the margin ε · D(V−1)/V is a fraction of the natural
        # view-dispersion at the regularizer's equilibrium. ε = 0
        # recovers strict invariance (per-element MSE); ε = 1 disables
        # invariance pressure up to the prior floor.
        per_sample_sq = (z_stack - mean_z).square().sum(dim=-1)   # (V, B)
        prior_floor = D * (V - 1) / V
        margin = self.inv_tol * prior_floor
        inv = torch.clamp(per_sample_sq - margin, min=0.0).mean() / D

        loss = self.lambd * reg + (1.0 - self.lambd) * inv
        out["loss"] = loss

        # Features + duplicated labels (once per live view) for probe callbacks.
        out["embedding"] = torch.cat(live_emb, dim=0)
        out["projection"] = torch.cat(live_z, dim=0)
        for k in label_keys:
            if k in views[0]:
                out[k] = torch.cat([v[k] for v in views], dim=0)

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/reg",  reg,  on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/inv",  inv,  on_step=True, on_epoch=True, sync_dist=True)

        self._last_output = out
        return out

    return forward


# ---------------------------------------------------------------------------
# Run allocation
# ---------------------------------------------------------------------------

def allocate_run(log_dir: Path, recipe: str, args_dict: dict) -> tuple[Path, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{recipe}_"
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
    (run_dir / "args.json").write_text(json.dumps(args_dict, indent=2, sort_keys=True))
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
    p.add_argument("--regularizer", default="sigreg",
                   choices=["sigreg", "w1", "w2"])
    p.add_argument("--lambd", type=float, default=0.05)
    p.add_argument("--inv-tol", type=float, default=0.0,
                   help="Invariance margin ε ∈ [0, 1]. Fraction of the "
                        "N(0, I) prior floor ‖z_i − z̄‖² = D·(V−1)/V below "
                        "which invariance has no penalty. 0 = strict "
                        "invariance (default); 1 = no pressure below "
                        "the regularizer equilibrium.")
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--num-proj", type=int, default=1024)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--probe", default="both",
                   choices=["backbone", "projector", "both"],
                   help="Online probe feature source.")
    p.add_argument("--probe-tasks", nargs="*", default=DEFAULT_PROBE_TASKS,
                   help="Tasks to probe online (emoji-identity classification "
                        "by default). Their train splits are mixed into the "
                        "training dataloader; probes train only on those "
                        "samples. Full multi-task eval lives in eval.py.")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run probe TEST eval on probe/<task>/test/ every N "
                        "training epochs (0 = off). Logs "
                        "probe/<task>/<src>/test_top1 alongside the "
                        "train-time running metric.")
    return p


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

    data = make_data(args.recipe, args.batch_size, args.num_workers,
                     probe_tasks, label_maps)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=False)
    projector = make_projector(EMB_DIM, args.proj_dim, args.proj_hidden)
    regularizer = make_regularizer(
        args.regularizer, num_proj=args.num_proj, knots=args.knots)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
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

    # One callback owns all probe heads + optional periodic test eval.
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

    # Save under <log_dir>/<run_name>/ directly. We do NOT use spt.Manager:
    # it auto-injects a RegistryLogger that REPLACES any user CSVLogger and
    # writes to trainer.default_root_dir (= CWD), so metrics.csv and
    # last.ckpt end up nowhere near run_dir. Calling trainer.fit directly
    # keeps logs + checkpoints rooted at run_dir.
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

    data.setup("fit")
    trainer.fit(module, train_dataloaders=data.train_dataloader())


if __name__ == "__main__":
    main()
