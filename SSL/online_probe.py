"""One callback that runs all online probes + periodic test evaluation.

Training (every SSL step)
    For each (task, source):
        - Read features + labels from the module's cached forward output
        - Drop samples with label < 0 (pretrain has -1 for every task)
        - One SGD+Nesterov step on a linear head

Test eval (every ``eval_every_n_epochs``)
    For each task:
        - Iterate probe/<task>/test/
        - Forward through the frozen backbone (+ projector if needed)
        - Apply each head, compute top-1
    Logs ``probe/<task>/<source>/{train_top1, test_top1}``.

One callback owns all heads, optimizers, metrics, and test dataloaders.
No multi-val dataloader, no ``dataloader_idx`` plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from PIL import Image

import stable_pretraining as spt
from stable_pretraining.data import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ProbeSplit(torch.utils.data.Dataset):
    """One probe task's split, (PIL image, int label)."""

    def __init__(self, split_dir: Path, label_to_idx: dict[str, int]):
        self.split_dir = Path(split_dir)
        with (self.split_dir / "metadata.jsonl").open() as f:
            meta = [json.loads(line) for line in f]
        self.samples = [
            (r["image"], label_to_idx[str(r["label"])])
            for r in meta if str(r["label"]) in label_to_idx
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, label = self.samples[i]
        img = Image.open(self.split_dir / name).convert("RGB")
        return img, label


def _eval_transform(image_size: int):
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((image_size, image_size)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

class OnlineProbe(pl.Callback):
    """Multi-task linear online probe with periodic test evaluation.

    Args:
        tasks: list of task names under ``probe_root``.
        sources: list of ``(src_name, input_key, in_dim)`` triples
            — usually ``[("backbone", "embedding", 512)]`` or the
            ``backbone + projector`` pair.
        label_maps: ``{task: {label_str: idx}}`` for every task.
        probe_root: ``data/SyntheticSSL/probe``.
        image_size: model input size (for the test loader transform).
        eval_every_n_epochs: run test eval every N train epochs (0 = off).
        batch_size / num_workers: test-loader settings.
        lr / momentum / weight_decay: SGD hyperparameters for all heads.
    """

    def __init__(
        self,
        tasks: list[str],
        sources: list[tuple[str, str, int]],
        label_maps: dict[str, dict[str, int]],
        probe_root: Path,
        image_size: int,
        eval_every_n_epochs: int = 5,
        batch_size: int = 256,
        num_workers: int = 2,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.tasks = list(tasks)
        self.sources = list(sources)
        self.label_maps = label_maps
        self.probe_root = Path(probe_root)
        self.image_size = image_size
        self.eval_every = eval_every_n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Populated in setup()
        self.heads: dict[tuple[str, str], nn.Linear] = {}
        self.opts: dict[tuple[str, str], torch.optim.Optimizer] = {}
        self.train_accs: dict[tuple[str, str], torchmetrics.Metric] = {}
        self.test_accs: dict[tuple[str, str], torchmetrics.Metric] = {}
        self.test_loaders: dict[str, torch.utils.data.DataLoader] = {}

    # Lifecycle --------------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if self.heads:
            return
        dev = pl_module.device
        tf = _eval_transform(self.image_size)

        for task in self.tasks:
            n_classes = len(self.label_maps[task])

            # Heads + optimizers + metrics
            for src_name, _input_key, in_dim in self.sources:
                key = (task, src_name)
                head = nn.Linear(in_dim, n_classes).to(dev)
                self.heads[key] = head
                self.opts[key] = torch.optim.SGD(
                    head.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                    nesterov=self.momentum > 0,
                )
                self.train_accs[key] = torchmetrics.classification.MulticlassAccuracy(
                    n_classes).to(dev)
                self.test_accs[key] = torchmetrics.classification.MulticlassAccuracy(
                    n_classes).to(dev)

            # Test dataloader (for periodic eval)
            if self.eval_every > 0:
                raw = ProbeSplit(self.probe_root / task / "test",
                                 self.label_maps[task])
                ds = spt.data.FromTorchDataset(
                    raw, names=["image", "label"], transform=tf)
                self.test_loaders[task] = torch.utils.data.DataLoader(
                    ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.num_workers > 0,
                )

            # Also ensure test_accs metric exists (built in heads loop above)
            for src_name, _, _ in self.sources:
                if (task, src_name) not in self.test_accs:
                    n_classes = len(self.label_maps[task])
                    self.test_accs[(task, src_name)] = (
                        torchmetrics.classification.MulticlassAccuracy(
                            n_classes).to(dev))

    # Online training (every step) -------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        out = getattr(pl_module, "_last_output", None)
        if not out:
            return

        for task in self.tasks:
            label_key = f"label_{task}"
            if label_key not in out:
                continue
            labels = out[label_key].detach()
            mask = labels >= 0
            if int(mask.sum().item()) == 0:
                continue
            labels_m = labels[mask]

            for src_name, input_key, _in_dim in self.sources:
                if input_key not in out:
                    continue
                head = self.heads[(task, src_name)]
                opt = self.opts[(task, src_name)]
                feats = out[input_key].detach()[mask].to(head.weight.dtype)
                if feats.dim() > 2:
                    feats = feats.flatten(1)
                head.train()
                logits = head(feats)
                loss = F.cross_entropy(logits, labels_m)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self.train_accs[(task, src_name)].update(
                    logits.detach(), labels_m)

    # End-of-epoch: log train top-1 + maybe run test eval --------------------

    def on_train_epoch_end(self, trainer, pl_module):
        # Train top-1 (running over the epoch's labeled samples)
        lines = []
        for key, metric in self.train_accs.items():
            try:
                acc = metric.compute()
            except Exception:
                continue
            task, src = key
            pl_module.log(
                f"probe/{task}/{src}/train_top1", acc,
                on_epoch=True, sync_dist=False)
            lines.append(f"  train  {task}/{src}: {acc:.4f}")
            metric.reset()

        # Periodic test eval
        if self.eval_every > 0:
            epoch = trainer.current_epoch
            due = (epoch % self.eval_every == 0
                   or epoch == trainer.max_epochs - 1)
            if due:
                test_lines = self._run_test_eval(pl_module)
                lines += test_lines

        if lines:
            # Pull the SSL loss components (logged from train.py's forward
            # via ``self.log("fit/{loss,reg,inv}", ...)``). Lightning adds
            # an "_epoch" suffix to per-epoch-reduced values. Also surface
            # any ``*_active`` diagnostics (e.g. ``inv_view_active`` from
            # train_equiv.py — fraction of samples whose margin clamp
            # actually triggered).
            metrics = trainer.callback_metrics
            ssl_bits = []
            for name in ("loss", "reg", "inv"):
                for key in (f"fit/{name}_epoch", f"fit/{name}"):
                    if key in metrics:
                        ssl_bits.append(f"{name}={float(metrics[key]):.4f}")
                        break
            # Lightning logs both ``fit/x`` (last step) and ``fit/x_epoch``
            # (epoch mean) — prefer the epoch one, fall back to the step
            # one, deduped by short name.
            seen: set[str] = set()
            for key in sorted(metrics, key=lambda k: not k.endswith("_epoch")):
                if not key.startswith("fit/"):
                    continue
                short = key[len("fit/"):]
                if short.endswith("_epoch"):
                    short = short[:-len("_epoch")]
                if not short.endswith("_active") or short in seen:
                    continue
                seen.add(short)
                ssl_bits.append(f"{short}={float(metrics[key]):.3f}")

            header = f"[probe] epoch {trainer.current_epoch}"
            if ssl_bits:
                header += "  " + "  ".join(ssl_bits)
            print(header)
            for ln in lines:
                print(ln)

    def _run_test_eval(self, pl_module: pl.LightningModule) -> list[str]:
        dev = pl_module.device
        need_projector = any(s[0] == "projector" for s in self.sources)
        backbone_was_training = pl_module.backbone.training
        projector_was_training = pl_module.projector.training
        pl_module.backbone.eval()
        pl_module.projector.eval()

        lines: list[str] = []
        for task in self.tasks:
            loader = self.test_loaders.get(task)
            if loader is None:
                continue
            for key, metric in self.test_accs.items():
                if key[0] == task:
                    metric.reset()

            with torch.no_grad():
                for batch in loader:
                    imgs = batch["image"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    emb = pl_module.backbone(imgs)
                    proj = pl_module.projector(emb) if need_projector else None
                    for src_name, _input_key, _in_dim in self.sources:
                        head = self.heads[(task, src_name)]
                        feats = emb if src_name == "backbone" else proj
                        feats = feats.to(head.weight.dtype)
                        if feats.dim() > 2:
                            feats = feats.flatten(1)
                        head.eval()
                        logits = head(feats)
                        self.test_accs[(task, src_name)].update(logits, labels)

            for src_name, _input_key, _in_dim in self.sources:
                acc = self.test_accs[(task, src_name)].compute()
                pl_module.log(
                    f"probe/{task}/{src_name}/test_top1", acc,
                    on_epoch=True, sync_dist=False)
                lines.append(f"  test   {task}/{src_name}: {acc:.4f}")

        if backbone_was_training:
            pl_module.backbone.train()
        if projector_was_training:
            pl_module.projector.train()
        return lines
