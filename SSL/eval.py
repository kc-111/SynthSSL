"""Multi-task linear probe over a frozen SSL backbone.

Loads a ``SSL/logs/<run>/checkpoints/last.ckpt``, freezes the backbone,
extracts features on every probe task's train + test split in
``data/SyntheticSSL/probe/<task>/``, fits a multinomial logistic
regression (scikit-learn) on the train features, and reports test
accuracy per task.

Usage::

    python SSL/eval.py --run SSL/logs/small_1
    python SSL/eval.py --run SSL/logs/small_1 --tasks group style

Writes ``<run>/eval_results.json`` with one record per task.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stable_pretraining as spt


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO_ROOT / "data" / "SyntheticSSL" / "probe"
IMAGE_SIZE = 128


# ---------------------------------------------------------------------------
# Probe dataset (on-disk layout from SynthSSL.generate)
# ---------------------------------------------------------------------------

class ProbeSplitDataset(torch.utils.data.Dataset):
    """One probe task's train/ or test/ split.

    Reads ``<task_dir>/<split>/metadata.jsonl`` for (image, label) pairs
    and ``<task_dir>/<split>/<filename>.jpg`` for the image bytes.

    If ``label_to_idx`` is provided (built from the train split), the
    test split will drop any sample whose label is unseen; otherwise a
    new mapping is constructed from the labels present in the data.
    """

    def __init__(self, task_dir: Path, split: str,
                 label_to_idx: dict[str, int] | None = None):
        self.split_dir = task_dir / split
        meta_path = self.split_dir / "metadata.jsonl"
        with meta_path.open() as f:
            meta = [json.loads(line) for line in f]

        pairs = [(r["image"], str(r["label"])) for r in meta]

        if label_to_idx is None:
            labels = sorted({lbl for _, lbl in pairs})
            label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
            self.samples = pairs
        else:
            # Keep only samples whose label is in the provided mapping.
            self.samples = [(name, lbl) for name, lbl in pairs
                            if lbl in label_to_idx]

        self.label_to_idx = label_to_idx
        self.tf = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(**_imagenet_stats()),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, label = self.samples[i]
        img = Image.open(self.split_dir / name).convert("RGB")
        return self.tf(img), self.label_to_idx[label]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)


def _imagenet_stats() -> dict:
    stats = spt.data.static.ImageNet
    return {"mean": stats["mean"], "std": stats["std"]}


# ---------------------------------------------------------------------------
# Checkpoint loading: strip to just the backbone
# ---------------------------------------------------------------------------

def _load_submodule(state: dict, prefix: str, module: nn.Module,
                    label: str) -> nn.Module:
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sub, strict=False)
    if missing:
        head = list(missing)[:5]
        tail = "…" if len(missing) > 5 else ""
        print(f"[warn] {label}: missing {len(missing)} key(s) {head}{tail}")
    if unexpected:
        print(f"[warn] {label}: unexpected {list(unexpected)[:5]}")
    module.eval()
    return module


def _build_projector(in_dim: int, out_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def load_network_from_ckpt(
    ckpt_path: Path,
    need_projector: bool,
    proj_dim: int,
    proj_hidden: int,
) -> tuple[nn.Module, nn.Module | None]:
    """Rebuild the backbone (and, if requested, the projector) and restore weights.

    We reconstruct both architectures from scratch rather than
    instantiating the full Lightning module — eval only needs forward
    passes, not optimizer state.
    """
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    _load_submodule(state, "backbone.", backbone, "backbone")
    projector = None
    if need_projector:
        projector = _build_projector(512, proj_dim, proj_hidden)
        _load_submodule(state, "projector.", projector, "projector")
    return backbone, projector


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    projector: nn.Module | None,
    ds: torch.utils.data.Dataset,
    device: torch.device,
    sources: list[str],
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run ``ds`` through the frozen network, return (``{source: feats}``, labels)."""
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    backbone = backbone.to(device)
    if projector is not None:
        projector = projector.to(device)
    out: dict[str, list[np.ndarray]] = {s: [] for s in sources}
    labs: list[np.ndarray] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = backbone(imgs)
        if "backbone" in sources:
            out["backbone"].append(emb.float().cpu().numpy())
        if "projector" in sources:
            if projector is None:
                raise RuntimeError("feature_source requires projector but none loaded")
            out["projector"].append(projector(emb).float().cpu().numpy())
        labs.append(labels.numpy())
    feats = {s: np.concatenate(out[s], axis=0) for s in sources}
    return feats, np.concatenate(labs, axis=0)


def _resolve_sources(feature_source: str) -> list[str]:
    if feature_source == "both":
        return ["backbone", "projector"]
    if feature_source in ("backbone", "projector"):
        return [feature_source]
    raise ValueError(f"feature_source {feature_source!r} not in "
                     "{'backbone', 'projector', 'both'}")


# ---------------------------------------------------------------------------
# Per-task probe
# ---------------------------------------------------------------------------

def probe_task(
    backbone: nn.Module,
    projector: nn.Module | None,
    task: str,
    device: torch.device,
    sources: list[str],
    max_iter: int = 1000,
    C: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict:
    """Extract features + fit + evaluate a single task for each source.

    Returns a dict keyed by source, each with its own top-1/top-5.
    """
    task_dir = PROBE_ROOT / task
    if not (task_dir / "train" / "metadata.jsonl").exists():
        return {"task": task, "error": "missing train/metadata.jsonl"}

    train_ds = ProbeSplitDataset(task_dir, "train")
    test_ds = ProbeSplitDataset(task_dir, "test", label_to_idx=train_ds.label_to_idx)

    t0 = time.time()
    X_train, y_train = extract_features(
        backbone, projector, train_ds, device, sources,
        batch_size=batch_size, num_workers=num_workers)
    X_test, y_test = extract_features(
        backbone, projector, test_ds, device, sources,
        batch_size=batch_size, num_workers=num_workers)
    t_feat = time.time() - t0

    n_classes = train_ds.num_classes
    solver = "lbfgs" if n_classes <= 100 else "saga"

    per_source: dict[str, dict] = {}
    for src in sources:
        t0 = time.time()
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, C=C, solver=solver),
        )
        clf.fit(X_train[src], y_train)
        t_fit = time.time() - t0

        pred = clf.predict(X_test[src])
        acc = accuracy_score(y_test, pred)

        inner = clf.named_steps["logisticregression"]

        top5 = None
        if n_classes > 5:
            proba = clf.predict_proba(X_test[src])
            try:
                top5 = top_k_accuracy_score(
                    y_test, proba, k=5, labels=inner.classes_)
            except ValueError:
                top5 = None

        per_source[src] = {
            "top1": float(acc),
            "top5": float(top5) if top5 is not None else None,
            "n_classes_seen_by_clf": int(len(inner.classes_)),
            "feature_dim": int(X_train[src].shape[1]),
            "t_fit_s": round(t_fit, 2),
        }

    return {
        "task": task,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(n_classes),
        "solver": solver,
        "t_feature_extract_s": round(t_feat, 2),
        "sources": per_source,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_tasks() -> list[str]:
    if not PROBE_ROOT.is_dir():
        return []
    return sorted(d.name for d in PROBE_ROOT.iterdir()
                  if (d / "train" / "metadata.jsonl").exists())


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path,
                   help="Run directory under SSL/logs (e.g. SSL/logs/small_1).")
    p.add_argument("--ckpt", default="last.ckpt",
                   help="Checkpoint filename under <run>/checkpoints/.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of tasks to evaluate. Default: all tasks under data/SyntheticSSL/probe/.")
    p.add_argument("--out", default="eval_results.json",
                   help="Result filename (written under <run>/).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverse L2 regularization for logistic regression.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feature-source", default="backbone",
                   choices=["backbone", "projector", "both"],
                   help="Which representation to probe: the encoder's "
                        "penultimate features, the projector MLP's output, "
                        "or both (for an encoder-vs-projector comparison).")
    p.add_argument("--proj-dim", type=int, default=128,
                   help="Projector output dim — must match the checkpoint's.")
    p.add_argument("--proj-hidden", type=int, default=2048,
                   help="Projector hidden dim — must match the checkpoint's.")
    return p


def main():
    args = build_parser().parse_args()
    ckpt_path = args.run / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    sources = _resolve_sources(args.feature_source)
    need_projector = "projector" in sources

    device = torch.device(args.device)
    print(f"[eval] loading checkpoint {ckpt_path}")
    backbone, projector = load_network_from_ckpt(
        ckpt_path, need_projector=need_projector,
        proj_dim=args.proj_dim, proj_hidden=args.proj_hidden)

    tasks = args.tasks or discover_tasks()
    if not tasks:
        raise RuntimeError(f"No probe tasks found under {PROBE_ROOT}")
    print(f"[eval] probing {len(tasks)} tasks × {'+'.join(sources)} on {device}")

    results = []
    for task in tasks:
        print(f"\n=== {task} ===")
        r = probe_task(backbone, projector, task, device, sources,
                       max_iter=args.max_iter, C=args.C,
                       batch_size=args.batch_size, num_workers=args.num_workers)
        if "error" in r:
            print(f"  skipped: {r['error']}")
        else:
            for src, m in r["sources"].items():
                top5 = (f"  top5={m['top5']:.4f}" if m["top5"] is not None
                        else "")
                print(f"  {src:<10s} top1={m['top1']:.4f}{top5}  "
                      f"(classes={r['n_classes']}, D={m['feature_dim']}, "
                      f"fit={m['t_fit_s']}s)")
        results.append(r)

    out_path = args.run / args.out
    out_path.write_text(json.dumps({
        "run": str(args.run),
        "ckpt": args.ckpt,
        "feature_source": args.feature_source,
        "results": results,
    }, indent=2))
    print(f"\n[eval] wrote {out_path}")

    # Summary table — one row per (task, source).
    print("\n=== summary ===")
    print(f"{'task':<22s} {'source':<10s} {'classes':>8s} "
          f"{'top1':>8s} {'top5':>8s}")
    for r in results:
        if "error" in r:
            continue
        for src, m in r["sources"].items():
            top5 = f"{m['top5']:.4f}" if m["top5"] is not None else "—"
            print(f"{r['task']:<22s} {src:<10s} {r['n_classes']:>8d} "
                  f"{m['top1']:>8.4f} {top5:>8s}")


if __name__ == "__main__":
    main()
