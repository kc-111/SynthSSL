"""Audit factor coverage on a generated SyntheticSSL dataset.

Answers the question "how do we know everything got covered in
generation?" by actually *counting* unique emojis, backgrounds,
object-counts, styles, and noise types across every split on disk.

What it measures
================

For each directory (``pretrain_*`` and every ``probe/<task>/{train,test}``):

- **Unique anchor emojis** — should equal the pool size the generator
  used. For pretrain this is the full ~3,944. For probe train it's
  the anchor-stratified pool. For probe test it's the subset that made
  the 80/20 cut.
- **Unique backgrounds (asset ids + categories)** — ambientCG files
  get random-sampled per scene so we want to see every file used.
- **Object-count distribution** — scenes should cover N ∈ {1..5}.
- **Style distribution** — should span all 5 emoji styles in pretrain.
- **Noise-type distribution** — should span {none, gaussian, …} where
  applicable.

For probe splits it also **cross-checks train vs test**: every test
emoji must be in train (anchor-stratified generation guarantees this;
the audit verifies it actually held).

Usage
-----
::

    python scripts/audit_coverage.py                    # audit default dataset
    python scripts/audit_coverage.py --root data/SyntheticSSL/small

Reports a summary table per split + a cross-split check at the end.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Scanning one metadata.jsonl
# ---------------------------------------------------------------------------

def _scan_metadata(meta_path: Path) -> dict:
    """Walk a metadata.jsonl and tally factor frequencies.

    Returns a dict::

        {
            "n_records":        total record count,
            "n_errors":         records with 'error' field,
            "emoji":            Counter of anchor emoji hex codes,
            "object_count":     Counter of len(objects),
            "style":            Counter of anchor object's style,
            "bg_category":      Counter of background.base_category,
            "bg_id":            Counter of background.base_id (ambientCG filenames),
            "noise_type":       Counter of background.noise_type,
        }
    """
    emoji, obj_count, style = Counter(), Counter(), Counter()
    bg_cat, bg_id, noise = Counter(), Counter(), Counter()
    n, errors = 0, 0

    with meta_path.open() as f:
        for line in f:
            n += 1
            r = json.loads(line)
            if "error" in r:
                errors += 1
                continue

            # Anchor object (objects[anchor_index], default 0).
            idx = r.get("anchor_index", 0)
            objs = r.get("objects", [])
            if objs and 0 <= idx < len(objs):
                anchor = objs[idx]
                emoji[anchor.get("hex")] += 1
                style[anchor.get("style")] += 1
            obj_count[len(objs)] += 1

            # Background metadata (may be flat or nested depending on gen path).
            bg = r.get("background", {}) or {}
            bg_cat[bg.get("base_category")] += 1
            if bg.get("base_id") is not None:
                bg_id[bg["base_id"]] += 1
            noise[bg.get("noise_type", "none")] += 1

    return {
        "n_records": n,
        "n_errors": errors,
        "emoji": emoji,
        "object_count": obj_count,
        "style": style,
        "bg_category": bg_cat,
        "bg_id": bg_id,
        "noise_type": noise,
    }


# ---------------------------------------------------------------------------
# Pretty printer — one report block per split
# ---------------------------------------------------------------------------

def _print_block(label: str, s: dict):
    """Human-readable summary for one split's scan result."""
    print(f"\n=== {label} ===")
    print(f"  records: {s['n_records']:>7d}   errors: {s['n_errors']}")

    print(f"  unique anchor emojis: {len(s['emoji']):>5d}   "
          f"min/max occurrences: {min(s['emoji'].values(), default=0)} / "
          f"{max(s['emoji'].values(), default=0)}")

    print(f"  unique bg categories: {len(s['bg_category']):>5d}   "
          f"unique ambientCG ids: {len(s['bg_id']):>5d}")

    # Distribution bars for small-class factors.
    def _bars(counter: Counter, width: int = 40) -> str:
        if not counter:
            return "  (none)"
        total = sum(counter.values())
        max_n = max(counter.values())
        lines = []
        for k, v in counter.most_common():
            bar = "█" * max(1, int(width * v / max_n))
            pct = 100 * v / total
            lines.append(f"    {str(k):<20s} {v:>7d}  {pct:>5.1f}%  {bar}")
        return "\n".join(lines)

    print("  object-count distribution:")
    print(_bars(s["object_count"]))
    print("  style distribution:")
    print(_bars(s["style"]))
    print("  noise-type distribution:")
    print(_bars(s["noise_type"]))


# ---------------------------------------------------------------------------
# Cross-split consistency checks (probe-only)
# ---------------------------------------------------------------------------

def _probe_cross_check(task: str, train: dict, test: dict):
    """Verify invariants that anchor-stratified generation should give.

    Specifically: every emoji in test must also be in train. Reports
    violations if any, otherwise confirms coverage.
    """
    train_emojis = set(train["emoji"].keys())
    test_emojis = set(test["emoji"].keys())
    unseen = test_emojis - train_emojis
    shared = test_emojis & train_emojis

    if test_emojis:
        pct_shared = 100 * len(shared) / len(test_emojis)
    else:
        pct_shared = 0.0

    status = "✓" if not unseen else "✗"
    print(f"  [{status}] probe/{task}: {len(test_emojis)} test emojis, "
          f"{len(shared)} also in train ({pct_shared:.1f}%)"
          + (f", {len(unseen)} test-only" if unseen else ""))
    if unseen:
        sample = ", ".join(sorted(unseen)[:8])
        print(f"        examples never in train: {sample}"
              + ("..." if len(unseen) > 8 else ""))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _candidate_splits(root: Path) -> list[tuple[str, Path]]:
    """Find all (label, metadata_path) pairs under the dataset root.

    Walks root for:
      - ``pretrain_*/metadata.jsonl``
      - ``probe/<task>/{train,test}/metadata.jsonl``
    """
    found: list[tuple[str, Path]] = []

    # Pretrain directories (pretrain_small, pretrain_large, etc.).
    for p in sorted(root.glob("pretrain_*")):
        meta = p / "metadata.jsonl"
        if meta.exists():
            found.append((p.name, meta))

    # Probe tasks.
    probe_root = root / "probe"
    if probe_root.is_dir():
        for task_dir in sorted(probe_root.iterdir()):
            for split in ("train", "test"):
                meta = task_dir / split / "metadata.jsonl"
                if meta.exists():
                    found.append((f"probe/{task_dir.name}/{split}", meta))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "data" / "SyntheticSSL",
                    help="Dataset root (default: data/SyntheticSSL)")
    ap.add_argument("--quiet", action="store_true",
                    help="Skip per-split distribution tables; just "
                         "report headline counts + cross-split checks.")
    args = ap.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"{args.root} does not exist")

    splits = _candidate_splits(args.root)
    if not splits:
        raise RuntimeError(f"no metadata.jsonl files found under {args.root}")

    scans: dict[str, dict] = {}
    for label, meta_path in splits:
        scans[label] = _scan_metadata(meta_path)
        if not args.quiet:
            _print_block(label, scans[label])

    # Probe cross-checks: every test split's emoji set should be a
    # subset of its train split's emoji set.
    print("\n=== cross-split checks ===")
    for label in scans:
        if not label.startswith("probe/") or not label.endswith("/test"):
            continue
        task = label.split("/")[1]
        train_label = f"probe/{task}/train"
        if train_label not in scans:
            continue
        _probe_cross_check(task, scans[train_label], scans[label])

    # Summary table.
    print("\n=== summary ===")
    print(f"{'split':<40s} {'records':>8s} {'emojis':>7s} "
          f"{'bg_cats':>8s} {'bg_ids':>7s}")
    for label, s in scans.items():
        print(f"{label:<40s} {s['n_records']:>8d} "
              f"{len(s['emoji']):>7d} {len(s['bg_category']):>8d} "
              f"{len(s['bg_id']):>7d}")


if __name__ == "__main__":
    main()
