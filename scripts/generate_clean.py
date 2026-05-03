"""Generate the CLEAN SyntheticSSL recipe — emojis only, no background, no noise.

Output::

    data/SyntheticSSL/
      pretrain_clean/                     # K-anchored, flat-white background
      probe/
        group_clean/    train/ test/ metadata.jsonl
        subgroup_clean/ train/ test/ metadata.jsonl

The "clean" variant strips procedural / ambientCG backgrounds, drop
shadows, and the noise overlay. Every scene is a single emoji on a
white canvas — useful as an emoji-recognition baseline / ablation.

Run from repo root::

    python scripts/generate_clean.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from SynthSSL.generate import generate_pretrain, generate_probe  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "SyntheticSSL"

# Knobs — match SMALL recipe sizes by default; bump K for a larger pretrain.
K = 10
PROBE_PER_ANCHOR = 5
WORKERS = 8
PRETRAIN_SEED = 0
PROBE_SEED = 100

CLEAN_PROBE_TASKS = ["group_clean", "subgroup_clean"]


def main():
    print(f"=== SyntheticSSL CLEAN recipe → {OUT_ROOT} ===")
    generate_pretrain(
        out_dir=OUT_ROOT / "pretrain_clean",
        K=K,
        seed=PRETRAIN_SEED,
        workers=WORKERS,
        task="pretrain_clean",
    )
    for i, task in enumerate(CLEAN_PROBE_TASKS):
        task_dir = OUT_ROOT / "probe" / task
        if (task_dir / "train" / "metadata.jsonl").exists():
            print(f"[probe:{task}] skip (already present)")
            continue
        generate_probe(
            task=task,
            out_dir=task_dir,
            per_anchor=PROBE_PER_ANCHOR,
            seed=PROBE_SEED + i * 101,
            workers=WORKERS,
        )


if __name__ == "__main__":
    main()
