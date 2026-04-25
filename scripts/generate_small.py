"""Generate the SMALL SyntheticSSL recipe — K=10 pretrain + all probes.

Output::

    data/SyntheticSSL/
      pretrain_small/             # ~39,440 unlabeled images + metadata.jsonl
      probe/                      # shared across recipes
        group/     train/ test/ metadata.jsonl
        subgroup/  train/ test/ metadata.jsonl
        …

The probe directory is shared between recipes (see generate_large.py).
If it already exists (every task has ``train/metadata.jsonl``) the probe
step skips — run either recipe first, then the other only re-does
the pretrain half.

Run from repo root::

    python scripts/generate_small.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from SynthSSL.generate import generate_all_probes, generate_pretrain  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "SyntheticSSL"

# Knobs
K = 10
PROBE_PER_ANCHOR = 5   # scenes per emoji; total ≈ 5 × n_anchors ≈ 20k
WORKERS = 8
PRETRAIN_SEED = 0
PROBE_SEED = 100


def main():
    print(f"=== SyntheticSSL SMALL recipe → {OUT_ROOT} ===")
    generate_pretrain(
        out_dir=OUT_ROOT / "pretrain_small",
        K=K,
        seed=PRETRAIN_SEED,
        workers=WORKERS,
    )
    generate_all_probes(
        out_root=OUT_ROOT / "probe",
        per_anchor=PROBE_PER_ANCHOR,
        seed=PROBE_SEED,
        workers=WORKERS,
        skip_existing=True,
    )


if __name__ == "__main__":
    main()
