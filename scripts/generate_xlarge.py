"""Generate the LARGE SyntheticSSL recipe — K=50 pretrain + all probes.

Output::

    data/SyntheticSSL/
      pretrain_large/             # ~197,200 unlabeled images + metadata.jsonl
      probe/                      # shared across recipes
        group/     train/ test/ metadata.jsonl
        subgroup/  train/ test/ metadata.jsonl
        …

The probe directory is shared with the SMALL recipe. Running this after
``generate_small.py`` re-does the pretrain half only; probe generation
skips when each task's ``train/metadata.jsonl`` already exists.

Run from repo root::

    python scripts/generate_large.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from SynthSSL.generate import generate_all_probes, generate_pretrain  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "SyntheticSSL"

# Knobs
K = 250
PROBE_PER_ANCHOR = 5
WORKERS = 8
PRETRAIN_SEED = 0
PROBE_SEED = 100


def main():
    print(f"=== SyntheticSSL XLARGE recipe → {OUT_ROOT} ===")
    generate_pretrain(
        out_dir=OUT_ROOT / "pretrain_xlarge",
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
