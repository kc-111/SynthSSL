"""Generate the per-task probe evaluation datasets (shared across recipes).

Output::

    data/SyntheticSSL/probe/
      group/     train/ test/ metadata.jsonl
      subgroup/  train/ test/ metadata.jsonl
      base_leaf/ …
      style/ grid3x3/ scale/ object-count/
      background-base/ background-noise/ background-color/
      unicode-color/

Safe to run before, after, or between the pretrain recipes. Existing
task directories are skipped so re-running is a no-op.

Run from repo root::

    python scripts/generate_probes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from SynthSSL.generate import generate_all_probes  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "SyntheticSSL" / "probe"

PROBE_PER_ANCHOR = 5
WORKERS = 8
PROBE_SEED = 100


def main():
    print(f"=== SyntheticSSL probe datasets → {OUT_ROOT} ===")
    generate_all_probes(
        out_root=OUT_ROOT,
        per_anchor=PROBE_PER_ANCHOR,
        seed=PROBE_SEED,
        workers=WORKERS,
        skip_existing=True,
    )


if __name__ == "__main__":
    main()
