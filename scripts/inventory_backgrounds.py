"""Print the background categories present on disk.

Walks ``src/ambientcg/*.jpg`` via ``utilities.BACKGROUND_INDEX``, groups
by category (derived from the asset-ID prefix, e.g. ``Rock064`` →
``Rock``), and prints a per-category count plus a few example asset
IDs. Intended as a quick local audit after running the download.

Run from the repo root::

    python scripts/inventory_backgrounds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from SynthSSL.utilities import (  # noqa: E402  (path tweak above)
    BACKGROUND_CATEGORIES,
    BACKGROUND_DIR,
    BACKGROUND_INDEX,
    BACKGROUND_PATHS,
)


def main():
    if not BACKGROUND_PATHS:
        print(f"No backgrounds found in {BACKGROUND_DIR}.")
        print("Run the ambientCG download from DOWNLOAD.md first.")
        return

    print(f"Background root:     {BACKGROUND_DIR}")
    print(f"Total files:         {len(BACKGROUND_PATHS)}")
    print(f"Categories on disk:  {len(BACKGROUND_CATEGORIES)}")
    print()
    print(f"{'category':<25s} {'count':>6s}  sample asset IDs")
    print("-" * 72)
    ordered = sorted(BACKGROUND_INDEX.items(), key=lambda kv: -len(kv[1]))
    for cat, paths in ordered:
        sample = ", ".join(p.stem for p in paths[:2])
        print(f"{cat:<25s} {len(paths):>6d}  {sample}")


if __name__ == "__main__":
    main()
