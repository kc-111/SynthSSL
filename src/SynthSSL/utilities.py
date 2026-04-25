"""Paths, hierarchy loading, and asset lookup for SynthSSL.

This module is the single source of truth for:

- Where each emoji style and the ambientCG background set live on disk.
- The loaded Unicode hierarchy (``HIERARCHY``, ``EMOJIS``).
- Per-style SVG path resolution (``svg_path_for``).
- Background enumeration and sampling (``sample_background``).

All five emoji sources are SVG-based:

- ``OpenMoji``  — ``src/OpenMoji/openmoji-svg-color/<HEX>.svg`` (uppercase, dashes)
- ``Noto``     — ``src/noto-emoji/svg/emoji_u<hex>_<hex>.svg`` (lowercase, underscores)
- ``Fluent``   — ``src/fluentui-emoji/assets/<Human Name>/Flat/*.svg`` (folder-indexed)
- ``Twemoji``  — ``src/twemoji/assets/svg/<hex-with-dashes>.svg`` (lowercase, dashes)
- ``Blobmoji`` — ``src/blobmoji/{svg,svg15,derived}/<emoji_u...|human name>.svg``

Fluent and Blobmoji need a scan-once-at-import index because their
filenames aren't purely derivable from hex. All others compute their
path from hex on the fly.
"""

import json
import random
import re
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent           # src/SynthSSL/
REPO = HERE.parent.parent                        # repo root
SRC = REPO / "src"

OPENMOJI_DIR    = SRC / "OpenMoji" / "openmoji-svg-color"
NOTO_DIR        = SRC / "noto-emoji"
FLUENT_DIR      = SRC / "fluentui-emoji" / "assets"
TWEMOJI_DIR     = SRC / "twemoji" / "assets" / "svg"
BLOBMOJI_DIR    = SRC / "blobmoji"
BACKGROUND_DIR  = SRC / "ambientcg"

HIERARCHY_PATH = HERE / "hierarchy.json"

STYLES = ["openmoji", "noto", "fluent", "twemoji", "blobmoji"]


# ---------------------------------------------------------------------------
# Hierarchy (loaded once at import)
# ---------------------------------------------------------------------------

HIERARCHY = json.loads(HIERARCHY_PATH.read_text())
EMOJIS = HIERARCHY["emojis"]
"""dict[str, dict]: Mapping from normalized hex key to emoji metadata.

Each value has keys: ``emoji``, ``group``, ``subgroup``, ``name``,
``version``, ``base_hex``, ``available`` (dict with per-style bool).
"""


# ---------------------------------------------------------------------------
# Per-style SVG path resolution
# ---------------------------------------------------------------------------

def _index_both_forms(idx: dict[str, Path], key: str, path: Path) -> None:
    """Register ``path`` under ``key`` and its trailing-FE0F toggled form.

    Tolerates source-side inconsistencies around the FE0F variation
    selector so lookups succeed whether the caller uses the
    fully-qualified form (with ``-fe0f``) or the bare form.
    """
    idx.setdefault(key, path)
    if key.endswith("-fe0f"):
        idx.setdefault(key[: -len("-fe0f")], path)
    else:
        idx.setdefault(key + "-fe0f", path)


def _build_fluent_index() -> dict[str, Path]:
    """Walk Fluent metadata.json files and index by normalized hex.

    Returns:
        dict[str, Path]: ``{hex_key: Path to the Flat-style SVG}`` for
        every Fluent emoji that has a ``Flat/`` subfolder with at least
        one ``.svg`` file. Empty dict if Fluent isn't downloaded.
    """
    idx: dict[str, Path] = {}
    if not FLUENT_DIR.is_dir():
        return idx
    for meta in FLUENT_DIR.glob("*/metadata.json"):
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cps = [int(x, 16) for x in data.get("unicode", "").split()]
        if not cps:
            continue
        hex_key = "-".join(f"{cp:x}" for cp in cps)
        flat_dir = meta.parent / "Flat"
        if not flat_dir.is_dir():
            continue
        svg = next(flat_dir.glob("*.svg"), None)
        if svg is not None:
            _index_both_forms(idx, hex_key, svg)
    return idx


# Blobmoji's own filename normalization rule — lifted from its
# convert_filenames.py — used to match human-readable SVG filenames
# ("accordion.svg", "artist dark skin tone.svg") back to Unicode names.
_BLOB_DELIM  = re.compile(r"[-_. ]")
_BLOB_REMOVE = re.compile(r"""[,*\\/:'"()]""")


def _blob_normalize(s: str) -> str:
    """Normalize a name string using Blobmoji's canonical rule.

    Splits on ``[-_. space]``, strips ``[,*\\/:'"()]``, rejoins with a
    single space, lowercases. Reproduces the normalization that
    Blobmoji's own ``convert_filenames.py`` uses internally.

    Args:
        s: Any name to normalize (Unicode emoji name or Blobmoji filename stem).

    Returns:
        Normalized string — lowercase, space-separated, punctuation stripped.
    """
    parts = _BLOB_DELIM.split(s)
    parts = [_BLOB_REMOVE.sub("", p) for p in parts]
    return " ".join(parts).lower().strip()


def _build_blobmoji_index() -> dict[str, Path]:
    """Build ``{hex_key: svg_path}`` for Blobmoji, covering both naming conventions.

    Blobmoji ships SVGs with two naming styles:

    1. Hex-keyed ``emoji_u<hex>_<hex>.svg`` — parsed directly.
    2. Human-readable (e.g. ``accordion.svg``) — matched to a Unicode
       emoji name via :func:`_blob_normalize` against ``EMOJIS``.

    Scans ``svg/``, ``svg15/``, and ``derived/`` subdirectories in that
    order; the first match for a given hex wins.

    Returns:
        dict[str, Path]: ``{hex_key: path}``. Empty if Blobmoji isn't
        downloaded.
    """
    idx: dict[str, Path] = {}
    if not BLOBMOJI_DIR.is_dir():
        return idx

    name_to_hex = {_blob_normalize(v["name"]): k for k, v in EMOJIS.items()}

    for sub in ("svg", "svg15", "derived"):
        sub_dir = BLOBMOJI_DIR / sub
        if not sub_dir.is_dir():
            continue
        for svg in sub_dir.glob("*.svg"):
            stem = svg.stem
            if stem == "placeholder":
                continue

            target: str | None = None
            if stem.startswith("emoji_u"):
                hex_part = stem[len("emoji_u"):]
                try:
                    cps = [int(x, 16) for x in hex_part.split("_")]
                    target = "-".join(f"{cp:x}" for cp in cps)
                except ValueError:
                    target = None
            if target is None:
                target = name_to_hex.get(_blob_normalize(stem))
            if target is not None:
                _index_both_forms(idx, target, svg)
    return idx


FLUENT_INDEX = _build_fluent_index()
BLOBMOJI_INDEX = _build_blobmoji_index()


def _fe0f_variants(hex_key: str) -> list[str]:
    """Return candidate hex-key forms to try when resolving a path.

    Sources disagree on whether to include a trailing FE0F variation
    selector in filenames. We try the key as-is first, then with the
    trailing ``-fe0f`` stripped (or added, if absent). Also handles
    the case where FE0F appears mid-sequence for sources that strip
    it entirely — we try a fully-stripped form as a last resort.
    """
    candidates = [hex_key]
    if hex_key.endswith("-fe0f"):
        candidates.append(hex_key[: -len("-fe0f")])
    else:
        candidates.append(hex_key + "-fe0f")
    stripped_all = hex_key.replace("-fe0f", "")
    if stripped_all and stripped_all not in candidates:
        candidates.append(stripped_all)
    return candidates


def svg_path_for(hex_key: str, style: str) -> Path | None:
    """Resolve the SVG path for a given emoji and style.

    Each style names its files differently — OpenMoji / Noto / Blobmoji
    pad every codepoint to a minimum of 4 hex digits, while Twemoji and
    our normalized hex key drop leading zeros. When inverting the hex
    key back into a filename, we pad per-segment for the padded styles.

    FE0F variation selectors in filenames are inconsistent across sources
    (Unicode's fully-qualified forms require them, most SVG filenames omit
    them). We try multiple candidate keys via :func:`_fe0f_variants` to
    tolerate this.

    Args:
        hex_key: Normalized hex key (lowercase, dash-separated). Example:
            ``"1f436"`` for 🐶, ``"1f468-200d-1f469-200d-1f467"`` for 👨‍👩‍👧.
        style: One of ``STYLES``.

    Returns:
        Path to the SVG file, or ``None`` if no variant resolves for this
        style. Callers should typically check the ``available`` flag in
        ``EMOJIS[hex_key]`` first to avoid redundant file-existence checks.
    """
    for candidate in _fe0f_variants(hex_key):
        if style == "openmoji":
            parts = [p.upper().zfill(4) for p in candidate.split("-")]
            p = OPENMOJI_DIR / f"{'-'.join(parts)}.svg"
            if p.exists():
                return p
        elif style == "noto":
            parts = [p.zfill(4) for p in candidate.split("-")]
            p = NOTO_DIR / "svg" / f"emoji_u{'_'.join(parts)}.svg"
            if p.exists():
                return p
        elif style == "twemoji":
            p = TWEMOJI_DIR / f"{candidate}.svg"
            if p.exists():
                return p
        elif style == "fluent":
            p = FLUENT_INDEX.get(candidate)
            if p is not None:
                return p
        elif style == "blobmoji":
            p = BLOBMOJI_INDEX.get(candidate)
            if p is not None:
                return p
    return None


# ---------------------------------------------------------------------------
# Backgrounds (ambientCG)
# ---------------------------------------------------------------------------

_CATEGORY_RE = re.compile(r"^([A-Za-z]+)\d")


def _background_category(asset_id: str) -> str:
    """Extract the ambientCG category from an asset ID.

    Asset IDs follow ``<Category><Number>[Variant]`` (e.g. ``Rock064``,
    ``WoodFloor051``, ``Road012A``). The category is the alphabetic
    prefix up to the first digit.

    Args:
        asset_id: Stem of the JPG file, e.g. ``"PavingStones150"``.

    Returns:
        The category string, e.g. ``"PavingStones"``. Returns the input
        unchanged if no digit is found (shouldn't happen for real
        ambientCG IDs).
    """
    m = _CATEGORY_RE.match(asset_id)
    return m.group(1) if m else asset_id


def _build_background_index() -> dict[str, list[Path]]:
    """Group ambientCG background JPGs by category.

    Returns:
        dict[str, list[Path]]: ``{category: [jpg_path, ...]}`` sorted
        alphabetically by path within each category. Empty dict if the
        background directory isn't present.
    """
    idx: dict[str, list[Path]] = {}
    if not BACKGROUND_DIR.is_dir():
        return idx
    for p in sorted(BACKGROUND_DIR.glob("*.jpg")):
        cat = _background_category(p.stem)
        idx.setdefault(cat, []).append(p)
    return idx


BACKGROUND_INDEX = _build_background_index()
"""dict[str, list[Path]]: Category → sorted list of JPG paths."""

BACKGROUND_PATHS = [p for paths in BACKGROUND_INDEX.values() for p in paths]
"""list[Path]: Flat list of every background path (any category)."""

BACKGROUND_CATEGORIES = sorted(BACKGROUND_INDEX.keys())
"""list[str]: Sorted category names present on disk."""


def sample_background(
    size: int,
    category: str | None = None,
    rng: random.Random | None = None,
) -> tuple[Image.Image, str]:
    """Sample a random ambientCG background, cropped to a square at ``size``.

    Loads the JPG, picks a random ``size × size`` crop, and returns the
    crop along with the asset ID (filename stem) of the source file.
    Source tiles are typically 1024×1024 — if ``size`` exceeds the
    source dimensions, the image is resized up first (not ideal;
    prefer sampling with ``size <= 1024``).

    Args:
        size: Output side length in pixels. Square crop.
        category: Optional ambientCG category name (e.g. ``"Wood"``,
            ``"Bricks"``). If ``None``, samples uniformly across all
            backgrounds. If the category has no files, falls back to
            the full pool.
        rng: Optional seeded ``random.Random`` for reproducibility. If
            ``None``, uses the global ``random`` module.

    Returns:
        Tuple of:
          - PIL ``Image`` in RGB mode, sized ``(size, size)``.
          - Asset ID (filename stem, e.g. ``"Wood024"``) — useful for
            writing provenance metadata alongside the crop.

    Raises:
        RuntimeError: If no backgrounds are available on disk at all.
    """
    rng = rng or random

    if category and BACKGROUND_INDEX.get(category):
        pool = BACKGROUND_INDEX[category]
    else:
        pool = BACKGROUND_PATHS

    if not pool:
        raise RuntimeError(
            f"No backgrounds available in {BACKGROUND_DIR}. "
            "Run the ambientCG download from DOWNLOAD.md."
        )

    path = rng.choice(pool)
    img = Image.open(path).convert("RGB")
    asset_id = path.stem          # e.g. "Wood024"

    W, H = img.size
    if W < size or H < size:
        img = img.resize((max(W, size), max(H, size)), Image.LANCZOS)
        W, H = img.size

    x = rng.randint(0, W - size)
    y = rng.randint(0, H - size)
    return img.crop((x, y, x + size, y + size)), asset_id
