"""Label-extraction functions per probe task.

Each probe task has a pure function ``label_for_<task>(metadata_record)``
that reads a scene's metadata dict and returns the class label. The
generator calls the relevant one when writing a per-task dataset's
``metadata.jsonl`` (one line per sample with ``{image_path, label, ...}``).

A single ``SCALE_BINS`` / ``GRID_N`` / ``COLOR_BUCKETS_ORDER`` constant
per task keeps the bin edges canonical; probe code can import these to
sanity-check label distributions.
"""

from __future__ import annotations

from typing import Any, Callable

from SynthSSL.scene import COLOR_BUCKETS, OBJECT_SIZE_RANGE, UNICODE_COLOR_VARIANTS


# ---------------------------------------------------------------------------
# Bin edges / ordering constants
# ---------------------------------------------------------------------------

SCALE_BINS = 10
SCALE_BIN_EDGES = tuple(
    OBJECT_SIZE_RANGE[0] + i * (OBJECT_SIZE_RANGE[1] - OBJECT_SIZE_RANGE[0]) / SCALE_BINS
    for i in range(1, SCALE_BINS)
)

GRID_N = 3   # 3x3 grid

# Ordered lists so class indices are stable across runs.
COLOR_BUCKETS_ORDER = tuple(COLOR_BUCKETS.keys())
UNICODE_COLORS_ORDER = tuple(UNICODE_COLOR_VARIANTS.keys())

OBJECT_COUNT_CLASSES = (1, 2, 3, 4, 5)

NOISE_TYPES_ORDER = ("none", "gaussian", "laplacian", "uniform", "salt-pepper", "pink")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anchor(record: dict) -> dict:
    """Return ``objects[anchor_index]`` for an anchor-based task."""
    idx = record.get("anchor_index", 0)
    return record["objects"][idx]


def _bin_scale(scale_px: int) -> int:
    """Bin a pixel size into 0..SCALE_BINS-1 over [100, 320]."""
    lo, hi = OBJECT_SIZE_RANGE
    scale_px = max(lo, min(hi - 1, scale_px))
    bin_idx = int((scale_px - lo) / (hi - lo) * SCALE_BINS)
    return min(SCALE_BINS - 1, bin_idx)


def _grid_cell(x: float, y: float, n: int = GRID_N) -> int:
    """3×3 grid cell index from normalized (x, y) ∈ [0, 1]², row-major."""
    x = max(0.0, min(0.9999, x))
    y = max(0.0, min(0.9999, y))
    col = int(x * n)
    row = int(y * n)
    return row * n + col


# ---------------------------------------------------------------------------
# Per-task label functions
# ---------------------------------------------------------------------------

def label_group(r: dict) -> str:
    return _anchor(r)["group"]


def label_subgroup(r: dict) -> str:
    return _anchor(r)["subgroup"]


def label_leaf(r: dict) -> str:
    return _anchor(r)["hex"]


def label_base_leaf(r: dict) -> str:
    return _anchor(r)["base_hex"]


def label_style(r: dict) -> str:
    return _anchor(r)["style"]


def label_grid3x3(r: dict) -> int:
    """3×3 grid cell of the anchor emoji's center (not top-left).

    position_xy is the top-left corner in normalized [0, 1] coordinates;
    center = position_xy + 0.5 * (scale_px / CANVAS, scale_px / CANVAS).
    CANVAS = 512, so scale_fraction = scale_px / 512.
    """
    obj = _anchor(r)
    x_tl, y_tl = obj["position_xy"]
    scale_frac = obj["scale_px"] / 512
    cx = x_tl + 0.5 * scale_frac
    cy = y_tl + 0.5 * scale_frac
    return _grid_cell(cx, cy)


def label_scale(r: dict) -> int:
    return _bin_scale(_anchor(r)["scale_px"])


def label_object_count(r: dict) -> int:
    """0-indexed class: returns count - 1 since counts range 1..5."""
    n = len(r["objects"])
    n = max(1, min(5, n))
    return n - 1


def label_background_base(r: dict) -> str:
    return r["background"]["base_category"]


def label_background_noise(r: dict) -> str:
    return r["background"]["noise_type"]


def label_background_color(r: dict) -> str:
    bucket = r["background"].get("color_bucket")
    if bucket is None:
        raise ValueError(
            "background-color task requires color_bucket in metadata "
            "(was the scene generated with solid_color_bucket set?)"
        )
    return bucket


def label_unicode_color(r: dict) -> str:
    """Return the Unicode-color-variant bucket for the anchor emoji."""
    anchor_hex = _anchor(r)["hex"]
    for color, hexes in UNICODE_COLOR_VARIANTS.items():
        if anchor_hex in hexes:
            return color
    raise ValueError(
        f"unicode-color task anchor {anchor_hex!r} isn't in "
        "UNICODE_COLOR_VARIANTS — scene spec was misconfigured."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LABEL_FUNCTIONS: dict[str, Callable[[dict], Any]] = {
    "group":            label_group,
    "subgroup":         label_subgroup,
    "leaf":             label_leaf,             # used against pretrain data
    "base_leaf":        label_base_leaf,
    "style":            label_style,
    "grid3x3":          label_grid3x3,
    "scale":            label_scale,
    "object-count":     label_object_count,
    "background-base":  label_background_base,
    "background-noise": label_background_noise,
    "background-color": label_background_color,
    "unicode-color":    label_unicode_color,
}
