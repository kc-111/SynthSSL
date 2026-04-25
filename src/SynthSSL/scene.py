"""Scene composition — build one (image, metadata) sample.

Four-layer pipeline, strict z-order bottom → top:

    1. Background — solid color / gradient / perlin-like / voronoi /
       gabor / ambientCG texture.
    2. Noise overlay — optional per-pixel modulation (gaussian /
       laplacian / uniform / salt-pepper / pink).
    3. Anchor emoji — guaranteed present, `anchor_index=0` in metadata.
    4. Clutter emojis — `N − 1` more, placed with non-overlap rejection.

No training-time augmentation here — that's `SSL/train.py`'s job.
Every probe task is expressed as a ``SceneSpec`` that constrains object
count, background source, noise behavior, or emoji pool.

Run this module directly for a visual demo of one scene per probe task::

    python src/SynthSSL/scene.py
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from SynthSSL.objects import render
from SynthSSL.utilities import (
    BACKGROUND_CATEGORIES,
    EMOJIS,
    REPO,
    STYLES,
    sample_background,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS = 256
OBJECT_SIZE_RANGE = (50, 160)           # min/max short-side px per object
ALPHA_RANGE = (0.85, 1.0)
NOISE_PROB = 0.5                         # for pretrain; probes override
MAX_PLACE_RETRIES = 50
# Reject a placement if intersection/min(area) exceeds this threshold.
# 0.0 = no overlap at all. 0.2 allows brushing contact / corner clipping but
# rejects anything close to half-coverage.
MAX_OVERLAP_IOMIN = 0.2

STRUCTURED_BACKGROUNDS = ["solid", "gradient", "perlin", "voronoi", "gabor"]
NOISE_TYPES = ["gaussian", "laplacian", "uniform", "salt-pepper", "pink"]

# HSV color buckets (H in degrees, S and V in [0, 1]). Mutually exclusive
# by construction — the label is the bucket name, not the RGB.
COLOR_BUCKETS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "red":    ((  0,  20), (0.6, 1.0), (0.6, 1.0)),
    "orange": (( 20,  45), (0.7, 1.0), (0.7, 1.0)),
    "yellow": (( 45,  70), (0.6, 1.0), (0.8, 1.0)),
    "green":  (( 70, 160), (0.5, 1.0), (0.4, 0.9)),
    "cyan":   ((160, 200), (0.5, 1.0), (0.6, 1.0)),
    "blue":   ((200, 250), (0.5, 1.0), (0.4, 0.9)),
    "purple": ((250, 290), (0.4, 0.9), (0.4, 0.9)),
    "pink":   ((290, 340), (0.3, 0.7), (0.8, 1.0)),
    "brown":  (( 15,  40), (0.4, 0.7), (0.2, 0.5)),
    "gray":   ((  0, 360), (0.0, 0.1), (0.3, 0.7)),
}

# Unicode color-variant emoji pools — hex keys for circles, squares,
# hearts. Ground-truth color labels taken from Unicode's own names.
UNICODE_COLOR_VARIANTS: dict[str, list[str]] = {
    "red":    ["1f534", "1f7e5", "2764-fe0f"],
    "orange": ["1f7e0", "1f7e7", "1f9e1"],
    "yellow": ["1f7e1", "1f7e8", "1f49b"],
    "green":  ["1f7e2", "1f7e9", "1f49a"],
    "blue":   ["1f535", "1f7e6", "1f499"],
    "purple": ["1f7e3", "1f7ea", "1f49c"],
    "brown":  ["1f7e4", "1f7eb", "1f90e"],
    "black":  ["26ab",  "2b1b",  "1f5a4"],
    "white":  ["26aa",  "2b1c",  "1f90d"],
    "pink":   ["1fa77"],
}


# ---------------------------------------------------------------------------
# Scene spec
# ---------------------------------------------------------------------------

@dataclass
class SceneSpec:
    """Constraints on the scene generator for a particular probe task.

    Defaults reproduce the pretrain distribution (full §2 pipeline).
    Probe tasks override individual fields; any field left unset samples
    from the full pretrain distribution.

    Attributes:
        n_objects: Fixed count (int) or range (tuple[int, int], inclusive).
        anchor_hex: If set, the anchor emoji is this exact hex key.
        anchor_style: If set, the anchor emoji is rendered in this style.
        emoji_pool: Restrict sampled emoji keys to this set. None = all.
        base_source_pool: Restrict background sources to this subset
            (a mix of ``STRUCTURED_BACKGROUNDS`` and ``"ambientcg"``).
        solid_color_bucket: If set, force the background to solid color
            from this COLOR_BUCKETS bucket. Implies ``base_source = "solid"``.
        noise_enabled: False disables the noise overlay entirely.
        noise_type_forced: Force a specific noise type (from NOISE_TYPES).
        shadows: Whether per-object drop shadows are allowed.
        style_pool: Restrict anchor style to this subset (useful for the
            style probe which cycles over all 5 styles).
    """
    n_objects: int | tuple[int, int] = (1, 5)
    anchor_hex: str | None = None
    anchor_style: str | None = None
    emoji_pool: list[str] | None = None
    base_source_pool: list[str] | None = None
    solid_color_bucket: str | None = None
    noise_enabled: bool = True
    noise_type_forced: str | None = None
    shadows: bool = True
    style_pool: list[str] | None = None


# ---------------------------------------------------------------------------
# Background samplers (each returns a (PIL.Image RGB, metadata dict))
# ---------------------------------------------------------------------------

def _sample_rgb_in_bucket(rng: random.Random, bucket: str) -> tuple[int, int, int]:
    """Draw a single RGB tuple uniformly within a color bucket's HSV box."""
    H_range, S_range, V_range = COLOR_BUCKETS[bucket]
    h = rng.uniform(*H_range) / 360.0
    s = rng.uniform(*S_range)
    v = rng.uniform(*V_range)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def bg_solid(rng: random.Random, color_bucket: str | None = None) -> tuple[Image.Image, dict]:
    bucket = color_bucket or rng.choice(list(COLOR_BUCKETS))
    rgb = _sample_rgb_in_bucket(rng, bucket)
    img = Image.new("RGB", (CANVAS, CANVAS), rgb)
    return img, {
        "base_source": "procedural",
        "base_id": None,
        "base_category": "solid",
        "color_bucket": bucket,
    }


def bg_gradient(rng: random.Random) -> tuple[Image.Image, dict]:
    c1 = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    theta = rng.uniform(0.0, 2 * np.pi)
    yy, xx = np.meshgrid(np.arange(CANVAS), np.arange(CANVAS), indexing="ij")
    proj = (xx - CANVAS / 2) * np.cos(theta) + (yy - CANVAS / 2) * np.sin(theta)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    arr = (1 - proj)[:, :, None] * c1 + proj[:, :, None] * c2
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img, {"base_source": "procedural", "base_id": None,
                 "base_category": "gradient", "color_bucket": None}


def bg_perlin(rng: random.Random) -> tuple[Image.Image, dict]:
    """Smoothed Gaussian-noise proxy for Perlin (low-frequency cloudy texture)."""
    seed = rng.randrange(2**31)
    nrng = np.random.default_rng(seed)
    arr = nrng.normal(0, 1, (CANVAS, CANVAS, 3)).astype(np.float32)
    img = Image.fromarray(((arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(16, 48)))
    return img, {"base_source": "procedural", "base_id": None,
                 "base_category": "perlin", "color_bucket": None}


def bg_voronoi(rng: random.Random) -> tuple[Image.Image, dict]:
    """Random Voronoi cells — pick K seed points, color each cell."""
    K = rng.randint(8, 32)
    seed = rng.randrange(2**31)
    nrng = np.random.default_rng(seed)
    pts = nrng.uniform(0, CANVAS, size=(K, 2))
    colors = nrng.integers(0, 256, size=(K, 3))
    yy, xx = np.meshgrid(np.arange(CANVAS), np.arange(CANVAS), indexing="ij")
    coords = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32)
    dists = np.linalg.norm(coords[:, None, :] - pts[None, :, :], axis=-1)
    labels = np.argmin(dists, axis=-1)
    arr = colors[labels].reshape(CANVAS, CANVAS, 3).astype(np.uint8)
    return Image.fromarray(arr), {"base_source": "procedural", "base_id": None,
                                  "base_category": "voronoi", "color_bucket": None}


def bg_gabor(rng: random.Random) -> tuple[Image.Image, dict]:
    """Gabor patch — sinusoidal grating modulated by a Gaussian envelope."""
    freq = rng.uniform(0.02, 0.08)
    theta = rng.uniform(0.0, np.pi)
    yy, xx = np.meshgrid(np.arange(CANVAS), np.arange(CANVAS), indexing="ij")
    yy = yy - CANVAS / 2
    xx = xx - CANVAS / 2
    xp = xx * np.cos(theta) + yy * np.sin(theta)
    grating = np.sin(2 * np.pi * freq * xp)
    envelope = np.exp(-(xx ** 2 + yy ** 2) / (2 * (CANVAS / 3) ** 2))
    patch = grating * envelope
    # Map to RGB with two random endpoint colors
    c1 = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    p = (patch - patch.min()) / (patch.max() - patch.min() + 1e-9)
    arr = (1 - p)[:, :, None] * c1 + p[:, :, None] * c2
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img, {"base_source": "procedural", "base_id": None,
                 "base_category": "gabor", "color_bucket": None}


def bg_ambientcg(rng: random.Random, category: str | None = None) -> tuple[Image.Image, dict]:
    """Random crop from an ambientCG texture, with full provenance.

    ``sample_background`` returns (image, asset_id) so we can record
    which specific file was used — useful for coverage audits and
    for reproducing any particular scene.
    """
    img, asset_id = sample_background(CANVAS, category=category, rng=rng)
    return img, {"base_source": "ambientcg",
                 "base_id": asset_id,     # e.g. "Wood024"
                 "base_category": category or "ambientcg-mixed",
                 "color_bucket": None}


_BG_DISPATCH = {
    "solid": bg_solid,
    "gradient": bg_gradient,
    "perlin": bg_perlin,
    "voronoi": bg_voronoi,
    "gabor": bg_gabor,
}


def sample_background_for(rng: random.Random, spec: SceneSpec) -> tuple[Image.Image, dict]:
    """Sample a background according to the scene spec.

    Solid-color-bucket constraint wins: if ``spec.solid_color_bucket`` is
    set, always produces a solid of that bucket. Otherwise picks uniformly
    from ``spec.base_source_pool`` (or all sources if unset).
    """
    if spec.solid_color_bucket is not None:
        return bg_solid(rng, color_bucket=spec.solid_color_bucket)

    if spec.base_source_pool is not None:
        source = rng.choice(spec.base_source_pool)
    else:
        pool = STRUCTURED_BACKGROUNDS + ["ambientcg"]
        source = rng.choice(pool)

    if source == "ambientcg":
        category = rng.choice(BACKGROUND_CATEGORIES) if BACKGROUND_CATEGORIES else None
        return bg_ambientcg(rng, category=category)
    return _BG_DISPATCH[source](rng)


# ---------------------------------------------------------------------------
# Noise overlays
# ---------------------------------------------------------------------------

NOISE_STRENGTH_RANGES = {
    "gaussian":    (0.02, 0.12),
    "laplacian":   (0.02, 0.10),
    "uniform":     (0.03, 0.15),
    "salt-pepper": (0.002, 0.03),
    "pink":        (0.05, 0.20),
}


def apply_noise(img: Image.Image, rng: random.Random, noise_type: str,
                strength: float) -> Image.Image:
    """Return a copy of ``img`` with the given noise overlay applied.

    All noise operates on float RGB in [0, 1] then clamps back to [0, 255].
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0
    H, W, _ = arr.shape
    seed = rng.randrange(2**31)
    nrng = np.random.default_rng(seed)

    if noise_type == "gaussian":
        arr = arr + nrng.normal(0, strength, arr.shape)
    elif noise_type == "laplacian":
        arr = arr + nrng.laplace(0, strength, arr.shape)
    elif noise_type == "uniform":
        arr = arr + nrng.uniform(-strength, strength, arr.shape)
    elif noise_type == "salt-pepper":
        mask = nrng.random(arr.shape[:2]) < strength
        values = nrng.integers(0, 2, size=mask.sum()).astype(np.float32)  # 0 or 1
        arr[mask] = values[:, None]
    elif noise_type == "pink":
        # 1/f spectrum: white noise filtered with per-freq scaling.
        white = nrng.normal(0, 1, (H, W, 3))
        F = np.fft.fft2(white, axes=(0, 1))
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        mag = np.sqrt(fy * fy + fx * fx) + 1e-6
        F = F / mag[..., None]
        pink = np.real(np.fft.ifft2(F, axes=(0, 1)))
        pink = (pink - pink.mean()) / (pink.std() + 1e-9)
        arr = arr + strength * pink
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}")

    return Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))


def maybe_apply_noise(img: Image.Image, rng: random.Random,
                      spec: SceneSpec) -> tuple[Image.Image, dict]:
    """Apply a noise overlay according to the spec. Returns (img, metadata)."""
    if not spec.noise_enabled:
        return img, {"noise_type": "none", "noise_strength": 0.0}
    if spec.noise_type_forced is None and rng.random() >= NOISE_PROB:
        return img, {"noise_type": "none", "noise_strength": 0.0}

    noise_type = spec.noise_type_forced or rng.choice(NOISE_TYPES)
    lo, hi = NOISE_STRENGTH_RANGES[noise_type]
    strength = rng.uniform(lo, hi)
    return apply_noise(img, rng, noise_type, strength), {
        "noise_type": noise_type, "noise_strength": strength}


# ---------------------------------------------------------------------------
# Object placement (non-overlap)
# ---------------------------------------------------------------------------

def _bbox_iomin(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over min(area) — fraction of the smaller bbox covered.

    0.0 = disjoint. 1.0 = smaller bbox fully inside the larger.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / min(area_a, area_b)


def _bbox_exceeds_overlap(
    bbox: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    threshold: float = MAX_OVERLAP_IOMIN,
) -> bool:
    """Return True if ``bbox`` overlaps any existing bbox beyond ``threshold``."""
    return any(_bbox_iomin(bbox, b) > threshold for b in existing)


def place_object(
    canvas: Image.Image,
    hex_key: str,
    style: str,
    rng: random.Random,
    existing_bboxes: list[tuple[int, int, int, int]],
    shadow: bool = True,
) -> dict | None:
    """Attempt to render + paste one object with non-overlap.

    Returns the per-object metadata dict on success, or ``None`` if no
    valid placement was found after ``MAX_PLACE_RETRIES`` shrink-and-retry
    loops.
    """
    lo, hi = OBJECT_SIZE_RANGE
    scale_px = rng.randint(lo, hi)

    placed = False
    bbox = (0, 0, 0, 0)
    x = y = 0
    for _attempt in range(MAX_PLACE_RETRIES):
        # Keep bbox on-canvas.
        if scale_px >= CANVAS:
            scale_px = max(1, int(scale_px * 0.85))
            continue
        x = rng.randint(0, CANVAS - scale_px)
        y = rng.randint(0, CANVAS - scale_px)
        bbox = (x, y, x + scale_px, y + scale_px)
        if not _bbox_exceeds_overlap(bbox, existing_bboxes):
            placed = True
            break
        # Gentle shrink every few attempts if we're stuck.
        if _attempt > 0 and _attempt % 10 == 0:
            scale_px = max(OBJECT_SIZE_RANGE[0], int(scale_px * 0.9))
    if not placed:
        return None

    alpha = rng.uniform(*ALPHA_RANGE)
    img = render(hex_key, style, size=scale_px, scale=1.0, rotation=0.0, alpha=alpha)
    if img is None:
        return None

    # Drop shadow: darken the emoji's alpha mask, blur, paste offset behind.
    if shadow and rng.random() < 0.5:
        mask = img.split()[-1]
        blur = rng.uniform(3, 6)
        offset = (3, 4)
        shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_alpha = mask.filter(ImageFilter.GaussianBlur(blur))
        shadow_alpha = shadow_alpha.point(lambda v: int(v * 0.5))
        shadow_layer.putalpha(shadow_alpha)
        sx, sy = x + offset[0], y + offset[1]
        canvas.paste(shadow_layer, (sx, sy), shadow_layer)

    canvas.paste(img, (x, y), img)
    existing_bboxes.append(bbox)

    meta = EMOJIS[hex_key]
    return {
        "hex": hex_key,
        "base_hex": meta["base_hex"],
        "group": meta["group"],
        "subgroup": meta["subgroup"],
        "name": meta["name"],
        "style": style,
        "position_xy": [x / CANVAS, y / CANVAS],
        "scale_px": scale_px,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Emoji sampling
# ---------------------------------------------------------------------------

def _resolve_n(rng: random.Random, n: int | tuple[int, int]) -> int:
    if isinstance(n, int):
        return n
    lo, hi = n
    return rng.randint(lo, hi)


def _sample_emoji_hex(rng: random.Random, spec: SceneSpec) -> str:
    pool = spec.emoji_pool
    if pool is None:
        # Default pool = emojis with at least one available style.
        pool = [k for k, v in EMOJIS.items() if any(v["available"].values())]
    return rng.choice(pool)


def _sample_style_for(rng: random.Random, hex_key: str,
                      style_pool: list[str] | None = None) -> str | None:
    meta = EMOJIS[hex_key]
    avail = [s for s in STYLES if meta["available"].get(s, False)]
    if style_pool is not None:
        avail = [s for s in avail if s in style_pool]
    if not avail:
        return None
    return rng.choice(avail)


# ---------------------------------------------------------------------------
# Top-level: generate one scene
# ---------------------------------------------------------------------------

def generate_scene(rng: random.Random, spec: SceneSpec | None = None
                   ) -> tuple[Image.Image, dict]:
    """Build one (image, metadata) sample following the §2 pipeline.

    Args:
        rng: Seeded ``random.Random`` — all randomness routes through it.
        spec: Scene constraints; ``None`` or ``SceneSpec()`` uses the
            pretrain distribution.

    Returns:
        ``(PIL Image RGB 512x512, metadata dict)`` — the metadata has
        ``objects`` (list), ``background`` (dict), and ``anchor_index`` (0).
    """
    spec = spec or SceneSpec()

    # 1. Background
    canvas, bg_meta = sample_background_for(rng, spec)

    # 2. Noise overlay
    canvas, noise_meta = maybe_apply_noise(canvas, rng, spec)
    bg_meta.update(noise_meta)

    canvas = canvas.convert("RGBA")

    # 3 & 4. Anchor + clutter
    n = _resolve_n(rng, spec.n_objects)

    bboxes: list[tuple[int, int, int, int]] = []
    objects_meta: list[dict] = []

    # Anchor
    anchor_hex = spec.anchor_hex or _sample_emoji_hex(rng, spec)
    anchor_style = spec.anchor_style or _sample_style_for(
        rng, anchor_hex, style_pool=spec.style_pool)
    if anchor_style is None:
        raise RuntimeError(f"No style available for anchor {anchor_hex!r}")

    anchor_obj = place_object(canvas, anchor_hex, anchor_style, rng,
                              bboxes, shadow=spec.shadows)
    if anchor_obj is None:
        raise RuntimeError(f"Could not place anchor {anchor_hex!r}")
    objects_meta.append(anchor_obj)

    # Clutter
    for _ in range(n - 1):
        hex_key = _sample_emoji_hex(rng, spec)
        style = _sample_style_for(rng, hex_key)
        if style is None:
            continue
        obj = place_object(canvas, hex_key, style, rng, bboxes,
                           shadow=spec.shadows)
        if obj is not None:
            objects_meta.append(obj)

    return canvas.convert("RGB"), {
        "anchor_index": 0,
        "objects": objects_meta,
        "background": bg_meta,
    }


# ---------------------------------------------------------------------------
# Probe-task scene specs — one per task in DESIGN.md §5
# ---------------------------------------------------------------------------

def _intersection_pool() -> list[str]:
    return [k for k, v in EMOJIS.items() if all(v["available"].values())]


def task_specs() -> dict[str, SceneSpec]:
    """Return ``{task_name: SceneSpec}`` for every probe task.

    pretrain is included as a sanity-check task that uses the full
    distribution.
    """
    intersection = _intersection_pool()
    color_variant_pool = [h for hs in UNICODE_COLOR_VARIANTS.values() for h in hs]

    return {
        "pretrain":            SceneSpec(),  # full pipeline, no constraints
        "group":               SceneSpec(n_objects=1),
        "subgroup":            SceneSpec(n_objects=1),
        "base_leaf":           SceneSpec(n_objects=1),
        "style":               SceneSpec(n_objects=1, emoji_pool=intersection),
        "grid3x3":             SceneSpec(n_objects=1),
        "scale":               SceneSpec(n_objects=1),
        "object-count":        SceneSpec(n_objects=(1, 5)),
        "background-base":     SceneSpec(n_objects=1),
        "background-noise":    SceneSpec(n_objects=1, noise_type_forced=None,
                                          noise_enabled=True),
        "background-color":    SceneSpec(
            n_objects=1,
            solid_color_bucket=None,        # compositor will pick a bucket per sample
            noise_enabled=False,
        ),
        "unicode-color":       SceneSpec(n_objects=1, emoji_pool=color_variant_pool,
                                          base_source_pool=["solid"],
                                          noise_enabled=False),
    }


# ---------------------------------------------------------------------------
# Demo: one image per probe task, arranged into a grid
# ---------------------------------------------------------------------------

def _sample_override(rng: random.Random, name: str, spec: SceneSpec) -> SceneSpec:
    """Per-sample spec tweaks that can't live as static SceneSpec fields.

    - ``background-color``: pick a fresh color bucket every sample so the grid
      shows the full 10-way palette, not just one color repeated.
    - ``unicode-color``: pick a fresh (color bucket, hex) pair every sample.
    """
    if name == "background-color":
        bucket = rng.choice(list(COLOR_BUCKETS))
        return SceneSpec(**{**spec.__dict__, "solid_color_bucket": bucket})
    if name == "unicode-color":
        bucket = rng.choice(list(UNICODE_COLOR_VARIANTS))
        hex_key = rng.choice(UNICODE_COLOR_VARIANTS[bucket])
        return SceneSpec(**{**spec.__dict__, "anchor_hex": hex_key})
    return spec


def demo(
    out: Path | str = REPO / "demo_scenes.png",
    seed: int = 0,
    cell: int = 192,
    samples_per_task: int = 6,
):
    """Render ``samples_per_task`` scenes for every probe task as a grid.

    One row per task (labeled on the left); one column per sample. Lets
    you eyeball the *distribution* each task produces, not just a single
    draw.
    """
    from PIL import ImageDraw, ImageFont

    specs = task_specs()
    names = list(specs.keys())

    label_w = 170
    rows = len(names)
    cols = samples_per_task

    grid_w = label_w + cols * cell
    grid_h = rows * cell
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for r, name in enumerate(names):
        base_spec = specs[name]
        y = r * cell

        # Left-edge label strip
        draw.rectangle((0, y, label_w, y + cell), fill=(30, 30, 30))
        draw.text((8, y + 8), name, fill=(255, 255, 255), font=font)

        n_ok = 0
        anchors = []
        for c in range(cols):
            sub_rng = random.Random(seed + r * 10_000 + c)
            spec = _sample_override(sub_rng, name, base_spec)
            try:
                img, meta = generate_scene(sub_rng, spec)
            except Exception as e:
                img = Image.new("RGB", (CANVAS, CANVAS), (220, 60, 60))
                meta = {"objects": [], "background": {}}
                draw_err = ImageDraw.Draw(img)
                draw_err.text((10, 10), f"err: {e}"[:40],
                              fill=(255, 255, 255), font=font)
            img = img.resize((cell, cell), Image.LANCZOS)
            x = label_w + c * cell
            grid.paste(img, (x, y))

            if meta.get("objects"):
                anchors.append(meta["objects"][0].get("name", "?"))
                n_ok += 1

        # One-line summary per row for the console
        preview = ", ".join(a[:18] for a in anchors[:3])
        print(f"[{name:<20s}] {n_ok}/{cols} samples   "
              f"e.g. {preview}")

    out = Path(out)
    grid.save(out)
    print(f"\nSaved demo grid: {out}  ({cols} cols × {rows} rows, "
          f"cell={cell}px, label_w={label_w}px)")


if __name__ == "__main__":
    demo()
