"""Sample emojis and render them onto a canvas, in any of five styles.

All five styles are SVG now, so one code path (``cairosvg``) handles
rendering. ``render`` exposes per-sample controls for canvas size,
emoji-to-canvas scale, in-plane rotation, and alpha opacity.

Run this module directly to produce a demo grid::

    python src/SynthSSL/objects.py
"""

import random
from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image

from SynthSSL.utilities import EMOJIS, REPO, STYLES, svg_path_for


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_svg(svg_path: Path, size: int) -> Image.Image:
    """Rasterize an SVG file to a square RGBA PIL image.

    Args:
        svg_path: Path to any SVG file.
        size: Output width and height in pixels.

    Returns:
        PIL ``Image`` in RGBA mode, ``size × size``.
    """
    png = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
    return Image.open(BytesIO(png)).convert("RGBA")


def render(
    hex_key: str,
    style: str,
    size: int = 128,
    scale: float = 1.0,
    rotation: float = 0.0,
    alpha: float = 1.0,
) -> Image.Image | None:
    """Render one emoji onto a ``size × size`` RGBA canvas.

    Returns ``None`` when the requested (hex_key, style) pair isn't
    available — the compositor should check this and try another style
    or skip the emoji.

    Args:
        hex_key: Normalized hex key, e.g. ``"1f436"`` for 🐶.
        style: One of ``STYLES``
            (``"openmoji"``, ``"noto"``, ``"fluent"``, ``"twemoji"``, ``"blobmoji"``).
        size: Output canvas side length in pixels.
        scale: Emoji bounding box as a fraction of the canvas. ``1.0``
            fills the canvas; ``0.5`` occupies half the width
            (a quarter of the area); values ``> 1`` overflow and get
            clipped by the canvas edges.
        rotation: In-plane rotation in degrees, counterclockwise. The
            glyph is rotated around its center with the bitmap expanded
            to preserve the full shape, then center-pasted onto the
            output canvas.
        alpha: Opacity multiplier in ``[0, 1]``. Applied to the emoji's
            existing alpha channel.

    Returns:
        PIL ``Image`` in RGBA mode with the emoji on a fully
        transparent background, or ``None`` if the style doesn't have
        this emoji.
    """
    meta = EMOJIS.get(hex_key)
    if meta is None or not meta["available"].get(style, False):
        return None

    svg = svg_path_for(hex_key, style)
    if svg is None:
        return None

    emoji_px = max(1, int(size * scale))
    img = render_svg(svg, emoji_px)

    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)

    if alpha < 1.0:
        a = img.split()[-1].point(lambda v: int(v * alpha))
        img.putalpha(a)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_emojis(
    n: int,
    require: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Pick ``n`` random emoji hex keys, optionally constrained by style.

    Args:
        n: Number of keys to return. Must not exceed the pool size.
        require: Optional list of style names. Only emojis for which
            ``available[style]`` is ``True`` for *every* listed style
            are eligible. Default (``None``) samples from all 3,944
            fully-qualified Unicode emojis.
        seed: Optional seed for reproducibility.

    Returns:
        List of ``n`` hex keys, without replacement.

    Raises:
        ValueError: If ``n`` is larger than the eligible pool.
    """
    require = require or []
    pool = [
        k for k, v in EMOJIS.items()
        if all(v["available"][s] for s in require)
    ]
    rng = random.Random(seed)
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# Demo grid
# ---------------------------------------------------------------------------

def demo(
    n: int = 6,
    cell: int = 128,
    out: Path | str = REPO / "demo.png",
    seed: None = None,
    scale: float = 1.0,
    rotation: float = 0.0,
    alpha: float = 1.0,
):
    """Render ``n`` random emojis across all five styles, save as a grid PNG.

    Rows are sampled emojis; columns are styles in ``STYLES`` order.
    The same ``scale`` / ``rotation`` / ``alpha`` are applied to every
    cell — useful for eyeballing how transforms look across styles.

    Args:
        n: Number of emojis (grid rows).
        cell: Side length of each grid cell in pixels.
        out: Output PNG path.
        seed: Seed for the emoji sampler.
        scale: Passed through to :func:`render`.
        rotation: Passed through to :func:`render`.
        alpha: Passed through to :func:`render`.
    """
    # Pick random seed each time if None
    seed = random.randint(0, 1000000) if seed is None else seed
    print(f"Seed: {seed}")
    picks = sample_emojis(n, require=STYLES, seed=seed)

    cols, rows = len(STYLES), n
    grid = Image.new("RGBA", (cols * cell, rows * cell), (255, 255, 255, 255))

    for r, hex_key in enumerate(picks):
        meta = EMOJIS[hex_key]
        print(f"[{r}] {meta['emoji']}  {hex_key}  {meta['name']}")
        for c, style in enumerate(STYLES):
            img = render(hex_key, style, cell,
                         scale=scale, rotation=rotation, alpha=alpha)
            if img is None:
                continue
            grid.paste(img, (c * cell, r * cell), img)

    out = Path(out)
    grid.save(out)
    print(f"\nSaved grid: {out}  ({cols}x{rows} cells of {cell}px)")
    print(f"Columns (left→right): {STYLES}")


if __name__ == "__main__":
    demo()
