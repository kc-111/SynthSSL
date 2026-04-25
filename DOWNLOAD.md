# Downloading SynthSSL assets

Run these from the repo root. Everything lands under `src/` next to the Python package.

## What's already in the repo

- `src/OpenMoji/` — OpenMoji PNG/SVG set + `openmoji.json` (CC-BY-SA 4.0). Small enough to commit.

Nothing to download for that directory.

## 1. Noto Emoji (SVG, Google)

OFL license, ~3,700 emojis. Google's current emoji set, the one you see on Android / ChromeOS / Gmail. SVG files named by hex with underscores (`emoji_u<hex>_<hex>.svg`) — same convention as Blobmoji (which forked it).

The full repo is ~500 MB (includes PNGs, fonts, build tooling). We only need the `svg/` directory — sparse-clone to grab just that, ~30–50 MB:

```bash
cd src
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/googlefonts/noto-emoji.git
cd noto-emoji
git sparse-checkout init --no-cone
git sparse-checkout set 'svg'
git checkout
cd ../..
```

Layout after download:
```
src/noto-emoji/
  svg/emoji_u1f436.svg
  svg/emoji_u1f468_200d_1f469_200d_1f467.svg
  ...
```

`build_hierarchy.py`'s `scan_noto` reads `src/noto-emoji/svg/`. No fontTools dependency, no "assume sequences render" caveat — same file-level availability check as OpenMoji / Twemoji / Blobmoji.

### Why we switched from the font

`NotoColorEmoji.ttf` is a color **bitmap** font locked to 109px per glyph. Rendering at any other size goes through FreeType resizing → blurry. The official `googlefonts/noto-emoji` repo provides the same glyphs as SVG vectors, renderable at any size cleanly via `cairosvg`. Simpler code path, better quality, removes a special case in the renderer.

## 2. Fluent Emoji

Microsoft Fluent Emoji — MIT-licensed. Three SVG variants per emoji:

- **Color** — fully shaded
- **Flat** — flat fills
- **High Contrast** — black/white line art

Plus a **3D** folder containing PNGs (large, ~1.5 GB total, skipped by default).

### Option A — sparse clone (~300 MB, recommended)

```bash
cd src
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/microsoft/fluentui-emoji.git
cd fluentui-emoji
git sparse-checkout init --no-cone
git sparse-checkout set 'assets/*/metadata.json' 'assets/*/Color' 'assets/*/Flat' 'assets/*/High Contrast'
git checkout
cd ../..
```

Modern git's sparse-checkout defaults to **cone mode**, which only accepts directory prefixes (no `*` patterns). `--no-cone` is required here because we want per-emoji subfolders like `assets/*/Color`.

### Option B — full clone then trim (if sparse-checkout misbehaves)

```bash
cd src
git clone --depth 1 https://github.com/microsoft/fluentui-emoji.git
rm -rf fluentui-emoji/assets/*/3D
cd ..
```

Downloads ~2 GB up front, ~300 MB on disk after cleanup.

### Layout after download

```
src/fluentui-emoji/assets/
  Dog face/
    metadata.json
    Color/dog_face_color.svg
    Flat/dog_face_flat.svg
    High Contrast/dog_face_high_contrast.svg
  Cat face/
    ...
```

Folder names are human-readable strings. Join against OpenMoji / Noto via `metadata.json`'s `unicode` field (hex, e.g. `"1f436"`).

## 3. Twemoji (Twitter emoji — community fork)

CC-BY 4.0, ~3,700 emojis through Unicode 15.1. Closes most of the gap Fluent leaves on newer emojis. SVGs named by lowercase hex with dashes (`1f436.svg`, `1f468-200d-1f469-200d-1f467.svg`).

```bash
cd src
git clone --depth 1 https://github.com/jdecked/twemoji.git
cd ..
```

Layout:
```
src/twemoji/
  assets/svg/*.svg        # vector, scale freely — use these
  assets/72x72/*.png      # 72px rasters (we don't use these)
```

`build_hierarchy.py` looks at `src/twemoji/assets/svg/`.

## 4. Blobmoji (maintained old Android "blob" style)

Apache-2.0, ~2,700 emojis. Retro aesthetic — visually very different from the flat modern sets. Ships SVGs (we use those, ignore the PNG / font build artifacts).

```bash
cd src
git clone --depth 1 https://github.com/c1710/blobmoji.git
cd ..
```

Layout (what `build_hierarchy.py` scans): three directories of SVGs.

```
src/blobmoji/
  svg/       # main set — mix of emoji_u<hex>.svg and human-named files
  svg15/     # newer additions (human-named, e.g. "black bird.svg")
  derived/   # auto-generated skin-tone/ZWJ combinations (hex-named)
```

Blobmoji uses two naming conventions in the same tree:
- **Hex-keyed**: `emoji_u1f436.svg`, `emoji_u1f468_200d_1f469_200d_1f467.svg` (underscores between codepoints, not dashes like OpenMoji/Twemoji).
- **Human-readable**: `accordion.svg`, `artist dark skin tone.svg`, `black bird.svg`.

`scan_blobmoji` in `build_hierarchy.py` handles both: hex-named files parse directly; human-named files are matched against Unicode emoji names via the same normalization Blobmoji's own `convert_filenames.py` uses (split on `[-_. space]`, strip punctuation, lowercase).

Blobmoji also ships `third_party/region-flags/svg/` with ~250 flag SVGs named by ISO 3166 codes (e.g. `US.svg`). These would need a separate ISO-to-regional-indicator mapping; currently skipped. If you need flag coverage, OpenMoji + Noto + Twemoji already cover them.

### Safe to delete from the clone

If you want to save disk space after cloning, these are not used by the pipeline:

```bash
cd src/blobmoji
rm -rf .git fonts images tables third_party ComicNeue *.tmpl* *.gpl *.afpalette
rm -f placeholder.svg build.ps1 Dockerfile convert_filenames.py update_changed.py
cd ../..
```

## 5. ambientCG backgrounds

ambientCG ships ~2000 CC0 PBR materials. For SynthSSL we only need the **albedo / color map** at **1K JPEG**, not the full PBR set. Uses ambientCG's **v3 API**.

A couple hundred materials is plenty for the 10k/50k recipes; grab the full set if aiming at 500k.

### Check inventory first (optional, one HTTP call)

Query `/api/v3/categories` to see total material count and per-category breakdown before committing to a download. Single request, no pagination.

```bash
uv add requests   # if not already installed
python scripts/inventory_ambientcg.py
```

Script (`scripts/inventory_ambientcg.py`):

```python
import requests

API = "https://ambientCG.com/api/v3/categories"

def main():
    r = requests.get(API)
    r.raise_for_status()
    entries = r.json()

    materials = [e for e in entries if e["type"] == "material"]
    total = sum(e["numberOfAssets"] for e in materials)

    print(f"Total material assets: {total}")
    print(f"Material categories: {len(materials)}\n")
    print(f"{'category':<25s} {'count':>6s}  title")
    print("-" * 70)
    for e in sorted(materials, key=lambda x: -x["numberOfAssets"]):
        print(f"{e['id']:<25s} {e['numberOfAssets']:>6d}  {e['title']}")

if __name__ == "__main__":
    main()
```

Expected output (as of April 2026):
```
Total material assets: 1993
Material categories: 95

category                   count  title
----------------------------------------------------------------------
Tiles                         158  Tiles
PavingStones                  154  Paving Stones
Ground                        121  Ground
Bricks                        115  Bricks
Metal                         101  Metal
...
```

Use this to decide `--limit` and whether to stratify. Asset IDs follow `<Category><Number>` (e.g. `Rock064`, `WoodFloor051`), so category is recoverable from the ID using these category names as a longest-prefix match.

### Download (recommended)

```bash
uv add requests
python scripts/download_ambientcg.py --out src/ambientcg --limit 500
```

Minimal script (`scripts/download_ambientcg.py`):

```python
import argparse
import io
import zipfile
from pathlib import Path

import requests

API = "https://ambientCG.com/api/v3/assets"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=500,
                    help="Max materials to download (ambientCG has ~2000)")
    ap.add_argument("--resolution", default="1K-JPG")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    offset, got = 0, 0
    page_size = 100
    while got < args.limit:
        r = requests.get(API, params={
            "type": "Material", "limit": page_size, "offset": offset,
            "include": "downloads",
        })
        r.raise_for_status()
        assets = r.json().get("assets", [])
        if not assets:
            break
        for a in assets:
            if got >= args.limit:
                break
            asset_id = a["id"]
            out_path = args.out / f"{asset_id}.jpg"
            if out_path.exists():
                got += 1
                continue
            zip_url = next(
                (d["url"] for d in a.get("downloads", [])
                 if d.get("attributes") == args.resolution
                 and d.get("extension") == "zip"),
                None,
            )
            if not zip_url:
                continue
            print(f"[{got+1}/{args.limit}] {asset_id}")
            blob = requests.get(zip_url).content
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for name in z.namelist():
                    if "_Color" in name and name.lower().endswith((".jpg", ".jpeg")):
                        out_path.write_bytes(z.read(name))
                        got += 1
                        break
        offset += page_size

if __name__ == "__main__":
    main()
```

Each 1K JPEG is ~500 KB → 500 materials ≈ 250 MB, 2000 materials ≈ 1 GB. ambientCG responds quickly but rate-limits gently; expect ~15–30 min for the full 2000.

### Manual alternative

Browse [ambientcg.com/list](https://ambientcg.com/list), download whichever materials you want as 1K-JPG zips, extract only the `*_Color.jpg` file into `src/ambientcg/`.

## 6. Procedural backgrounds

No download — generated at runtime by `src/SynthSSL/backgrounds.py` (Perlin noise, gradients, Voronoi, Gabor, solid colors).

## Verification

```bash
ls src/noto-emoji/svg/emoji_u*.svg    | wc -l     # expect ~3700
ls src/fluentui-emoji/assets          | wc -l     # expect ~1595
ls src/twemoji/assets/svg/*.svg       | wc -l     # expect ~3700
find src/blobmoji/svg src/blobmoji/svg15 src/blobmoji/derived -name '*.svg' | wc -l  # expect ~7000 (sum across dirs)
ls src/ambientcg/*.jpg                | wc -l     # expect ~1992
```

Then run `python scripts/build_hierarchy.py` — the printed summary will tell you exactly how many emojis each source covers.
