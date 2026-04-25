"""Build emoji hierarchy + per-style availability index.

Parses Unicode's emoji-test.txt for the group/subgroup/leaf tree, then
cross-references the three image sources (OpenMoji, Noto, Fluent) to
mark which emoji is renderable in which style.

Background terminology
======================

Codepoint
  An integer that identifies a single Unicode character. For example,
  the "dog face" emoji 🐶 has codepoint U+1F436, which is the integer
  0x1F436 == 128054. Python's `chr(128054)` returns '🐶'.

Hex notation
  Codepoints are usually written in hex. `0x1F436` is the literal; the
  display form is `U+1F436`. Conversion:
      int("1F436", 16)      -> 128054      (hex string -> int)
      f"{128054:x}"          -> "1f436"     (int -> lowercase hex string)

Emoji sequence
  Some emojis are a *single* codepoint (🐶 = [0x1F436]). Others are a
  *sequence* of codepoints joined by the Zero-Width Joiner (ZWJ,
  U+200D). Example:
      👨‍👩‍👧  =  [0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467]
                    (man,    ZWJ,    woman,   ZWJ,    girl)
  The keyboard types this as one visual glyph, but it's five codepoints.

Variation selector-16 (VS16, U+FE0F)
  An invisible codepoint that forces emoji-style rendering on characters
  that have both a text and emoji form (like ☺ vs ☺️). Fully-qualified
  emoji sequences include FE0F where relevant; unqualified/minimally
  forms omit it. We only keep *fully-qualified* entries.

Skin-tone modifiers (U+1F3FB .. U+1F3FF)
  Five codepoints (Light, Medium-Light, Medium, Medium-Dark, Dark) that
  follow a person emoji to color it. "Woman: medium skin tone" is
  [0x1F469, 0x1F3FD]. We strip these to get the canonical form for
  grouping (see `compute_base_hex`).

Hex-key normalization
=====================

Different sources write the same codepoints in different formats:

  emoji-test.txt :  "1F468 200D 1F469"   (uppercase, space-separated)
  OpenMoji file  :  "1F468-200D-1F469.svg"  (uppercase, dash-separated)
  Fluent JSON    :  "1f468 200d 1f469"   (lowercase, space-separated)

To cross-reference them we normalize everything to **lowercase,
dash-separated** — the `codepoints_to_key` function does this. The
normalized key is what we use as the dict key in `hierarchy.json`.

Output
======

Writes one JSON to src/SynthSSL/hierarchy.json with:
  - metadata : version info + summary counts + group/subgroup lists
  - emojis   : dict keyed by normalized hex, each value has
               {emoji, group, subgroup, name, version, base_hex,
                available: {openmoji, noto, fluent}}
"""

import argparse
import datetime
import json
import re
from pathlib import Path

# Repo root is two parents up from this script (scripts/build_hierarchy.py).
REPO = Path(__file__).resolve().parent.parent

# The five skin-tone modifier codepoints (U+1F3FB..U+1F3FF). If any of
# these appears in an emoji's codepoint list, the emoji is a skin-tone
# variant of some base emoji. We strip them in `compute_base_hex` to get
# the canonical "person emoji" without color.
SKIN_TONES = {0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF}

# Zero-Width Joiner — connects codepoints into a single visual emoji
# (e.g. family, profession sequences).
ZWJ = 0x200D

# ---------------------------------------------------------------------------
# Attribute rules (derived from CLDR group/subgroup labels)
#
# Attribute probes (has_face, is_living, is_vehicle, is_food) are computed
# from the Unicode-assigned group/subgroup rather than hand-curated flags.
# The rules below capture "the common reading" at the subgroup level. Edge
# cases (a bone counts as is_living because its subgroup is body-parts; a
# glass of water counts as is_food because its subgroup is drink) are
# accepted for simplicity — probes are about coarse category structure,
# not philosophical boundaries.
# ---------------------------------------------------------------------------

HAS_FACE_SUBGROUPS = frozenset({
    # Smileys & Emotion: every face-* subgroup plus stylized animal faces
    "face-smiling", "face-affection", "face-tongue", "face-hand",
    "face-neutral-skeptical", "face-sleepy", "face-unwell", "face-hat",
    "face-glasses", "face-concerned", "face-negative", "face-costume",
    "cat-face", "monkey-face",
    # People & Body: person figures and families (hands / body-parts excluded)
    "person", "person-gesture", "person-role", "person-fantasy",
    "person-activity", "person-sport", "person-resting", "family",
    # Animals & Nature: animal emojis are typically drawn face-forward
    "animal-mammal", "animal-bird", "animal-amphibian", "animal-reptile",
    "animal-marine", "animal-bug",
})

IS_LIVING_GROUPS = frozenset({"People & Body", "Animals & Nature"})

IS_VEHICLE_SUBGROUPS = frozenset({
    "transport-ground", "transport-water", "transport-air",
})

IS_FOOD_SUBGROUPS = frozenset({
    "food-fruit", "food-vegetable", "food-prepared",
    "food-asian", "food-marine", "food-sweet", "drink",
    # Note: "dishware" excluded — plates/forks aren't food.
})


def compute_attributes(group: str, subgroup: str) -> dict[str, bool]:
    """Derive binary attribute flags from an emoji's CLDR group and subgroup.

    Args:
        group: Top-level CLDR group (e.g. ``"Animals & Nature"``).
        subgroup: CLDR subgroup (e.g. ``"animal-mammal"``).

    Returns:
        dict with keys ``has_face``, ``is_living``, ``is_vehicle``, ``is_food``.
    """
    return {
        "has_face":   subgroup in HAS_FACE_SUBGROUPS,
        "is_living":  group in IS_LIVING_GROUPS,
        "is_vehicle": subgroup in IS_VEHICLE_SUBGROUPS,
        "is_food":    subgroup in IS_FOOD_SUBGROUPS,
    }


# Regex for one data line in emoji-test.txt. Example line:
#   1F600                                      ; fully-qualified     # 😀 E1.0 grinning face
#
# Groups captured:
#   1. codepoints   — hex numbers with spaces, e.g. "1F600" or "1F468 200D 1F469"
#   2. status       — "fully-qualified", "minimally-qualified", "unqualified", "component"
#   3. glyph        — the actual emoji character (rendered, so it's ONE visual token)
#   4. version      — Unicode emoji version the emoji was introduced in (e.g. "1.0")
#   5. name         — human-readable name (e.g. "grinning face")
LINE_RE = re.compile(
    r"^([0-9A-F][0-9A-F ]+?)\s*;\s*(\S+)\s*#\s*(\S+)\s+E([0-9.]+)\s+(.+)$"
)


def codepoints_to_key(cps: list[int]) -> str:
    """Convert a list of codepoint ints into the normalized hex key.

    Example:
        [0x1F468, 0x200D, 0x1F469]  ->  "1f468-200d-1f469"

    We use lowercase so keys compare cleanly regardless of source case,
    and dashes so the key is safe as a filename stem (OpenMoji SVGs use
    this exact format already).

    `f"{cp:x}"` formats an int as lowercase hex with no padding. We then
    join the list with dashes.
    """
    return "-".join(f"{cp:x}" for cp in cps)


def parse_emoji_test(path: Path) -> tuple[list[dict], str | None]:
    """Walk emoji-test.txt and return fully-qualified entries + file version.

    The file is laid out as a flat text list with section comments:

        # group: Smileys & Emotion
        # subgroup: face-smiling
        1F600    ; fully-qualified     # 😀 E1.0 grinning face
        1F603    ; fully-qualified     # 😃 E0.6 grinning face with big eyes
        ...
        # subgroup: face-affection
        1F970    ; fully-qualified     # 🥰 E11.0 smiling face with hearts
        ...
        # group: People & Body
        ...

    We iterate linewise, keeping `group` and `subgroup` as running state
    whenever we hit a `# group:` or `# subgroup:` header, and attach
    those labels to every emoji line that follows.
    """
    group = subgroup = None
    version = None
    entries = []

    for line in Path(path).read_text().splitlines():
        # Section headers: update the running group/subgroup context.
        if line.startswith("# group:"):
            # "# group: Smileys & Emotion" -> "Smileys & Emotion"
            group = line.split(":", 1)[1].strip()
            continue
        if line.startswith("# subgroup:"):
            subgroup = line.split(":", 1)[1].strip()
            continue

        # The file also has a `# Version: 17.0` header near the top.
        if line.startswith("# Version:") and version is None:
            version = line.split(":", 1)[1].strip()
            continue

        # Data line? Parse with the regex, skip anything that doesn't match
        # (blank lines, other comments, etc.).
        m = LINE_RE.match(line)
        if not m:
            continue
        codepoints_raw, status, glyph, ver, name = m.groups()

        # Keep only fully-qualified forms. Unqualified/minimally are the
        # same emoji without the VS16 variation selector — we don't want
        # duplicate entries of the same visual emoji.
        if status != "fully-qualified":
            continue

        # `codepoints_raw` is the hex codepoints separated by spaces, e.g.
        #   "1F600"                    (one codepoint: grinning face)
        #   "1F468 200D 1F469 200D 1F467"  (five codepoints: family)
        #
        # .split() with no args splits on any whitespace. Then int(x, 16)
        # parses each hex string as an int. So:
        #   "1F600"       -> ["1F600"]           -> [0x1F600]
        #   "1F468 200D"  -> ["1F468", "200D"]   -> [0x1F468, 0x200D]
        cps = [int(x, 16) for x in codepoints_raw.split()]

        entries.append({
            "hex": codepoints_to_key(cps),
            "codepoints": cps,
            "emoji": glyph,
            "group": group,
            "subgroup": subgroup,
            "name": name,
            "version": ver,
        })

    return entries, version


# ---------------------------------------------------------------------------
# SVG validation — actually render the file at 32px to confirm cairosvg
# can parse it. A small number of SVGs in various sources have malformed
# numeric attributes or path commands that crash cairosvg; we mark those
# files as unavailable here so the generator never attempts them at runtime.
# ---------------------------------------------------------------------------

try:
    import cairosvg  # type: ignore
    _HAVE_CAIROSVG = True
except ImportError:
    _HAVE_CAIROSVG = False


def _add_both_forms(keys: set[str], hex_key: str) -> None:
    """Add ``hex_key`` to ``keys`` along with its trailing-FE0F-stripped form.

    Sources disagree on whether filenames include a trailing FE0F
    variation selector. Unicode's fully-qualified forms require it
    (e.g. ``1f590-fe0f`` = 🖐️), but most image sources store the file
    as ``1f590`` / ``emoji_u1f590.svg`` without it. By registering both
    keys we make the availability set agnostic to this convention.
    """
    keys.add(hex_key)
    if hex_key.endswith("-fe0f"):
        keys.add(hex_key[: -len("-fe0f")])
    else:
        keys.add(hex_key + "-fe0f")


def _svg_renders_ok(path: Path) -> bool:
    """Return True if cairosvg can rasterize this SVG at 32x32.

    Renders at a small size to make validation cheap (~1–5 ms per file).
    Catches ValueError (malformed numeric attrs), TypeError, and generic
    parse errors. If cairosvg isn't available at all, return True
    (we can't validate, so trust the file).
    """
    if not _HAVE_CAIROSVG:
        return True
    try:
        cairosvg.svg2png(url=str(path), output_width=32, output_height=32)
        return True
    except Exception:
        return False


def compute_base_hex(codepoints: list[int]) -> str:
    """Strip skin-tone modifiers and clean up dangling ZWJs.

    The goal is: all skin-tone variants of the same emoji collapse to
    one canonical "base" key.

    Example:
        Input : [0x1F469, 0x1F3FD, 0x200D, 0x1F52C]
                (woman, medium skin tone, ZWJ, microscope)  = 👩🏽‍🔬
        After stripping skin tones:
                [0x1F469, 0x200D, 0x1F52C]
                (woman, ZWJ, microscope)  = 👩‍🔬  "woman scientist"
        Output: "1f469-200d-1f52c"

    Edge cases handled:
      - Multiple skin tones in one sequence (e.g. two-person emojis)
      - Resulting dangling ZWJ (ZWJ at start or end, or doubled ZWJ
        where two codepoints between ZWJs were both skin tones)

    Gender is NOT collapsed here — "woman scientist" and "man scientist"
    still get different base_hex. Collapse further downstream if needed.
    """
    # Step 1: drop any codepoint that is a skin-tone modifier.
    stripped = [cp for cp in codepoints if cp not in SKIN_TONES]

    # Step 2: de-duplicate ZWJs. Removing a skin tone can leave two ZWJs
    # next to each other (e.g. [A, ZWJ, skin_tone, ZWJ, B] became
    # [A, ZWJ, ZWJ, B]). Collapse runs of ZWJs to a single ZWJ.
    cleaned = []
    for cp in stripped:
        if cp == ZWJ and cleaned and cleaned[-1] == ZWJ:
            continue  # skip duplicate ZWJ
        cleaned.append(cp)

    # Step 3: trim ZWJs at the start or end (they'd be dangling).
    while cleaned and cleaned[-1] == ZWJ:
        cleaned.pop()
    while cleaned and cleaned[0] == ZWJ:
        cleaned.pop(0)

    return codepoints_to_key(cleaned)


def scan_openmoji(root: Path) -> set[str]:
    """Return normalized hex keys for every OpenMoji SVG file we find.

    OpenMoji files are named `<HEX>-<HEX>-...-<HEX>.svg` where each
    segment is the hex of one codepoint in the emoji, e.g.
        1F436.svg                          -> dog face
        1F468-200D-1F469-200D-1F467.svg   -> family: man, woman, girl

    We parse each filename stem back into a codepoint list and
    re-normalize via `codepoints_to_key`. The round-trip converts
    OpenMoji's uppercase filenames into our lowercase keys.
    """
    if not root.is_dir():
        return set()
    keys: set[str] = set()
    broken = 0
    for p in root.glob("*.svg"):
        # p.stem is the filename without the .svg extension.
        # split("-") breaks it into per-codepoint hex strings.
        # int(x, 16) parses each one. If any segment isn't valid hex,
        # we skip (shouldn't happen for OpenMoji but be defensive).
        try:
            cps = [int(x, 16) for x in p.stem.split("-")]
        except ValueError:
            continue
        if not _svg_renders_ok(p):
            broken += 1
            continue
        _add_both_forms(keys, codepoints_to_key(cps))
    if broken:
        print(f"  [openmoji] skipped {broken} SVG(s) that failed to render")
    return keys


def scan_twemoji(root: Path) -> set[str]:
    """Return normalized hex keys for every Twemoji SVG we find.

    Twemoji (jdecked fork) layout:
        src/twemoji/assets/svg/<lowercase-hex-with-dashes>.svg

    So `1f436.svg` is 🐶, `1f468-200d-1f469-200d-1f467.svg` is 👨‍👩‍👧.
    The filename format is already our normalized key (lowercase dash),
    so we just need to verify each segment parses as hex.
    """
    svg_dir = root / "assets" / "svg"
    if not svg_dir.is_dir():
        # Also tolerate the user pointing --twemoji straight at the svg dir.
        if not (root / "svg").is_dir() and not any(root.glob("*.svg")):
            return set()
        svg_dir = root / "svg" if (root / "svg").is_dir() else root
    keys: set[str] = set()
    broken = 0
    for p in svg_dir.glob("*.svg"):
        try:
            cps = [int(x, 16) for x in p.stem.split("-")]
        except ValueError:
            continue
        if not _svg_renders_ok(p):
            broken += 1
            continue
        _add_both_forms(keys, codepoints_to_key(cps))
    if broken:
        print(f"  [twemoji] skipped {broken} SVG(s) that failed to render")
    return keys


# Blobmoji's own filename normalization rule, lifted from its
# convert_filenames.py so we can reproduce matches against the human-
# readable SVG names (e.g. "accordion.svg" for U+1FA97).
# Split on these characters:
_BLOB_DELIM = re.compile(r"[-_. ]")
# Then strip these characters from each split token:
_BLOB_REMOVE = re.compile(r"""[,*\\/:'"()]""")


def _blob_normalize_name(name: str) -> str:
    """Apply Blobmoji's name-normalization rule.

    Example:
      Unicode "artist: dark skin tone"  -> "artist dark skin tone"
      Blobmoji "artist dark skin tone"  -> "artist dark skin tone"
    """
    parts = _BLOB_DELIM.split(name)
    parts = [_BLOB_REMOVE.sub("", p) for p in parts]
    return " ".join(parts).lower().strip()


def scan_blobmoji(root: Path, entries: list[dict]) -> set[str]:
    """Return normalized hex keys for every Blobmoji SVG we recognize.

    Blobmoji ships SVG files with TWO naming conventions in the same
    source tree:

      (a) Hex-keyed (underscores between codepoints):
            svg/emoji_u1f436.svg
            derived/emoji_u1f436_1f3fb.svg
          → we parse the hex directly.

      (b) Human-readable English names:
            svg/accordion.svg
            svg/artist dark skin tone.svg
            svg15/black bird.svg
          → we can't parse these to hex without a lookup. We build
          `{ normalize(unicode_name): hex }` from the Unicode
          emoji-test.txt we already parsed (`entries`), then match
          each human-named file by normalizing its stem the same way.

    The normalization rule is Blobmoji's own (see `_blob_normalize_name`
    above): split on ``[-_. space]``, strip ``[,*\\/:'"()]``, rejoin with
    single space, lowercase. This is what Blobmoji's own build pipeline
    uses to map between human names and hex filenames.

    Directories scanned (in order): svg/, svg15/, derived/.
    Skipped: third_party/ (region-flag SVGs named by ISO 3166 codes
    like "US.svg" — these map to Regional Indicator sequences but
    need a separate ISO-to-codepoint table; not worth it for now).

    `placeholder.svg` at the root is always skipped.
    """
    if not root.is_dir():
        return set()

    # Build the reverse lookup: normalized Unicode name -> hex key.
    # Multiple Unicode entries could normalize to the same string in
    # pathological cases — last one wins; in practice they're disjoint.
    name_to_hex = {_blob_normalize_name(e["name"]): e["hex"] for e in entries}

    # Blobmoji's directories that hold "real" emoji SVGs.
    search_dirs = [root / "svg", root / "svg15", root / "derived"]

    keys: set[str] = set()
    broken = 0
    for sub in search_dirs:
        if not sub.is_dir():
            continue
        for svg in sub.glob("*.svg"):
            stem = svg.stem
            if stem == "placeholder":
                continue

            target_key: str | None = None

            # (a) Hex-keyed file: parse codepoints directly.
            if stem.startswith("emoji_u"):
                hex_part = stem[len("emoji_u"):]
                try:
                    cps = [int(x, 16) for x in hex_part.split("_")]
                    target_key = codepoints_to_key(cps)
                except ValueError:
                    target_key = None

            # (b) Human-named file: look up via Unicode name normalization.
            if target_key is None:
                target_key = name_to_hex.get(_blob_normalize_name(stem))

            if target_key is None:
                continue
            if not _svg_renders_ok(svg):
                broken += 1
                continue
            _add_both_forms(keys, target_key)

    if broken:
        print(f"  [blobmoji] skipped {broken} SVG(s) that failed to render")
    return keys


def scan_fluent(root: Path) -> set[str]:
    """Return normalized hex keys for every Fluent folder that has a Flat SVG.

    Fluent's folder names are human-readable ("Dog face", "Family - Man, Woman, Girl")
    so we can't parse them. Each folder has a ``metadata.json`` whose
    ``unicode`` field holds the hex codepoints, lowercase and
    space-separated::

        {"unicode": "1f468 200d 1f469 200d 1f467", ...}

    We read the field, split on whitespace, parse each hex, and
    normalize to the same dash-separated lowercase key as the other sources.

    **Availability gate**: Some Fluent folders ship only ``metadata.json``
    with no ``Flat/`` subdirectory — these are parent entries whose actual
    designs live in per-skin-tone / per-gender child folders. We require
    a non-empty ``Flat/`` directory so the availability flag matches what
    the runtime renderer can actually load (see
    ``utilities._build_fluent_index``).
    """
    if not root.is_dir():
        return set()
    keys: set[str] = set()
    broken = 0
    for meta in root.glob("*/metadata.json"):
        flat_dir = meta.parent / "Flat"
        if not flat_dir.is_dir():
            continue
        svg = next(flat_dir.glob("*.svg"), None)
        if svg is None:
            continue
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        unicode_field = data.get("unicode", "")
        if not unicode_field:
            continue
        try:
            cps = [int(x, 16) for x in unicode_field.split()]
        except ValueError:
            continue
        if not _svg_renders_ok(svg):
            broken += 1
            continue
        _add_both_forms(keys, codepoints_to_key(cps))
    if broken:
        print(f"  [fluent] skipped {broken} SVG(s) that failed to render")
    return keys


def scan_noto(root: Path) -> set[str]:
    """Return normalized hex keys for every Noto Emoji SVG we find.

    Noto Emoji (googlefonts/noto-emoji) layout:
        src/noto-emoji/svg/emoji_u<hex>_<hex>.svg

    Same `emoji_u<hex>_<hex>.svg` convention as Blobmoji (Blobmoji forked
    from this repo pre-2017), so the parser is identical to the
    hex-keyed branch of scan_blobmoji: strip the 'emoji_u' prefix, split
    on underscores, parse each segment as hex, re-normalize.
    """
    svg_dir = root / "svg"
    if not svg_dir.is_dir():
        return set()
    keys: set[str] = set()
    broken = 0
    for p in svg_dir.glob("emoji_u*.svg"):
        hex_part = p.stem[len("emoji_u"):]
        try:
            cps = [int(x, 16) for x in hex_part.split("_")]
        except ValueError:
            continue
        if not _svg_renders_ok(p):
            broken += 1
            continue
        _add_both_forms(keys, codepoints_to_key(cps))
    if broken:
        print(f"  [noto] skipped {broken} SVG(s) that failed to render")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emoji-test", type=Path, default=REPO / "emoji.txt")
    ap.add_argument("--openmoji", type=Path,
                    default=REPO / "src/OpenMoji/openmoji-svg-color")
    ap.add_argument("--fluent", type=Path,
                    default=REPO / "src/fluentui-emoji/assets")
    ap.add_argument("--twemoji", type=Path,
                    default=REPO / "src/twemoji")
    ap.add_argument("--blobmoji", type=Path,
                    default=REPO / "src/blobmoji")
    ap.add_argument("--noto", type=Path,
                    default=REPO / "src/noto-emoji")
    ap.add_argument("--out", type=Path,
                    default=REPO / "src/SynthSSL/hierarchy.json")
    args = ap.parse_args()

    # Parse the Unicode source of truth.
    entries, uver = parse_emoji_test(args.emoji_test)

    # Each scan_* returns a set of normalized hex keys. Membership in
    # the set == source has that emoji. Missing directories return
    # empty sets so the script runs even before all sources are present.
    openmoji_keys = scan_openmoji(args.openmoji)
    fluent_keys = scan_fluent(args.fluent)
    twemoji_keys = scan_twemoji(args.twemoji)
    blobmoji_keys = scan_blobmoji(args.blobmoji, entries)
    noto_keys = scan_noto(args.noto)

    # Build the per-emoji records for the output JSON.
    emojis = {}
    for e in entries:
        emojis[e["hex"]] = {
            "emoji": e["emoji"],
            "group": e["group"],
            "subgroup": e["subgroup"],
            "name": e["name"],
            "version": e["version"],
            "base_hex": compute_base_hex(e["codepoints"]),
            "codepoint_count": len(e["codepoints"]),
            "attributes": compute_attributes(e["group"], e["subgroup"]),
            "available": {
                "openmoji": e["hex"] in openmoji_keys,
                "noto": e["hex"] in noto_keys,
                "fluent": e["hex"] in fluent_keys,
                "twemoji": e["hex"] in twemoji_keys,
                "blobmoji": e["hex"] in blobmoji_keys,
            },
        }

    # Summary structures for convenience in metadata.
    groups = sorted({e["group"] for e in entries})
    subgroups_by_group: dict[str, set[str]] = {}
    for e in entries:
        subgroups_by_group.setdefault(e["group"], set()).add(e["subgroup"])
    subgroups_by_group = {g: sorted(s) for g, s in subgroups_by_group.items()}

    # Counts for the header.
    # "available_in_N_or_more" lets you see the distribution across
    # styles without picking a specific intersection size.
    avail_all = sum(1 for v in emojis.values() if all(v["available"].values()))
    avail_any = sum(1 for v in emojis.values() if any(v["available"].values()))
    unique_bases = len({v["base_hex"] for v in emojis.values()})

    style_names = ["openmoji", "noto", "fluent", "twemoji", "blobmoji"]
    avail_by_k = {
        k: sum(
            1 for v in emojis.values()
            if sum(v["available"][s] for s in style_names) >= k
        )
        for k in range(1, len(style_names) + 1)
    }

    attr_counts = {
        attr: sum(1 for v in emojis.values() if v["attributes"][attr])
        for attr in ("has_face", "is_living", "is_vehicle", "is_food")
    }

    single_cp = sum(1 for v in emojis.values() if v["codepoint_count"] == 1)
    multi_cp = sum(1 for v in emojis.values() if v["codepoint_count"] > 1)

    output = {
        "metadata": {
            "unicode_version": uver or "unknown",
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds").replace("+00:00", "Z"),
            "counts": {
                "total": len(emojis),
                "unique_base_hex": unique_bases,
                "groups": len(groups),
                "subgroups": sum(len(v) for v in subgroups_by_group.values()),
                "available_openmoji": sum(
                    1 for v in emojis.values() if v["available"]["openmoji"]),
                "available_noto": sum(
                    1 for v in emojis.values() if v["available"]["noto"]),
                "available_fluent": sum(
                    1 for v in emojis.values() if v["available"]["fluent"]),
                "available_twemoji": sum(
                    1 for v in emojis.values() if v["available"]["twemoji"]),
                "available_blobmoji": sum(
                    1 for v in emojis.values() if v["available"]["blobmoji"]),
                "available_all_styles": avail_all,
                "available_any_style": avail_any,
                "available_in_k_or_more_styles": avail_by_k,
                "attribute_counts": attr_counts,
                "single_codepoint": single_cp,
                "multi_codepoint": multi_cp,
            },
            "groups": groups,
            "subgroups_by_group": subgroups_by_group,
            "notes": [
                "All five style availabilities are SVG-file-based now "
                "(Noto uses googlefonts/noto-emoji, not the bitmap font).",
                "base_hex strips skin-tone modifiers (U+1F3FB..U+1F3FF) "
                "and collapses dangling ZWJ joiners. Gender is NOT "
                "collapsed — group_by(base_hex) then by gender if desired.",
            ],
        },
        "emojis": emojis,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    c = output["metadata"]["counts"]
    print(f"Wrote {args.out}")
    print(f"  unicode version: {output['metadata']['unicode_version']}")
    print(f"  total: {c['total']}  (unique base_hex: {c['unique_base_hex']})")
    print(f"  groups: {c['groups']}, subgroups: {c['subgroups']}")
    print(f"  openmoji: {c['available_openmoji']}")
    print(f"  noto:     {c['available_noto']}")
    print(f"  fluent:   {c['available_fluent']}")
    print(f"  twemoji:  {c['available_twemoji']}")
    print(f"  blobmoji: {c['available_blobmoji']}")
    print(f"  any style:      {c['available_any_style']}")
    print(f"  all 5 styles:   {c['available_all_styles']}")
    print("  available in k-or-more styles:")
    for k, n in sorted(c["available_in_k_or_more_styles"].items()):
        print(f"    k>={k}: {n}")
    print("  attributes (derived from group/subgroup):")
    for attr, n in c["attribute_counts"].items():
        pct = 100 * n / c["total"]
        print(f"    {attr:<12s} {n:>5d}  ({pct:.1f}%)")
    print(f"  single-codepoint: {c['single_codepoint']}")
    print(f"  multi-codepoint:  {c['multi_codepoint']}")


if __name__ == "__main__":
    main()
