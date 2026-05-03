"""Dataset generation — pretrain and per-task probe datasets.

Overview
========

Three Python entry points, no CLI::

    from SynthSSL.generate import (
        generate_pretrain,       # SSL pretrain (anchored enumeration)
        generate_probe,          # one probe-task dataset, 80/20 split
        generate_all_probes,     # convenience: every labelable task
    )

Why enumerate anchors? Because we want **guaranteed coverage** of every
emoji. Random sampling from the 3,944-emoji pool would leave some
emojis unseen. Instead, both pretrain and (by default) probe generation
iterate explicitly over the emoji list and render a fixed number of
scenes per emoji (``K`` for pretrain, ``per_anchor`` for probes).
That way "every emoji appears ≥K times" is a structural property of
the generation, not a probabilistic one.

Determinism
-----------
Every function is reproducible under its ``seed`` argument. The
per-scene seed is derived as ``(master * PRIME + idx) & 0xFFFFFFFF``,
where ``PRIME`` is Knuth's multiplicative hash constant. This gives
well-distributed per-scene seeds that don't correlate with the
shuffle order — so workers can process scenes in any order and
produce byte-identical images.

On-disk layout
--------------

Pretrain::

    {out_dir}/
        0000000.jpg
        0000001.jpg
        ...
        metadata.jsonl        # one JSON object per line, one per sample

Probe (per task)::

    {out_dir}/
        train/
            000000.jpg
            ...
            metadata.jsonl    # the train split, includes "label" field
        test/
            000000.jpg
            ...
            metadata.jsonl    # the test split
        metadata.jsonl        # combined train + test, same format

``metadata.jsonl`` lines follow §2.1 of ``DESIGN.md`` (the scene's
full composition) plus:

- ``image``: relative filename within the split dir
- ``seed``: per-scene RNG seed (useful for debugging)
- ``label`` (probe only): task-specific class label
- ``split`` (probe only): "train" | "test"
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from SynthSSL.labels import LABEL_FUNCTIONS
from SynthSSL.scene import (
    COLOR_BUCKETS,
    SceneSpec,
    UNICODE_COLOR_VARIANTS,
    generate_scene,
    task_specs,
)
from SynthSSL.utilities import EMOJIS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Knuth's multiplicative hash constant. Used to spread per-sample seeds
# across the 32-bit range — ensures consecutive sample indices produce
# uncorrelated RNG states even with small master seeds like 0, 1, 2.
_SEED_PRIME = 2_654_435_761

# JPEG quality for on-disk images. 90 is a common SSL-training default
# that balances file size against decode quality.
JPEG_QUALITY = 90


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _seed_for(master: int, idx: int) -> int:
    """Per-scene seed from (master_seed, sample_index).

    Uses a multiplicative hash so adjacent indices produce uncorrelated
    seeds. This matters because scenes are typically numbered 0..N
    sequentially — if we just did ``master + idx`` the first few scenes
    would have adjacent seeds and render nearly-identical augmentations.

    Masked to 32 bits because downstream ``random.Random`` and
    ``numpy.random.default_rng`` both accept 32-bit seeds cleanly.
    """
    return (master * _SEED_PRIME + idx) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Per-sample spec overrides
# ---------------------------------------------------------------------------

def _resolve_spec(task: str, base_spec: SceneSpec, rng: random.Random) -> SceneSpec:
    """Return a possibly-modified ``SceneSpec`` for one scene.

    Most tasks use their default spec for every scene (set once in
    ``scene.task_specs()``). Two tasks need per-scene overrides so that
    the dataset covers their full class set rather than whatever the
    default sampler happens to draw:

    - **background-color**: pick a fresh color bucket per scene so the
      generated dataset covers all 10 buckets, not just one.
    - **unicode-color**: pick a (bucket, emoji-hex) pair per scene so
      every Unicode color variant appears.

    Other tasks return their base spec unchanged.
    """
    if task == "background-color":
        bucket = rng.choice(list(COLOR_BUCKETS))
        return SceneSpec(**{**base_spec.__dict__, "solid_color_bucket": bucket})
    if task == "unicode-color":
        bucket = rng.choice(list(UNICODE_COLOR_VARIANTS))
        hex_key = rng.choice(UNICODE_COLOR_VARIANTS[bucket])
        return SceneSpec(**{**base_spec.__dict__, "anchor_hex": hex_key})
    return base_spec


# ---------------------------------------------------------------------------
# Worker: renders a single scene
# ---------------------------------------------------------------------------
#
# Each worker process gets a "payload" dict describing one scene to
# render. The payload carries everything the worker needs in pickled
# form — no shared module state besides the already-loaded hierarchy
# and asset indices (which are module-level constants, forked cleanly
# into worker processes on Linux).
#
# Payload schema:
#     out_dir     str      directory where the .jpg should be written
#     filename    str      name of the .jpg (assigned by the caller)
#     seed        int      per-scene RNG seed, from _seed_for()
#     task        str      key into scene.task_specs()
#     anchor_hex  str|None if set, forces this emoji as the anchor
#                          (used for both pretrain K-enumeration and
#                           probe per_anchor enumeration)
# ---------------------------------------------------------------------------

def _render_one(payload: dict) -> dict:
    """Render + save one scene. Return its metadata record.

    The result dict is what ``scene.generate_scene`` produced (with the
    full composition metadata) plus ``image`` + ``seed`` fields. If
    rendering raises — rare, but can happen on pathological emoji +
    background combinations — we return an ``error``-tagged record
    instead of crashing the whole batch.
    """
    out_dir = Path(payload["out_dir"])
    filename = payload["filename"]
    seed = payload["seed"]
    task = payload["task"]
    anchor_override = payload.get("anchor_hex")

    rng = random.Random(seed)
    base_spec = task_specs()[task]

    # If the caller pinned an anchor emoji (pretrain's K-enumeration or
    # probe's per_anchor enumeration), bake it into the spec before
    # _resolve_spec runs — it may add other per-scene overrides.
    if anchor_override is not None:
        base_spec = SceneSpec(**{**base_spec.__dict__, "anchor_hex": anchor_override})
    spec = _resolve_spec(task, base_spec, rng)

    try:
        img, meta = generate_scene(rng, spec)
    except Exception as e:
        # Graceful failure — we log the error on the record so the
        # caller can count failures and continue. The file is NOT
        # written in this case.
        return {"image": filename, "error": f"{type(e).__name__}: {e}"}

    path = out_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=JPEG_QUALITY)

    meta["image"] = filename
    meta["seed"] = seed
    return meta


# ---------------------------------------------------------------------------
# Parallel render driver
# ---------------------------------------------------------------------------

def _render_all(
    payloads: list[dict],
    workers: int = 8,
    progress_every: int = 500,
    label: str = "samples",
) -> list[dict]:
    """Dispatch ``_render_one`` across a process pool.

    The return order matches the input order regardless of which
    worker finished first (we rebuild the list by index). This matters
    for pretrain so that filename ordering matches metadata ordering
    and downstream tools can do ``records[i]`` ↔ ``000...{i}.jpg``.

    Progress is reported every ``progress_every`` completions.

    ``workers <= 1`` runs in the main process (useful for debugging
    since worker exceptions surface in-line).
    """
    results: list[dict | None] = [None] * len(payloads)
    t0 = time.time()

    if workers <= 1:
        for i, p in enumerate(payloads):
            results[i] = _render_one(p)
            if (i + 1) % progress_every == 0:
                _report_progress(i + 1, len(payloads), t0, label)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Submit every payload and remember which index each future
            # corresponds to, so we can place results in original order.
            futs = {pool.submit(_render_one, p): i for i, p in enumerate(payloads)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                done += 1
                if done % progress_every == 0:
                    _report_progress(done, len(payloads), t0, label)

    _report_progress(len(payloads), len(payloads), t0, label)

    # Filter out errors (keep them in the records but count them up).
    out = []
    errors = 0
    for r in results:
        if r is None:
            errors += 1
            continue
        if "error" in r:
            errors += 1
        out.append(r)
    if errors:
        print(f"  [!] {errors} render errors / missing results")
    return out


def _report_progress(done: int, total: int, t0: float, label: str):
    """Print one-line progress update with rate and ETA."""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(f"    {done}/{total} {label}  "
          f"({rate:.1f}/s, eta {eta:.0f}s)")


# ---------------------------------------------------------------------------
# Metadata writing
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: Iterable[dict]):
    """Write JSONL (one JSON object per line, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# Entry point: pretrain
# ---------------------------------------------------------------------------

def generate_pretrain(
    out_dir: Path | str,
    K: int = 10,
    seed: int = 0,
    workers: int = 8,
    emoji_subset: list[str] | None = None,
    task: str = "pretrain",
) -> Path:
    """Render the SSL pretrain dataset.

    Strategy: **anchor enumeration**. For every emoji in the corpus we
    render ``K`` scenes with that emoji as the anchor. This guarantees
    every emoji appears in exactly ``K`` scenes — a structural
    property that random sampling wouldn't provide.

    Total size = ``K × |emoji corpus|``. For ``K=10`` and the ~3,944
    fully-qualified Unicode emojis this is ~39,440 scenes.

    Args:
        out_dir: Output directory. Created if missing; existing files
            inside will be overwritten on re-run.
        K: Anchor scenes per emoji (10 = Small recipe, 50 = Large).
        seed: Master RNG seed for reproducibility.
        workers: Process-pool size. 1 = single-threaded (debugging).
        emoji_subset: If given, only these hex keys are used as anchors
            (useful for smoke tests). Defaults to every emoji that has
            at least one style available.

    Returns:
        ``out_dir`` as a ``Path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the anchor list. By default include every emoji that at
    # least one of the 5 image sources can render — anything else
    # would fail in the compositor.
    if emoji_subset is not None:
        anchors = list(emoji_subset)
    else:
        anchors = [k for k, v in EMOJIS.items() if any(v["available"].values())]

    total = len(anchors) * K
    skipped = len(EMOJIS) - len(anchors) if emoji_subset is None else 0
    print(f"[pretrain] {len(anchors)} emojis × K={K} = {total} images → {out_dir}"
          + (f" (skipped {skipped} emojis with no available style)" if skipped else ""))

    # Build the payload list: one entry per scene we want to render.
    # Every scene gets a unique index so the worker can name files
    # deterministically and so the per-scene seed is stable.
    if task not in task_specs():
        raise KeyError(
            f"unknown task {task!r}; choose from {list(task_specs())}")

    payloads = []
    idx = 0
    for anchor_hex in anchors:
        for _k in range(K):
            payloads.append({
                "out_dir": str(out_dir),
                "filename": f"{idx:07d}.jpg",
                "seed": _seed_for(seed, idx),
                "task": task,
                "anchor_hex": anchor_hex,
            })
            idx += 1

    records = _render_all(payloads, workers=workers, label="scenes")
    _write_jsonl(out_dir / "metadata.jsonl", records)
    print(f"[pretrain] wrote metadata.jsonl with {len(records)} records")
    return out_dir


# ---------------------------------------------------------------------------
# Entry point: one probe-task dataset
# ---------------------------------------------------------------------------

def generate_probe(
    task: str,
    out_dir: Path | str,
    n: int | None = None,
    per_anchor: int = 0,
    seed: int = 0,
    workers: int = 8,
    train_frac: float = 0.8,
) -> Path:
    """Render one probe-task dataset, stratified 80/20 into train + test.

    Two sampling modes (specify exactly one):

    - **Anchor-stratified** (``per_anchor=K``): enumerate every emoji
      compatible with the task and render ``K`` scenes per emoji. This
      is the default for probe generation and guarantees *every* emoji
      appears in the dataset — and, after the 80/20 split, in both
      train and test (assuming ``K >= 2``).

    - **Random-anchor** (``n=N``): render ``N`` scenes, each sampling
      its anchor emoji freely. Fewer coverage guarantees; kept for
      the occasional case where anchor coverage isn't the bottleneck.

    Split
    -----
    Stratified on the **task's own label** (e.g. ``group`` for the
    group task), with a rounding rule designed to give test coverage
    even on rare classes::

        samples in class   → samples in train  samples in test
        ----------------------------------------------------------
        1                  → 1                 0
        2                  → 1                 1
        3                  → 2                 1
        N ≥ 4              → ⌊N × 0.8⌋         N − ⌊N × 0.8⌋

    So every class with ≥2 samples has at least 1 test example, and
    train is always a strict superset of test's class vocabulary.

    Args:
        task: Task name from ``scene.task_specs()``. Must also have a
            label function in ``labels.LABEL_FUNCTIONS``.
        out_dir: Output dir. Creates ``{train,test}/`` subdirectories.
        n: Total scenes (random-anchor mode). Leave ``None`` if
            using ``per_anchor``.
        per_anchor: Scenes per emoji (anchor-stratified mode). Leave 0
            if using ``n``.
        seed: Master RNG seed.
        workers: Process-pool size.
        train_frac: Fraction in train. Default 0.8 ⇒ 80/20 split.

    Returns:
        ``out_dir`` as a ``Path``.
    """
    if task not in task_specs():
        raise KeyError(f"unknown task {task!r}; choose from {list(task_specs())}")
    if task not in LABEL_FUNCTIONS:
        raise KeyError(f"task {task!r} has no label function")
    if (n is None) == (per_anchor <= 0):
        raise ValueError(
            "specify exactly one of n= (random-anchor) or per_anchor= "
            "(anchor-stratified)"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: build the payload list.
    # ------------------------------------------------------------------
    # For anchor-stratified mode we enumerate emojis explicitly so
    # every one gets exactly per_anchor scenes. For random-anchor
    # mode we just emit N payloads with anchor=None and let the
    # scene generator sample it.
    if per_anchor > 0:
        spec = task_specs()[task]
        if spec.emoji_pool is not None:
            # Task constrains the pool (e.g., style probe uses only
            # emojis in the 5-style intersection); respect that.
            anchors = list(spec.emoji_pool)
        else:
            # Default: every emoji with at least one renderable style.
            anchors = [k for k, v in EMOJIS.items()
                       if any(v["available"].values())]

        total = len(anchors) * per_anchor
        print(f"[probe:{task}] per_anchor={per_anchor} × {len(anchors)} emojis "
              f"= {total} scenes → {out_dir}")

        payloads = []
        i = 0
        for anchor_hex in anchors:
            for _ in range(per_anchor):
                payloads.append({
                    "out_dir": str(out_dir / "_scratch"),
                    "filename": f"{i:07d}.jpg",
                    "seed": _seed_for(seed, i),
                    "task": task,
                    "anchor_hex": anchor_hex,
                })
                i += 1
    else:
        # Random-anchor mode — anchor is sampled per scene in the
        # compositor.
        print(f"[probe:{task}] {n} scenes (random anchor) → {out_dir}")
        payloads = [{
            "out_dir": str(out_dir / "_scratch"),
            "filename": f"{i:06d}.jpg",
            "seed": _seed_for(seed, i),
            "task": task,
            "anchor_hex": None,
        } for i in range(n)]

    # ------------------------------------------------------------------
    # Step 2: render everything into a scratch directory, then extract
    # the task's label from each rendered scene's metadata.
    # ------------------------------------------------------------------
    records = _render_all(payloads, workers=workers, label="scenes")

    label_fn = LABEL_FUNCTIONS[task]
    labeled: list[tuple[dict, Any]] = []
    for r in records:
        if "error" in r:
            # Render failed — skip. (Already reported by _render_all.)
            continue
        try:
            lbl = label_fn(r)
        except Exception as e:
            # Label extraction failed — the metadata record doesn't
            # have the field this task expects. Flag it on the record
            # and drop from the labeled set.
            r["error"] = f"label_fn: {e}"
            continue
        labeled.append((r, lbl))

    # ------------------------------------------------------------------
    # Step 3: stratified 80/20 train/test split.
    # ------------------------------------------------------------------
    # We group samples by their class label and split each group
    # independently. The rounding rule (see docstring) keeps rare
    # classes represented in both splits.
    by_label: dict[Any, list[dict]] = {}
    for r, lbl in labeled:
        by_label.setdefault(lbl, []).append(r)

    rng_split = random.Random(seed ^ 0xBEEF)  # independent from render seeds
    train_records: list[dict] = []
    test_records: list[dict] = []
    for lbl, group in by_label.items():
        rng_split.shuffle(group)
        n_in_class = len(group)
        if n_in_class <= 1:
            # Only one sample of this class: put it in train.
            # Test will have zero samples of this class — but the probe
            # head will never have to predict it on test either.
            cut = n_in_class
        else:
            # Floor of train_frac × n, but always ≥ 1 train sample and
            # always ≤ n−1 (i.e. ≥ 1 test sample).
            cut = max(1, min(n_in_class - 1, int(n_in_class * train_frac)))
        train_records.extend((r, lbl) for r in group[:cut])
        test_records.extend((r, lbl) for r in group[cut:])

    # ------------------------------------------------------------------
    # Step 4: move images from scratch/ into train/ and test/ with
    # renumbered filenames, and record the split assignment + label in
    # the per-sample metadata. The scratch folder is then removed.
    # ------------------------------------------------------------------
    scratch_dir = out_dir / "_scratch"
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    final_train = _finalize_split(scratch_dir, train_dir, train_records, "train")
    final_test = _finalize_split(scratch_dir, test_dir, test_records, "test")
    scratch_dir.rmdir()

    # Write three metadata.jsonl files: one per split, plus a combined
    # one at the task root (convenient for coverage audits).
    _write_jsonl(train_dir / "metadata.jsonl", final_train)
    _write_jsonl(test_dir / "metadata.jsonl", final_test)
    _write_jsonl(out_dir / "metadata.jsonl", final_train + final_test)

    n_classes = len(by_label)
    print(f"[probe:{task}] train={len(final_train)}  test={len(final_test)}  "
          f"classes={n_classes}")
    return out_dir


def _finalize_split(
    scratch_dir: Path,
    split_dir: Path,
    records: list[tuple[dict, Any]],
    split_name: str,
) -> list[dict]:
    """Move rendered images into their split dir, renumber filenames,
    stamp ``split`` and ``label`` onto each metadata record.

    Records are returned with their final on-disk ``image`` filename
    so the caller can write a matching ``metadata.jsonl``.
    """
    finalized: list[dict] = []
    for i, (r, lbl) in enumerate(records):
        old_path = scratch_dir / r["image"]
        new_name = f"{i:06d}.jpg"
        new_path = split_dir / new_name
        old_path.rename(new_path)
        # Work on a copy so we don't mutate the combined list later.
        r = dict(r)
        r["image"] = new_name
        r["split"] = split_name
        r["label"] = lbl
        finalized.append(r)
    return finalized


# ---------------------------------------------------------------------------
# Entry point: generate every labelable probe task
# ---------------------------------------------------------------------------

def generate_all_probes(
    out_root: Path | str,
    n: int | None = None,
    per_anchor: int = 5,
    seed: int = 0,
    workers: int = 8,
    exclude_tasks: list[str] | None = None,
    skip_existing: bool = True,
) -> Path:
    """Run ``generate_probe`` for every task with a label function.

    Default is anchor-stratified with ``per_anchor=5`` — every emoji
    gets 5 scenes per task, so train has ~4 / emoji and test has ~1.
    At that count every emoji is guaranteed in train, and nearly every
    (task, emoji) pair contributes to the 80/20 test as well.

    Args:
        out_root: Parent directory. Each task gets its own subdir.
        n: Use random-anchor mode with N scenes per task. Leave None
            for anchor-stratified mode.
        per_anchor: Anchor-stratified scenes per emoji (default 5).
        seed: Master seed. Each task gets ``seed + i * 101`` so runs
            across tasks don't share scene seeds.
        workers: Process-pool size.
        exclude_tasks: Task names to skip. ``leaf`` is excluded by
            default — it's redundant with ``base_leaf`` and the
            pretrain set already provides leaf-level labels.
        skip_existing: If True, skip tasks whose output already has a
            ``train/metadata.jsonl`` marker.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Default-skip ``leaf`` (redundant with ``base_leaf``) and the
    # ``*_clean`` variants — those are opt-in via ``generate_clean.py``.
    exclude = set(exclude_tasks or ["leaf", "group_clean", "subgroup_clean"])
    tasks = [t for t in task_specs() if t in LABEL_FUNCTIONS and t not in exclude]

    if per_anchor > 0 and n is None:
        print(f"[all-probes] {len(tasks)} tasks × "
              f"{per_anchor} scenes/anchor → {out_root}")
    else:
        print(f"[all-probes] {len(tasks)} tasks × {n} random-anchor "
              f"scenes → {out_root}")

    for i, task in enumerate(tasks):
        task_dir = out_root / task
        marker = task_dir / "train" / "metadata.jsonl"
        if skip_existing and marker.exists():
            print(f"[probe:{task}] skip (already present)")
            continue
        generate_probe(
            task=task,
            out_dir=task_dir,
            n=n if per_anchor <= 0 else None,
            per_anchor=per_anchor,
            seed=seed + i * 101,
            workers=workers,
        )
    return out_root


# ---------------------------------------------------------------------------
# Entry point: smoke test (small dataset for quick visual audit)
# ---------------------------------------------------------------------------

def generate_smoke(
    out_dir: Path | str,
    samples_per_task: int = 50,
    seed: int = 0,
    workers: int = 4,
    include_tasks: list[str] | None = None,
) -> Path:
    """Tiny run: one probe-style dataset per task for visual audit.

    Uses random-anchor mode (``n=50``) rather than the per_anchor mode
    used for real probe generation — 50 samples is too few to cover
    every emoji and we just want to eyeball the pipeline end-to-end.
    """
    out_dir = Path(out_dir)
    tasks = include_tasks or list(task_specs().keys())
    # ``pretrain`` is listed in task_specs() but doesn't have a label
    # function, so skip it here (smoke is probe-flavored).
    tasks = [t for t in tasks if t in LABEL_FUNCTIONS]

    print(f"[smoke] {len(tasks)} tasks × {samples_per_task} samples → {out_dir}")
    for i, task in enumerate(tasks):
        generate_probe(
            task=task,
            out_dir=out_dir / task,
            n=samples_per_task,
            seed=seed + i,
            workers=workers,
        )
    return out_dir
