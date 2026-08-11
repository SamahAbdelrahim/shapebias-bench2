#!/usr/bin/env python3
"""Build the fixed trial pool for the human experiment and export its images.

The human app used to read a stimulus manifest at run time, which only worked
for the flat 30-shape packages. The texture grid nests a texture level under
each shape, which the JavaScript side cannot walk. This script resolves the
selection offline instead: it picks the triads once, copies their images out to
display-sized WebP, and writes a single pool file that the frontend reads
verbatim. Nothing at run time depends on the ``/scratch`` symlinks.

Grid triads reuse the shape-stratified round-robin from
``playgrounds.embedding_robust.build_grid_triplets`` at ``seed=0``, so the human
items are the same 114 that back the embedding panel.

Cue-conflict triads are built only under ``--include-cc``, which is off. Humans
run the novel grid alone. The Geirhos classes are familiar named categories, so
a pseudo-word label in the noun condition competes with a name the participant
already has, which pushes responses for lexical rather than perceptual reasons
and does so only in that set. The selection code is kept so the set can be
switched back on without rewriting it.

Outputs:
  results/data/human_trial_pool_v2.csv       tidy manifest (one row per triad)
  human-experiment/public/trial_pool.json    what the frontend fetches
  human-experiment/public/stimuli/...        resized WebP images

Usage:
  .venv/bin/python scripts/build_human_trial_pool.py
  .venv/bin/python scripts/build_human_trial_pool.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from evaluation_pipe.eval_core import (  # noqa: E402
    load_stimuli_grid,
    load_stimuli_triad_dir,
)

POOL_VERSION = "v2"

GRID_PKG = "stimuli_texture_grid_v1_scratch"
GRID_STIM_SET = "stimuli_A_auto_contrast"
GRID_N = 114
GRID_SEED = 0

CC_ROOT = REPO_ROOT / "previous-lit-stimuli" / "cue-conflict" / "cc_triads_n320"
CC_SUBSET_CSV = REPO_ROOT / "results" / "data" / "cueconflict_triad_subset_n320.csv"
CC_PER_CLASS = 10
CC_SEED = 0

# One catch trial per this many test trials, resolved on the frontend; the pool
# only needs enough catch items that no participant repeats one.
N_CATCH = 24
CATCH_SEED = 0

# Grid renders are 1024 px, which is far more than the task shows and roughly
# 600 KB per PNG. Cue-conflict triads are already 224 px and are never upscaled.
MAX_PX = 384
WEBP_QUALITY = 88

OUT_CSV = REPO_ROOT / "results" / "data" / f"human_trial_pool_{POOL_VERSION}.csv"
PUBLIC_DIR = REPO_ROOT / "human-experiment" / "public"
OUT_JSON = PUBLIC_DIR / "trial_pool.json"
IMG_ROOT = PUBLIC_DIR / "stimuli"
IMG_URL_PREFIX = "/human-experiment/stimuli"

CSV_FIELDS = [
    "pool_version",
    "stim_set_name",
    "role",
    "stim_id",
    "stl_id",
    "texture_set",
    "shape_class",
    "texture_class",
    "congruent",
    "reference_url",
    "shape_match_url",
    "texture_match_url",
    "catch_correct_is",
]


def _class_of(name: str) -> str:
    """Strip the exemplar index off a Geirhos id (``airplane1`` -> ``airplane``)."""
    return re.sub(r"\d+$", "", name)


def select_grid(n: int = GRID_N, seed: int = GRID_SEED) -> list[dict]:
    """Pick grid triads with the stratification used for the embedding sample.

    Mirrors ``build_grid_triplets``: group by shape, sort shapes numerically,
    shuffle each shape's textures with one shared RNG in that order, then take
    triads round-robin across shapes. Reproducing the operation order matters,
    since a single RNG stream is consumed across all 30 shuffles.
    """
    records = load_stimuli_grid(GRID_PKG, GRID_STIM_SET)
    rng = random.Random(seed)
    by_shape: dict[str, list[dict]] = {}
    for rec in records:
        by_shape.setdefault(rec["stl_id"], []).append(rec)
    shapes = sorted(by_shape, key=lambda s: int(s))
    for s in shapes:
        rng.shuffle(by_shape[s])

    picked: list[dict] = []
    idxs = {s: 0 for s in shapes}
    while len(picked) < n and any(idxs[s] < len(by_shape[s]) for s in shapes):
        for s in shapes:
            i = idxs[s]
            if i < len(by_shape[s]):
                picked.append(by_shape[s][i])
                idxs[s] = i + 1
            if len(picked) >= n:
                break
    if len(picked) < n:
        raise ValueError(f"Only {len(picked)} grid triads available, wanted {n}")
    return picked


def select_cc(per_class: int = CC_PER_CLASS, seed: int = CC_SEED) -> list[dict]:
    """Pick cue-conflict triads from the frozen n=320 model subset.

    Congruent triads (``dog1-dog2``) are skipped: both options come from the
    same class, so there is no shape-versus-texture answer for a participant to
    give. They stay in the model subset, where they are labelled rather than
    excluded, so the human pool is a subset of the model ids either way.
    """
    if not CC_SUBSET_CSV.is_file():
        raise FileNotFoundError(f"Cue-conflict subset manifest not found: {CC_SUBSET_CSV}")
    with open(CC_SUBSET_CSV, newline="", encoding="utf-8") as f:
        subset_rows = list(csv.DictReader(f))
    allowed = {
        r["stim_id"]: r
        for r in subset_rows
        if str(r.get("congruent", "")).strip().lower() != "true"
    }

    records = [r for r in load_stimuli_triad_dir(CC_ROOT) if r["stim_id"] in allowed]
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for rec in records:
        by_class.setdefault(_class_of(rec["stl_id"]), []).append(rec)
    classes = sorted(by_class)
    for c in classes:
        rng.shuffle(by_class[c])

    for c in classes:
        if len(by_class[c]) < per_class:
            raise ValueError(
                f"Class {c} has only {len(by_class[c])} non-congruent triads, "
                f"wanted {per_class}"
            )

    # Emit round-robin across classes rather than class by class. Participants
    # receive a contiguous window of the pool, so a class-grouped order would
    # hand one person ten airplanes; round-robin makes any window span classes.
    picked: list[dict] = []
    for i in range(per_class):
        for c in classes:
            rec = dict(by_class[c][i])
            rec["shape_class"] = c
            rec["texture_class"] = _class_of(rec["texture_set"])
            picked.append(rec)
    return picked


def build_catch(grid: list[dict], cc: list[dict], n: int = N_CATCH,
                seed: int = CATCH_SEED) -> list[dict]:
    """Build identity-match attention checks from triads already in the pool.

    A catch trial shows a reference and offers an exact duplicate of it against
    an unrelated object, so the correct answer does not depend on any
    shape-versus-texture judgement. Foils come from a different shape (grid) or
    a different Geirhos class (cue-conflict) so nothing about the check is
    ambiguous. Catch items reuse pool images, so they cost no extra export.

    Checks are split across whichever sets are in the pool. With the grid alone
    every check is a grid check, which also removes an ordering wrinkle from the
    two-set version: there, a check drawn from one set could land inside the
    other set's block.
    """
    rng = random.Random(seed)
    catch: list[dict] = []

    def _pairs(records: list[dict], group_of, count: int, set_name: str) -> None:
        if count <= 0 or not records:
            return
        pool = list(records)
        rng.shuffle(pool)
        used = 0
        for rec in pool:
            if used >= count:
                break
            foils = [r for r in records if group_of(r) != group_of(rec)]
            if not foils:
                continue
            foil = foils[rng.randrange(len(foils))]
            catch.append({
                "stim_set_name": set_name,
                "stim_id": f"catch_{set_name}_{rec['stim_id']}",
                "stl_id": rec.get("stl_id", ""),
                "texture_set": rec.get("texture_set", ""),
                "shape_class": rec.get("shape_class", ""),
                "texture_class": rec.get("texture_class", ""),
                "reference_src": rec["reference_path"],
                # The duplicate is the reference itself; the foil is another
                # object's reference. Reusing both keeps the export set small.
                "match_src": rec["reference_path"],
                "foil_src": foil["reference_path"],
            })
            used += 1
        if used < count:
            raise ValueError(f"Could not build {count} catch trials for {set_name}")

    n_cc = n // 2 if cc else 0
    _pairs(grid, lambda r: r["stl_id"], n - n_cc, "grid")
    _pairs(cc, lambda r: r["shape_class"], n_cc, "cc_triads")
    return catch


def export_image(src: Path, dest: Path, dry_run: bool = False) -> None:
    """Write ``src`` to ``dest`` as WebP, downscaled to ``MAX_PX`` at most."""
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_PX:
            im.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
        im.save(dest, "WEBP", quality=WEBP_QUALITY, method=6)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve the selection and report counts without writing images")
    ap.add_argument("--grid-n", type=int, default=GRID_N)
    ap.add_argument("--include-cc", action="store_true",
                    help="Also build cue-conflict triads. Off by default: humans run "
                         "the novel grid alone, because a pseudo-word over a familiar "
                         "Geirhos category competes with a name participants already have")
    ap.add_argument("--cc-per-class", type=int, default=CC_PER_CLASS)
    ap.add_argument("--catch-n", type=int, default=N_CATCH)
    args = ap.parse_args()

    grid = select_grid(args.grid_n)
    cc = select_cc(args.cc_per_class) if args.include_cc else []
    catch = build_catch(grid, cc, args.catch_n)
    print(f"grid: {len(grid)} triads over {len({r['stl_id'] for r in grid})} shapes, "
          f"{len({r['texture_set'] for r in grid})} textures")
    if cc:
        print(f"cc_triads: {len(cc)} triads over {len({r['shape_class'] for r in cc})} classes")
    else:
        print("cc_triads: excluded (pass --include-cc to add them back)")
    print(f"catch: {len(catch)} identity-match trials")

    csv_rows: list[dict] = []
    pool_test: list[dict] = []
    pool_catch: list[dict] = []

    def _url(set_name: str, stim_id: str, role: str, src: Path) -> str:
        rel = Path(set_name) / f"{_slug(stim_id)}__{role}.webp"
        export_image(src, IMG_ROOT / rel, args.dry_run)
        return f"{IMG_URL_PREFIX}/{rel.as_posix()}"

    for rec in grid:
        urls = {
            role: _url("grid", rec["stim_id"], role, rec[f"{role}_path"])
            for role in ("reference", "shape_match", "texture_match")
        }
        entry = {
            "stim_set_name": "grid",
            "stim_id": rec["stim_id"],
            "stl_id": str(rec["stl_id"]),
            "texture_set": str(rec["texture_set"]),
            "reference_url": urls["reference"],
            "shape_match_url": urls["shape_match"],
            "texture_match_url": urls["texture_match"],
        }
        pool_test.append(entry)
        csv_rows.append({**entry, "pool_version": POOL_VERSION, "role": "test",
                         "shape_class": "", "texture_class": "", "congruent": "false",
                         "catch_correct_is": ""})

    for rec in cc:
        urls = {
            role: _url("cc_triads", rec["stim_id"], role, rec[f"{role}_path"])
            for role in ("reference", "shape_match", "texture_match")
        }
        entry = {
            "stim_set_name": "cc_triads",
            "stim_id": rec["stim_id"],
            "stl_id": str(rec["stl_id"]),
            "texture_set": str(rec["texture_set"]),
            "reference_url": urls["reference"],
            "shape_match_url": urls["shape_match"],
            "texture_match_url": urls["texture_match"],
        }
        pool_test.append(entry)
        csv_rows.append({**entry, "pool_version": POOL_VERSION, "role": "test",
                         "shape_class": rec["shape_class"],
                         "texture_class": rec["texture_class"],
                         "congruent": "false", "catch_correct_is": ""})

    for rec in catch:
        set_name = rec["stim_set_name"]
        ref_url = _url(set_name, rec["stim_id"], "reference", rec["reference_src"])
        entry = {
            "stim_set_name": set_name,
            "stim_id": rec["stim_id"],
            "stl_id": rec["stl_id"],
            "texture_set": rec["texture_set"],
            "reference_url": ref_url,
            # The duplicate option is the same exported file as the reference.
            "match_url": ref_url,
            "foil_url": _url(set_name, rec["stim_id"], "foil", rec["foil_src"]),
        }
        pool_catch.append(entry)
        csv_rows.append({
            "pool_version": POOL_VERSION, "stim_set_name": set_name, "role": "catch",
            "stim_id": rec["stim_id"], "stl_id": rec["stl_id"],
            "texture_set": rec["texture_set"], "shape_class": rec["shape_class"],
            "texture_class": rec["texture_class"], "congruent": "",
            "reference_url": entry["reference_url"],
            "shape_match_url": entry["match_url"],
            "texture_match_url": entry["foil_url"],
            "catch_correct_is": "match",
        })

    payload = {
        "pool_version": POOL_VERSION,
        "grid_stim_set": GRID_STIM_SET,
        "grid_seed": GRID_SEED,
        "cc_seed": CC_SEED,
        "counts": {"grid": len(grid), "cc_triads": len(cc), "catch": len(catch)},
        "test": pool_test,
        "catch": pool_catch,
    }

    if args.dry_run:
        print("dry run: no files written")
        return 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    total_bytes = sum(p.stat().st_size for p in IMG_ROOT.rglob("*.webp"))
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(csv_rows)} rows)")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"images under {IMG_ROOT.relative_to(REPO_ROOT)}: "
          f"{sum(1 for _ in IMG_ROOT.rglob('*.webp'))} files, "
          f"{total_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
