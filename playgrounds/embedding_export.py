#!/usr/bin/env python3
"""Persist vision-encoder embeddings for every cell of the stimulus grid.

``embedding_robust.py`` computes cosines in memory and throws the vectors away,
so there is no cached representation to analyse. This script walks the full
30 mesh x 38 texture grid, extracts all four read-outs per image, and writes
them to ``.npz`` so that ``linear_probe.py`` can run on CPU afterwards without
touching a GPU again.

Read-outs come from ``extract_reps`` in ``embedding_robust.py`` (imported, not
re-implemented) so the vectors here are the same objects the published cosine
numbers were computed from.

Also writes a raw-pixel baseline (32x32 greyscale) so a probe result can be
checked against trivial image statistics.

Deduplication: the shape-match image of cell (s, t) is the mesh s rendered with
texture t+19, which is also a cell of the grid. Whether that render is byte
identical to the reference render of cell (s, t+19) depends on the pose-matching
step in the stimulus pipeline, so this script hashes file contents and reuses an
embedding whenever two roles resolve to the same bytes. Pass ``--no-dedupe`` to
force one forward pass per role.

Output (per model, per read-out):
    <out-dir>/<model>_<readout>.npz   X[N,D] float16, role, stl_id, texture_set
    <out-dir>/<model>_rows.csv        human-readable index
    <out-dir>/pixels32.npz            pixel baseline, model-independent

Usage:
    python playgrounds/embedding_export.py --models qwen3.5-9b \
        --grid-pkg stimuli_texture_grid_v1_scratch
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:  # torch and the model registry are only needed for the GPU path
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

ROLES = ("reference", "shape_match", "texture_match")

# The four models that carry the argument: the two clean within-model
# dissociations, the model the original claim was built on, and the strongest
# texture lean.
DEFAULT_MODELS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3-vl-8b", "smolvlm"]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grid_rows(grid_pkg: str, stim_set: str) -> list[dict]:
    """One row per (role, mesh, texture) cell, carrying the image path."""
    from evaluation_pipe.eval_core import load_stimuli_grid

    records = load_stimuli_grid(grid_pkg, stim_set)
    rows = []
    for rec in records:
        for role in ROLES:
            rows.append(
                {
                    "role": role,
                    "stl_id": int(rec["stl_id"]),
                    "texture_set": str(rec["texture_set"]),
                    "stim_id": rec["stim_id"],
                    "path": Path(rec[f"{role}_path"]),
                }
            )
    return rows


def load_flat_rows(root: Path) -> list[dict]:
    """Fallback for the local 30-set package (no texture level in the layout).

    Used only for a shape-and-metadata sanity run before spending GPU time on
    the grid; the texture label is unavailable here so it is left as 'unknown'.
    """
    rows = []
    for d in sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        for role in ROLES:
            p = d / f"{role}.png"
            if p.is_file():
                rows.append(
                    {
                        "role": role,
                        "stl_id": int(d.name),
                        "texture_set": "unknown",
                        "stim_id": d.name,
                        "path": p,
                    }
                )
    return rows


def pixel_baseline(rows: list[dict], size: int = 32) -> np.ndarray:
    """Downsampled greyscale vectors: what a probe can get without an encoder."""
    out = []
    for r in rows:
        img = Image.open(r["path"]).convert("L").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
    return np.stack(out)


def _write_npz(path: Path, X: np.ndarray, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float16),
        role=np.array([r["role"] for r in rows]),
        stl_id=np.array([r["stl_id"] for r in rows], dtype=np.int32),
        texture_set=np.array([r["texture_set"] for r in rows]),
    )


def export_model(
    name: str, rows: list[dict], out_dir: Path, dedupe: bool = True
) -> dict:
    import torch

    from evaluation_pipe.models import create_model
    from playgrounds.embedding_robust import extract_reps

    device = "cuda" if torch.cuda.is_available() else "mps"
    wrapper = create_model(name, device=device)

    # role images that resolve to identical bytes share one forward pass
    digests = [_sha1(r["path"]) for r in rows] if dedupe else [str(i) for i in
                                                               range(len(rows))]
    first_seen: dict[str, int] = {}
    for i, d in enumerate(digests):
        first_seen.setdefault(d, i)
    unique_idx = sorted(set(first_seen.values()))
    n_unique = len(unique_idx)
    print(f"{name}: {len(rows)} rows, {n_unique} unique images "
          f"({len(rows) - n_unique} deduped)")

    by_rep: dict[str, dict[int, np.ndarray]] = {}
    errs: dict[str, str] = {}
    for count, i in enumerate(unique_idx, 1):
        img = Image.open(rows[i]["path"]).convert("RGB")
        reps = extract_reps(wrapper, img)
        for k, v in reps.items():
            if k.endswith("__err"):
                errs[k] = v
                continue
            by_rep.setdefault(k, {})[i] = v.squeeze().float().cpu().numpy()
        if count % 200 == 0:
            print(f"  {count}/{n_unique}", flush=True)

    written = {}
    for rep, vecs in by_rep.items():
        if len(vecs) != n_unique:
            print(f"  skip {rep}: {len(vecs)}/{n_unique} images produced a vector")
            continue
        dim = len(next(iter(vecs.values())))
        X = np.zeros((len(rows), dim), dtype=np.float32)
        for i, d in enumerate(digests):
            X[i] = vecs[first_seen[d]]
        path = out_dir / f"{name}_{rep}.npz"
        _write_npz(path, X, rows)
        written[rep] = {"path": str(path), "dim": dim, "n": len(rows)}
        print(f"  wrote {path.name}  X={X.shape}")

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"written": written, "errors": errs, "n_unique": n_unique}


def _has_gpu() -> bool:
    import torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--grid-pkg", default="stimuli_texture_grid_v1_scratch")
    ap.add_argument("--stim-set", default="stimuli_A_auto_contrast")
    ap.add_argument("--flat-root", type=Path,
                    help="use a flat 30-set package instead of the grid (sanity run)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap cells for a smoke test")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--pixels-only", action="store_true",
                    help="write the pixel baseline and exit (no GPU needed)")
    args = ap.parse_args()

    if args.flat_root:
        rows = load_flat_rows(args.flat_root)
        tag = "flat30"
    else:
        rows = load_grid_rows(args.grid_pkg, args.stim_set)
        tag = args.stim_set
    if args.limit:
        keep = {r["stim_id"] for r in rows}
        keep = set(sorted(keep)[: args.limit])
        rows = [r for r in rows if r["stim_id"] in keep]

    missing = [r["path"] for r in rows if not Path(r["path"]).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} image files not found, e.g. {missing[0]}")
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        from evaluation_pipe.eval_core import default_session_results_dir

        out_dir = Path(default_session_results_dir("probe")) / "embeddings_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} rows from {tag} -> {out_dir}")

    with open(out_dir / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "stl_id", "texture_set", "stim_id", "path"])
        for r in rows:
            w.writerow([r["role"], r["stl_id"], r["texture_set"], r["stim_id"],
                        r["path"]])

    px_path = out_dir / "pixels32.npz"
    if not px_path.exists():
        _write_npz(px_path, pixel_baseline(rows), rows)
        print(f"wrote {px_path.name}")
    if args.pixels_only:
        return 0

    if not _has_gpu():
        print("No GPU available")
        return 1

    for name in args.models:
        print("\n" + "#" * 60 + f"\nMODEL: {name}\n" + "#" * 60)
        try:
            info = export_model(name, rows, out_dir, dedupe=not args.no_dedupe)
            if info["errors"]:
                print(f"  read-out errors: {sorted(info['errors'])}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"MODEL FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
