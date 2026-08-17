#!/usr/bin/env python3
"""Export continuous per-triad read-out margins from saved grid embeddings.

The read-out ladder (``linear_probe.py``) stores only binary shape/texture
preferences per rung, without stimulus alignment. The linkage analyses need the
graded quantity instead: the signed margin

    margin = cos(reference, shape_match) - cos(reference, texture_match)

per triad, under each unsupervised read-out (raw, centred, train-centred, ZCA
whitened), keyed by ``(stl_id, texture_set)`` so it joins the behavioral CSVs.

Fold structure, centering, and whitening are byte-for-byte the ladder's own:
ZCA is fit on the train-block references only and margins are taken on the
held-out block, so every triad appears exactly once and never sees its own
statistics. Validation reproduces the published rung shape rates from the probe
JSONs before anything is written.

Usage
-----
    python playgrounds/margin_export.py                  # all NPZs
    python playgrounds/margin_export.py --models qwen3.5-4b qwen3.5-9b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe import (  # noqa: E402
    GridEmbeddings,
    _cos,
    _zca,
    assert_disjoint,
    build_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = REPO_ROOT / "results/probe.results/session_readout_power/embeddings_grid"
PROBE_DIR = REPO_ROOT / "results/probe.results/session_readout_power"
OUT_DIR = PROBE_DIR / "margins"

RUNG_OF_COLUMN = {
    "margin_raw": "1_raw_cosine",
    "margin_centered": "2_centered_cosine",
    "margin_centered_train": "2b_centered_train",
    "margin_zca": "3_zca_cosine",
}


def export_margins(emb: GridEmbeddings) -> pd.DataFrame:
    """One row per triad, margins under the four unsupervised read-outs."""
    splits = build_splits(emb)
    assert_disjoint(splits, emb)
    all_m = [int(m) for m in emb.meshes]
    all_t = [str(t) for t in emb.textures]

    rows: list[dict] = []
    for mi, mesh_fold in enumerate(splits["mesh_folds"]):
        for ti, tex_fold in enumerate(splits["tex_pair_folds"]):
            triads = [
                (s, t, emb.row("reference", s, t), emb.row("shape_match", s, t),
                 emb.row("texture_match", s, t))
                for s in mesh_fold
                for t in tex_fold
            ]
            triads = [x for x in triads if None not in x[2:]]
            if not triads:
                continue
            R = emb.X[np.array([x[2] for x in triads])]
            S = emb.X[np.array([x[3] for x in triads])]
            T = emb.X[np.array([x[4] for x in triads])]

            tr_m = [m for m in all_m if m not in set(mesh_fold)]
            tr_t = [t for t in all_t if t not in set(tex_fold)]
            Xtr, _, _ = emb.reference_block(tr_m, tr_t)
            if len(Xtr) < 10:
                continue

            def margin(Rz, Sz, Tz):
                return _cos(Rz, Sz) - _cos(Rz, Tz)

            m_raw = margin(R, S, T)

            mu_eval = np.concatenate([R, S, T]).mean(axis=0, keepdims=True)
            m_cent = margin(R - mu_eval, S - mu_eval, T - mu_eval)

            mu_tr = Xtr.mean(axis=0, keepdims=True)
            m_cent_tr = margin(R - mu_tr, S - mu_tr, T - mu_tr)

            mu_z, Wz = _zca(Xtr)
            m_zca = margin((R - mu_z) @ Wz, (S - mu_z) @ Wz, (T - mu_z) @ Wz)

            for k, (s, t, *_rest) in enumerate(triads):
                rows.append({
                    "stl_id": int(s),
                    "texture_set": str(t),
                    "mesh_fold": mi,
                    "tex_fold": ti,
                    "margin_raw": float(m_raw[k]),
                    "margin_centered": float(m_cent[k]),
                    "margin_centered_train": float(m_cent_tr[k]),
                    "margin_zca": float(m_zca[k]),
                })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, probe_json: Path, name: str) -> bool:
    """Sign of the margins must reproduce the published rung shape rates."""
    if not probe_json.exists():
        print(f"  {name}: no probe JSON, skipping validation")
        return True
    rungs = json.loads(probe_json.read_text())["ladder"]["rungs"]
    ok = True
    for col, rung in RUNG_OF_COLUMN.items():
        if rung not in rungs:
            continue
        got = float((df[col] > 0).mean())
        want = float(rungs[rung]["shape_rate"])
        line = f"  {name}: {rung} recomputed={got:.4f} published={want:.4f}"
        if abs(got - want) > 5e-3:
            print(line + "  MISMATCH")
            ok = False
        else:
            print(line + "  ok")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--models", nargs="*", help="restrict to these model names")
    args = ap.parse_args()

    npzs = sorted(
        p for p in args.emb_dir.glob("*.npz")
        if p.stem != "pixels32" and not p.stem.endswith("_tokens")
    )
    if args.models:
        keep = set(args.models)
        npzs = [p for p in npzs if p.stem.rsplit("_", 2)[0].split("_")[0] in keep
                or any(p.stem.startswith(m + "_") for m in keep)]
    if not npzs:
        print("no embedding NPZs matched")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    for path in npzs:
        out = args.out_dir / f"{path.stem}_margins.csv"
        if out.exists():
            print(f"skip (exists): {out.name}")
            continue
        emb = GridEmbeddings.load(path)
        df = export_margins(emb)
        if not validate(df, PROBE_DIR / f"probe_{path.stem}.json", path.stem):
            n_bad += 1
        df.to_csv(out, index=False)
        print(f"  wrote {out.name}  ({len(df)} triads, dim={emb.X.shape[1]})")
    if n_bad:
        print(f"\n{n_bad} file(s) failed validation against probe JSONs")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
