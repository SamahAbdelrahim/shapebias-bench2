#!/usr/bin/env python3
"""Diagnostics prompted by the Test 2 / Test 4 critique.

  A  corr(top128 margin, tail128 margin) on the same items: is the linkage
     shape-specific or a shared diffuse component?
  B  calibration: raw margin shape rate before/after removing its mean, and
     AUC of raw/ZCA margins against the sign of the logit gap (ranking
     without the sign-at-zero convention).
  C  fixed-width sliding windows over the PC spectrum (equal rank), shape
     rate and gap linkage per window.
  D  unified rung table: raw / centered / centered_train / ZCA x locus x
     cell gap correlations in one code path, for qwen3.5-4b and 9b.
  E  cross-model swap: 4b margins predicting 9b gaps and vice versa,
     against the within-model numbers (item-difficulty confound check).

CPU only, reuses margin CSVs and grid embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis_pipe"))
sys.path.insert(0, str(REPO_ROOT / "playgrounds"))
from linkage_analysis import CELLS, FIG_DIR, discover_margins, load_logit_items  # noqa: E402
from linear_probe import GridEmbeddings, _cos, assert_disjoint, build_splits  # noqa: E402
from power_vs_relevance import _pca_fit, export_pca_slice_margins  # noqa: E402

EMB_DIR = REPO_ROOT / "results/probe.results/session_readout_power/embeddings_grid"
DATA_DIR = REPO_ROOT / "results/data"
MODEL = "qwen3.5-4b"
MODEL_B = "qwen3.5-9b"
LOCUS = "vit_penult_mean"
WIN = 60  # 720 = 12 x 60, equal rank, full coverage


def window_margins(emb: GridEmbeddings, width: int) -> pd.DataFrame:
    """Per-triad margins from fixed-width PC windows (train-fold PCA)."""
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
                for s in mesh_fold for t in tex_fold
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
            mu, Vt = _pca_fit(Xtr)
            n_pc = Vt.shape[0]
            Rc, Sc, Tc = R - mu, S - mu, T - mu
            for w0 in range(0, n_pc - width + 1, width):
                V = Vt[w0:w0 + width].T
                marg = _cos(Rc @ V, Sc @ V) - _cos(Rc @ V, Tc @ V)
                for k, (s, t, *_r) in enumerate(triads):
                    rows.append({
                        "stl_id": int(s), "texture_set": str(t),
                        "pc_start": w0, "pc_stop": w0 + width,
                        "margin": float(marg[k]),
                    })
    return pd.DataFrame(rows)


def main() -> int:
    margins_index = discover_margins()
    emb = GridEmbeddings.load(EMB_DIR / f"{MODEL}_{LOCUS}.npz")

    # ---------------- A: slice-vs-slice correlations ----------------
    print("=== A. slice margin correlations (same items) ===")
    sl = export_pca_slice_margins(emb)
    wide = sl.pivot_table(index=["stl_id", "texture_set"], columns="slice",
                          values="margin")
    order = ["top8", "top32", "top128", "mid32_128", "mid128_512",
             "tail128", "tail32", "all"]
    order = [o for o in order if o in wide.columns]
    corr = wide[order].corr()
    print(corr.round(3).to_string())
    print(f"\nKEY: r(top128, tail128) = {corr.loc['top128','tail128']:.3f}   "
          f"r(top32, tail32) = {corr.loc['top32','tail32']:.3f}")

    # ---------------- B: calibration / AUC ----------------
    print("\n=== B. raw margin calibration ===")
    mdf = pd.read_csv(margins_index[(MODEL, LOCUS)])
    raw = mdf["margin_raw"].to_numpy()
    zca = mdf["margin_zca"].to_numpy()
    print(f"raw shape rate (sign at 0):        {(raw > 0).mean():.3f}")
    print(f"raw shape rate (sign at mean):     {(raw > raw.mean()).mean():.3f}")
    fold_adj = mdf.groupby(["mesh_fold", "tex_fold"])["margin_raw"].transform(
        lambda s: s - s.mean())
    print(f"raw shape rate (per-fold demeaned): {(fold_adj > 0).mean():.3f}")
    print(f"zca shape rate:                    {(zca > 0).mean():.3f}")

    aucs = []
    for cell in CELLS:
        logit = load_logit_items(MODEL, cell)
        if logit is None:
            continue
        j = logit.merge(mdf, on=["stl_id", "texture_set"], how="inner")
        y = (j["logit_gap"] > 0).astype(int)
        if y.nunique() < 2:
            continue
        aucs.append({
            "cell": cell,
            "auc_raw": roc_auc_score(y, j["margin_raw"]),
            "auc_zca": roc_auc_score(y, j["margin_zca"]),
            "auc_centered": roc_auc_score(y, j["margin_centered"]),
        })
    adf = pd.DataFrame(aucs)
    print("\nAUC of margins vs sign(logit gap), per cell:")
    print(adf.round(3).to_string(index=False))
    print("means:", adf[["auc_raw", "auc_zca", "auc_centered"]]
          .mean().round(3).to_dict())

    # ---------------- C: fixed-width windows ----------------
    print(f"\n=== C. sliding windows, width={WIN} ===")
    wm = window_margins(emb, WIN)
    win_rows = []
    for (w0, w1), g in wm.groupby(["pc_start", "pc_stop"]):
        srate = float((g["margin"] > 0).mean())
        rs = []
        for cell in CELLS:
            logit = load_logit_items(MODEL, cell)
            if logit is None:
                continue
            j = logit.merge(g, on=["stl_id", "texture_set"], how="inner")
            if len(j) < 50:
                continue
            rs.append(float(sps.spearmanr(j["margin"], j["logit_gap"]).statistic))
        win_rows.append({
            "pc_start": w0, "pc_stop": w1, "shape_rate": srate,
            "spearman_mean": float(np.mean(rs)), "spearman_sd": float(np.std(rs)),
        })
    wdf = pd.DataFrame(win_rows).sort_values("pc_start")
    print(wdf.round(3).to_string(index=False))
    wdf.to_csv(DATA_DIR / "power_vs_relevance_pc_windows.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    x = wdf["pc_start"] + WIN / 2
    ax1.errorbar(x, wdf["spearman_mean"], yerr=wdf["spearman_sd"],
                 fmt="ko-", ms=4, lw=1.2, label="gap linkage (spearman)")
    ax1.set_ylabel("spearman r(margin, logit gap)")
    ax1.set_xlabel(f"PC window (width {WIN}, train-fold PCA, rank 720)")
    ax1.axhline(0, color="0.7", lw=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, wdf["shape_rate"], "s--", color="tab:red", ms=4, lw=1,
             label="shape rate P(margin>0)")
    ax2.axhline(0.5, color="tab:red", lw=0.5, alpha=0.4)
    ax2.set_ylabel("shape rate", color="tab:red")
    ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    ax1.set_title(f"Equal-rank PC windows: decode vs linkage  [{MODEL} | {LOCUS}]")
    fig.tight_layout()
    out = FIG_DIR / "power_vs_relevance_pc_windows.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---------------- D: unified rung table ----------------
    print("\n=== D. unified rung x locus gap linkage (spearman, mean over cells) ===")
    rung_cols = ["margin_raw", "margin_centered", "margin_centered_train",
                 "margin_zca"]
    rows = []
    for model in (MODEL, MODEL_B):
        loci = sorted({l for (m, l) in margins_index if m == model
                       and l != "proj_mean"})
        for locus in loci:
            mm = pd.read_csv(margins_index[(model, locus)])
            for cell in CELLS:
                logit = load_logit_items(model, cell)
                if logit is None:
                    continue
                j = logit.merge(mm, on=["stl_id", "texture_set"], how="inner")
                if len(j) < 50:
                    continue
                for col in rung_cols:
                    rows.append({
                        "model": model, "locus": locus, "cell": cell,
                        "rung": col.replace("margin_", ""),
                        "spearman_r": float(
                            sps.spearmanr(j[col], j["logit_gap"]).statistic),
                        "shape_rate": float((j[col] > 0).mean()),
                    })
    rt = pd.DataFrame(rows)
    rt.to_csv(DATA_DIR / "power_vs_relevance_rung_table.csv", index=False)
    piv = rt.pivot_table(index=["model", "locus"], columns="rung",
                         values="spearman_r", aggfunc="mean")
    print(piv.round(3).to_string())
    piv2 = rt.pivot_table(index=["model", "locus"], columns="rung",
                          values="shape_rate", aggfunc="mean")
    print("\nshape rates for the same rungs:")
    print(piv2.round(3).to_string())

    # ---------------- E: cross-model swap ----------------
    print("\n=== E. cross-model margin swap (penult, spearman mean over cells) ===")
    swap_rows = []
    m4 = pd.read_csv(margins_index[(MODEL, LOCUS)])
    m9 = pd.read_csv(margins_index[(MODEL_B, LOCUS)])
    for rung in ("margin_raw", "margin_zca"):
        for margin_owner, mdfx in ((MODEL, m4), (MODEL_B, m9)):
            for behavior_owner in (MODEL, MODEL_B):
                rs = []
                for cell in CELLS:
                    logit = load_logit_items(behavior_owner, cell)
                    if logit is None:
                        continue
                    j = logit.merge(mdfx, on=["stl_id", "texture_set"],
                                    how="inner")
                    if len(j) < 50:
                        continue
                    rs.append(float(
                        sps.spearmanr(j[rung], j["logit_gap"]).statistic))
                swap_rows.append({
                    "rung": rung.replace("margin_", ""),
                    "margins_from": margin_owner,
                    "behavior_from": behavior_owner,
                    "within": margin_owner == behavior_owner,
                    "spearman_mean": float(np.mean(rs)),
                })
    sw = pd.DataFrame(swap_rows)
    sw.to_csv(DATA_DIR / "power_vs_relevance_crossmodel.csv", index=False)
    print(sw.round(3).to_string(index=False))
    # also gap-gap correlation between the two models
    gg = []
    for cell in CELLS:
        a = load_logit_items(MODEL, cell)
        b = load_logit_items(MODEL_B, cell)
        if a is None or b is None:
            continue
        j = a.merge(b, on=["stl_id", "texture_set"], suffixes=("_4b", "_9b"))
        gg.append(float(sps.spearmanr(j["logit_gap_4b"],
                                      j["logit_gap_9b"]).statistic))
    print(f"\nr(4b gap, 9b gap) across items, mean over cells: {np.mean(gg):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
