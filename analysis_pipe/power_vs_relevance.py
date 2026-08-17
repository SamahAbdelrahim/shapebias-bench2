#!/usr/bin/env python3
"""Separate read-out *power* from *relevance* for the ZCA > raw > centered
linkage ranking.

  1 incremental variance   (done): nest logit_gap ~ raw + ZCA
  2 accuracy matching      : noise-handicap ZCA until its shape rate matches
                             raw's, then re-compare linkage to the logit gap
  3 attenuation correction : deferred
  4 PCA-spectrum margins   : build margins from top / middle / tail PCs and
                             correlate each with the logit gap

Reuses loaders from linkage_analysis.py and fold helpers from linear_probe.
Default model: qwen3.5-4b.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis_pipe"))
sys.path.insert(0, str(REPO_ROOT / "playgrounds"))
from linkage_analysis import (  # noqa: E402
    CELLS,
    FIG_DIR,
    discover_margins,
    load_gates,
    load_logit_items,
)
from linear_probe import (  # noqa: E402
    GridEmbeddings,
    _cos,
    assert_disjoint,
    build_splits,
)

EMB_DIR = REPO_ROOT / "results/probe.results/session_readout_power/embeddings_grid"
DATA_DIR = REPO_ROOT / "results/data"

OUT_INC = DATA_DIR / "power_vs_relevance_incremental.csv"
OUT_ACC = DATA_DIR / "power_vs_relevance_accuracy_match.csv"
OUT_PCA = DATA_DIR / "power_vs_relevance_pca_spectrum.csv"
FIG_INC = FIG_DIR / "power_vs_relevance_incremental.png"
FIG_ACC = FIG_DIR / "power_vs_relevance_accuracy_match.png"
FIG_PCA = FIG_DIR / "power_vs_relevance_pca_spectrum.png"

MARGIN_CENTERED = "margin_centered"
N_NOISE_SEEDS = 20


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def _loci_for(model: str, margins: dict) -> list[str]:
    loci = sorted({l for (m, l) in margins if m == model})
    if "proj_mean" in loci and "vit_last_mean" in loci:
        loci = [l for l in loci if l != "proj_mean"]
    return loci


# --------------------------------------------------------------------------
# 1. incremental variance
# --------------------------------------------------------------------------


def nested_gap_models(df: pd.DataFrame) -> dict:
    d = df[["logit_gap", "margin_raw", "margin_zca", "stl_id"]].dropna().copy()
    if len(d) < 50:
        return {}
    d["raw_z"] = _z(d["margin_raw"])
    d["zca_z"] = _z(d["margin_zca"])
    groups = d["stl_id"]

    def fit(formula: str):
        return smf.ols(formula, d).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )

    m_raw = fit("logit_gap ~ raw_z")
    m_zca = fit("logit_gap ~ zca_z")
    m_both = fit("logit_gap ~ raw_z + zca_z")
    return {
        "n": len(d),
        "r_raw_zca": float(np.corrcoef(d["margin_raw"], d["margin_zca"])[0, 1]),
        "r_gap_raw": float(np.corrcoef(d["margin_raw"], d["logit_gap"])[0, 1]),
        "r_gap_zca": float(np.corrcoef(d["margin_zca"], d["logit_gap"])[0, 1]),
        "r2_raw": float(m_raw.rsquared),
        "r2_zca": float(m_zca.rsquared),
        "r2_both": float(m_both.rsquared),
        "delta_r2_zca_given_raw": float(m_both.rsquared - m_raw.rsquared),
        "delta_r2_raw_given_zca": float(m_both.rsquared - m_zca.rsquared),
        "beta_raw_alone": float(m_raw.params["raw_z"]),
        "beta_zca_alone": float(m_zca.params["zca_z"]),
        "beta_raw_joint": float(m_both.params["raw_z"]),
        "beta_zca_joint": float(m_both.params["zca_z"]),
        "p_raw_joint": float(m_both.pvalues["raw_z"]),
        "p_zca_joint": float(m_both.pvalues["zca_z"]),
        "se_raw_joint": float(m_both.bse["raw_z"]),
        "se_zca_joint": float(m_both.bse["zca_z"]),
    }


def centered_companion(df: pd.DataFrame) -> dict:
    if MARGIN_CENTERED not in df.columns:
        return {}
    d = df[["logit_gap", MARGIN_CENTERED, "margin_zca", "stl_id"]].dropna().copy()
    if len(d) < 50:
        return {}
    d["cen_z"] = _z(d[MARGIN_CENTERED])
    d["zca_z"] = _z(d["margin_zca"])
    groups = d["stl_id"]

    def fit(formula: str):
        return smf.ols(formula, d).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )

    m_cen = fit("logit_gap ~ cen_z")
    m_zca = fit("logit_gap ~ zca_z")
    m_both = fit("logit_gap ~ cen_z + zca_z")
    return {
        "r_centered_zca": float(np.corrcoef(
            d[MARGIN_CENTERED], d["margin_zca"])[0, 1]),
        "r2_centered": float(m_cen.rsquared),
        "r2_zca_vs_cen": float(m_zca.rsquared),
        "r2_cen_zca_both": float(m_both.rsquared),
        "delta_r2_zca_given_centered": float(m_both.rsquared - m_cen.rsquared),
        "delta_r2_centered_given_zca": float(m_both.rsquared - m_zca.rsquared),
        "p_centered_joint": float(m_both.pvalues["cen_z"]),
        "p_zca_joint_vs_cen": float(m_both.pvalues["zca_z"]),
    }


def run_incremental(models: list[str]) -> pd.DataFrame:
    margins = discover_margins()
    gates = load_gates()
    gate_of = {
        (r.model, r.prompt_condition): bool(r.logit_gate_pass)
        for r in gates.itertuples()
    }
    rows = []
    for model in models:
        for locus in _loci_for(model, margins):
            mdf = pd.read_csv(margins[(model, locus)])
            for cell in CELLS:
                logit = load_logit_items(model, cell)
                if logit is None:
                    continue
                j = logit.merge(mdf, on=["stl_id", "texture_set"], how="inner")
                stats = nested_gap_models(j)
                if not stats:
                    continue
                rows.append({
                    "model": model, "prompt_condition": cell, "locus": locus,
                    "logit_gate": gate_of.get((model, cell), False),
                    "analysis": "incremental_raw_zca",
                    **stats, **centered_companion(j),
                })
    return pd.DataFrame(rows)


def fig_incremental(df: pd.DataFrame) -> None:
    if df.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    loci = list(df["locus"].unique())
    cells = list(CELLS)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)

    def series(col):
        out = {}
        for locus in loci:
            sub = df[df["locus"] == locus]
            out[locus] = [
                float(sub.loc[sub.prompt_condition == c, col].mean())
                if (sub.prompt_condition == c).any() else np.nan
                for c in cells
            ]
        return out

    ax = axes[0]
    for locus, ys in series("r_raw_zca").items():
        ax.plot(ys, range(len(cells)), "o-", label=locus, ms=5)
    ax.axvline(0.95, color="0.7", ls="--", lw=0.8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=8)
    ax.set_xlabel("r(raw, ZCA)")
    ax.set_title("A. margin collinearity\n(dashed = 0.95, low nest power)")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()

    ax = axes[1]
    for locus, ys in series("delta_r2_zca_given_raw").items():
        ax.plot(ys, range(len(cells)), "o-", label=locus, ms=5)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([""] * len(cells))
    ax.set_xlabel("ΔR²: ZCA | raw")
    ax.set_title("B. unique variance from ZCA\nonce raw is known")
    ax.invert_yaxis()

    ax = axes[2]
    for locus, ys in series("delta_r2_raw_given_zca").items():
        ax.plot(ys, range(len(cells)), "o-", label=locus, ms=5)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([""] * len(cells))
    ax.set_xlabel("ΔR²: raw | ZCA")
    ax.set_title("C. unique variance from raw\nonce ZCA is known")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"Power vs relevance (1): incremental variance  [{df.model.iloc[0]}]",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_INC, dpi=160, bbox_inches="tight")
    fig.savefig(FIG_INC.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_INC}")


# --------------------------------------------------------------------------
# 2. accuracy matching
# --------------------------------------------------------------------------


def _shape_rate(margin: np.ndarray) -> float:
    return float((margin > 0).mean())


def _noise_sd_for_target(
    margin: np.ndarray, target_rate: float, n_seeds: int = 8
) -> float:
    """Binary-search Gaussian noise SD so E[P(m+ε>0)] ≈ target_rate."""
    m = np.asarray(margin, dtype=float)
    base = _shape_rate(m)
    # already at or below target: no noise (or cannot lower further usefully)
    if abs(base - target_rate) < 0.005:
        return 0.0
    if base < target_rate:
        # noise pulls toward 0.5; cannot raise a below-chance rate above itself
        # toward a higher target without bias. Return 0 and flag later.
        return 0.0

    lo, hi = 0.0, float(m.std() * 20 + 1e-6)

    def rate_at(sd: float) -> float:
        rates = []
        for s in range(n_seeds):
            eps = np.random.default_rng(1000 + s).normal(0.0, sd, size=m.shape)
            rates.append(_shape_rate(m + eps))
        return float(np.mean(rates))

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = rate_at(mid)
        if r > target_rate:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def run_accuracy_match(models: list[str], n_seeds: int = N_NOISE_SEEDS) -> pd.DataFrame:
    """Handicap ZCA shape rate to match raw, then compare linkage to logit gap."""
    margins = discover_margins()
    gates = load_gates()
    gate_of = {
        (r.model, r.prompt_condition): bool(r.logit_gate_pass)
        for r in gates.itertuples()
    }
    rows = []
    # noise SD is a property of the margin distribution (locus), not the cell
    noise_of: dict[tuple[str, str], dict] = {}
    for model in models:
        for locus in _loci_for(model, margins):
            mdf = pd.read_csv(margins[(model, locus)])
            raw = mdf["margin_raw"].to_numpy()
            zca = mdf["margin_zca"].to_numpy()
            raw_rate = _shape_rate(raw)
            zca_rate = _shape_rate(zca)
            sd = _noise_sd_for_target(zca, raw_rate)
            # verify achieved rate
            achieved = []
            for s in range(n_seeds):
                achieved.append(_shape_rate(
                    zca + np.random.default_rng(s).normal(0.0, sd, size=zca.shape)
                ))
            noise_of[(model, locus)] = {
                "raw_shape_rate": raw_rate,
                "zca_shape_rate": zca_rate,
                "noise_sd": sd,
                "handicapped_shape_rate_mean": float(np.mean(achieved)),
                "handicapped_shape_rate_sd": float(np.std(achieved)),
            }
            print(
                f"  {model} {locus}: raw_rate={raw_rate:.3f} zca_rate={zca_rate:.3f} "
                f"noise_sd={sd:.4f} -> handicapped={np.mean(achieved):.3f}"
            )

            for cell in CELLS:
                logit = load_logit_items(model, cell)
                if logit is None:
                    continue
                j = logit.merge(mdf, on=["stl_id", "texture_set"], how="inner")
                if len(j) < 50:
                    continue
                r_raw = float(sps.pearsonr(j["margin_raw"], j["logit_gap"]).statistic)
                r_zca = float(sps.pearsonr(j["margin_zca"], j["logit_gap"]).statistic)
                s_raw = float(sps.spearmanr(j["margin_raw"], j["logit_gap"]).statistic)
                s_zca = float(sps.spearmanr(j["margin_zca"], j["logit_gap"]).statistic)

                r_h, s_h = [], []
                for seed in range(n_seeds):
                    noisy = j["margin_zca"].to_numpy() + np.random.default_rng(
                        seed
                    ).normal(0.0, sd, size=len(j))
                    r_h.append(float(sps.pearsonr(noisy, j["logit_gap"]).statistic))
                    s_h.append(float(sps.spearmanr(noisy, j["logit_gap"]).statistic))

                meta = noise_of[(model, locus)]
                rows.append({
                    "model": model,
                    "prompt_condition": cell,
                    "locus": locus,
                    "logit_gate": gate_of.get((model, cell), False),
                    "analysis": "accuracy_match",
                    "n": len(j),
                    "n_noise_seeds": n_seeds,
                    **meta,
                    "pearson_raw": r_raw,
                    "pearson_zca": r_zca,
                    "pearson_handicapped_mean": float(np.mean(r_h)),
                    "pearson_handicapped_sd": float(np.std(r_h)),
                    "spearman_raw": s_raw,
                    "spearman_zca": s_zca,
                    "spearman_handicapped_mean": float(np.mean(s_h)),
                    "spearman_handicapped_sd": float(np.std(s_h)),
                    "handicapped_beats_raw_pearson": float(
                        np.mean(r_h) > r_raw
                    ),
                    "handicapped_beats_raw_spearman": float(
                        np.mean(s_h) > s_raw
                    ),
                })
    return pd.DataFrame(rows)


def fig_accuracy_match(df: pd.DataFrame) -> None:
    if df.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    loci = list(df["locus"].unique())
    fig, axes = plt.subplots(1, len(loci), figsize=(3.6 * len(loci), 3.6),
                             sharey=True)
    if len(loci) == 1:
        axes = [axes]
    x = np.arange(3)
    labels = ["raw", "ZCA", "ZCA\n(handicapped)"]
    for ax, locus in zip(axes, loci):
        sub = df[df["locus"] == locus]
        means = [
            sub["spearman_raw"].mean(),
            sub["spearman_zca"].mean(),
            sub["spearman_handicapped_mean"].mean(),
        ]
        # cell-level scatter
        for _, r in sub.iterrows():
            ax.plot(
                x,
                [r.spearman_raw, r.spearman_zca, r.spearman_handicapped_mean],
                color="0.75", lw=0.8, marker="o", ms=3, alpha=0.7,
            )
        ax.plot(x, means, "ko-", ms=7, lw=1.5, label="mean across cells")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(locus, fontsize=9)
        ax.axhline(0, color="0.7", lw=0.6)
        rate_note = (
            f"rates raw={sub.raw_shape_rate.iloc[0]:.2f} "
            f"zca={sub.zca_shape_rate.iloc[0]:.2f} "
            f"hcap={sub.handicapped_shape_rate_mean.iloc[0]:.2f}"
        )
        ax.text(0.02, 0.02, rate_note, transform=ax.transAxes, fontsize=6,
                va="bottom")
    axes[0].set_ylabel("Spearman r(margin, logit gap)")
    fig.suptitle(
        f"Power vs relevance (2): accuracy-matched ZCA  [{df.model.iloc[0]}]",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_ACC, dpi=160, bbox_inches="tight")
    fig.savefig(FIG_ACC.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_ACC}")


# --------------------------------------------------------------------------
# 4. PCA-spectrum margins
# --------------------------------------------------------------------------


def _pc_slices(n_dim: int) -> list[tuple[str, slice]]:
    """Named bands over the PC spectrum (0 = highest variance)."""
    # keep bands that fit; skip empties for small dims
    candidates = [
        ("top8", slice(0, min(8, n_dim))),
        ("top32", slice(0, min(32, n_dim))),
        ("top128", slice(0, min(128, n_dim))),
        ("mid32_128", slice(min(32, n_dim), min(128, n_dim))),
        ("mid128_512", slice(min(128, n_dim), min(512, n_dim))),
        ("tail128", slice(max(0, n_dim - 128), n_dim)),
        ("tail32", slice(max(0, n_dim - 32), n_dim)),
        ("all", slice(0, n_dim)),
    ]
    out = []
    seen = set()
    for name, sl in candidates:
        if sl.start >= sl.stop:
            continue
        key = (sl.start, sl.stop)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, sl))
    return out


def _pca_fit(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, components[D, D] with rows = PCs, descending variance)."""
    mu = Xtr.mean(axis=0)
    Xc = Xtr - mu
    # economy SVD; Vh rows are PCs
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt


def export_pca_slice_margins(emb: GridEmbeddings) -> pd.DataFrame:
    """Per-triad margins from PC bands; PCA fit on train-fold references only."""
    splits = build_splits(emb)
    assert_disjoint(splits, emb)
    all_m = [int(m) for m in emb.meshes]
    all_t = [str(t) for t in emb.textures]
    rows: list[dict] = []
    # slice menu is rebuilt per fold from the actual PC count (≤ min(N_train, D))
    slices_template = None

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
            mu, Vt = _pca_fit(Xtr)
            n_pc = Vt.shape[0]
            if slices_template is None:
                slices_template = _pc_slices(n_pc)
                print(f"    PCA rank={n_pc} (emb dim={emb.X.shape[1]}); "
                      f"slices={[s for s,_ in slices_template]}")
            Rc, Sc, Tc = R - mu, S - mu, T - mu

            for name, sl in slices_template:
                # clamp in case a later fold has fewer PCs
                start = min(sl.start, n_pc)
                stop = min(sl.stop, n_pc)
                if start >= stop:
                    continue
                V = Vt[start:stop].T  # [D, k]
                Rm, Sm, Tm = Rc @ V, Sc @ V, Tc @ V
                marg = _cos(Rm, Sm) - _cos(Rm, Tm)
                for k, (s, t, *_rest) in enumerate(triads):
                    rows.append({
                        "stl_id": int(s),
                        "texture_set": str(t),
                        "mesh_fold": mi,
                        "tex_fold": ti,
                        "slice": name,
                        "n_pcs": int(V.shape[1]),
                        "pc_start": int(start),
                        "pc_stop": int(stop),
                        "margin": float(marg[k]),
                    })
    return pd.DataFrame(rows)


def run_pca_spectrum(models: list[str], loci_pref: list[str] | None = None) -> pd.DataFrame:
    """Correlate each PC-slice margin with logit gap; one locus at a time."""
    margins_index = discover_margins()
    gates = load_gates()
    gate_of = {
        (r.model, r.prompt_condition): bool(r.logit_gate_pass)
        for r in gates.itertuples()
    }
    rows = []
    for model in models:
        loci = _loci_for(model, margins_index)
        if loci_pref:
            loci = [l for l in loci if l in loci_pref] or loci
        # default focal locus: vit_penult if present, else first
        if loci_pref is None:
            if "vit_penult_mean" in loci:
                loci = ["vit_penult_mean"]
            else:
                loci = loci[:1]
        for locus in loci:
            path = EMB_DIR / f"{model}_{locus}.npz"
            if not path.exists():
                # pooler naming
                alt = EMB_DIR / f"{model}_{locus.replace('_mean','')}.npz"
                path = alt if alt.exists() else path
            if not path.exists():
                print(f"  missing emb {path}")
                continue
            print(f"  PCA spectrum: {path.name}")
            emb = GridEmbeddings.load(path)
            mdf = export_pca_slice_margins(emb)
            for cell in CELLS:
                logit = load_logit_items(model, cell)
                if logit is None:
                    continue
                for sl_name, g in mdf.groupby("slice", sort=False):
                    j = logit.merge(
                        g[["stl_id", "texture_set", "margin", "n_pcs",
                           "pc_start", "pc_stop"]],
                        on=["stl_id", "texture_set"], how="inner",
                    )
                    if len(j) < 50:
                        continue
                    pear = sps.pearsonr(j["margin"], j["logit_gap"])
                    spear = sps.spearmanr(j["margin"], j["logit_gap"])
                    shape_rate = _shape_rate(j["margin"].to_numpy())
                    rows.append({
                        "model": model,
                        "prompt_condition": cell,
                        "locus": locus,
                        "logit_gate": gate_of.get((model, cell), False),
                        "analysis": "pca_spectrum",
                        "slice": sl_name,
                        "n_pcs": int(g["n_pcs"].iloc[0]),
                        "pc_start": int(g["pc_start"].iloc[0]),
                        "pc_stop": int(g["pc_stop"].iloc[0]),
                        "n": len(j),
                        "shape_rate": shape_rate,
                        "pearson_r": float(pear.statistic),
                        "pearson_p": float(pear.pvalue),
                        "spearman_r": float(spear.statistic),
                        "spearman_p": float(spear.pvalue),
                    })
    return pd.DataFrame(rows)


def fig_pca_spectrum(df: pd.DataFrame) -> None:
    if df.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # order slices by pc_start then n_pcs
    slice_order = (
        df.groupby("slice")[["pc_start", "pc_stop", "n_pcs"]]
        .first()
        .sort_values(["pc_start", "pc_stop"])
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for cell in CELLS:
        sub = df[df.prompt_condition == cell]
        if sub.empty:
            continue
        ys = [float(sub.loc[sub["slice"] == s, "spearman_r"].mean())
              if (sub["slice"] == s).any() else np.nan for s in slice_order]
        ax.plot(range(len(slice_order)), ys, "o-", ms=4, lw=1, label=cell)
    # mean across cells
    means = [float(df.loc[df["slice"] == s, "spearman_r"].mean())
             for s in slice_order]
    ax.plot(range(len(slice_order)), means, "k|-", ms=12, lw=2, label="mean")
    ax.set_xticks(range(len(slice_order)))
    ax.set_xticklabels(slice_order, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_ylabel("Spearman r(slice margin, logit gap)")
    ax.set_title(
        f"Power vs relevance (4): PCA-spectrum margins  "
        f"[{df.model.iloc[0]} | {df.locus.iloc[0]}]"
    )
    ax.legend(fontsize=6, ncol=2, loc="best")
    # annotate shape rates on mean line
    for i, s in enumerate(slice_order):
        sr = float(df.loc[df["slice"] == s, "shape_rate"].mean())
        ax.annotate(f"{sr:.2f}", (i, means[i]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6, color="0.35")
    fig.tight_layout()
    fig.savefig(FIG_PCA, dpi=160, bbox_inches="tight")
    fig.savefig(FIG_PCA.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_PCA}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3.5-4b"])
    ap.add_argument(
        "--tests", nargs="+", type=int, default=[2, 4],
        help="which tests to run (1 incremental, 2 accuracy match, 4 PCA); "
             "3 attenuation deferred",
    )
    ap.add_argument(
        "--pca-loci", nargs="+", default=None,
        help="loci for PCA test (default: vit_penult_mean only)",
    )
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if 1 in args.tests:
        print("=== Test 1: incremental variance ===")
        df = run_incremental(args.models)
        if df.empty:
            print("no rows")
        else:
            df.to_csv(OUT_INC, index=False)
            print(f"wrote {OUT_INC} ({len(df)} rows)")
            print(df.groupby("locus")[
                ["r_raw_zca", "delta_r2_zca_given_raw", "delta_r2_raw_given_zca"]
            ].mean().round(4).to_string())
            fig_incremental(df)

    if 2 in args.tests:
        print("=== Test 2: accuracy matching ===")
        df = run_accuracy_match(args.models)
        if df.empty:
            print("no rows")
        else:
            df.to_csv(OUT_ACC, index=False)
            print(f"wrote {OUT_ACC} ({len(df)} rows)")
            show = df.groupby("locus")[
                ["raw_shape_rate", "zca_shape_rate", "handicapped_shape_rate_mean",
                 "spearman_raw", "spearman_zca", "spearman_handicapped_mean",
                 "handicapped_beats_raw_spearman"]
            ].mean()
            print(show.round(4).to_string())
            fig_accuracy_match(df)

    if 3 in args.tests:
        print("Test 3 (attenuation) deferred by request.")

    if 4 in args.tests:
        print("=== Test 4: PCA-spectrum margins ===")
        df = run_pca_spectrum(args.models, loci_pref=args.pca_loci)
        if df.empty:
            print("no rows")
        else:
            df.to_csv(OUT_PCA, index=False)
            print(f"wrote {OUT_PCA} ({len(df)} rows)")
            print(df.groupby("slice")[
                ["n_pcs", "shape_rate", "spearman_r", "pearson_r"]
            ].mean().round(4).to_string())
            fig_pca_spectrum(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
