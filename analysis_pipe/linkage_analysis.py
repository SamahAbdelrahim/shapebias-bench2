#!/usr/bin/env python3
"""Item-level linkage between encoder read-out margins and model behavior.

Three analyses, all on existing data (Williams, Dang & Kanwisher 2007 logic:
decodable information should track behavior if it is actually read out):

  1a  Williams-style split: within model x gate-passing prompt cell, is the
      whitened margin larger on trials where the model chose shape than where
      it chose texture? Logistic mixed model with crossed random intercepts
      for mesh and texture (BinomialBayesMixedGLM, VB fit), margin
      standardized within cell x locus so slopes are comparable across loci.

  1b  Distance-to-bound analogue: the signed logit gap
      log p(shape) - log p(texture), averaged over the two option orders,
      regressed on the margin per locus. Graded on both sides, so stronger
      than 1a. Computed for every cell; gate flags from both paths are
      carried in the output.

  1c  Order-swap flips as trial-level variance: items whose generated choice
      flips between the two option orders should sit closer to the read-out
      boundary (smaller |margin|) than items stable in both orders.

Inputs
------
  margins        results/probe.results/session_readout_power/margins/*.csv
                 (from playgrounds/margin_export.py)
  generation     results/model.results/session_full_grid_v1a/
  logit          results/model.results/session_full_grid_v1a_logit/
  gates          results/data/full_grid_v1a_summary.csv (generation path)
                 results/data/full_grid_logit_validity.csv (logit path)

Outputs
-------
  results/data/linkage_summary.csv        long-format statistics
  results/figures/linkage/*.png/.pdf      summary figures
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]
MARGIN_DIR = REPO_ROOT / "results/probe.results/session_readout_power/margins"
GEN_DIR = REPO_ROOT / "results/model.results/session_full_grid_v1a"
LOGIT_DIR = REPO_ROOT / "results/model.results/session_full_grid_v1a_logit"
SUMMARY_CSV = REPO_ROOT / "results/data/full_grid_v1a_summary.csv"
LOGIT_VALIDITY_CSV = REPO_ROOT / "results/data/full_grid_logit_validity.csv"
OUT_CSV = REPO_ROOT / "results/data/linkage_summary.csv"
FIG_DIR = REPO_ROOT / "results/figures/linkage"

LOCI = ("proj_mean", "vit_last_mean", "vit_penult_mean", "vit_pooler",
        "proj_lm_mean")
MARGIN_COL = "margin_zca"  # primary read-out; others carried as sensitivity
CELLS = (
    "no_word_similarity", "no_word_similarity_AB",
    "no_word_category", "no_word_category_AB",
    "noun_label", "noun_label_AB",
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def discover_margins() -> dict[tuple[str, str], Path]:
    """{(model, locus): path} parsed from '<model>_<locus>_margins.csv'."""
    out: dict[tuple[str, str], Path] = {}
    for p in sorted(MARGIN_DIR.glob("*_margins.csv")):
        stem = p.name[: -len("_margins.csv")]
        for locus in LOCI:
            if stem.endswith("_" + locus):
                out[(stem[: -len(locus) - 1], locus)] = p
                break
    return out


def behavioral_csv(session_dir: Path, model: str, cell: str) -> Path | None:
    hits = sorted(session_dir.glob(f"{model}__{cell}*.csv"))
    # noun_label glob also catches noun_label_AB_shiple; disambiguate exactly
    exact = [h for h in hits if h.stem in (f"{model}__{cell}", f"{model}__{cell}_shiple")]
    return exact[0] if exact else None


def load_generation_items(model: str, cell: str) -> pd.DataFrame | None:
    """Per-item generation outcome: shape-choice count and swap stability."""
    path = behavioral_csv(GEN_DIR, model, cell)
    if path is None:
        return None
    df = pd.read_csv(path, usecols=["stl_id", "texture_set", "ordering", "choice"])
    df = df[df["choice"].isin(["shape", "texture"])]
    df["is_shape"] = (df["choice"] == "shape").astype(int)
    g = df.groupby(["stl_id", "texture_set"])["is_shape"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "p_shape_gen", "count": "n_orders"}).reset_index()
    g["flip"] = ((g["p_shape_gen"] > 0) & (g["p_shape_gen"] < 1)).astype(int)
    return g


def load_logit_items(model: str, cell: str) -> pd.DataFrame | None:
    """Per-item signed logit gap log p(shape) - log p(texture), order-averaged."""
    path = behavioral_csv(LOGIT_DIR, model, cell)
    if path is None:
        return None
    df = pd.read_csv(
        path, usecols=["stl_id", "texture_set", "a_is", "prob_1_abs", "prob_2_abs"]
    )
    df = df.dropna(subset=["prob_1_abs", "prob_2_abs"])
    shape_is_1 = df["a_is"] == "shape"
    p_shape = np.where(shape_is_1, df["prob_1_abs"], df["prob_2_abs"])
    p_tex = np.where(shape_is_1, df["prob_2_abs"], df["prob_1_abs"])
    df["gap"] = np.log(np.maximum(p_shape, 1e-9)) - np.log(np.maximum(p_tex, 1e-9))
    g = (
        df.groupby(["stl_id", "texture_set"])["gap"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "logit_gap", "count": "n_orders_logit"})
        .reset_index()
    )
    return g


def load_gates() -> pd.DataFrame:
    gen = pd.read_csv(SUMMARY_CSV)[["model", "prompt_condition", "gate_pass"]]
    logit = pd.read_csv(LOGIT_VALIDITY_CSV)[
        ["model", "prompt_condition", "logit_gate_pass"]
    ]
    return gen.merge(logit, on=["model", "prompt_condition"], how="outer")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def mixed_logistic_slope(df: pd.DataFrame, y: str, x: str) -> dict:
    """choice ~ margin with crossed random intercepts (mesh, texture), VB fit."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df[[y, x, "stl_id", "texture_set"]].dropna().copy()
    if d[y].nunique() < 2 or len(d) < 50:
        return {"slope": float("nan"), "slope_sd": float("nan"), "n": len(d)}
    d["xz"] = (d[x] - d[x].mean()) / (d[x].std() + 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = BinomialBayesMixedGLM.from_formula(
            f"{y} ~ xz",
            {"mesh": "0 + C(stl_id)", "tex": "0 + C(texture_set)"},
            d,
        )
        fit = md.fit_vb()
    i = list(fit.model.exog_names).index("xz")
    return {"slope": float(fit.fe_mean[i]), "slope_sd": float(fit.fe_sd[i]), "n": len(d)}


def gap_margin_stats(df: pd.DataFrame) -> dict:
    d = df[["logit_gap", MARGIN_COL, "stl_id"]].dropna()
    if len(d) < 30:
        return {}
    pear = sps.pearsonr(d[MARGIN_COL], d["logit_gap"])
    spear = sps.spearmanr(d[MARGIN_COL], d["logit_gap"])
    # OLS slope on standardized margin with cluster-robust SE by mesh
    import statsmodels.formula.api as smf

    d = d.copy()
    d["xz"] = (d[MARGIN_COL] - d[MARGIN_COL].mean()) / (d[MARGIN_COL].std() + 1e-12)
    ols = smf.ols("logit_gap ~ xz", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["stl_id"]}
    )
    return {
        "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue),
        "ols_slope": float(ols.params["xz"]), "ols_slope_se": float(ols.bse["xz"]),
        "ols_p": float(ols.pvalues["xz"]), "n": len(d),
    }


def flip_stats(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=[MARGIN_COL, "flip"])
    stable = d.loc[d["flip"] == 0, MARGIN_COL].abs().to_numpy()
    flipped = d.loc[d["flip"] == 1, MARGIN_COL].abs().to_numpy()
    if len(stable) < 5 or len(flipped) < 5:
        return {}
    u = sps.mannwhitneyu(flipped, stable, alternative="less")
    rank_biserial = 1.0 - 2.0 * u.statistic / (len(stable) * len(flipped))
    return {
        "n_stable": len(stable), "n_flipped": len(flipped),
        "abs_margin_stable": float(np.mean(stable)),
        "abs_margin_flipped": float(np.mean(flipped)),
        "mw_p_flipped_smaller": float(u.pvalue),
        "rank_biserial": float(rank_biserial),
    }


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------


def run(models_filter: list[str] | None) -> pd.DataFrame:
    margins = discover_margins()
    gates = load_gates()
    gate_of = {
        (r.model, r.prompt_condition): (bool(r.gate_pass), bool(r.logit_gate_pass))
        for r in gates.itertuples()
    }
    models = sorted({m for m, _ in margins})
    if models_filter:
        models = [m for m in models if m in models_filter]

    rows: list[dict] = []
    for model in models:
        loci = [l for l in LOCI if (model, l) in margins]
        margin_df = {l: pd.read_csv(margins[(model, l)]) for l in loci}
        for cell in CELLS:
            gen_gate, logit_gate = gate_of.get((model, cell), (False, False))
            gen = load_generation_items(model, cell)
            logit = load_logit_items(model, cell)
            for locus in loci:
                base = dict(model=model, prompt_condition=cell, locus=locus,
                            gen_gate=gen_gate, logit_gate=logit_gate)
                mdf = margin_df[locus]

                if gen is not None:
                    j = gen.merge(mdf, on=["stl_id", "texture_set"], how="inner")
                    # 1a: only meaningful where the generated answer is valid
                    if gen_gate and len(j):
                        shape_m = j.loc[j["p_shape_gen"] == 1, MARGIN_COL].to_numpy()
                        tex_m = j.loc[j["p_shape_gen"] == 0, MARGIN_COL].to_numpy()
                        d = cohens_d(shape_m, tex_m)
                        j["chose_shape"] = (j["p_shape_gen"] >= 0.5).astype(int)
                        mm = mixed_logistic_slope(j, "chose_shape", MARGIN_COL)
                        rows.append({**base, "analysis": "1a_williams_split",
                                     "cohens_d": d,
                                     "n_shape_items": len(shape_m),
                                     "n_texture_items": len(tex_m), **mm})
                        # sensitivity: split under the other margin rungs
                        for col in ("margin_raw", "margin_centered",
                                    "margin_centered_train"):
                            rows.append({**base, "analysis": "1a_sensitivity",
                                         "rung": col,
                                         "cohens_d": cohens_d(
                                             j.loc[j.p_shape_gen == 1, col].to_numpy(),
                                             j.loc[j.p_shape_gen == 0, col].to_numpy())})
                        # 1c
                        fs = flip_stats(j)
                        if fs:
                            rows.append({**base, "analysis": "1c_swap_flips", **fs})

                if logit is not None:
                    j = logit.merge(mdf, on=["stl_id", "texture_set"], how="inner")
                    gs = gap_margin_stats(j)
                    if gs:
                        rows.append({**base, "analysis": "1b_logit_linkage", **gs})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

LOCUS_LABEL = {
    "proj_mean": "proj_mean",
    "vit_last_mean": "vit_last",
    "vit_penult_mean": "vit_penult",
    "vit_pooler": "vit_pooler",
    "proj_lm_mean": "proj_lm",
}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {FIG_DIR / name}.png")


def fig_1a(df: pd.DataFrame) -> None:
    d = df[df["analysis"] == "1a_williams_split"].dropna(subset=["slope"])
    if d.empty:
        return
    d = d.copy()
    d["label"] = d["model"] + " | " + d["prompt_condition"]
    labels = sorted(d["label"].unique())
    loci = [l for l in LOCI if l in set(d["locus"])]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) * len(loci) + 1.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(loci)))
    y = 0
    yticks, ylabels = [], []
    for label in labels:
        sub = d[d["label"] == label]
        for li, locus in enumerate(loci):
            r = sub[sub["locus"] == locus]
            if r.empty:
                y += 1
                continue
            r = r.iloc[0]
            ax.errorbar(r["slope"], y, xerr=1.96 * r["slope_sd"], fmt="o",
                        color=colors[li], ms=5, capsize=2,
                        label=LOCUS_LABEL[locus] if label == labels[0] else None)
            y += 1
        yticks.append(y - (len(loci) + 1) / 2)
        ylabels.append(label)
        y += 1
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("mixed-logistic slope: item choice ~ standardized whitened margin")
    ax.set_title("1a Williams split: margin tracks generated choice (gate-passing cells)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "linkage_1a_williams_split")


def fig_1b(df: pd.DataFrame) -> None:
    d = df[df["analysis"] == "1b_logit_linkage"].dropna(subset=["spearman_r"])
    if d.empty:
        return
    loci = [l for l in LOCI if l in set(d["locus"])]
    models = sorted(d["model"].unique())
    fig, axes = plt.subplots(1, len(loci), figsize=(3.2 * len(loci), 
                             0.3 * len(models) * len(CELLS) / 2 + 2), sharey=True)
    if len(loci) == 1:
        axes = [axes]
    for ax, locus in zip(axes, loci):
        sub = d[d["locus"] == locus]
        ylabels, yvals, xvals, passed = [], [], [], []
        y = 0
        for model in models:
            for cell in CELLS:
                r = sub[(sub["model"] == model) & (sub["prompt_condition"] == cell)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ylabels.append(f"{model} | {cell}")
                yvals.append(y)
                xvals.append(r["spearman_r"])
                passed.append(bool(r["logit_gate"]))
                y += 1
        color = ["tab:blue" if p else "lightgray" for p in passed]
        ax.barh(yvals, xvals, color=color, height=0.7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(LOCUS_LABEL[locus], fontsize=10)
        ax.set_xlabel("Spearman r (gap ~ margin)")
        if ax is axes[0]:
            ax.set_yticks(yvals)
            ax.set_yticklabels(ylabels, fontsize=6)
    fig.suptitle("1b item-level linkage: logit gap ~ whitened margin "
                 "(colored = logit gate pass)", y=1.0)
    fig.tight_layout()
    _save(fig, "linkage_1b_gap_vs_margin")


def fig_1c(df: pd.DataFrame) -> None:
    d = df[df["analysis"] == "1c_swap_flips"].dropna(subset=["rank_biserial"])
    if d.empty:
        return
    d = d.copy()
    d["label"] = d["model"] + " | " + d["prompt_condition"]
    loci = [l for l in LOCI if l in set(d["locus"])]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) / max(len(loci), 1) + 2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(loci)))
    labels = sorted(d["label"].unique())
    for li, locus in enumerate(loci):
        sub = d[d["locus"] == locus].set_index("label")
        xs = [sub.loc[l, "rank_biserial"] if l in sub.index else np.nan
              for l in labels]
        sig = [sub.loc[l, "mw_p_flipped_smaller"] < 0.05 if l in sub.index else False
               for l in labels]
        ys = np.arange(len(labels)) + li * 0.18
        ax.scatter(xs, ys, color=colors[li], s=[40 if s else 15 for s in sig],
                   label=LOCUS_LABEL[locus])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(labels)) + 0.27)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("rank-biserial (positive = flipped items have smaller |margin|)")
    ax.set_title("1c order-swap flips sit closer to the read-out boundary\n"
                 "(large dot = Mann-Whitney p < .05, one-sided)")
    ax.legend(fontsize=8)
    _save(fig, "linkage_1c_swap_flips")


def fig_scatter_example(df: pd.DataFrame) -> None:
    """Exemplar scatter: qwen3.5-4b noun_label, gap vs margin per locus."""
    margins = discover_margins()
    model, cell = "qwen3.5-4b", "noun_label"
    logit = load_logit_items(model, cell)
    if logit is None:
        return
    loci = [l for l in LOCI if (model, l) in margins]
    fig, axes = plt.subplots(1, len(loci), figsize=(3.4 * len(loci), 3.4),
                             sharey=True)
    if len(loci) == 1:
        axes = [axes]
    for ax, locus in zip(axes, loci):
        mdf = pd.read_csv(margins[(model, locus)])
        j = logit.merge(mdf, on=["stl_id", "texture_set"], how="inner").dropna(
            subset=[MARGIN_COL, "logit_gap"])
        ax.scatter(j[MARGIN_COL], j["logit_gap"], s=6, alpha=0.4)
        r = sps.spearmanr(j[MARGIN_COL], j["logit_gap"])
        ax.set_title(f"{LOCUS_LABEL[locus]}  rho={r.statistic:.2f}", fontsize=10)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("whitened margin")
    axes[0].set_ylabel("logit gap  log p(shape) - log p(texture)")
    fig.suptitle(f"{model} | {cell}: per-item linkage across loci")
    fig.tight_layout()
    _save(fig, "linkage_example_scatter")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", help="restrict to these models")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    df = run(args.models)
    if df.empty:
        print("no results; are the margin CSVs exported?")
        return 1
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(df)} rows)")

    if not args.no_figures:
        fig_1a(df)
        fig_1b(df)
        fig_1c(df)
        fig_scatter_example(df)

    # console digest: locus contrast on the two strongest models
    d1 = df[df["analysis"] == "1a_williams_split"]
    if not d1.empty:
        print("\n1a mixed slopes (choice ~ margin), gate-passing cells:")
        for (m, c), sub in d1.groupby(["model", "prompt_condition"]):
            parts = [
                f"{LOCUS_LABEL[r.locus]}={r.slope:.3f}±{1.96 * r.slope_sd:.3f}"
                for r in sub.itertuples() if np.isfinite(r.slope)
            ]
            print(f"  {m:14s} {c:24s} " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
