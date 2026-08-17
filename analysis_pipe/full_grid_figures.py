#!/usr/bin/env python3
"""Paper-quality figures for the full texture-grid evaluation (v1, mode A).

Reads the generation CSVs / summary produced by analysis_pipe/full_grid_summary.py
and, when present, PriDe and embedding JSON from the full-grid probe session.

Outputs:
  results/figures/full_grid/fig*.png|pdf
  results/data/full_grid_tidy_behavioral.csv

Fig 7 / 7b compare four stimulus sets (novel grid, Smith, cc_triads,
decomposition); the original Geirhos class-folder set is not included.

Run:  .venv/bin/python analysis_pipe/full_grid_figures.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GATE = 0.70
# Keep in sync with scripts/model_ladder.sh (family-grouped, ascending).
MODEL_ORDER = (
    "smolvlm-256m", "smolvlm",
    "internvl", "internvl-2b", "internvl-8b", "internvl-14b",
    "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b",
)
MODEL_COLORS = {
    "smolvlm-256m": "#F0C75E", "smolvlm": "#E69F00",
    "internvl": "#A6D8F0", "internvl-2b": "#56B4E9",
    "internvl-8b": "#2B8CBF", "internvl-14b": "#08519C",
    "qwen3-vl-2b": "#66C2A5", "qwen3-vl-4b": "#009E73", "qwen3-vl-8b": "#006D4C",
    "qwen3.5-0.8b": "#FDBB84", "qwen3.5-2b": "#D55E00",
    "qwen3.5-4b": "#CC79A7", "qwen3.5-9b": "#999999", "qwen3.5-27b": "#000000",
}
MODEL_MARKERS = {
    "smolvlm-256m": "h", "smolvlm": "o",
    "internvl": "s", "internvl-2b": "D", "internvl-8b": "d", "internvl-14b": "H",
    "qwen3-vl-2b": "^", "qwen3-vl-4b": "v", "qwen3-vl-8b": "<",
    "qwen3.5-0.8b": "P", "qwen3.5-2b": ">", "qwen3.5-4b": "X",
    "qwen3.5-9b": "p", "qwen3.5-27b": "*",
}

CELL_ORDER = [
    ("numeric", "no_word_similarity", "similarity"),
    ("numeric", "no_word_category", "category"),
    ("numeric", "noun_label", "noun"),
    ("AB", "no_word_similarity_AB", "similarity"),
    ("AB", "no_word_category_AB", "category"),
    ("AB", "noun_label_AB", "noun"),
]

# Framings for fig6 / fig7 / fig7b facets. Numeric is the default row; AB gets
# its own twin in fig6b because label format is not null on the logit path.
FRAMINGS = (
    ("no_word_similarity", "similarity"),
    ("no_word_category", "category"),
    ("noun_label", "noun"),
)
FRAMINGS_AB = (
    ("no_word_similarity_AB", "similarity"),
    ("no_word_category_AB", "category"),
    ("noun_label_AB", "noun"),
)

# Pooling layers in the embedding JSONs, shallow to deep. proj_mean is the layer
# fig7 / fig7b report, so every axis that shows an embedding rate names it: the
# rate moves across layers (see fig8), and an unlabelled "embedding shape rate"
# hides which read-out produced it.
EMB_LAYERS = ("vit_penult_mean", "vit_last_mean", "vit_pooler", "proj_mean")
EMB_LAYER = "proj_mean"
EMB_LAYER_COLORS = {
    "vit_penult_mean": "#A6CEE3", "vit_last_mean": "#1F78B4",
    "vit_pooler": "#B2DF8A", "proj_mean": "#33A02C",
}

SUMMARY = REPO / "results" / "data" / "full_grid_v1a_summary.csv"
SUMMARY_SMITH = REPO / "results" / "data" / "smith_ladder_summary.csv"
SUMMARY_CC = REPO / "results" / "data" / "cueconflict_cc_triads_summary.csv"
SUMMARY_DECOMP = REPO / "results" / "data" / "cueconflict_decomposition_triads_summary.csv"
BREAKDOWN = REPO / "results" / "data" / "full_grid_v1a_by_shape_texture.csv"
PRIDE = REPO / "results" / "data" / "full_grid_v1a_pride.csv"
LOGIT_VALIDITY = REPO / "results" / "data" / "full_grid_logit_validity.csv"
EMBED_GRID = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_grid.json"
EMBED_SMITH = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_smith_probe.json"
EMBED_SMITH_FALLBACK = REPO / "results" / "probe.results" / "session_2026-07-31_farmshare" / "embedding_smith_probe.json"
EMBED_CC_TRIADS = REPO / "results" / "probe.results" / "session_cueconflict_triads" / "embedding_cc_triads.json"
EMBED_DECOMP = REPO / "results" / "probe.results" / "session_cueconflict_triads" / "embedding_decomposition_triads.json"
FIG_DIR = REPO / "results" / "figures" / "full_grid"
DATA_DIR = REPO / "results" / "data"
HUMAN_SUMMARY = DATA_DIR / "human_friendly_summary.csv"
# Per stimulus set and framing, written by summarize_human_by_set_condition in
# analysis_pipe/src/human_analysis.R. The older HUMAN_SUMMARY above collapses to
# a single number from the March 2026 pilot and is only a fallback now.
HUMAN_BY_SET = DATA_DIR / "human_summary_by_set.csv"
HUMAN_COLOR = "#1e7d3c"
# Figure keys for the stimulus sets versus the names used in the human export.
HUMAN_SET_KEY = {"ours": "grid", "cc_triads": "cc_triads"}

# Four stimulus sets used in fig7 / fig7b (old Geirhos folder left out on purpose).
STIM_SETS = (
    ("ours", "novel grid", "#0072B2", EMBED_GRID, SUMMARY, None),
    ("smith", "Smith", "#009E73", EMBED_SMITH, SUMMARY_SMITH, EMBED_SMITH_FALLBACK),
    ("cc_triads", "cc_triads", "#D55E00", EMBED_CC_TRIADS, SUMMARY_CC, None),
    ("decomp", "decomposition", "#CC79A7", EMBED_DECOMP, SUMMARY_DECOMP, None),
)

# Per-set figure builds. fig1-fig6d and fig9/fig9b are set-local: each reads one
# stimulus set's summary, PriDe and logit-validity CSVs and writes into its own
# directory. fig7 / fig7b / fig7c / fig8 are cross-set by construction and fig10
# needs matched human items, so both groups stay on the grid only.
#   key: (display label, summary, pride, logit validity, breakdown, subdir,
#         human key in human_summary_by_set.csv)
FIGURE_SETS = {
    "grid": (
        "Full grid (v1A)", SUMMARY,
        DATA_DIR / "full_grid_v1a_pride.csv",
        DATA_DIR / "full_grid_logit_validity.csv",
        DATA_DIR / "full_grid_v1a_by_shape_texture.csv",
        "full_grid", "grid",
    ),
    "smith": (
        "Smith probe ladder", SUMMARY_SMITH,
        DATA_DIR / "smith_ladder_pride.csv",
        DATA_DIR / "smith_ladder_logit_validity.csv",
        None,
        "smith", None,
    ),
    "cc_triads": (
        "Cue-conflict cc_triads", SUMMARY_CC,
        DATA_DIR / "cueconflict_cc_triads_pride.csv",
        DATA_DIR / "cueconflict_cc_triads_logit_validity.csv",
        DATA_DIR / "cueconflict_cc_triads_by_shape_texture.csv",
        "cc_triads", "cc_triads",
    ),
    "decomposition": (
        "Cue-conflict decomposition", SUMMARY_DECOMP,
        DATA_DIR / "cueconflict_decomposition_triads_pride.csv",
        DATA_DIR / "cueconflict_decomposition_triads_logit_validity.csv",
        DATA_DIR / "cueconflict_decomposition_triads_by_shape_texture.csv",
        "decomposition", None,
    ),
}

# Rebound by use_stimulus_set(); the figure functions read these rather than
# taking a set argument, so the plotting code is shared verbatim across sets.
SET_KEY = "grid"
SET_LABEL = "Full grid (v1A)"
SET_HUMAN_KEY: str | None = "grid"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def wilson_ci(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.png/.pdf")


@contextlib.contextmanager
def use_stimulus_set(key: str):
    """Point the module-level paths and titles at one stimulus set.

    The figure functions read SUMMARY / PRIDE / LOGIT_VALIDITY / BREAKDOWN /
    FIG_DIR as globals. Rebinding them here lets every set reuse the same
    plotting code instead of each figure growing a per-set branch.
    """
    global SET_KEY, SET_LABEL, SET_HUMAN_KEY
    global SUMMARY, PRIDE, LOGIT_VALIDITY, BREAKDOWN, FIG_DIR
    label, summary, pride, validity, breakdown, subdir, human_key = FIGURE_SETS[key]
    saved = (SET_KEY, SET_LABEL, SET_HUMAN_KEY,
             SUMMARY, PRIDE, LOGIT_VALIDITY, BREAKDOWN, FIG_DIR)
    SET_KEY, SET_LABEL, SET_HUMAN_KEY = key, label, human_key
    SUMMARY, PRIDE, LOGIT_VALIDITY = summary, pride, validity
    # A set with no by-shape/texture export gets a path that cannot exist, so
    # fig5 takes its normal "missing input" skip rather than needing a guard.
    BREAKDOWN = breakdown if breakdown is not None else DATA_DIR / "__absent__.csv"
    FIG_DIR = REPO / "results" / "figures" / subdir
    try:
        yield
    finally:
        (SET_KEY, SET_LABEL, SET_HUMAN_KEY,
         SUMMARY, PRIDE, LOGIT_VALIDITY, BREAKDOWN, FIG_DIR) = saved


def load_summary(path: Path | None = None) -> list[dict]:
    path = SUMMARY if path is None else path
    if not path.is_file():
        raise FileNotFoundError(f"Run full_grid_summary.py first: missing {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = "AB" if r["prompt_condition"].endswith("_AB") else "numeric"
            rows.append({
                "model": r["model"],
                "prompt_condition": r["prompt_condition"],
                "word": r.get("word", ""),
                "label_set": label,
                "tracking": float(r["tracking"]),
                "shape": float(r["shape_rate"]),
                "pos_first": float(r["pos_first"]),
                "gate": "PASS" if r["gate_pass"].lower() in ("true", "1") else "fail",
                "n_decided": int(float(r["n_decided"])),
                "n_stimuli": int(float(r.get("n_stimuli") or 0)),
                "shape_ci_lo": float(r["shape_ci_lo"]),
                "shape_ci_hi": float(r["shape_ci_hi"]),
            })
    return rows


def _n_stimuli(rows: list[dict]) -> int:
    vals = [r["n_stimuli"] for r in rows if r.get("n_stimuli")]
    return max(vals) if vals else 0


def cell_lookup(rows: list[dict]) -> dict:
    return {(r["label_set"], r["prompt_condition"], r["model"]): r for r in rows}


def model_legend(ax, models, extra_handles=None, **kw):
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                   linestyle="", markersize=6, label=m)
        for m in models if m in MODEL_COLORS
    ]
    if extra_handles:
        handles.extend(extra_handles)
    ax.legend(handles=handles, **kw)


def fig1_validity(rows: list[dict]) -> None:
    cells = cell_lookup(rows)
    models = list(MODEL_ORDER)
    mat = np.full((len(models), len(CELL_ORDER)), np.nan)
    locked = np.zeros_like(mat, dtype=bool)
    for j, (label, cond, _) in enumerate(CELL_ORDER):
        for i, m in enumerate(models):
            r = cells.get((label, cond, m))
            if r is None:
                continue
            mat[i, j] = r["tracking"]
            p = r["pos_first"]
            locked[i, j] = p >= 0.9 or p <= 0.1

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(CELL_ORDER)):
            if np.isnan(mat[i, j]):
                continue
            v = mat[i, j]
            txt = f"{v:.2f}" + ("\u2020" if locked[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    fontweight="bold" if v >= GATE else "normal",
                    color="black" if 0.25 < v < 0.85 else "white")
            if v >= GATE:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=1.4))
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([c[2] for c in CELL_ORDER], fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.axvline(2.5, color="white", linewidth=3)
    for x, lab in ((1, "numeric (1/2)"), (4, "AB")):
        ax.text(x, -1.05, lab, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01).set_label("image tracking across orders")
    n_stim = _n_stimuli(rows)
    n_note = f"n = {n_stim:,} \u00d7 2 orders" if n_stim else "both orders"
    ax.set_title(
        f"{SET_LABEL}: generation-path tracking "
        f"(gate \u2265 {GATE:.2f} boxed; \u2020 = position lock; {n_note})",
        fontsize=9.5, pad=36,
    )
    save(fig, "fig1_validity_gates")


def fig2_shape_bias(rows: list[dict], human_cells: dict[tuple[str, str], dict],
                    human_scalar: float | None) -> None:
    """Numeric labels only; fig4 shows that A/B gives the same answer here."""
    cells = cell_lookup(rows)
    panel_cells = [c for c in CELL_ORDER if c[0] == "numeric"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    used: set[str] = set()
    for j, (label, cond, short) in enumerate(panel_cells):
        for m in MODEL_ORDER:
            r = cells.get((label, cond, m))
            if r is None:
                continue
            shp = r["shape"]
            x = j + (MODEL_ORDER.index(m) - 6.5) * 0.055
            kw = dict(color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], markersize=5.5)
            if r["gate"] == "PASS":
                lo, hi = r["shape_ci_lo"], r["shape_ci_hi"]
                ax.errorbar(x, shp, yerr=[[max(0.0, shp - lo)], [max(0.0, hi - shp)]],
                            linestyle="", elinewidth=1, capsize=2, **kw)
                used.add(m)
            else:
                ax.plot(x, shp, linestyle="", alpha=0.18, **kw)
    ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")

    # Humans ran the grid in the noun and no-word-category framings only, so the
    # anchor is placed in the cells it was measured in rather than drawn across
    # the whole panel. The old flat line implied a comparison for framings and
    # AB variants that no human ever saw.
    plotted_human = False
    for j, (label, cond, _short) in enumerate(panel_cells):
        cell = (human_cells.get((SET_HUMAN_KEY, cond))
                if SET_HUMAN_KEY else None)
        if cell is None:
            continue
        _human_errorbar(ax, j + 0.42, cell)
        plotted_human = True
    if not plotted_human and human_scalar is not None:
        ax.axhline(human_scalar, color=HUMAN_COLOR, linewidth=1.1, linestyle=":")
        ax.text(len(panel_cells) - 0.45, human_scalar + 0.02,
                f"adult humans, 2026 pilot ({human_scalar:.2f})",
                fontsize=7, color=HUMAN_COLOR, ha="right")

    ax.set_xticks(range(len(panel_cells)))
    ax.set_xticklabels([c[2] for c in panel_cells], fontsize=9)
    ax.set_xlabel("prompt framing")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(shape match)")
    human_key = [
        plt.Line2D([0], [0], linestyle="", marker="D", color=HUMAN_COLOR, markersize=6,
                   markeredgecolor="#222", markeredgewidth=0.6, label="adult humans")
    ] if plotted_human else None
    model_legend(ax, list(MODEL_ORDER), extra_handles=human_key,
                 loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    ax.set_title(
        f"{SET_LABEL} shape bias, numeric labels\n"
        "(filled = gate PASS with 95% Wilson CI; faded = fail)",
        fontsize=10,
    )
    save(fig, "fig2_shape_bias")


def fig3_naming(rows: list[dict]) -> None:
    """The naming null, as the double-gated pairs alone.

    Numeric labels only, since fig4 establishes that A/B gives the same answer on
    the generated response. Only pairs where both cells clear the gate are drawn:
    a dumbbell between two invalid numbers says nothing, and in the earlier
    version those faded pairs outnumbered the real ones eleven to one.
    """
    cells = cell_lookup(rows)
    contrasts = [("no_word_similarity", "similarity"),
                 ("no_word_category", "category")]
    pairs = []
    for base_c, base_lab in contrasts:
        for m in MODEL_ORDER:
            b = cells.get(("numeric", base_c, m))
            n = cells.get(("numeric", "noun_label", m))
            if b is None or n is None:
                continue
            if b["gate"] != "PASS" or n["gate"] != "PASS":
                continue
            pairs.append((m, base_lab, b, n, n["shape"] - b["shape"]))
    if not pairs:
        print("  skip fig3: no double-gated pairs")
        return
    pairs.sort(key=lambda p: p[4])

    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.46 * len(pairs) + 2.2)))
    for y, (m, base_lab, b, n, delta) in enumerate(pairs):
        ax.plot([b["shape"], n["shape"]], [y, y], color=MODEL_COLORS[m], linewidth=1.8)
        ax.plot(b["shape"], y, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                markersize=7)
        ax.plot(n["shape"], y, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                markersize=7, markerfacecolor="white", markeredgewidth=1.5)
        ax.text(1.015, y, f"{delta:+.3f}", fontsize=7.5, va="center")
    ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{m}  ({lab})" for m, lab, _b, _n, _d in pairs], fontsize=8)
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(len(pairs) - 0.4, -0.6)
    ax.set_xlabel("P(shape match)")
    handles = [
        plt.Line2D([0], [0], color="#555", marker="o", linestyle="", markersize=6,
                   label="no-word framing"),
        plt.Line2D([0], [0], color="#555", marker="o", linestyle="", markersize=6,
                   markerfacecolor="white", markeredgewidth=1.5, label="noun + shiple"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.set_title(
        f"{SET_LABEL}: naming effect, double-gated pairs only, ordered by effect size\n"
        "(numeric labels; adding the novel word does not move shape choosing)",
        fontsize=9.5,
    )
    # The pooled test and the detectable-shift bound were computed on the grid,
    # so they are only quoted there.
    pooled = ("Pooled over numeric and AB: +0.017, Wilcoxon p = 0.56 "
              "(stats_summary.csv).\nSmallest shift this design could detect: 0.046. "
              if SET_KEY == "grid" else "")
    fig.text(0.01, -0.05,
             f"mean Δ = {np.mean([p[4] for p in pairs]):+.3f} over these "
             f"{len(pairs)} numeric pairs. {pooled}"
             "Numbers at the right are the noun-minus-no-word difference.",
             fontsize=7.5, va="bottom")
    save(fig, "fig3_naming_effect")


def fig4_label_format(rows: list[dict]) -> None:
    """The screen that lets the other figures report one label format.

    Numeric and A/B labels are crossed with every framing, which doubles the
    points in every panel that carries them. This figure tests the factor once,
    on both measures, so fig2 and fig3 can report numeric alone. The one place
    label format does change the answer is the logit path, which is fig9.
    """
    cells = cell_lookup(rows)
    framings = [
        ("no_word_similarity", "no_word_similarity_AB", "similarity"),
        ("no_word_category", "no_word_category_AB", "category"),
        ("noun_label", "noun_label_AB", "noun"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    deltas: dict[str, list[float]] = {"tracking": [], "shape": []}
    n_agree = n_tot = 0

    for ax, (metric, xlabel) in zip(axes, (("tracking", "image tracking"),
                                          ("shape", "P(shape match)"))):
        for y, m in enumerate(MODEL_ORDER):
            for num_c, ab_c, _title in framings:
                a = cells.get(("numeric", num_c, m))
                b = cells.get(("AB", ab_c, m))
                if a is None or b is None:
                    continue
                both = a["gate"] == "PASS" and b["gate"] == "PASS"
                alpha = 1.0 if both else 0.22
                ax.plot([a[metric], b[metric]], [y, y], color=MODEL_COLORS[m],
                        alpha=alpha, linewidth=1.2)
                ax.plot(a[metric], y, "o", color=MODEL_COLORS[m], alpha=alpha,
                        markersize=4.5)
                ax.plot(b[metric], y, "s", color=MODEL_COLORS[m], alpha=alpha,
                        markersize=4.5, markerfacecolor="white",
                        markeredgecolor=MODEL_COLORS[m])
                if metric == "tracking":
                    deltas["tracking"].append(b[metric] - a[metric])
                    n_agree += int(a["gate"] == b["gate"])
                    n_tot += 1
                else:
                    deltas["shape"].append(b[metric] - a[metric])
        if metric == "tracking":
            ax.axvline(GATE, color="#B00", linewidth=0.9, linestyle=":")
        else:
            ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        key = "tracking" if metric == "tracking" else "shape"
        d = deltas[key]
        if d:
            ax.set_title(
                f"mean Δ (AB − numeric) = {np.mean(d):+.3f}, "
                f"max |Δ| = {max(abs(x) for x in d):.3f}", fontsize=9)
    axes[0].set_yticks(range(len(MODEL_ORDER)))
    axes[0].set_yticklabels(MODEL_ORDER, fontsize=8)
    axes[0].text(0.02, len(MODEL_ORDER) - 0.4,
                 f"gate decision agrees in {n_agree}/{n_tot} pairs",
                 fontsize=7.5, va="bottom")
    fmt_handles = [
        plt.Line2D([0], [0], color="#555", marker="o", linestyle="", markersize=5,
                   label="numeric (1/2)"),
        plt.Line2D([0], [0], color="#555", marker="s", linestyle="", markersize=5,
                   markerfacecolor="white", label="A/B"),
    ]
    axes[1].legend(handles=fmt_handles, loc="lower right", fontsize=7.5)
    fig.suptitle(
        f"{SET_LABEL}: label format screen, 1/2 against A/B on the generated answer, "
        "all three framings pooled.\nOn the grid the two formats agree, which is why "
        "fig2 and fig3 report numeric only. On the logit path they do not (fig9).",
        fontsize=9.5, y=1.06,
    )
    save(fig, "fig4_label_format")


def fig5_by_shape_texture(path: Path | None = None) -> None:
    """Mean shape rate across gate-passing cells, by STL and by texture."""
    path = BREAKDOWN if path is None else path
    if not path.is_file():
        print(f"  skip fig5: missing {path}")
        return
    by_shape: dict[str, list[float]] = {}
    by_tex: dict[str, list[float]] = {}
    # Restrict to models/cells that pass the gate in the summary.
    rows = load_summary()
    pass_keys = {
        (r["model"], r["prompt_condition"])
        for r in rows if r["gate"] == "PASS"
    }
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["prompt_condition"])
            if key not in pass_keys:
                continue
            rate = float(r["shape_rate"])
            if r["grouping"] == "stl_id":
                by_shape.setdefault(r["level"], []).append(rate)
            else:
                by_tex.setdefault(r["level"], []).append(rate)

    def _means(d):
        items = sorted(((k, float(np.mean(v))) for k, v in d.items()),
                       key=lambda x: -x[1])
        return items

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, items, title in (
        (axes[0], _means(by_shape), "by shape (STL id)"),
        (axes[1], _means(by_tex)[:20], "by texture (top 20)"),
    ):
        if not items:
            ax.set_title(f"{title} (no gate-pass cells)")
            continue
        ys = np.arange(len(items))
        ax.barh(ys, [v for _, v in items], color="#4C78A8", height=0.75)
        ax.set_yticks(ys)
        ax.set_yticklabels([k for k, _ in items], fontsize=7)
        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("mean P(shape) over gate-pass cells")
        ax.set_title(title, fontsize=9.5)
    fig.suptitle(
        "Full grid: shape preference varies across shapes and textures "
        "(gate-passing cells only)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig5_by_shape_texture")


def _embed_proj(payload: dict, model: str, layer: str = EMB_LAYER,
                metric: str = "centered") -> tuple[float | None, float | None, float | None]:
    """Centred shape_rate and CI for one pooling layer of an embedding JSON."""
    models = payload.get("models") or payload
    block = models.get(model) if isinstance(models, dict) else None
    if not isinstance(block, dict):
        return (None, None, None)
    reps = block.get("reps") or block.get("representations") or {}
    rep = reps.get(layer) or {}
    cen = rep.get(metric) or rep
    rate = cen.get("shape_rate")
    if rate is None:
        return (None, None, None)
    ci = cen.get("ci95") or []
    lo = float(ci[0]) if len(ci) >= 2 else None
    hi = float(ci[1]) if len(ci) >= 2 else None
    return (float(rate), lo, hi)


def _embed_shape(payload: dict, model: str, layer: str = EMB_LAYER) -> float | None:
    rate, _, _ = _embed_proj(payload, model, layer)
    return rate


def _load_embed_path(primary: Path, fallback: Path | None) -> dict | None:
    path = primary if primary.is_file() else fallback
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def _stim_embed_payloads() -> list[tuple[str, str, str, dict]]:
    """(key, label, color, payload) for each available stimulus set."""
    out = []
    for key, label, color, emb_path, _summary, fallback in STIM_SETS:
        payload = _load_embed_path(emb_path, fallback)
        if payload is None:
            print(f"  fig7/7b pending embed: {emb_path.name}")
            continue
        out.append((key, label, color, payload))
    return out


def _stim_behavior_lookups() -> dict[str, dict]:
    """key -> cell_lookup for no_word_similarity rows of that set."""
    lookups: dict[str, dict] = {}
    for key, _label, _color, _emb, summary_path, _fb in STIM_SETS:
        if not summary_path.is_file():
            print(f"  fig7/7b pending summary: {summary_path.name}")
            continue
        rows = load_summary(summary_path)
        lookups[key] = cell_lookup(rows)
    return lookups


def fig7_vision_vs_behavior(rows: list[dict]) -> None:
    """Top: vision-tower rates (prompt-free). Bottom: emb vs behavior for
    similarity / category / noun (numeric)."""
    embeds = _stim_embed_payloads()
    cells = cell_lookup(rows)
    models = list(MODEL_ORDER)
    xs = np.arange(len(models))

    fig = plt.figure(figsize=(11.4, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_bs = [fig.add_subplot(gs[1, i]) for i in range(3)]

    n_sets = max(len(embeds), 1)
    half = (n_sets - 1) / 2.0
    any_a = False
    for si, (_key, label, color, payload) in enumerate(embeds):
        off = (si - half) * 0.18
        vals, los, his, pos = [], [], [], []
        for i, m in enumerate(models):
            rate, lo, hi = _embed_proj(payload, m)
            if rate is None:
                continue
            vals.append(rate)
            los.append(0.0 if lo is None else max(0.0, rate - lo))
            his.append(0.0 if hi is None else max(0.0, hi - rate))
            pos.append(i + off)
        if not vals:
            continue
        any_a = True
        ax_a.errorbar(pos, vals, yerr=[los, his], linestyle="", marker="o",
                      color=color, markersize=4.5, elinewidth=0.9, capsize=1.5,
                      label=label)
    ax_a.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
    ax_a.set_ylabel(f"embedding shape rate\n(centred cosine, {EMB_LAYER})")
    ax_a.set_ylim(-0.03, 1.03)
    if any_a:
        ax_a.legend(fontsize=7.5, loc="upper left", ncols=4)
    else:
        ax_a.text(0.5, 0.5, "embeddings pending", ha="center", va="center",
                  transform=ax_a.transAxes, color="#888")
    ax_a.set_title("A. Vision tower across stimulus sets (same for all prompts)", fontsize=9.5)

    grid_payload = next((p for k, _l, _c, p in embeds if k == "ours"), None)
    for ax, (cond, short) in zip(ax_bs, FRAMINGS):
        for m in models:
            if grid_payload is None:
                break
            emb = _embed_shape(grid_payload, m)
            beh = cells.get(("numeric", cond, m))
            if emb is None or beh is None:
                continue
            alpha = 1.0 if beh["gate"] == "PASS" else 0.25
            ax.plot(emb, beh["shape"], marker=MODEL_MARKERS[m], linestyle="",
                    color=MODEL_COLORS[m], markersize=6.5, alpha=alpha)
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="#ccc", linewidth=0.6)
        ax.axvline(0.5, color="#ccc", linewidth=0.6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"embedding P(shape), {EMB_LAYER}")
        ax.set_title(f"B. {short}", fontsize=9.5)
        if ax is ax_bs[0]:
            ax.set_ylabel("behavioral P(shape)")
    model_legend(ax_bs[-1], models, loc="center left",
                 bbox_to_anchor=(1.05, 0.5), fontsize=6.5)
    fig.suptitle(
        f"Vision tower ({EMB_LAYER}, centred cosine) vs generation (numeric): "
        "similarity / category / noun (faded points fail the tracking gate)",
        fontsize=10, y=0.98,
    )
    save(fig, "fig7_vision_vs_behavior")


PROBE_DIR = REPO / "results" / "probe.results" / "session_readout_power"
# Defensible read-outs only: the leaky ceiling and the PCA robustness sweep are
# excluded, so the band is "anything from raw cosine to a held-out learned metric".
BAND_RUNGS = ("1_raw_cosine", "2_centered_cosine", "2b_centered_train",
              "3_zca_cosine", "5_learned_metric")


def _readout_band(model: str) -> tuple[float, float, float] | None:
    """(min, max, zca_mean) of ladder shape rates across rungs and loci."""
    vals, zca = [], []
    for p in PROBE_DIR.glob(f"probe_{model}_*.json"):
        try:
            rungs = json.loads(p.read_text())["ladder"]["rungs"]
        except (json.JSONDecodeError, KeyError):
            continue
        vals.extend(rungs[k]["shape_rate"] for k in BAND_RUNGS if k in rungs)
        if "3_zca_cosine" in rungs:
            zca.append(rungs["3_zca_cosine"]["shape_rate"])
    if not vals:
        return None
    return (min(vals), max(vals), float(np.mean(zca)) if zca else float("nan"))


def fig7c_readout_band(rows: list[dict]) -> None:
    """Fig 7 as a bound: embedding axis spans the read-out ladder, behavioral
    axis spans the gate-passing prompt cells, so neither side rests on one
    arbitrary read-out choice."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if r["gate"] == "PASS":
            by_model.setdefault(r["model"], []).append(r)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    plotted = []
    for m in MODEL_ORDER:
        band = _readout_band(m)
        cells = by_model.get(m)
        if band is None or not cells:
            continue
        e_lo, e_hi, e_zca = band
        b_rates = [c["shape"] for c in cells]
        b_lo, b_hi, b_mid = min(b_rates), max(b_rates), float(np.mean(b_rates))
        color = MODEL_COLORS.get(m, "#444")
        ax.plot([e_lo, e_hi], [b_mid, b_mid], color=color, lw=2.2, alpha=0.55,
                solid_capstyle="butt")
        ax.plot([e_zca, e_zca], [b_lo, b_hi], color=color, lw=2.2, alpha=0.55,
                solid_capstyle="butt")
        ax.plot(e_zca, b_mid, marker=MODEL_MARKERS.get(m, "o"), color=color,
                markersize=7, linestyle="")
        plotted.append(m)

    ax.plot([0, 1], [0, 1], color="#888", lw=0.8, ls="--")
    ax.axhline(0.5, color="#ccc", lw=0.6)
    ax.axvline(0.5, color="#ccc", lw=0.6)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("embedding shape rate: band across read-out ladder rungs\n"
                  "(raw cosine to held-out learned metric, all loci; "
                  "marker at ZCA)")
    ax.set_ylabel("behavioral P(shape): range across gate-passing prompt cells")
    ax.set_title(
        f"Embedding vs behavior as bounds, not points ({len(plotted)} models "
        "with a gate-passing cell)", fontsize=9.5)
    model_legend(ax, plotted, loc="upper left", fontsize=7)
    save(fig, "fig7c_readout_band")


def fig7b_sets_behavior(human_cells: dict[tuple[str, str], dict] | None = None) -> None:
    """1×3: behavioral P(shape) per model, four stimulus sets, one panel per framing.

    Humans take a final x slot wherever the matched_v2 sample covers that set and
    framing, so human and model points share an axis and the colour of the
    stimulus set they came from.
    """
    human_cells = human_cells or {}
    beh_lookups = _stim_behavior_lookups()
    models = list(MODEL_ORDER)
    set_meta = [(k, lab, col) for k, lab, col, _e, _s, _f in STIM_SETS]
    half = (len(set_meta) - 1) / 2.0

    show_human = bool(human_cells)
    tick_labels = models + (["adult\nhumans"] if show_human else [])
    xs = np.arange(len(tick_labels))
    human_x = len(models)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
    for ax, (cond, short) in zip(axes, FRAMINGS):
        any_plotted = False
        for si, (key, label, color) in enumerate(set_meta):
            off = (si - half) * 0.18
            cells = beh_lookups.get(key)
            if cells is not None:
                vals, los, his, pos = [], [], [], []
                for i, m in enumerate(models):
                    r = cells.get(("numeric", cond, m))
                    if r is None:
                        continue
                    vals.append(r["shape"])
                    los.append(max(0.0, r["shape"] - r["shape_ci_lo"]))
                    his.append(max(0.0, r["shape_ci_hi"] - r["shape"]))
                    pos.append(i + off)
                if vals:
                    any_plotted = True
                    ax.errorbar(pos, vals, yerr=[los, his], linestyle="", marker="o",
                                color=color, markersize=4.0, elinewidth=0.8, capsize=1.3,
                                label=label if ax is axes[0] else None, alpha=0.95)

            hcell = human_cells.get((HUMAN_SET_KEY.get(key, ""), cond))
            if hcell is not None:
                lo, hi = hcell.get("lo"), hcell.get("hi")
                yerr = None
                if lo is not None and hi is not None:
                    yerr = [[max(0.0, hcell["shape"] - lo)], [max(0.0, hi - hcell["shape"])]]
                ax.errorbar([human_x + off], [hcell["shape"]], yerr=yerr, linestyle="",
                            marker="D", color=color, markersize=5.5, elinewidth=1.1,
                            capsize=2, markeredgecolor="#222", markeredgewidth=0.6)

        if show_human:
            ax.axvline(len(models) - 0.5, color="#ccc", linewidth=1)
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(short, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("behavioral P(shape)")
            if any_plotted:
                if show_human:
                    ax.plot([], [], linestyle="", marker="D", color="#444",
                            markersize=5.5, label="humans")
                ax.legend(fontsize=7, loc="upper left", ncols=2)
    fig.suptitle(
        "Behavior across stimulus sets by numeric framing "
        "(novel grid / Smith / cc_triads / decomposition; 95% Wilson CI)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig7b_sets_behavior")


def fig7b_sets_emb_vs_behavior() -> None:
    """1×3: embedding vs behavior for the four sets, one panel per framing."""
    embeds = _stim_embed_payloads()
    beh_lookups = _stim_behavior_lookups()
    models = list(MODEL_ORDER)
    set_meta = [(k, lab, col) for k, lab, col, _e, _s, _f in STIM_SETS]
    emb_by_key = {k: p for k, _l, _c, p in embeds}

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
    plotted_keys: list[str] = []
    for ax, (cond, short) in zip(axes, FRAMINGS):
        for key, label, color in set_meta:
            payload = emb_by_key.get(key)
            cells = beh_lookups.get(key)
            if payload is None or cells is None:
                continue
            if key not in plotted_keys:
                plotted_keys.append(key)
            for m in models:
                emb = _embed_shape(payload, m)
                beh = cells.get(("numeric", cond, m))
                if emb is None or beh is None:
                    continue
                alpha = 1.0 if beh["gate"] == "PASS" else 0.22
                ax.plot(emb, beh["shape"], marker=MODEL_MARKERS[m], linestyle="",
                        color=color, markersize=6.0, alpha=alpha)
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="#ccc", linewidth=0.6)
        ax.axvline(0.5, color="#ccc", linewidth=0.6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"embedding P(shape), {EMB_LAYER}")
        ax.set_title(short, fontsize=10)
        ax.set_aspect("equal")
    axes[0].set_ylabel("behavioral P(shape)")
    if plotted_keys:
        set_handles = [
            plt.Line2D([0], [0], color=colr, marker="o", linestyle="",
                       markersize=5.5, label=lab)
            for k, lab, colr in set_meta if k in plotted_keys
        ]
        leg1 = axes[0].legend(handles=set_handles, loc="upper left",
                              fontsize=7, title="set", title_fontsize=7)
        axes[0].add_artist(leg1)
    # Colour encodes the stimulus set here, so the model legend shows shape only.
    shape_handles = [
        plt.Line2D([0], [0], color="#555", marker=MODEL_MARKERS[m], linestyle="",
                   markersize=5.5, label=m)
        for m in models if m in MODEL_MARKERS
    ]
    axes[-1].legend(handles=shape_handles, loc="center left",
                    bbox_to_anchor=(1.03, 0.5), fontsize=6.5)
    fig.suptitle(
        f"Vision tower ({EMB_LAYER}, centred cosine) vs behavior across stimulus sets "
        "(color = set, marker = model; faded points fail the gate)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig7b_sets_emb_vs_behavior")


def fig8_embedding_layers(payload: dict | None = None) -> None:
    """Which pooling layer the embedding read-out uses, and whether it matters.

    Panel A gives every model's centred-cosine shape rate at each layer; panel B
    gives the paired difference from proj_mean, the layer fig7 / fig7b report.
    Layers that return the same vectors as proj_mean are dropped from both panels
    and named in the caption, since plotting a vector against itself adds a series
    and no information.
    """
    if payload is None:
        payload = _load_embed_path(EMBED_GRID, None)
    if payload is None:
        print(f"  skip fig8: missing {EMBED_GRID.name}")
        return
    models = list(MODEL_ORDER)
    xs = np.arange(len(models))

    aliases = []
    for layer in EMB_LAYERS:
        if layer == EMB_LAYER:
            continue
        d = [a - b for a, b in
             ((_embed_shape(payload, m, layer), _embed_shape(payload, m, EMB_LAYER))
              for m in models)
             if a is not None and b is not None]
        if len(d) >= 3 and all(x == 0.0 for x in d):
            aliases.append(layer)
    layers = [L for L in EMB_LAYERS if L not in aliases]
    n_layers = len(layers)
    half = (n_layers - 1) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2),
                             gridspec_kw={"width_ratios": [1.75, 1.0]})
    ax_a, ax_b = axes

    coverage: dict[str, int] = {}
    for si, layer in enumerate(layers):
        off = (si - half) * 0.19
        vals, los, his, pos = [], [], [], []
        for i, m in enumerate(models):
            rate, lo, hi = _embed_proj(payload, m, layer)
            if rate is None:
                continue
            vals.append(rate)
            los.append(0.0 if lo is None else max(0.0, rate - lo))
            his.append(0.0 if hi is None else max(0.0, hi - rate))
            pos.append(i + off)
        coverage[layer] = len(vals)
        if not vals:
            continue
        ax_a.errorbar(pos, vals, yerr=[los, his], linestyle="", marker="o",
                      color=EMB_LAYER_COLORS[layer], markersize=4.5,
                      elinewidth=0.9, capsize=1.5,
                      label=f"{layer} (n={len(vals)})")
    ax_a.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
    ax_a.set_ylabel("embedding shape rate\n(centred cosine)")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.legend(fontsize=7, loc="upper left", ncols=2)
    ax_a.set_title("A. Shape rate by pooling layer (novel grid, prompt-free)", fontsize=9.5)

    # Panel B: paired difference from the reported layer, one point per model.
    other = [L for L in layers if L != EMB_LAYER]
    for si, layer in enumerate(other):
        deltas = []
        for m in models:
            a = _embed_shape(payload, m, layer)
            b = _embed_shape(payload, m, EMB_LAYER)
            if a is None or b is None:
                continue
            deltas.append(a - b)
        if not deltas:
            continue
        jitter = (np.random.default_rng(0).random(len(deltas)) - 0.5) * 0.16
        ax_b.plot(np.full(len(deltas), si) + jitter, deltas, linestyle="",
                  marker="o", color=EMB_LAYER_COLORS[layer], markersize=5,
                  alpha=0.75)
        mean_d = float(np.mean(deltas))
        ax_b.plot([si - 0.22, si + 0.22], [mean_d, mean_d], color="#222", linewidth=1.8)
        ax_b.text(si, 0.30, f"mean {mean_d:+.3f}\nmax |Δ| {max(abs(d) for d in deltas):.3f}",
                  ha="center", va="top", fontsize=7)
    ax_b.axhline(0.0, color="#888", linewidth=0.8, linestyle="--")
    ax_b.set_xticks(range(len(other)))
    ax_b.set_xticklabels(other, fontsize=7.5)
    ax_b.set_ylim(-0.35, 0.35)
    ax_b.set_ylabel(f"shape rate minus {EMB_LAYER}")
    ax_b.set_title(f"B. Paired difference from {EMB_LAYER}\n(one point per model)",
                   fontsize=9.5)

    missing = [f"{L} n={coverage.get(L, 0)} models" for L in layers
               if coverage.get(L, 0) < len(models)]
    notes = []
    if missing:
        notes.append("uneven coverage: " + ", ".join(missing))
    if aliases:
        notes.append(", ".join(aliases) + f" not shown: identical to {EMB_LAYER} "
                                          "in every model")
    n_stim = (payload.get("config") or {}).get("n_stimuli")
    n_txt = f", n = {n_stim} triads" if n_stim else ""
    fig.suptitle(
        f"Embedding read-out depends on the pooling layer; fig7 / fig7b report "
        f"{EMB_LAYER}{n_txt}\n" + ("; ".join(notes) if notes else ""),
        fontsize=9.5, y=1.05,
    )
    save(fig, "fig8_embedding_layers")


def _load_logit_validity() -> dict[tuple[str, str], dict]:
    if not LOGIT_VALIDITY.is_file():
        return {}
    out = {}
    with open(LOGIT_VALIDITY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["prompt_condition"])] = r
    return out


def _fig6_bars(
    rows: list[dict] | None,
    framings: tuple[tuple[str, str], ...],
    label_set: str,
    outfile: str,
    title_prefix: str,
) -> None:
    """Shared three-bar layout for fig6 (numeric) and fig6b (A/B)."""
    if not PRIDE.is_file():
        print(f"  skip {outfile}: pending {PRIDE} (run logit job + full_grid_pride.py)")
        return
    pride = {(r["model"], r["prompt_condition"]): r
             for r in csv.DictReader(open(PRIDE, newline="", encoding="utf-8"))}
    logit = _load_logit_validity()
    cells = cell_lookup(rows) if rows else {}
    conds = {c for c, _ in framings}
    agree = [
        abs(float(r["swap_shape_rate"]) - float(r["pride_shape_rate"]))
        for (m, c), r in pride.items()
        if c in conds and r.get("pride_shape_rate")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    xs = np.arange(len(MODEL_ORDER))
    width = 0.26
    for ax, (cond, short) in zip(axes, framings):
        series = {
            "generated answer": ("#4C78A8", []),
            "logit argmax": ("#B279A2", []),
            "logit, order-swap corrected": ("#F58518", []),
        }
        low_mass = []
        for m in MODEL_ORDER:
            gen = cells.get((label_set, cond, m))
            lg = logit.get((m, cond))
            series["generated answer"][1].append(gen["shape"] if gen else np.nan)
            series["logit argmax"][1].append(
                float(lg["logit_shape"]) if lg else np.nan)
            series["logit, order-swap corrected"][1].append(
                float(lg["swap_shape_rate"]) if lg else np.nan)
            low_mass.append(bool(lg) and float(lg["option_mass"]) < 0.5)

        for i, (lab, (color, vals)) in enumerate(series.items()):
            ax.bar(xs + (i - 1) * width, vals, width=width,
                   label=lab if ax is axes[0] else None, color=color)
        for i, flag in enumerate(low_mass):
            if flag:
                ax.text(xs[i] + width, 0.02, "\u00d7", ha="center",
                        va="bottom", fontsize=8, color="#B00")
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_ORDER, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.set_title(short, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("shape rate")
            ax.legend(loc="upper left", fontsize=7)
    note = (f"PriDe differs from order-swap by {np.mean(agree):.3f} on average "
            f"(max {max(agree):.3f}), both as decision rates; "
            "see fig9b for the correction" if agree else "")
    fig.suptitle(
        f"{title_prefix}: the generated answer against the option logits, "
        f"all as decision rates. \u00d7 = option mass < 0.5.\n{note}",
        fontsize=9.5, y=1.05,
    )
    save(fig, outfile)


def fig6_pride(path: Path | None = None, rows: list[dict] | None = None) -> None:
    """Generated answer, logit argmax, and order-swap: numeric labels.

    All three bars are decision rates. PriDe stays in fig9b; the A/B twin is
    fig6b; how far correcting moves each format is fig6c.
    """
    _fig6_bars(rows, FRAMINGS, "numeric",
               "fig6_position_bias_correction",
               f"{SET_LABEL} (numeric labels)")


def fig6b_pride_ab(rows: list[dict] | None = None) -> None:
    """Same three bars as fig6, A/B labels only.

    Label format is null on the generation path but not on the logit path, so
    the A/B cells get their own correction figure rather than being dropped.
    """
    _fig6_bars(rows, FRAMINGS_AB, "AB",
               "fig6b_position_bias_correction_ab",
               f"{SET_LABEL} (A/B labels)")


def fig6c_correction_by_label_format() -> None:
    """How far prior-bias correction moves numeric vs A/B cells.

    Panel A: mean |PriDe rate − raw logit argmax| by label format.
    Panel B: count of cells moved by more than 0.10. Both use decision rates.
    """
    if not PRIDE.is_file():
        print(f"  skip fig6c: pending {PRIDE}")
        return
    pride = list(csv.DictReader(open(PRIDE, newline="", encoding="utf-8")))
    by_fmt: dict[str, list[float]] = {"numeric": [], "AB": []}
    for r in pride:
        if not r.get("pride_shape_rate"):
            continue
        fmt = "AB" if r["prompt_condition"].endswith("_AB") else "numeric"
        by_fmt[fmt].append(
            abs(float(r["pride_shape_rate"]) - float(r["logit_argmax_shape"]))
        )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    colors = {"numeric": "#4C78A8", "AB": "#F58518"}
    xs = [0, 1]
    labels = ["numeric (1/2)", "A/B"]
    means = [float(np.mean(by_fmt[k])) if by_fmt[k] else float("nan")
             for k in ("numeric", "AB")]
    counts = [sum(1 for d in by_fmt[k] if d > 0.10) for k in ("numeric", "AB")]
    ns = [len(by_fmt[k]) for k in ("numeric", "AB")]

    ax_a.bar(xs, means, color=[colors["numeric"], colors["AB"]], width=0.55)
    for i, (mu, n) in enumerate(zip(means, ns)):
        ax_a.text(i, mu + 0.008, f"{mu:.3f}\nn={n}", ha="center", va="bottom",
                  fontsize=8)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, max(means) * 1.35 if means else 1)
    ax_a.set_ylabel("mean |PriDe − raw logit argmax|")
    ax_a.set_title("A. How far correcting moves a cell", fontsize=9.5)

    ax_b.bar(xs, counts, color=[colors["numeric"], colors["AB"]], width=0.55)
    for i, (c, n) in enumerate(zip(counts, ns)):
        ax_b.text(i, c + 0.3, f"{c}/{n}", ha="center", va="bottom", fontsize=8)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, max(counts) * 1.35 + 1 if counts else 10)
    ax_b.set_ylabel("cells with |Δ| > 0.10")
    ax_b.set_title("B. Cells moved by more than 0.10", fontsize=9.5)

    fig.suptitle(
        f"{SET_LABEL}: prior-bias correction by label format. "
        "Both panels use decision rates.",
        fontsize=9.5, y=1.04,
    )
    save(fig, "fig6c_correction_by_label_format")


def fig6d_generation_follows_logit_ab(rows: list[dict] | None = None) -> None:
    """After fig6b: does generation track raw logits or swap-corrected ones?

    Same A/B framings and model order as fig6b. Each model has two bars:
    |generation shape rate − raw logit argmax| and
    |generation shape rate − swap-corrected logit rate|.
    The shorter bar is the logit measure generation follows more closely.
    """
    logit = _load_logit_validity()
    if not logit:
        print(f"  skip fig6d: missing {LOGIT_VALIDITY.name}")
        return
    cells = cell_lookup(rows) if rows else {}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    xs = np.arange(len(MODEL_ORDER))
    width = 0.36
    all_raw, all_corr = [], []

    for ax, (cond, short) in zip(axes, FRAMINGS_AB):
        d_raw, d_corr = [], []
        for m in MODEL_ORDER:
            gen = cells.get(("AB", cond, m))
            lg = logit.get((m, cond))
            if gen is None or lg is None:
                d_raw.append(np.nan)
                d_corr.append(np.nan)
                continue
            g = gen["shape"]
            raw = abs(g - float(lg["logit_shape"]))
            corr = abs(g - float(lg["swap_shape_rate"]))
            d_raw.append(raw)
            d_corr.append(corr)
            all_raw.append(raw)
            all_corr.append(corr)

        ax.bar(xs - width / 2, d_raw, width=width, color="#B279A2",
               label="|gen − raw logit|" if ax is axes[0] else None)
        ax.bar(xs + width / 2, d_corr, width=width, color="#F58518",
               label="|gen − swap-corrected|" if ax is axes[0] else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_ORDER, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(short, fontsize=10)
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("|Δ| shape rate")
            ax.legend(loc="upper left", fontsize=7)

    n_closer = sum(1 for a, b in zip(all_corr, all_raw) if a < b)
    note = (
        f"mean |gen−raw| = {np.mean(all_raw):.3f}; "
        f"mean |gen−swap| = {np.mean(all_corr):.3f}; "
        f"closer to swap in {n_closer}/{len(all_raw)} cells"
    )
    fig.suptitle(
        f"{SET_LABEL} (A/B labels): which logit measure does generation follow?\n"
        f"Shorter bar = closer agreement. {note}.",
        fontsize=9.5, y=1.05,
    )
    save(fig, "fig6d_generation_follows_logit_ab")


def fig9_logit_vs_generation(rows: list[dict]) -> None:
    """Do the logits agree with the generated answer, and does the gate move?

    A: shape rate from generated text against shape rate from the logit argmax.
    B: the tracking gate computed on each path, with the quadrant counts.
    C: first-option rate on each path, which is where the AB cells separate.
    """
    logit = _load_logit_validity()
    if not logit:
        print(f"  skip fig9: missing {LOGIT_VALIDITY.name} (run logit_validity.py)")
        return
    cells = cell_lookup(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax_a, ax_b, ax_c = axes
    marks = {"numeric": "o", "AB": "s"}
    diffs: dict[str, list[float]] = {"numeric": [], "AB": []}
    quad = {"both": 0, "gen only": 0, "logit only": 0, "neither": 0}

    for (label, cond, _short) in CELL_ORDER:
        for m in MODEL_ORDER:
            gen = cells.get((label, cond, m))
            lg = logit.get((m, cond))
            if gen is None or lg is None:
                continue
            g_shape, l_shape = gen["shape"], float(lg["logit_shape"])
            g_trk, l_trk = gen["tracking"], float(lg["logit_tracking"])
            diffs[label].append(l_shape - g_shape)
            kw = dict(color=MODEL_COLORS[m], marker=marks[label], markersize=6,
                      linestyle="")
            ax_a.plot(g_shape, l_shape, alpha=0.9, **kw)
            ax_b.plot(g_trk, l_trk, alpha=0.9, **kw)
            ax_c.plot(gen["pos_first"], float(lg["logit_pos_first"]), alpha=0.9, **kw)

            g_pass, l_pass = g_trk >= GATE, l_trk >= GATE
            key = ("both" if g_pass and l_pass else "gen only" if g_pass
                   else "logit only" if l_pass else "neither")
            quad[key] += 1

    for ax, lab_x, lab_y, title in (
        (ax_a, "P(shape) from generated text", "P(shape) from logit argmax",
         "A. Same cell, two response measures"),
        (ax_b, "tracking, generation path", "tracking, logit path",
         "B. Does the validity gate move?"),
        (ax_c, "first-option rate, generation", "first-option rate, logits",
         "C. Position bias by response measure"),
    ):
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(lab_x)
        ax.set_ylabel(lab_y)
        ax.set_title(title, fontsize=9.5)
        ax.set_aspect("equal")

    txt = "\n".join(
        f"mean Δ {k}: {np.mean(v):+.3f}  (|Δ|>0.10 in {sum(1 for d in v if abs(d) > 0.10)}/{len(v)})"
        for k, v in diffs.items() if v
    )
    ax_a.text(0.03, 0.97, txt, transform=ax_a.transAxes, fontsize=7,
              va="top", ha="left")

    for ax in (ax_b,):
        ax.axhline(GATE, color="#B00", linewidth=0.9, linestyle=":")
        ax.axvline(GATE, color="#B00", linewidth=0.9, linestyle=":")
        ax.text(0.97, 0.97, f"both {quad['both']}\ngeneration only {quad['gen only']}\n"
                            f"logits only {quad['logit only']}\nneither {quad['neither']}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right")

    fmt_handles = [
        plt.Line2D([0], [0], color="#555", marker=marks[k], linestyle="",
                   markersize=6, label=f"{k} labels")
        for k in ("numeric", "AB")
    ]
    ax_a.legend(handles=fmt_handles, loc="lower right", fontsize=7)
    model_legend(ax_c, list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.04, 0.5), fontsize=6.5)
    fig.suptitle(
        f"{SET_LABEL}: generated answers and option logits are the same trials "
        "scored two ways; they part company under A/B labels",
        fontsize=10, y=1.02,
    )
    save(fig, "fig9_logit_vs_generation")


def fig9b_logit_vs_generation_pride(rows: list[dict]) -> None:
    """fig9 after genuine PriDe correction on the logit shape rate.

    Panel A replaces the raw logit argmax with PriDe's held-out, prior-corrected
    decision rate. Panel B shows how far that correction moved each cell.
    Panel C keeps the raw first-option rates, which is the bias PriDe estimates.
    Tracking is not redrawn: PriDe corrects single-order option probabilities
    and aggregates their decisions, so the original paired tracking gate is no
    longer defined.
    """
    logit = _load_logit_validity()
    if not logit or not PRIDE.is_file():
        print("  skip fig9b: run logit_validity.py and full_grid_pride.py first")
        return
    with open(PRIDE, newline="", encoding="utf-8") as f:
        pride = {(r["model"], r["prompt_condition"]): r for r in csv.DictReader(f)}
    cells = cell_lookup(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax_a, ax_b, ax_c = axes
    marks = {"numeric": "o", "AB": "s"}
    diffs_a: dict[str, list[float]] = {"numeric": [], "AB": []}
    diffs_b: dict[str, list[float]] = {"numeric": [], "AB": []}
    low_mass_n = 0

    for (label, cond, _short) in CELL_ORDER:
        for m in MODEL_ORDER:
            gen = cells.get((label, cond, m))
            lg = logit.get((m, cond))
            pr = pride.get((m, cond))
            if gen is None or lg is None or pr is None:
                continue
            g_shape = gen["shape"]
            l_raw = float(lg["logit_shape"])
            l_pride = float(pr["pride_shape_rate"])
            diffs_a[label].append(l_pride - g_shape)
            diffs_b[label].append(l_pride - l_raw)
            if float(lg["option_mass"]) < 0.5:
                low_mass_n += 1
            kw = dict(color=MODEL_COLORS[m], marker=marks[label], markersize=6,
                      linestyle="")
            ax_a.plot(g_shape, l_pride, alpha=0.9, **kw)
            ax_b.plot(l_raw, l_pride, alpha=0.9, **kw)
            ax_c.plot(gen["pos_first"], float(lg["logit_pos_first"]), alpha=0.9, **kw)

    for ax, lab_x, lab_y, title in (
        (ax_a, "P(shape) from generated text",
         "P(shape) from logits, PriDe-corrected",
         "A. Generation against PriDe-corrected logits"),
        (ax_b, "P(shape) from logit argmax (raw)",
         "P(shape) from logits, PriDe-corrected",
         "B. How far did the correction move each cell?"),
        (ax_c, "first-option rate, generation",
         "first-option rate, logits",
         "C. Position bias the correction removes (raw)"),
    ):
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(lab_x)
        ax.set_ylabel(lab_y)
        ax.set_title(title, fontsize=9.5)
        ax.set_aspect("equal")

    txt_a = "\n".join(
        f"mean Δ {k}: {np.mean(v):+.3f}  (|Δ|>0.10 in {sum(1 for d in v if abs(d) > 0.10)}/{len(v)})"
        for k, v in diffs_a.items() if v
    )
    ax_a.text(0.03, 0.97, txt_a, transform=ax_a.transAxes, fontsize=7,
              va="top", ha="left")

    txt_b = "\n".join(
        f"mean Δ {k}: {np.mean(v):+.3f}  (|Δ|>0.10 in {sum(1 for d in v if abs(d) > 0.10)}/{len(v)})"
        for k, v in diffs_b.items() if v
    )
    if low_mass_n:
        txt_b += f"\n× option mass < 0.5 in {low_mass_n} cells"
    ax_b.text(0.03, 0.97, txt_b, transform=ax_b.transAxes, fontsize=7,
              va="top", ha="left")

    fmt_handles = [
        plt.Line2D([0], [0], color="#555", marker=marks[k], linestyle="",
                   markersize=6, label=f"{k} labels")
        for k in ("numeric", "AB")
    ]
    ax_a.legend(handles=fmt_handles, loc="lower right", fontsize=7)
    model_legend(ax_c, list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.04, 0.5), fontsize=6.5)
    fig.suptitle(
        f"{SET_LABEL}: same layout as fig9, with a first-option prior estimated on "
        "a random 10% of stimuli and PriDe applied to the held-out 90%. "
        "Tracking is undefined after prior correction and aggregation.",
        fontsize=9.5, y=1.02,
    )
    save(fig, "fig9b_logit_vs_generation_pride")


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fig10_human_model_items(rows: list[dict]) -> None:
    """Item-level agreement: do models find the same triads hard that humans do?

    The aggregate rates are close (best models 0.80-0.94 against an adult 0.95),
    so the question is whether the agreement survives at the level of individual
    triads, which is what a claim about shared representation would need.
    """
    item_path = DATA_DIR / "full_grid_item_rates.csv"
    human_path = DATA_DIR / "human_item_means.csv"
    if not item_path.is_file() or not human_path.is_file():
        print("  skip fig10: run item_rates.py and the human export first")
        return

    human: dict[tuple[str, str], dict] = {}
    with open(human_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("stim_set_name") != "grid":
                continue
            human[(r["condition"], r["stim_id"])] = r

    model_items: dict[tuple[str, str], dict[str, float]] = {}
    with open(item_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["prompt_condition"])
            model_items.setdefault(key, {})[r["stim_id"]] = float(r["shape_rate"])

    cells = cell_lookup(rows)
    conds = [("no_word_category", "category (no novel word)"), ("noun_label", "noun + shiple")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    rng = np.random.default_rng(0)

    for ax, (cond, short) in zip(axes, conds):
        shared = sorted(sid for (c, sid) in human if c == cond)
        passers = [m for m in MODEL_ORDER
                   if (cells.get(("numeric", cond, m)) or {}).get("gate") == "PASS"]
        if not shared or not passers:
            ax.text(0.5, 0.5, "no gate-passing cell", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            continue

        hx = [float(human[(cond, sid)]["shape_prop"]) for sid in shared]
        stacked = []
        for m in passers:
            mi = model_items.get((m, cond), {})
            ys = [mi.get(sid) for sid in shared]
            if any(v is None for v in ys):
                continue
            stacked.append(ys)
        if not stacked:
            continue

        # Each model answers a triad twice, so a single model's item rate can
        # only be 0, 0.5 or 1. Pooling the gate-passing models is the finest
        # grain the design supports.
        mean_y = list(np.mean(np.asarray(stacked, float), axis=0))
        r_mean = _pearson(hx, mean_y)
        jx = np.asarray(hx) + (rng.random(len(hx)) - 0.5) * 0.015
        jy = np.asarray(mean_y) + (rng.random(len(mean_y)) - 0.5) * 0.015
        ax.plot(jx, jy, linestyle="", marker="o", markerfacecolor="none",
                markeredgecolor="#333", markersize=5.0, markeredgewidth=0.9,
                alpha=0.75, label="one triad")

        # Binned mean, since the human axis is bunched near ceiling.
        edges = np.array([0.0, 0.7, 0.8, 0.9, 0.95, 1.001])
        bx, by = [], []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = [(a, b) for a, b in zip(hx, mean_y) if lo_e <= a < hi_e]
            if len(sel) >= 5:
                bx.append(float(np.mean([s[0] for s in sel])))
                by.append(float(np.mean([s[1] for s in sel])))
        if len(bx) >= 2:
            ax.plot(bx, by, color="#D55E00", linewidth=1.8, marker="s",
                    markersize=5, label="binned mean")

        ax.text(0.03, 0.03,
                f"n = {len(mean_y)} triads, {len(stacked)} gate-passing models\n"
                f"r = {r_mean:+.3f}   human mean {np.mean(hx):.3f}, "
                f"model mean {np.mean(mean_y):.3f}\n"
                f"mean |human - model| = "
                f"{np.mean(np.abs(np.array(hx) - np.array(mean_y))):.3f}",
                transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left")

        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="#ddd", linewidth=0.6)
        ax.axvline(0.5, color="#ddd", linewidth=0.6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("human P(shape) for this triad")
        ax.set_title(short, fontsize=9.5)
        ax.set_aspect("equal")
    axes[0].set_ylabel("model P(shape) for this triad\n(mean of gate-passing models)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Same 114 triads, humans against gate-passing models. Both axes are noisy "
        "(2 model trials and 6-23 human trials per triad),\nso see the "
        "attenuation-corrected r in stats_summary.csv before reading the size of "
        "the correlation",
        fontsize=9.5, y=1.04,
    )
    save(fig, "fig10_human_model_items")


def write_tidy(rows: list[dict]) -> None:
    name = ("full_grid_tidy_behavioral.csv" if SET_KEY == "grid"
            else f"{SET_KEY}_tidy_behavioral.csv")
    path = DATA_DIR / name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["model", "label_set", "prompt_condition", "word", "tracking", "shape",
              "pos_first", "gate", "n_decided", "shape_ci_lo", "shape_ci_hi"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def load_human_scalar() -> float | None:
    """The single March 2026 pilot number, kept as a fallback anchor."""
    if not HUMAN_SUMMARY.is_file():
        return None
    with open(HUMAN_SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "shape_prop" in r and r["shape_prop"]:
                try:
                    return float(r["shape_prop"])
                except ValueError:
                    continue
    return None


def load_human_by_set() -> dict[tuple[str, str], dict]:
    """Human shape rates keyed by (stim_set_name, condition).

    Empty when the matched_v2 sample has not been exported yet, in which case
    the figures fall back to the scalar pilot anchor.
    """
    if not HUMAN_BY_SET.is_file():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with open(HUMAN_BY_SET, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                shape = float(r["shape_prop"])
            except (KeyError, TypeError, ValueError):
                continue

            def _opt(key: str) -> float | None:
                try:
                    return float(r[key])
                except (KeyError, TypeError, ValueError):
                    return None

            out[(r.get("stim_set_name", ""), r.get("condition", ""))] = {
                "shape": shape,
                "lo": _opt("shape_lo"),
                "hi": _opt("shape_hi"),
                "participants": _opt("participants"),
                "trials": _opt("trials"),
            }
    return out


def _human_errorbar(ax, x: float, cell: dict, **kwargs) -> None:
    lo, hi = cell.get("lo"), cell.get("hi")
    yerr = None
    if lo is not None and hi is not None:
        yerr = [[max(0.0, cell["shape"] - lo)], [max(0.0, hi - cell["shape"])]]
    ax.errorbar(x, cell["shape"], yerr=yerr, linestyle="", marker="D",
                color=HUMAN_COLOR, markersize=5.0, elinewidth=1.1, capsize=2,
                markeredgecolor="#222", markeredgewidth=0.6, zorder=5, **kwargs)


def build_set(key: str, human_cells: dict, human_scalar: float | None) -> None:
    """The figures that only need one stimulus set's own CSVs."""
    with use_stimulus_set(key):
        if not SUMMARY.is_file():
            print(f"skip {key}: missing {SUMMARY.name}")
            return
        print(f"\n{SET_LABEL} figures -> {FIG_DIR}")
        rows = load_summary()
        write_tidy(rows)
        fig1_validity(rows)
        fig2_shape_bias(rows, human_cells, human_scalar)
        fig3_naming(rows)
        fig4_label_format(rows)
        fig5_by_shape_texture()
        fig6_pride(rows=rows)
        fig6b_pride_ab(rows=rows)
        fig6c_correction_by_label_format()
        fig6d_generation_follows_logit_ab(rows=rows)
        fig9_logit_vs_generation(rows)
        fig9b_logit_vs_generation_pride(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="sets", action="append",
                    choices=sorted(FIGURE_SETS) + ["all"],
                    help="stimulus set to build (repeatable; default all)")
    args = ap.parse_args()
    keys = list(FIGURE_SETS) if not args.sets or "all" in args.sets else args.sets

    human_cells = load_human_by_set()
    human_scalar = load_human_scalar()
    if human_cells:
        print(f"  human anchors: {len(human_cells)} set x condition cells")
    elif human_scalar is not None:
        print(f"  human anchor: 2026 pilot scalar {human_scalar:.3f} (no matched_v2 export yet)")

    for key in keys:
        build_set(key, human_cells, human_scalar)

    if "grid" in keys:
        # Cross-set and human-item figures are defined on the grid build only.
        print("\nCross-set figures")
        rows = load_summary()
        fig7_vision_vs_behavior(rows)
        fig7c_readout_band(rows)
        fig7b_sets_behavior(human_cells)
        fig7b_sets_emb_vs_behavior()
        fig8_embedding_layers()
        fig10_human_model_items(rows)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
