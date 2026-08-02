#!/usr/bin/env python3
"""Paper-quality figures for the FarmShare playground results.

Covers the result types the Quarto/R pipeline (analysis.qmd) does not:
30-trial smoke ladders (generation picks), the AB label-fix rerun,
PriDe/swap-corrected logit scoring, and vision-tower embedding readouts.

Reads whatever sessions exist right now and skips pending cells, so it can be
rerun as-is when the 2026-07-30 Smith AB and finalize (PriDe) jobs land.

Outputs:
  results/figures/playground/fig*.png|pdf
  results/data/playground_tidy_behavioral.csv

Run:  python analysis_pipe/playground_figures.py
"""

from __future__ import annotations

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

from scripts.build_master_interpretation_report import (  # noqa: E402
    GATE,
    MODEL_ORDER,
    _embed_shape_rate,
    _merge_embed,
    collect_behavioral_rows,
)
from scripts.build_playground_results_html import (  # noqa: E402
    ARCHIVE_AB,
    PLAY,
    SESS_AB_FIXED,
    SESS_SMOKE,
    parse_smoke,
    summarize_pair,
)

PROBE = REPO / "results" / "probe.results" / "session_2026-07-10_farmshare"
FIG_DIR = REPO / "results" / "figures" / "playground"
DATA_DIR = REPO / "results" / "data"
HUMAN_SUMMARY = DATA_DIR / "human_friendly_summary.csv"
SUDO_WORDS = ("shiple", "adinefults", "clapher", "plailass", "procation")

# Okabe-Ito palette (colorblind-safe), one color per model in size order.
MODEL_COLORS = {
    "smolvlm": "#E69F00",
    "internvl": "#56B4E9",
    "qwen3-vl-2b": "#009E73",
    "qwen3-vl-4b": "#F0E442",
    "qwen3-vl-8b": "#0072B2",
    "qwen3.5-0.8b": "#D55E00",
    "qwen3.5-4b": "#CC79A7",
    "qwen3.5-9b": "#999999",
    "qwen3.5-27b": "#000000",
}
MODEL_MARKERS = {
    "smolvlm": "o", "internvl": "s", "qwen3-vl-2b": "^", "qwen3-vl-4b": "v",
    "qwen3-vl-8b": "D", "qwen3.5-0.8b": "P", "qwen3.5-4b": "X",
    "qwen3.5-9b": "p", "qwen3.5-27b": "*",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- helpers

def wilson_ci(k: float, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
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


def pair_rows(session: Path, condition: str) -> list[dict]:
    """summarize_pair rows for one condition in one session ([] if either log missing)."""
    sf = session / f"playground_smoke_30trials_shape_first_{condition}.txt"
    tf = session / f"playground_smoke_30trials_texture_first_{condition}.txt"
    if not sf.is_file() or not tf.is_file():
        return []
    sf_d, tf_d = parse_smoke(sf), parse_smoke(tf)
    return [
        summarize_pair(sf_d, tf_d, m)
        for m in sf_d["models"]
        if m in tf_d["models"]
    ]


def model_legend(ax, models: list[str], **kw) -> None:
    handles = [
        plt.Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                   linestyle="", markersize=6, label=m)
        for m in MODEL_ORDER if m in models
    ]
    ax.legend(handles=handles, **kw)


def human_shape_prop() -> float | None:
    if not HUMAN_SUMMARY.is_file():
        return None
    with HUMAN_SUMMARY.open(newline="") as fh:
        row = next(iter(csv.DictReader(fh)), None)
    return float(row["shape_prop"]) if row else None


# ---------------------------------------------------------------- data

def behavioral_df() -> list[dict]:
    """Current per-cell rows + archived (pre-fix) AB rows for comparison."""
    rows = [dict(r, era="current") for r in collect_behavioral_rows()]
    arch_sess = ARCHIVE_AB / "session_2026-07-17_farmshare"
    for condition in ("no_word_similarity_AB", "no_word_category_AB", "noun_label_AB_shiple"):
        for r in pair_rows(arch_sess, condition):
            cond = condition.replace("_shiple", "")
            rows.append({
                "model": r["model"], "condition": cond, "stim_set": "benchmark",
                "label_set": "AB", "n": r["gen"]["n"],
                "gen_tracking": r["gen"]["trk"], "gen_shape": r["gen"]["shp_avg"],
                "pos_first": r["gen"]["pos_a"],
                "gate": "PASS" if (r["gen"]["trk"] or 0) >= GATE else "fail",
                "era": "archived_mismatch",
            })
    return rows


def sudo_word_rows() -> list[dict]:
    """Per-word noun_label cells: numeric (07-17) and fixed AB (07-30)."""
    out: list[dict] = []
    for label, session, prefix in (
        ("numeric", SESS_SMOKE, "noun_label"),
        ("AB", SESS_AB_FIXED, "noun_label_AB"),
    ):
        for word in SUDO_WORDS:
            for r in pair_rows(session, f"{prefix}_{word}"):
                out.append({
                    "model": r["model"], "word": word, "label_set": label,
                    "gen_tracking": r["gen"]["trk"], "gen_shape": r["gen"]["shp_avg"],
                    "n": r["gen"]["n"],
                    "gate": "PASS" if (r["gen"]["trk"] or 0) >= GATE else "fail",
                })
    return out


def pride_rows() -> list[dict]:
    """PriDe CSVs: current 07-30 outputs when present, else archived numeric rows.

    Archived AB rows carried the label mismatch, so only numeric rows are kept
    from pre-fix files.
    """
    out: list[dict] = []
    current = sorted(PLAY.glob("*pride_debias_2026-07-30*.csv"))
    archived = [
        ARCHIVE_AB / "derived" / "prompt_pride_debias_2026-07-17.csv",
        ARCHIVE_AB / "derived" / "smith_prompt_pride_debias_2026-07-25.csv",
    ]
    for path in current + [p for p in archived if p.is_file()]:
        is_archived = ARCHIVE_AB in path.parents
        stim = "smith" if "smith" in path.name else "benchmark"
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if is_archived and row.get("label_set", "").lower() != "numeric":
                    continue
                row["stim_set"] = stim
                row["source_file"] = path.name
                out.append(row)
    return out


def _embed_payload_row(name: str, stim: str, models: dict[str, dict]) -> dict | None:
    rate = _embed_shape_rate(models, name)
    if rate is None:
        return None
    payload = models[name]
    if "error" in payload and "reps" not in payload:
        return None
    reps = payload.get("reps", {})
    rep = reps.get("proj_mean") or next(iter(reps.values()), None)
    if not rep:
        return None
    ci = rep.get("centered", {}).get("ci95", [None, None])
    return {
        "model": name, "stimuli": stim, "shape_rate": rate,
        "ci_lo": ci[0], "ci_hi": ci[1], "n": rep.get("n"),
    }


def embedding_rows() -> list[dict]:
    """Fig 7: novel/benchmark vs Geirhos cue-conflict (existing + large-model fill)."""
    novel = _merge_embed(
        PROBE / "embedding_robust.json",
        PROBE.parent / "session_2026-07-17_farmshare" / "embedding_robust_fill.json",
    )
    large = PROBE.parent / "session_2026-07-31_farmshare" / "embedding_robust_fill_large.json"
    if large.is_file():
        for name, payload in json.loads(large.read_text()).get("models", {}).items():
            novel[name] = payload
    cue = _merge_embed(
        PROBE / "embedding_cueconflict.json",
        PROBE.parent / "session_2026-07-17_farmshare" / "embedding_cueconflict_fill.json",
    )
    out = []
    for name in MODEL_ORDER:
        for stim, models in (("novel", novel), ("cue_conflict", cue)):
            row = _embed_payload_row(name, stim, models)
            if row:
                out.append(row)
    return out


def embedding_bench_smith_rows() -> list[dict]:
    """Fig 7b: novel/benchmark vs Smith probe embeddings for the full ladder."""
    novel = _merge_embed(
        PROBE / "embedding_robust.json",
        PROBE.parent / "session_2026-07-17_farmshare" / "embedding_robust_fill.json",
    )
    large = PROBE.parent / "session_2026-07-31_farmshare" / "embedding_robust_fill_large.json"
    if large.is_file():
        for name, payload in json.loads(large.read_text()).get("models", {}).items():
            novel[name] = payload
    smith_path = PROBE.parent / "session_2026-07-31_farmshare" / "embedding_smith_probe.json"
    smith: dict[str, dict] = {}
    if smith_path.is_file():
        smith = json.loads(smith_path.read_text()).get("models", {})
    out = []
    for name in MODEL_ORDER:
        for stim, models in (("benchmark", novel), ("smith", smith)):
            row = _embed_payload_row(name, stim, models)
            if row:
                out.append(row)
    return out


def write_tidy(behavioral: list[dict], sudo: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "playground_tidy_behavioral.csv"
    fields = ["model", "condition", "stim_set", "label_set", "word", "n",
              "gen_tracking", "gen_shape", "pos_first", "gate", "era"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in behavioral:
            w.writerow({k: r.get(k, "") for k in fields})
        for r in sudo:
            w.writerow({k: r.get(k, "") for k in fields}
                       | {"condition": "noun_label", "stim_set": "benchmark",
                          "era": "current"})
    print(f"  wrote {path}")


# ---------------------------------------------------------------- figures

CELL_ORDER = [
    ("benchmark", "numeric", "no_word_similarity", "similarity"),
    ("benchmark", "numeric", "no_word_category", "category"),
    ("benchmark", "numeric", "noun_label", "noun"),
    ("benchmark", "AB", "no_word_similarity_AB", "similarity"),
    ("benchmark", "AB", "no_word_category_AB", "category"),
    ("benchmark", "AB", "noun_label_AB", "noun"),
    ("smith", "numeric", "no_word_similarity", "similarity"),
    ("smith", "numeric", "no_word_category", "category"),
    ("smith", "numeric", "noun_label", "noun"),
    ("smith", "AB", "no_word_similarity_AB", "similarity"),
    ("smith", "AB", "no_word_category_AB", "category"),
    ("smith", "AB", "noun_label_AB", "noun"),
]


def cell_lookup(rows: list[dict], era: str = "current") -> dict:
    return {
        (r["stim_set"], r["label_set"], r["condition"], r["model"]): r
        for r in rows if r.get("era", "current") == era
    }


def fig1_validity(rows: list[dict]) -> None:
    """Image-tracking gate across every model x condition x label x stimulus cell."""
    cells = cell_lookup(rows)
    models = list(MODEL_ORDER)
    mat = np.full((len(models), len(CELL_ORDER)), np.nan)
    locked = np.zeros_like(mat, dtype=bool)
    for j, (stim, label, cond, _) in enumerate(CELL_ORDER):
        for i, m in enumerate(models):
            r = cells.get((stim, label, cond, m))
            if r is None or r["gen_tracking"] is None:
                continue
            mat[i, j] = float(r["gen_tracking"])
            p = r["pos_first"]
            locked[i, j] = p is not None and (float(p) >= 0.9 or float(p) <= 0.1)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(CELL_ORDER)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "pending", ha="center", va="center",
                        fontsize=6, color="#666")
                continue
            v = mat[i, j]
            txt = f"{v:.2f}" + ("\u2020" if locked[i, j] else "")
            weight = "bold" if v >= GATE else "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    fontweight=weight,
                    color="black" if 0.25 < v < 0.85 else "white")
            if v >= GATE:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=1.4))
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([c[3] for c in CELL_ORDER], fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    # group separators + headers
    for x in (2.5, 5.5, 8.5):
        ax.axvline(x, color="white", linewidth=3)
    for x, lab in ((1, "benchmark\nnumeric (1/2)"), (4, "benchmark\nAB"),
                   (7, "Smith\nnumeric (1/2)"), (10, "Smith\nAB")):
        ax.text(x, -1.1, lab, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold")
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("image tracking across orders")
    ax.set_title(
        f"Which cells are interpretable? Generation-path tracking "
        f"(gate \u2265 {GATE:.2f} boxed; \u2020 = position/label lock; n = 30 trials \u00d7 2 orders)",
        fontsize=9.5, pad=46,
    )
    save(fig, "fig1_validity_gates")


def fig2_shape_bias(rows: list[dict], human: float | None) -> None:
    """Shape choice for gate-passing cells, with Wilson CIs."""
    cells = cell_lookup(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, stim in zip(axes, ("benchmark", "smith")):
        cols = [c for c in CELL_ORDER if c[0] == stim]
        used_models: set[str] = set()
        for j, (_, label, cond, short) in enumerate(cols):
            for m in MODEL_ORDER:
                r = cells.get((stim, label, cond, m))
                if r is None or r["gen_shape"] is None:
                    continue
                shp = float(r["gen_shape"])
                n_tot = 2 * int(r["n"] or 0)
                lo, hi = wilson_ci(shp * n_tot, n_tot)
                x = j + (MODEL_ORDER.index(m) - 4) * 0.075
                passed = r["gate"] == "PASS"
                kw = dict(color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], markersize=5.5)
                if passed:
                    yerr = [[max(0.0, shp - lo)], [max(0.0, hi - shp)]]
                    ax.errorbar(x, shp, yerr=yerr, linestyle="",
                                elinewidth=1, capsize=2, **kw)
                    used_models.add(m)
                else:
                    ax.plot(x, shp, linestyle="", alpha=0.18, **kw)
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.text(len(cols) - 0.45, 0.505, "chance", fontsize=7, color="#888")
        if human is not None and stim == "benchmark":
            ax.axhline(human, color="#1e7d3c", linewidth=1.1, linestyle=":")
            ax.text(len(cols) - 0.45, human + 0.02, f"adult humans ({human:.2f})",
                    fontsize=7, color="#1e7d3c", ha="right")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{c[3]}\n{c[1]}" for c in cols], fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Novel-object benchmark" if stim == "benchmark"
                     else "Smith lab stimuli", fontsize=9.5)
    axes[0].set_ylabel("P(shape match) — generation")
    model_legend(axes[1], list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    fig.suptitle(
        "Shape bias where choices are interpretable "
        "(filled = tracking gate PASS with 95% Wilson CI; faded = gate fail)",
        fontsize=10, y=1.04,
    )
    save(fig, "fig2_shape_bias")


def fig3_naming_effect(rows: list[dict]) -> None:
    """Noun vs no-word similarity: does a count noun raise shape choice?"""
    cells = cell_lookup(rows)
    panels = [
        ("benchmark", "numeric", "no_word_similarity", "noun_label"),
        ("benchmark", "AB", "no_word_similarity_AB", "noun_label_AB"),
        ("smith", "numeric", "no_word_similarity", "noun_label"),
        ("smith", "AB", "no_word_similarity_AB", "noun_label_AB"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.4), sharey=True)
    for ax, (stim, label, base_c, noun_c) in zip(axes, panels):
        plotted = False
        for y, m in enumerate(MODEL_ORDER):
            b = cells.get((stim, label, base_c, m))
            n = cells.get((stim, label, noun_c, m))
            if b is None or n is None or b["gen_shape"] is None or n["gen_shape"] is None:
                continue
            both_pass = b["gate"] == "PASS" and n["gate"] == "PASS"
            alpha = 1.0 if both_pass else 0.22
            bs, ns = float(b["gen_shape"]), float(n["gen_shape"])
            ax.plot([bs, ns], [y, y], color="#999", linewidth=1, alpha=alpha, zorder=1)
            ax.plot(bs, y, "o", color="#56B4E9", markersize=5, alpha=alpha, zorder=2)
            ax.plot(ns, y, "o", color="#D55E00", markersize=5, alpha=alpha, zorder=2)
            plotted = True
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER, fontsize=7.5)
        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"{stim} \u00b7 {label}", fontsize=9)
        ax.set_xlabel("P(shape match)")
        if not plotted:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
    handles = [
        plt.Line2D([], [], color="#56B4E9", marker="o", linestyle="", label="no word (similarity)"),
        plt.Line2D([], [], color="#D55E00", marker="o", linestyle="", label="count noun"),
    ]
    axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(
        "Naming effect: adding a count noun vs similarity-only prompt "
        "(solid = both cells pass the tracking gate; faded = interpret with caution)",
        fontsize=10, y=1.03,
    )
    save(fig, "fig3_naming_effect")


def fig4_label_format(rows: list[dict]) -> None:
    """Numeric (1/2) vs alphabetic (A/B) answer format: fragility of the measure."""
    cells = cell_lookup(rows)
    pairs = [("no_word_similarity", "no_word_similarity_AB", "similarity"),
             ("no_word_category", "no_word_category_AB", "category"),
             ("noun_label", "noun_label_AB", "noun")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, metric, label in zip(axes, ("gen_tracking", "gen_shape"),
                                 ("image tracking", "P(shape match)")):
        used = set()
        for stim, stim_marker_fill in (("benchmark", True), ("smith", False)):
            for num_c, ab_c, _short in pairs:
                for m in MODEL_ORDER:
                    a = cells.get((stim, "numeric", num_c, m))
                    b = cells.get((stim, "AB", ab_c, m))
                    if not a or not b or a[metric] is None or b[metric] is None:
                        continue
                    ax.plot(
                        float(a[metric]), float(b[metric]),
                        marker=MODEL_MARKERS[m], linestyle="",
                        color=MODEL_COLORS[m],
                        markerfacecolor=MODEL_COLORS[m] if stim_marker_fill else "none",
                        markersize=6, alpha=0.9,
                    )
                    used.add(m)
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{label} — numeric (1/2)")
        ax.set_ylabel(f"{label} — alphabetic (A/B)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
    model_legend(axes[1], list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.05, 0.5), fontsize=7.5)
    axes[0].set_title("filled = benchmark, open = Smith", fontsize=8.5)
    fig.suptitle(
        "Answer-format sensitivity: the same task with 1/2 vs A/B response labels "
        "(points off the diagonal indicate format-dependent behavior)",
        fontsize=10, y=1.03,
    )
    save(fig, "fig4_label_format")


def fig5_mismatch_fix(rows: list[dict]) -> None:
    """Archived (mismatched image labels) vs fixed AB runs — methods figure."""
    cur = cell_lookup(rows, era="current")
    old = cell_lookup(rows, era="archived_mismatch")
    conds = [("no_word_similarity_AB", "similarity"),
             ("no_word_category_AB", "category"),
             ("noun_label_AB", "noun")]
    # one shared row list so both panels align with the y tick labels
    entries: list[tuple[str, dict, dict]] = []
    seps: list[float] = []
    for cond, short in conds:
        for m in MODEL_ORDER:
            a = old.get(("benchmark", "AB", cond, m))
            b = cur.get(("benchmark", "AB", cond, m))
            if a and b:
                entries.append((f"{m} \u00b7 {short}", a, b))
        seps.append(len(entries) - 0.5)
    seps = seps[:-1]
    if not entries:
        print("  fig5 skipped: no archive/fixed AB pairs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 0.24 * len(entries) + 1.0), sharey=True)
    for ax, metric, label in zip(axes, ("gen_tracking", "gen_shape"),
                                 ("image tracking", "P(shape match)")):
        for y, (_name, a, b) in enumerate(entries):
            if a[metric] is None or b[metric] is None:
                continue
            av, bv = float(a[metric]), float(b[metric])
            color = "#1e7d3c" if bv > av else ("#b3261e" if bv < av else "#888")
            ax.annotate("", xy=(bv, y), xytext=(av, y),
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))
            ax.plot(av, y, "o", color="#aaa", markersize=4)
        for sep in seps:
            ax.axhline(sep, color="#ddd", linewidth=0.7)
        ax.axvline(GATE if metric == "gen_tracking" else 0.5,
                   color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel(label)
    axes[0].set_yticks(range(len(entries)))
    axes[0].set_yticklabels([e[0] for e in entries], fontsize=7)
    axes[0].set_ylim(len(entries) - 0.5, -0.5)
    fig.subplots_adjust(top=0.95, bottom=0.1)
    fig.suptitle(
        "Effect of fixing the image-label mismatch in AB prompts (benchmark stimuli): "
        "grey dot = archived mismatched run, arrowhead = fixed rerun",
        fontsize=10, y=1.02,
    )
    save(fig, "fig5_ab_label_fix")


def fig6_position_bias(rows: list[dict]) -> None:
    """Raw per-order shape rates vs swap / PriDe-corrected logit estimates."""
    if not rows:
        print("  fig6 skipped: no PriDe CSVs found yet")
        return
    conds = sorted({(r["stim_set"], r["condition"], r["label_set"]) for r in rows})
    fig, axes = plt.subplots(
        1, len(conds), figsize=(2.9 * len(conds) + 1.5, 3.6), sharey=True, squeeze=False
    )
    for ax, key in zip(axes[0], conds):
        stim, cond, label = key
        y = 0
        ylabels = []
        for m in MODEL_ORDER:
            r = next(
                (x for x in rows
                 if (x["stim_set"], x["condition"], x["label_set"]) == key
                 and x["model"] == m),
                None,
            )
            if r is None:
                continue
            try:
                sf = float(r["gen_shape"]) if "gen_shape" in r else None
                swap = float(r["swap_shape_rate"])
                pride = 0.5 * (float(r["pride_sf_shape_rate"]) + float(r["pride_tf_shape_rate"]))
                full = float(r["fullperm_shape_rate"])
            except (KeyError, ValueError, TypeError):
                continue
            if sf is not None:
                ax.plot(sf, y, "o", color="#999", markersize=5, label="_")
            ax.plot(swap, y, "s", color="#0072B2", markersize=5)
            ax.plot(full, y, "^", color="#009E73", markersize=5)
            ax.plot(pride, y, "D", color="#D55E00", markersize=5)
            ylabels.append(m)
            y += 1
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7.5)
        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"{stim}\n{cond} \u00b7 {label}", fontsize=8.5)
        ax.set_xlabel("shape rate")
    handles = [
        plt.Line2D([], [], color="#999", marker="o", linestyle="", label="generation (raw)"),
        plt.Line2D([], [], color="#0072B2", marker="s", linestyle="", label="logit, swap-averaged"),
        plt.Line2D([], [], color="#009E73", marker="^", linestyle="", label="logit, full permutation"),
        plt.Line2D([], [], color="#D55E00", marker="D", linestyle="", label="logit, PriDe-corrected"),
    ]
    axes[0][-1].legend(handles=handles, loc="center left",
                       bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(
        "Position-bias correction: latent (logit) shape preference after swap / "
        "permutation / PriDe debiasing vs raw generation",
        fontsize=10, y=1.05,
    )
    save(fig, "fig6_position_bias_correction")


def fig7_vision_vs_behavior(embed: list[dict], rows: list[dict]) -> None:
    """Vision-tower embedding shape preference vs behavioral shape choice."""
    if not embed:
        print("  fig7 skipped: no embedding JSONs found")
        return
    cells = cell_lookup(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    # Panel A: embedding shape rate, novel vs cue-conflict
    models = [m for m in MODEL_ORDER if any(e["model"] == m for e in embed)]
    xs = np.arange(len(models))
    for off, stim, color in ((-0.17, "novel", "#0072B2"), (0.17, "cue_conflict", "#D55E00")):
        vals, los, his, pos = [], [], [], []
        for i, m in enumerate(models):
            e = next((x for x in embed if x["model"] == m and x["stimuli"] == stim), None)
            if e is None:
                continue
            vals.append(e["shape_rate"])
            los.append(e["shape_rate"] - (e["ci_lo"] or e["shape_rate"]))
            his.append((e["ci_hi"] or e["shape_rate"]) - e["shape_rate"])
            pos.append(i + off)
        ax1.errorbar(pos, vals, yerr=[los, his], linestyle="", marker="o",
                     color=color, markersize=5, elinewidth=1, capsize=2,
                     label="novel objects" if stim == "novel" else "Geirhos cue-conflict")
    ax1.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
    ax1.set_ylabel("embedding shape rate (centered cosine)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("A. Vision-tower representation", fontsize=9.5)

    # Panel B: embedding (novel) vs behavior (numeric similarity, benchmark)
    for e in embed:
        if e["stimuli"] != "novel":
            continue
        r = cells.get(("benchmark", "numeric", "no_word_similarity", e["model"]))
        if r is None or r["gen_shape"] is None:
            continue
        passed = r["gate"] == "PASS"
        ax2.plot(e["shape_rate"], float(r["gen_shape"]),
                 marker=MODEL_MARKERS[e["model"]], linestyle="",
                 color=MODEL_COLORS[e["model"]], markersize=7,
                 alpha=1.0 if passed else 0.25)
    ax2.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
    ax2.axhline(0.5, color="#ccc", linewidth=0.6)
    ax2.axvline(0.5, color="#ccc", linewidth=0.6)
    ax2.set_xlabel("embedding shape rate (novel)")
    ax2.set_ylabel("behavioral P(shape) — similarity, numeric")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("B. Representation vs behavior", fontsize=9.5)
    model_legend(ax2, models, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    fig.suptitle(
        "Is shape information in the vision tower, and does behavior use it? "
        "(faded points in B fail the tracking gate)",
        fontsize=10, y=1.04,
    )
    save(fig, "fig7_vision_vs_behavior")


def fig7b_benchmark_vs_smith(embed: list[dict], rows: list[dict]) -> None:
    """Vision-tower shape preference: novel benchmark vs Smith, all ladder models."""
    models = list(MODEL_ORDER)
    cells = cell_lookup(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))

    # Panel A: paired embedding rates, all models (pending marked)
    xs = np.arange(len(models))
    any_plotted = False
    for off, stim, color, label in (
        (-0.17, "benchmark", "#0072B2", "novel benchmark"),
        (0.17, "smith", "#009E73", "Smith probe"),
    ):
        vals, los, his, pos = [], [], [], []
        for i, m in enumerate(models):
            e = next((x for x in embed if x["model"] == m and x["stimuli"] == stim), None)
            if e is None:
                continue
            vals.append(e["shape_rate"])
            lo = e["ci_lo"] if e["ci_lo"] is not None else e["shape_rate"]
            hi = e["ci_hi"] if e["ci_hi"] is not None else e["shape_rate"]
            los.append(max(0.0, e["shape_rate"] - lo))
            his.append(max(0.0, hi - e["shape_rate"]))
            pos.append(i + off)
        if not vals:
            continue
        any_plotted = True
        ax1.errorbar(pos, vals, yerr=[los, his], linestyle="", marker="o",
                     color=color, markersize=5, elinewidth=1, capsize=2, label=label)
    for i, m in enumerate(models):
        has_b = any(e["model"] == m and e["stimuli"] == "benchmark" for e in embed)
        has_s = any(e["model"] == m and e["stimuli"] == "smith" for e in embed)
        if not has_b:
            ax1.text(i - 0.17, 0.5, "·", ha="center", va="center",
                     color="#aaa", fontsize=8)
        if not has_s:
            ax1.text(i + 0.17, 0.5, "pending", ha="center", va="center",
                     color="#888", fontsize=6, rotation=90)
    ax1.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
    ax1.set_ylabel("embedding shape rate (centered cosine)")
    ax1.set_ylim(-0.03, 1.03)
    if any_plotted:
        ax1.legend(fontsize=8, loc="upper left")
    else:
        ax1.text(0.5, 0.5, "pending — embedding job not finished",
                 ha="center", va="center", transform=ax1.transAxes, color="#888")
    ax1.set_title("A. Vision tower: benchmark vs Smith", fontsize=9.5)

    # Panel B: embedding vs behavior, one point per model x stimulus set
    plotted_models: set[str] = set()
    for stim, beh_stim, marker_fill in (
        ("benchmark", "benchmark", True),
        ("smith", "smith", False),
    ):
        for e in embed:
            if e["stimuli"] != stim:
                continue
            r = cells.get((beh_stim, "numeric", "no_word_similarity", e["model"]))
            if r is None or r["gen_shape"] is None:
                continue
            passed = r["gate"] == "PASS"
            ax2.plot(
                e["shape_rate"], float(r["gen_shape"]),
                marker=MODEL_MARKERS[e["model"]], linestyle="",
                color=MODEL_COLORS[e["model"]], markersize=7,
                markerfacecolor=MODEL_COLORS[e["model"]] if marker_fill else "none",
                alpha=1.0 if passed else 0.25,
            )
    ax2.plot([0, 1], [0, 1], color="#888", linewidth=0.8, linestyle="--")
    ax2.axhline(0.5, color="#ccc", linewidth=0.6)
    ax2.axvline(0.5, color="#ccc", linewidth=0.6)
    ax2.set_xlabel("embedding shape rate")
    ax2.set_ylabel("behavioral P(shape) — similarity, numeric")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("B. Representation vs behavior\n(filled = benchmark, open = Smith)",
                  fontsize=9)
    model_legend(ax2, list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    fig.suptitle(
        "Does the vision tower encode shape across stimulus sets? "
        "Benchmark novel objects vs Smith probe (faded points in B fail the tracking gate)",
        fontsize=10, y=1.05,
    )
    save(fig, "fig7b_benchmark_vs_smith")


def fig8_word_generality(sudo: list[dict]) -> None:
    """Shape choice across five pseudo-words: does the effect depend on the word?"""
    if not sudo:
        print("  fig8 skipped: no per-word logs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, label in zip(axes, ("numeric", "AB")):
        sub = [r for r in sudo if r["label_set"] == label]
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in sub)]
        for m in models:
            xs, ys, passes = [], [], []
            for i, w in enumerate(SUDO_WORDS):
                r = next((x for x in sub if x["model"] == m and x["word"] == w), None)
                if r is None or r["gen_shape"] is None:
                    continue
                xs.append(i)
                ys.append(float(r["gen_shape"]))
                passes.append(r["gate"] == "PASS")
            if not xs:
                continue
            any_pass = any(passes)
            ax.plot(xs, ys, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    markersize=5, linewidth=1,
                    alpha=1.0 if any_pass else 0.2)
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(SUDO_WORDS)))
        ax.set_xticklabels(SUDO_WORDS, fontsize=8)
        ax.set_title(f"{label} labels", fontsize=9.5)
        ax.set_ylim(-0.03, 1.03)
        if not sub:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
    axes[0].set_ylabel("P(shape match) — noun_label")
    used = sorted({r["model"] for r in sudo}, key=MODEL_ORDER.index)
    model_legend(axes[1], used, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    fig.suptitle(
        "Word generality: shape choice for five pseudo-words (benchmark stimuli; "
        "faded lines never pass the tracking gate)",
        fontsize=10, y=1.03,
    )
    save(fig, "fig8_word_generality")


# ---------------------------------------------------------------- main

def main() -> None:
    print("Collecting behavioral rows from smoke logs ...")
    behavioral = behavioral_df()
    sudo = sudo_word_rows()
    pride = pride_rows()
    embed = embedding_rows()
    embed_bs = embedding_bench_smith_rows()
    human = human_shape_prop()
    n_cur = sum(1 for r in behavioral if r["era"] == "current")
    print(f"  {n_cur} current cells, "
          f"{len(behavioral) - n_cur} archived-AB cells, "
          f"{len(sudo)} per-word cells, {len(pride)} PriDe rows, "
          f"{len(embed)} embedding rows (fig7), "
          f"{len(embed_bs)} bench/Smith embed rows (fig7b), human={human}")

    write_tidy(behavioral, sudo)
    fig1_validity(behavioral)
    fig2_shape_bias(behavioral, human)
    fig3_naming_effect(behavioral)
    fig4_label_format(behavioral)
    fig5_mismatch_fix(behavioral)
    fig6_position_bias(pride)
    fig7_vision_vs_behavior(embed, behavioral)
    fig7b_benchmark_vs_smith(embed_bs, behavioral)
    fig8_word_generality(sudo)
    print("Done.")


if __name__ == "__main__":
    main()
