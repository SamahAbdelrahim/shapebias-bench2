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

# Numeric-only framings for fig6 / fig7 / fig7b facets (option 1).
FRAMINGS = (
    ("no_word_similarity", "similarity"),
    ("no_word_category", "category"),
    ("noun_label", "noun"),
)

SUMMARY = REPO / "results" / "data" / "full_grid_v1a_summary.csv"
SUMMARY_SMITH = REPO / "results" / "data" / "smith_ladder_summary.csv"
SUMMARY_CC = REPO / "results" / "data" / "cueconflict_cc_triads_summary.csv"
SUMMARY_DECOMP = REPO / "results" / "data" / "cueconflict_decomposition_triads_summary.csv"
BREAKDOWN = REPO / "results" / "data" / "full_grid_v1a_by_shape_texture.csv"
PRIDE = REPO / "results" / "data" / "full_grid_v1a_pride.csv"
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


def load_summary(path: Path = SUMMARY) -> list[dict]:
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
                "shape_ci_lo": float(r["shape_ci_lo"]),
                "shape_ci_hi": float(r["shape_ci_hi"]),
            })
    return rows


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
    ax.set_title(
        f"Full grid (v1A): generation-path tracking "
        f"(gate \u2265 {GATE:.2f} boxed; \u2020 = position lock; n = 1,140 \u00d7 2 orders)",
        fontsize=9.5, pad=36,
    )
    save(fig, "fig1_validity_gates")


def fig2_shape_bias(rows: list[dict], human_cells: dict[tuple[str, str], dict],
                    human_scalar: float | None) -> None:
    cells = cell_lookup(rows)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    used: set[str] = set()
    for j, (label, cond, short) in enumerate(CELL_ORDER):
        for m in MODEL_ORDER:
            r = cells.get((label, cond, m))
            if r is None:
                continue
            shp = r["shape"]
            x = j + (MODEL_ORDER.index(m) - 4) * 0.075
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
    for j, (label, cond, _short) in enumerate(CELL_ORDER):
        cell = human_cells.get(("grid", cond))
        if cell is None:
            continue
        _human_errorbar(ax, j + 0.42, cell)
        plotted_human = True
    if not plotted_human and human_scalar is not None:
        ax.axhline(human_scalar, color=HUMAN_COLOR, linewidth=1.1, linestyle=":")
        ax.text(len(CELL_ORDER) - 0.45, human_scalar + 0.02,
                f"adult humans, 2026 pilot ({human_scalar:.2f})",
                fontsize=7, color=HUMAN_COLOR, ha="right")

    ax.axvline(2.5, color="#ccc", linewidth=1)
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([f"{c[2]}\n{c[0]}" for c in CELL_ORDER], fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(shape match)")
    human_key = [
        plt.Line2D([0], [0], linestyle="", marker="D", color=HUMAN_COLOR, markersize=6,
                   markeredgecolor="#222", markeredgewidth=0.6, label="adult humans")
    ] if plotted_human else None
    model_legend(ax, list(MODEL_ORDER), extra_handles=human_key,
                 loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    ax.set_title(
        "Full grid shape bias (filled = gate PASS with 95% Wilson CI; faded = fail)",
        fontsize=10,
    )
    save(fig, "fig2_shape_bias")


def fig3_naming(rows: list[dict]) -> None:
    cells = cell_lookup(rows)
    panels = [
        ("numeric", "no_word_similarity", "noun_label", "numeric"),
        ("AB", "no_word_similarity_AB", "noun_label_AB", "AB"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, (label, base_c, noun_c, title) in zip(axes, panels):
        for y, m in enumerate(MODEL_ORDER):
            b = cells.get((label, base_c, m))
            n = cells.get((label, noun_c, m))
            if b is None or n is None:
                continue
            both = b["gate"] == "PASS" and n["gate"] == "PASS"
            alpha = 1.0 if both else 0.25
            ax.plot([b["shape"], n["shape"]], [y, y], color=MODEL_COLORS[m],
                    alpha=alpha, linewidth=1.5)
            ax.plot(b["shape"], y, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    markersize=6, alpha=alpha)
            ax.plot(n["shape"], y, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    markersize=6, alpha=alpha, markerfacecolor="white",
                    markeredgewidth=1.4)
        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER if ax is axes[0] else [], fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("P(shape match)")
        ax.set_title(f"{title}: similarity \u2192 noun+shiple", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle(
        "Naming effect on the full grid (solid = both cells pass gate; open = noun)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig3_naming_effect")


def fig4_label_format(rows: list[dict]) -> None:
    cells = cell_lookup(rows)
    framings = [
        ("no_word_similarity", "no_word_similarity_AB", "similarity"),
        ("no_word_category", "no_word_category_AB", "category"),
        ("noun_label", "noun_label_AB", "noun"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True)
    for ax, (num_c, ab_c, title) in zip(axes, framings):
        for y, m in enumerate(MODEL_ORDER):
            a = cells.get(("numeric", num_c, m))
            b = cells.get(("AB", ab_c, m))
            if a is None or b is None:
                continue
            both = a["gate"] == "PASS" and b["gate"] == "PASS"
            alpha = 1.0 if both else 0.25
            ax.plot([a["tracking"], b["tracking"]], [y, y], color=MODEL_COLORS[m],
                    alpha=alpha, linewidth=1.4)
            ax.plot(a["tracking"], y, "o", color=MODEL_COLORS[m], alpha=alpha, markersize=5)
            ax.plot(b["tracking"], y, "s", color=MODEL_COLORS[m], alpha=alpha, markersize=5)
        ax.axvline(GATE, color="#888", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("tracking")
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(MODEL_ORDER)))
    axes[0].set_yticklabels(MODEL_ORDER, fontsize=8)
    fig.suptitle(
        "Label format (1/2 vs A/B): tracking on the full grid "
        "(circles = numeric, squares = AB)",
        fontsize=10, y=1.03,
    )
    save(fig, "fig4_label_format")


def fig5_by_shape_texture(path: Path = BREAKDOWN) -> None:
    """Mean shape rate across gate-passing cells, by STL and by texture."""
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


def _embed_proj(payload: dict, model: str) -> tuple[float | None, float | None, float | None]:
    """proj_mean centered shape_rate and CI from an embedding_robust JSON payload."""
    models = payload.get("models") or payload
    block = models.get(model) if isinstance(models, dict) else None
    if not isinstance(block, dict):
        return (None, None, None)
    reps = block.get("reps") or block.get("representations") or {}
    pm = reps.get("proj_mean") or {}
    cen = pm.get("centered") or pm
    rate = cen.get("shape_rate")
    if rate is None:
        return (None, None, None)
    ci = cen.get("ci95") or []
    lo = float(ci[0]) if len(ci) >= 2 else None
    hi = float(ci[1]) if len(ci) >= 2 else None
    return (float(rate), lo, hi)


def _embed_shape(payload: dict, model: str) -> float | None:
    rate, _, _ = _embed_proj(payload, model)
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
    ax_a.set_ylabel("embedding shape rate (centered cosine)")
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
        ax.set_xlabel("embedding P(shape)")
        ax.set_title(f"B. {short}", fontsize=9.5)
        if ax is ax_bs[0]:
            ax.set_ylabel("behavioral P(shape)")
    model_legend(ax_bs[-1], models, loc="center left",
                 bbox_to_anchor=(1.05, 0.5), fontsize=6.5)
    fig.suptitle(
        "Vision tower vs generation (numeric): similarity / category / noun "
        "(faded points fail the tracking gate)",
        fontsize=10, y=0.98,
    )
    save(fig, "fig7_vision_vs_behavior")


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
        ax.set_xlabel("embedding P(shape)")
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
        "Vision tower vs behavior across stimulus sets "
        "(color = set, marker = model; faded points fail the gate)",
        fontsize=10, y=1.02,
    )
    save(fig, "fig7b_sets_emb_vs_behavior")


def fig6_pride(path: Path = PRIDE) -> None:
    if not path.is_file():
        print(f"  skip fig6_pride: pending {path} (run logit job + full_grid_pride.py)")
        return
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), sharey=True)
    xs = np.arange(len(MODEL_ORDER))
    width = 0.22
    metrics = [
        ("gen_shape", "generation", "#4C78A8"),
        ("swap_shape", "order-swap", "#F58518"),
        ("pride_shape", "PriDe", "#54A24B"),
    ]
    for ax, (cond, short) in zip(axes, FRAMINGS):
        focus = [r for r in rows if r.get("prompt_condition") == cond]
        by_model = {r["model"]: r for r in focus}
        for i, (key, lab, color) in enumerate(metrics):
            vals = []
            for m in MODEL_ORDER:
                r = by_model.get(m)
                vals.append(float(r[key]) if r and r.get(key) not in ("", None) else np.nan)
            ax.bar(xs + (i - 1) * width, vals, width=width,
                   label=lab if ax is axes[0] else None, color=color)
        ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_ORDER, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.set_title(short, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("P(shape)")
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Full grid (numeric): generation vs order-swap vs PriDe",
        fontsize=10, y=1.02,
    )
    save(fig, "fig6_position_bias_correction")


def write_tidy(rows: list[dict]) -> None:
    path = DATA_DIR / "full_grid_tidy_behavioral.csv"
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


def main() -> int:
    print("Full-grid figures")
    rows = load_summary()
    write_tidy(rows)
    human_cells = load_human_by_set()
    human_scalar = load_human_scalar()
    if human_cells:
        print(f"  human anchors: {len(human_cells)} set x condition cells")
    elif human_scalar is not None:
        print(f"  human anchor: 2026 pilot scalar {human_scalar:.3f} (no matched_v2 export yet)")
    fig1_validity(rows)
    fig2_shape_bias(rows, human_cells, human_scalar)
    fig3_naming(rows)
    fig4_label_format(rows)
    fig5_by_shape_texture()
    fig6_pride()
    fig7_vision_vs_behavior(rows)
    fig7b_sets_behavior(human_cells)
    fig7b_sets_emb_vs_behavior()
    print(f"Done. Figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
