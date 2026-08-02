#!/usr/bin/env python3
"""Paper-quality figures for the full texture-grid evaluation (v1, mode A).

Reads the generation CSVs / summary produced by analysis_pipe/full_grid_summary.py
and, when present, PriDe and embedding JSON from the full-grid probe session.

Outputs:
  results/figures/full_grid/fig*.png|pdf
  results/data/full_grid_tidy_behavioral.csv

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
MODEL_ORDER = (
    "smolvlm", "internvl", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
    "qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b",
)
MODEL_COLORS = {
    "smolvlm": "#E69F00", "internvl": "#56B4E9", "qwen3-vl-2b": "#009E73",
    "qwen3-vl-4b": "#F0E442", "qwen3-vl-8b": "#0072B2", "qwen3.5-0.8b": "#D55E00",
    "qwen3.5-4b": "#CC79A7", "qwen3.5-9b": "#999999", "qwen3.5-27b": "#000000",
}
MODEL_MARKERS = {
    "smolvlm": "o", "internvl": "s", "qwen3-vl-2b": "^", "qwen3-vl-4b": "v",
    "qwen3-vl-8b": "D", "qwen3.5-0.8b": "P", "qwen3.5-4b": "X",
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

SUMMARY = REPO / "results" / "data" / "full_grid_v1a_summary.csv"
BREAKDOWN = REPO / "results" / "data" / "full_grid_v1a_by_shape_texture.csv"
PRIDE = REPO / "results" / "data" / "full_grid_v1a_pride.csv"
EMBED_GRID = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_grid.json"
EMBED_SMITH = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_smith_probe.json"
EMBED_SMITH_FALLBACK = REPO / "results" / "probe.results" / "session_2026-07-31_farmshare" / "embedding_smith_probe.json"
EMBED_CUE = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_cueconflict.json"
EMBED_CUE_FALLBACK = REPO / "results" / "probe.results" / "session_2026-07-10_farmshare" / "embedding_cueconflict.json"
FIG_DIR = REPO / "results" / "figures" / "full_grid"
DATA_DIR = REPO / "results" / "data"
HUMAN_SUMMARY = DATA_DIR / "human_friendly_summary.csv"

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


def model_legend(ax, models, **kw):
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                   linestyle="", markersize=6, label=m)
        for m in models if m in MODEL_COLORS
    ]
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


def fig2_shape_bias(rows: list[dict], human: float | None) -> None:
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
    if human is not None:
        ax.axhline(human, color="#1e7d3c", linewidth=1.1, linestyle=":")
        ax.text(len(CELL_ORDER) - 0.45, human + 0.02, f"adult humans ({human:.2f})",
                fontsize=7, color="#1e7d3c", ha="right")
    ax.axvline(2.5, color="#ccc", linewidth=1)
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([f"{c[2]}\n{c[0]}" for c in CELL_ORDER], fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(shape match)")
    model_legend(ax, list(MODEL_ORDER), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
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


def _embed_shape(payload: dict, model: str) -> float | None:
    """Read proj_mean centered shape_rate from embedding_robust JSON."""
    models = payload.get("models") or payload
    block = models.get(model) if isinstance(models, dict) else None
    if not isinstance(block, dict):
        return None
    reps = block.get("reps") or block.get("representations") or {}
    pm = reps.get("proj_mean") or {}
    cen = pm.get("centered") or pm
    rate = cen.get("shape_rate")
    return float(rate) if rate is not None else None


def fig6_pride(path: Path = PRIDE) -> None:
    if not path.is_file():
        print(f"  skip fig6_pride: pending {path} (run logit job + full_grid_pride.py)")
        return
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    # Prefer no_word_similarity numeric for the main panel.
    focus = [r for r in rows if r.get("prompt_condition") == "no_word_similarity"]
    if not focus:
        focus = rows
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(MODEL_ORDER))
    width = 0.22
    metrics = [
        ("gen_shape", "generation", "#4C78A8"),
        ("swap_shape", "order-swap", "#F58518"),
        ("pride_shape", "PriDe", "#54A24B"),
    ]
    by_model = {r["model"]: r for r in focus}
    for i, (key, lab, color) in enumerate(metrics):
        vals = []
        for m in MODEL_ORDER:
            r = by_model.get(m)
            vals.append(float(r[key]) if r and r.get(key) not in ("", None) else np.nan)
        ax.bar(xs + (i - 1) * width, vals, width=width, label=lab, color=color)
    ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(shape)")
    ax.legend(loc="upper left")
    ax.set_title("Full grid: generation vs order-swap vs PriDe (no_word_similarity)", fontsize=10)
    save(fig, "fig6_position_bias_correction")


def fig7_vision_vs_behavior(rows: list[dict]) -> None:
    """Behavior (gate-pass similarity) vs vision-tower embedding shape rate."""
    embed_sources = []
    for label, path, fallback in (
        ("grid sample", EMBED_GRID, None),
        ("cue-conflict", EMBED_CUE, EMBED_CUE_FALLBACK),
        ("Smith", EMBED_SMITH, EMBED_SMITH_FALLBACK),
    ):
        use = path if path.is_file() else fallback
        if use is not None and use.is_file():
            embed_sources.append((label, json.loads(use.read_text())))
        else:
            print(f"  fig7 pending embed: {path.name}")

    cells = cell_lookup(rows)
    fig, axes = plt.subplots(1, max(len(embed_sources), 1), figsize=(3.4 * max(len(embed_sources), 1), 3.6),
                             squeeze=False)
    axes = axes[0]
    if not embed_sources:
        axes[0].text(0.5, 0.5, "embeddings pending\n(submit embedding sbatch)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
        save(fig, "fig7_vision_vs_behavior")
        return

    for ax, (label, payload) in zip(axes, embed_sources):
        for m in MODEL_ORDER:
            beh = cells.get(("numeric", "no_word_similarity", m))
            emb = _embed_shape(payload, m)
            if beh is None or emb is None:
                continue
            alpha = 1.0 if beh["gate"] == "PASS" else 0.3
            ax.scatter(emb, beh["shape"], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       s=45, alpha=alpha, label=m)
        ax.plot([0, 1], [0, 1], color="#ccc", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="#eee", linewidth=0.6)
        ax.axvline(0.5, color="#eee", linewidth=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("embedding P(shape)")
        ax.set_ylabel("behavior P(shape)" if ax is axes[0] else "")
        ax.set_title(label, fontsize=9.5)
        ax.set_aspect("equal")
    model_legend(axes[-1], list(MODEL_ORDER), loc="center left",
                 bbox_to_anchor=(1.05, 0.5), fontsize=7)
    fig.suptitle(
        "Vision-tower vs generation shape preference "
        "(filled = gate PASS on full-grid similarity)",
        fontsize=10, y=1.04,
    )
    save(fig, "fig7_vision_vs_behavior")


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


def load_human() -> float | None:
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


def main() -> int:
    print("Full-grid figures")
    rows = load_summary()
    write_tidy(rows)
    human = load_human()
    fig1_validity(rows)
    fig2_shape_bias(rows, human)
    fig3_naming(rows)
    fig4_label_format(rows)
    fig5_by_shape_texture()
    fig6_pride()
    fig7_vision_vs_behavior(rows)
    print(f"Done. Figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
