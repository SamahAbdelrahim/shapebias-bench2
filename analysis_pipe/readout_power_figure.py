#!/usr/bin/env python3
"""Read-out power figures: what the encoder contains vs. what cosine can see.

Reads the probe JSON written by ``playgrounds/linear_probe.py`` and draws:

  figA  read-out power ladder. Shape-match rate on identical held-out triads
        under read-outs of increasing power, with the behavioural rate as a
        reference line. If the curve climbs from chance at cosine to well above
        chance at the learned metric, the encoder contains shape information
        that the published measure cannot see.
  figB  decodability. Mesh-identity and texture-identity probe accuracy against
        their chance levels and the pixel baseline, per model.

Run:  .venv/bin/python analysis_pipe/readout_power_figure.py \
          --probe-dir results/probe.results/session_readout_power
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FIG_DIR = REPO / "results" / "figures" / "readout_power"

MODEL_COLORS = {
    "smolvlm": "#E69F00", "internvl": "#56B4E9", "qwen3-vl-2b": "#009E73",
    "qwen3-vl-4b": "#F0E442", "qwen3-vl-8b": "#0072B2", "qwen3.5-0.8b": "#D55E00",
    "qwen3.5-4b": "#CC79A7", "qwen3.5-9b": "#999999", "qwen3.5-27b": "#000000",
}

# Gate-passing behavioural shape rates, noun-label numeric, from
# results/data/full_grid_v1a_summary.csv. Reference lines only.
BEHAVIOURAL = {
    "qwen3.5-4b": 0.913, "qwen3.5-9b": 0.914, "qwen3-vl-8b": 0.757,
    "qwen3-vl-4b": 0.823, "qwen3.5-27b": 0.895,
}

RUNG_LABELS = {
    "1_raw_cosine": "raw\ncosine",
    "2_centered_cosine": "centred\ncosine\n(published)",
    "2b_centered_train": "centred\n(train fit)",
    "3_zca_cosine": "ZCA\nwhitened\n(unsupervised)",
    "5_learned_metric": "learned\nmetric\n(held out)",
    "6_ceiling_leaky": "ceiling\n(leaky)",
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.png/.pdf")


def parse_name(stem: str) -> tuple[str, str]:
    """probe_qwen3.5-9b_proj_mean -> ('qwen3.5-9b', 'proj_mean')."""
    s = stem[len("probe_"):] if stem.startswith("probe_") else stem
    for rep in ("proj_mean", "vit_last_mean", "vit_penult_mean", "vit_pooler",
                "pixels32"):
        if s.endswith("_" + rep):
            return s[: -len(rep) - 1], rep
    return s, "unknown"


def load_probes(probe_dir: Path) -> list[dict]:
    out = []
    for p in sorted(probe_dir.glob("probe_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  skip unreadable {p.name}")
            continue
        model, rep = parse_name(p.stem)
        d["_model"], d["_readout"] = model, rep
        out.append(d)
    return out


def _ordered_rungs(probes: list[dict]) -> list[str]:
    seen: list[str] = []
    for d in probes:
        for k in (d.get("ladder", {}).get("rungs") or {}):
            if k not in seen:
                seen.append(k)
    # keep the pca sweep out of the headline panel; it is a robustness detail
    seen = [k for k in seen if not k.startswith("4_pca")]
    return sorted(seen)


def figA_ladder(probes: list[dict], readout: str = "proj_mean") -> None:
    sel = [d for d in probes if d["_readout"] == readout]
    if not sel:
        print(f"  no probes for readout={readout}")
        return
    rungs = _ordered_rungs(sel)
    if not rungs:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(rungs))
    for d in sorted(sel, key=lambda z: z["_model"]):
        r = d["ladder"]["rungs"]
        y = [r[k]["shape_rate"] if k in r else np.nan for k in rungs]
        lo = [r[k]["ci95"][0] if k in r else np.nan for k in rungs]
        hi = [r[k]["ci95"][1] if k in r else np.nan for k in rungs]
        color = MODEL_COLORS.get(d["_model"], "#444444")
        ax.errorbar(
            x, y,
            yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
            marker="o", ms=5, lw=1.6, capsize=3, color=color, label=d["_model"],
        )
        b = BEHAVIOURAL.get(d["_model"])
        if b is not None:
            ax.axhline(b, color=color, ls=":", lw=1.0, alpha=0.55)

    ax.axhline(0.5, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.text(len(rungs) - 0.45, 0.505, "chance", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([RUNG_LABELS.get(k, k) for k in rungs], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("shape-match rate")
    ax.set_xlabel("read-out power  →")
    ax.set_title(
        f"Same held-out triads, increasing read-out power ({readout})\n"
        "dotted lines: that model's behavioural shape rate",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, ncol=2, loc="best")
    save(fig, f"figA_readout_power_ladder_{readout}")


def figB_decodability(probes: list[dict], readout: str = "proj_mean") -> None:
    sel = [d for d in probes if d["_readout"] == readout]
    if not sel:
        return
    sel.sort(key=lambda z: z["_model"])
    models = [d["_model"] for d in sel]
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    mesh = [d["P1_mesh_identity"]["accuracy"] for d in sel]
    tex = [d["P2_texture_identity"]["accuracy"] for d in sel]
    m_err = np.array([
        [d["P1_mesh_identity"]["accuracy"] - d["P1_mesh_identity"]["ci95"][0]
         for d in sel],
        [d["P1_mesh_identity"]["ci95"][1] - d["P1_mesh_identity"]["accuracy"]
         for d in sel],
    ])
    t_err = np.array([
        [d["P2_texture_identity"]["accuracy"] - d["P2_texture_identity"]["ci95"][0]
         for d in sel],
        [d["P2_texture_identity"]["ci95"][1] - d["P2_texture_identity"]["accuracy"]
         for d in sel],
    ])

    ax.bar(x - w / 2, mesh, w, yerr=m_err, capsize=3, color="#0072B2",
           label="mesh identity (held-out textures)")
    ax.bar(x + w / 2, tex, w, yerr=t_err, capsize=3, color="#D55E00",
           label="texture identity (held-out meshes)")

    chance_m = sel[0]["P1_mesh_identity"]["chance"]
    chance_t = sel[0]["P2_texture_identity"]["chance"]
    ax.axhline(chance_m, color="#0072B2", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(chance_t, color="#D55E00", ls=":", lw=1.0, alpha=0.7)
    ax.text(-0.45, chance_m + 0.01, f"chance {chance_m:.3f}", fontsize=6.5,
            color="#0072B2")

    for i, d in enumerate(sel):
        pm = d.get("P3_pixel_mesh")
        if pm:
            ax.plot([x[i] - w / 2], [pm["accuracy"]], marker="_", ms=14,
                    color="black", lw=2)
    ax.plot([], [], marker="_", ms=10, color="black", lw=2, ls="none",
            label="pixel baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("probe accuracy")
    ax.set_title(f"Is identity linearly decodable from the encoder? ({readout})",
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, f"figB_decodability_{readout}")


def print_table(probes: list[dict]) -> None:
    print("\nmodel            readout          meshP1  texP2  cos(cent)  learned  mesh@1")
    for d in sorted(probes, key=lambda z: (z["_model"], z["_readout"])):
        r = d.get("ladder", {}).get("rungs") or {}
        cen = r.get("2_centered_cosine", {}).get("shape_rate", float("nan"))
        lrn = r.get("5_learned_metric", {}).get("shape_rate", float("nan"))
        ret = (d.get("retrieval") or {}).get("mesh_at1", float("nan"))
        print(
            f"{d['_model']:16} {d['_readout']:16} "
            f"{d['P1_mesh_identity']['accuracy']:6.3f} "
            f"{d['P2_texture_identity']['accuracy']:6.3f} "
            f"{cen:9.3f} {lrn:8.3f} {ret:7.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--probe-dir",
        type=Path,
        default=REPO / "results" / "probe.results" / "session_readout_power",
    )
    ap.add_argument("--readout", default="proj_mean")
    args = ap.parse_args()

    if not args.probe_dir.is_dir():
        print(f"No probe directory at {args.probe_dir}")
        print("Run scripts/run_grid_embedding_export.sbatch on FarmShare first.")
        return 1

    probes = load_probes(args.probe_dir)
    if not probes:
        print(f"No probe_*.json found in {args.probe_dir}")
        return 1
    print(f"Loaded {len(probes)} probe results from {args.probe_dir}")

    readouts = sorted({d["_readout"] for d in probes})
    for rep in readouts:
        if rep == "pixels32":
            continue
        figA_ladder(probes, rep)
        figB_decodability(probes, rep)
    print_table(probes)
    print(f"\nDone. Figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
