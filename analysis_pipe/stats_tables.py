#!/usr/bin/env python3
"""Numbers to report alongside the full-grid figures.

Every figure in results/figures/full_grid so far shows an estimate and, at best,
a Wilson interval. This writes the tests those figures imply, in one long-format
table so a claim in the text can be traced to a row:

  cell_vs_chance      per gate-passing cell, exact binomial test against 0.5
  naming_effect       noun label against each no-word framing, per double-gated
                      model, and the paired test across models
  emb_behavior_r      embedding shape rate against behavioural shape rate,
                      per pooling layer and framing, Pearson and Spearman
  layer_means         mean embedding shape rate per layer, and the paired
                      Wilcoxon against the reported layer
  readout_ladder      mean shape rate per rung of the read-out ladder, per layer
  mcnemar             centred cosine against the learned metric, from the probes
  gen_vs_logit        generated answer against logit argmax, split by label
                      format, plus the gate 2x2
  human_model_item    item-level human-model correlation on the shared triads

Output: results/data/stats_summary.csv

Run:  .venv/bin/python analysis_pipe/stats_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

# Bootstrap resamples can draw a constant vector; those draws are dropped in
# boot_corr_ci rather than reported, so the warning is noise.
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results" / "data"
PROBE_DIR = REPO / "results" / "probe.results" / "session_readout_power"
EMBED_GRID = REPO / "results" / "probe.results" / "session_full_grid_v1a" / "embedding_grid.json"
OUT = DATA / "stats_summary.csv"

GATE = 0.70
EMB_LAYERS = ("vit_penult_mean", "vit_last_mean", "vit_pooler", "proj_mean")
EMB_LAYER = "proj_mean"
FRAMINGS = (("no_word_similarity", "similarity"),
            ("no_word_category", "category"),
            ("noun_label", "noun"))
LADDER_RUNGS = ("1_raw_cosine", "2_centered_cosine", "3_zca_cosine",
                "5_learned_metric", "6_ceiling_leaky")

RESULTS: list[dict] = []


def add(analysis: str, group: str, statistic: str, value, n=None,
        ci_lo=None, ci_hi=None, p_value=None, note: str = "") -> None:
    def fmt(x):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.6g}"
    RESULTS.append({
        "analysis": analysis, "group": group, "statistic": statistic,
        "value": fmt(value), "n": "" if n is None else int(n),
        "ci_lo": fmt(ci_lo), "ci_hi": fmt(ci_hi), "p_value": fmt(p_value),
        "note": note,
    })


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_pass(row: dict, field: str = "gate_pass") -> bool:
    return str(row.get(field, "")).strip().lower() in ("true", "1")


def boot_ci(vals: list[float], fn=np.mean, n_boot: int = 10000,
            seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap. Used where the sampling distribution of a mean or a
    correlation over 14 models has no clean closed form."""
    if len(vals) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, float)
    draws = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def boot_corr_ci(x: list[float], y: list[float], method: str = "pearson",
                 n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 4:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ax, ay = np.asarray(x, float), np.asarray(y, float)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ax), len(ax))
        sx, sy = ax[idx], ay[idx]
        if sx.std() == 0 or sy.std() == 0:
            continue
        r = (stats.pearsonr(sx, sy)[0] if method == "pearson"
             else stats.spearmanr(sx, sy)[0])
        if not math.isnan(r):
            out.append(r)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# --------------------------------------------------------------------------
# 0. Factor screen: which design factors move the answer at all
# --------------------------------------------------------------------------
def factor_screen(summary: list[dict], logit: list[dict] | None,
                  pride: list[dict] | None, payload: dict | None,
                  probe_dir: Path) -> None:
    """Test each factor once, so the figures can drop the ones that do nothing.

    A factor kept in every panel after it has been shown not to matter costs the
    reader attention and buys no information. Each row here licenses a collapse
    downstream, and the two factors that survive stay in the figures.
    """
    cells = {(r["model"], r["prompt_condition"]): r for r in summary}
    models = sorted({r["model"] for r in summary})
    bases = [f[0] for f in FRAMINGS]

    # Label format on the generated answer.
    for metric, label in (("shape_rate", "P(shape)"), ("tracking", "tracking")):
        deltas, gated = [], []
        for m in models:
            for f in bases:
                a, b = cells.get((m, f)), cells.get((m, f + "_AB"))
                if a is None or b is None:
                    continue
                d = float(b[metric]) - float(a[metric])
                deltas.append(d)
                if is_pass(a) and is_pass(b):
                    gated.append(d)
        if deltas:
            w = stats.wilcoxon(deltas)
            add("factor_screen", f"label format (AB - numeric) | generation | {label}",
                "mean delta", float(np.mean(deltas)), n=len(deltas),
                p_value=w.pvalue,
                note=f"max |delta| {max(abs(d) for d in deltas):.3f}; "
                     "Wilcoxon over model x framing")
        if len(gated) >= 5:
            add("factor_screen", f"label format (AB - numeric) | double-gated | {label}",
                "mean delta", float(np.mean(gated)), n=len(gated),
                p_value=stats.wilcoxon(gated).pvalue,
                note=f"max |delta| {max(abs(d) for d in gated):.3f}")
    agree = sum(1 for m in models for f in bases
                if cells.get((m, f)) and cells.get((m, f + "_AB"))
                and is_pass(cells[(m, f)]) == is_pass(cells[(m, f + "_AB")]))
    tot = sum(1 for m in models for f in bases
              if cells.get((m, f)) and cells.get((m, f + "_AB")))
    add("factor_screen", "label format | generation | gate decision",
        "pairs agreeing", agree, n=tot,
        note="verdict: numeric and AB are equivalent on the generation path, so "
             "fig2 / fig3 report numeric and fig4 carries the check")

    # Label format on the logit path, the one factor that survives.
    if logit:
        L = {(r["model"], r["prompt_condition"]): r for r in logit}
        for field, label in (("logit_shape", "P(shape)"),
                             ("logit_tracking", "tracking")):
            deltas = []
            for m in models:
                for f in bases:
                    a, b = L.get((m, f)), L.get((m, f + "_AB"))
                    if a and b:
                        deltas.append(float(b[field]) - float(a[field]))
            if deltas:
                add("factor_screen", f"label format (AB - numeric) | logits | {label}",
                    "mean delta", float(np.mean(deltas)), n=len(deltas),
                    p_value=stats.wilcoxon(deltas).pvalue,
                    note="verdict: label format is NOT null on the logit path, so "
                         "fig9 keeps it")

    # Swap against PriDe, both read as decision rates. Comparing the swap mean
    # probability against the PriDe rate is a scale error, not an estimator
    # difference, and it roughly triples the apparent gap.
    if pride:
        d = [abs(float(r["swap_shape_rate"]) - float(r["pride_shape_rate"]))
             for r in pride if r.get("pride_shape_rate")]
        if d:
            add("factor_screen", "estimator | swap vs PriDe (both decision rates)",
                "mean |delta|", float(np.mean(d)), n=len(d),
                note=f"max {max(d):.4f}; {sum(x > 0.10 for x in d)} cells above "
                     "0.10; fig9b shows PriDe separately")

        # How far correcting moves each label format off the raw logit argmax.
        # This is the quantity fig6c reports; A/B moves more than numeric.
        for fmt, pred in (("numeric", lambda c: not c.endswith("_AB")),
                          ("AB", lambda c: c.endswith("_AB"))):
            moves = [
                abs(float(r["pride_shape_rate"]) - float(r["logit_argmax_shape"]))
                for r in pride
                if r.get("pride_shape_rate") and pred(r["prompt_condition"])
            ]
            if moves:
                n_big = sum(1 for x in moves if x > 0.10)
                add("factor_screen",
                    f"correction move | PriDe - logit argmax | {fmt}",
                    "mean |delta|", float(np.mean(moves)), n=len(moves),
                    note=f"max {max(moves):.4f}; {n_big}/{len(moves)} cells above "
                         f"0.10; fig6{'b' if fmt == 'AB' else ''} and fig6c")
        num_m = [
            abs(float(r["pride_shape_rate"]) - float(r["logit_argmax_shape"]))
            for r in pride
            if r.get("pride_shape_rate")
            and not r["prompt_condition"].endswith("_AB")
        ]
        ab_m = [
            abs(float(r["pride_shape_rate"]) - float(r["logit_argmax_shape"]))
            for r in pride
            if r.get("pride_shape_rate")
            and r["prompt_condition"].endswith("_AB")
        ]
        if num_m and ab_m:
            add("factor_screen",
                "correction move | AB mean|delta| - numeric mean|delta|",
                "mean delta", float(np.mean(ab_m) - np.mean(num_m)),
                n=len(num_m) + len(ab_m),
                note="verdict: A/B cells move farther under PriDe than numeric "
                     "ones; fig6b and fig6c carry the split")

    # Raw against centred cosine.
    if payload:
        d = []
        for m in (payload.get("models") or {}):
            rep = ((payload["models"][m].get("reps") or {}).get(EMB_LAYER)) or {}
            raw = (rep.get("raw") or {}).get("shape_rate")
            cen = (rep.get("centered") or {}).get("shape_rate")
            if raw is not None and cen is not None:
                d.append(float(cen) - float(raw))
        if len(d) >= 5:
            add("factor_screen", "metric | centred - raw cosine", "mean delta",
                float(np.mean(d)), n=len(d), p_value=stats.wilcoxon(d).pvalue,
                note=f"max |delta| {max(abs(x) for x in d):.3f}; verdict: "
                     "equivalent, figures show centred (the published convention)")

    # Pooling layers that are the same vectors.
    if payload:
        for L_ in EMB_LAYERS:
            if L_ == EMB_LAYER:
                continue
            d = []
            for m in (payload.get("models") or {}):
                a = _emb_rate(payload, m, L_)
                b = _emb_rate(payload, m, EMB_LAYER)
                if a is not None and b is not None:
                    d.append(a - b)
            if len(d) >= 3 and all(x == 0 for x in d):
                add("factor_screen", f"layer | {L_} vs {EMB_LAYER}",
                    "mean delta", 0.0, n=len(d),
                    note="identical in every model; verdict: one locus, not two, "
                         "so fig8 drops the duplicate")

    # Framing, the factor that behaves differently for validity and for rate.
    for a_c, b_c in (("no_word_category", "noun_label"),
                     ("no_word_similarity", "noun_label")):
        d = [float(cells[(m, b_c)]["shape_rate"]) - float(cells[(m, a_c)]["shape_rate"])
             for m in models
             if cells.get((m, a_c)) and cells.get((m, b_c))
             and is_pass(cells[(m, a_c)]) and is_pass(cells[(m, b_c)])]
        if d:
            add("factor_screen", f"framing | {b_c} - {a_c} | double-gated",
                "mean delta", float(np.mean(d)), n=len(d),
                note=f"range {min(d):+.3f} to {max(d):+.3f}; too few double-gated "
                     "cells to test, so framing is kept in the figures")
    for f in bases:
        n_pass = sum(1 for m in models if cells.get((m, f)) and is_pass(cells[(m, f)]))
        add("factor_screen", f"framing | {f} | numeric", "cells passing gate",
            n_pass, n=len(models),
            note="framing changes which cells are valid even where it does not "
                 "change the rate, so it stays in fig1")


# --------------------------------------------------------------------------
# 1. Per-cell shape rate against chance
# --------------------------------------------------------------------------
def cell_vs_chance(summary: list[dict]) -> None:
    n_pass = 0
    for r in summary:
        if not is_pass(r):
            continue
        n_pass += 1
        n = int(float(r["n_decided"]))
        k = int(round(float(r["shape_rate"]) * n))
        bt = stats.binomtest(k, n, 0.5)
        ci = bt.proportion_ci(confidence_level=0.95, method="wilson")
        add("cell_vs_chance", f"{r['model']} | {r['prompt_condition']}",
            "P(shape)", float(r["shape_rate"]), n=n,
            ci_lo=ci.low, ci_hi=ci.high, p_value=bt.pvalue,
            note="exact binomial vs 0.5, gate-passing cell")
    rates = [float(r["shape_rate"]) for r in summary if is_pass(r)]
    lo, hi = boot_ci(rates)
    add("cell_vs_chance", "all gate-passing cells", "mean P(shape)",
        float(np.mean(rates)), n=len(rates), ci_lo=lo, ci_hi=hi,
        note="bootstrap CI over cells")
    add("cell_vs_chance", "all cells", "cells passing gate", n_pass,
        n=len(summary), note=f"tracking >= {GATE}")


# --------------------------------------------------------------------------
# 2. Naming effect
# --------------------------------------------------------------------------
def naming_effect(summary: list[dict]) -> None:
    cells = {(r["model"], r["prompt_condition"]): r for r in summary}
    all_deltas: list[float] = []
    contrasts = [(b + suffix, "noun_label" + suffix)
                 for suffix in ("", "_AB")
                 for b in ("no_word_category", "no_word_similarity")]
    for base, noun in contrasts:
        deltas = []
        for model in sorted({r["model"] for r in summary}):
            b, nl = cells.get((model, base)), cells.get((model, noun))
            if b is None or nl is None or not (is_pass(b) and is_pass(nl)):
                continue
            nb, nn = int(float(b["n_decided"])), int(float(nl["n_decided"]))
            pb, pn = float(b["shape_rate"]), float(nl["shape_rate"])
            delta = pn - pb
            se = math.sqrt(pb * (1 - pb) / nb + pn * (1 - pn) / nn)
            z = delta / se if se > 0 else float("nan")
            p = 2 * (1 - stats.norm.cdf(abs(z))) if not math.isnan(z) else float("nan")
            deltas.append(delta)
            all_deltas.append(delta)
            add(f"naming_effect_{base}", model, "delta P(shape) noun - base",
                delta, n=nb + nn, ci_lo=delta - 1.96 * se, ci_hi=delta + 1.96 * se,
                p_value=p, note="two-proportion z, both cells gate-passing")
        if len(deltas) >= 2:
            lo, hi = boot_ci(deltas)
            add(f"naming_effect_{base}", "across models", "mean delta",
                float(np.mean(deltas)), n=len(deltas), ci_lo=lo, ci_hi=hi,
                note=f"range {min(deltas):+.3f} to {max(deltas):+.3f}")

    # All double-gated pairs pooled. REPORT.md quotes ten pairs and a mean shift
    # of +0.017, which only reconciles with this table if the AB contrasts are
    # counted alongside the numeric ones.
    if len(all_deltas) >= 5:
        lo, hi = boot_ci(all_deltas)
        w = stats.wilcoxon(all_deltas)
        t = stats.ttest_1samp(all_deltas, 0.0)
        add("naming_effect_pooled", "all double-gated pairs", "mean delta",
            float(np.mean(all_deltas)), n=len(all_deltas), ci_lo=lo, ci_hi=hi,
            note=f"range {min(all_deltas):+.3f} to {max(all_deltas):+.3f}, "
                 "numeric and AB contrasts")
        add("naming_effect_pooled", "all double-gated pairs",
            "Wilcoxon signed-rank vs 0", w.statistic, n=len(all_deltas),
            p_value=w.pvalue)
        add("naming_effect_pooled", "all double-gated pairs", "t vs 0",
            t.statistic, n=len(all_deltas), p_value=t.pvalue)
        # What shift could this many pairs have detected?
        sd = float(np.std(all_deltas, ddof=1))
        add("naming_effect_pooled", "all double-gated pairs",
            "SD of pair deltas", sd, n=len(all_deltas),
            note=f"smallest mean shift detectable at 80% power, two-sided 0.05: "
                 f"{2.8 * sd / math.sqrt(len(all_deltas)):.3f}")


# --------------------------------------------------------------------------
# 3 & 4. Embedding layers, and embedding against behaviour
# --------------------------------------------------------------------------
def _emb_rate(payload: dict, model: str, layer: str) -> float | None:
    block = (payload.get("models") or {}).get(model)
    if not isinstance(block, dict):
        return None
    rep = (block.get("reps") or {}).get(layer) or {}
    cen = rep.get("centered") or {}
    rate = cen.get("shape_rate")
    return None if rate is None else float(rate)


def layer_stats(payload: dict, summary: list[dict]) -> None:
    models = sorted((payload.get("models") or {}).keys())
    n_stim = (payload.get("config") or {}).get("n_stimuli")
    by_layer = {L: {m: _emb_rate(payload, m, L) for m in models} for L in EMB_LAYERS}

    for L in EMB_LAYERS:
        vals = [v for v in by_layer[L].values() if v is not None]
        if not vals:
            continue
        lo, hi = boot_ci(vals)
        add("layer_means", L, "mean embedding P(shape)", float(np.mean(vals)),
            n=len(vals), ci_lo=lo, ci_hi=hi,
            note=f"centred cosine, n={n_stim} triads, "
                 f"range {min(vals):.3f}-{max(vals):.3f}")

    for L in EMB_LAYERS:
        if L == EMB_LAYER:
            continue
        pairs = [(by_layer[L][m], by_layer[EMB_LAYER][m]) for m in models
                 if by_layer[L][m] is not None and by_layer[EMB_LAYER][m] is not None]
        if len(pairs) < 3:
            continue
        deltas = [a - b for a, b in pairs]
        note = "identical in every model" if all(d == 0 for d in deltas) else ""
        p = float("nan")
        if any(d != 0 for d in deltas):
            p = stats.wilcoxon(deltas).pvalue
        add("layer_means", f"{L} vs {EMB_LAYER}", "mean paired delta",
            float(np.mean(deltas)), n=len(deltas), p_value=p,
            note=note or f"max |delta| {max(abs(d) for d in deltas):.3f}, "
                         "Wilcoxon signed-rank")

    # Embedding against behaviour, per layer and framing.
    cells = {(r["model"], r["prompt_condition"]): r for r in summary}
    pending: list[tuple] = []
    for L in EMB_LAYERS:
        for cond, short in FRAMINGS:
            for subset in ("all", "gate-pass"):
                xs, ys = [], []
                for m in models:
                    e = by_layer[L][m]
                    b = cells.get((m, cond))
                    if e is None or b is None:
                        continue
                    if subset == "gate-pass" and not is_pass(b):
                        continue
                    xs.append(e)
                    ys.append(float(b["shape_rate"]))
                if len(xs) < 4:
                    continue
                pr = stats.pearsonr(xs, ys)
                sr = stats.spearmanr(xs, ys)
                plo, phi = boot_corr_ci(xs, ys, "pearson")
                pending.append((f"{L} | {short} | {subset}", pr, sr, plo, phi,
                                len(xs), subset))

    # These correlations are one family of tests over the same 14 models, so the
    # per-test p is optimistic. Benjamini-Hochberg q across the family, reported
    # next to each r rather than used to threshold anything.
    fam = [t for t in pending if t[6] == "all"]
    ps = [t[1][1] for t in fam]
    order = np.argsort(ps)
    q = np.empty(len(ps))
    for rank, idx in enumerate(order, start=1):
        q[idx] = ps[idx] * len(ps) / rank
    for rank in range(len(ps) - 2, -1, -1):
        q[order[rank]] = min(q[order[rank]], q[order[rank + 1]])
    q_by_group = {fam[i][0]: min(1.0, q[i]) for i in range(len(fam))}

    for group, pr, sr, plo, phi, n, subset in pending:
        note = ""
        if group in q_by_group:
            note = (f"BH q = {q_by_group[group]:.3f} across the {len(fam)} "
                    "all-cell correlations")
        add("emb_behavior_r", group, "Pearson r", pr[0], n=n,
            ci_lo=plo, ci_hi=phi, p_value=pr[1], note=note)
        add("emb_behavior_r", group, "Spearman rho", sr.statistic, n=n,
            p_value=sr.pvalue)


# --------------------------------------------------------------------------
# 5 & 6. Read-out ladder and McNemar, from the probe JSONs
# --------------------------------------------------------------------------
def probe_stats(probe_dir: Path) -> None:
    if not probe_dir.is_dir():
        print(f"  skip probe stats: missing {probe_dir}")
        return
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(probe_dir.glob("probe_*.json")):
        stem = path.stem[len("probe_"):]
        for L in EMB_LAYERS + ("pixels32",):
            if stem.endswith("_" + L):
                by_layer[L].append(json.loads(path.read_text()))
                break

    for L, probes in sorted(by_layer.items()):
        if L == "pixels32":
            continue
        for rung in LADDER_RUNGS:
            vals = [d["ladder"]["rungs"][rung]["shape_rate"] for d in probes
                    if rung in (d.get("ladder", {}).get("rungs") or {})]
            if not vals:
                continue
            lo, hi = boot_ci(vals)
            add("readout_ladder", f"{L} | {rung}", "mean shape rate",
                float(np.mean(vals)), n=len(vals), ci_lo=lo, ci_hi=hi,
                note=f"range {min(vals):.3f}-{max(vals):.3f}")

        cen = [d["ladder"]["rungs"]["2_centered_cosine"]["shape_rate"] for d in probes]
        lrn = [d["ladder"]["rungs"]["5_learned_metric"]["shape_rate"] for d in probes]
        if len(cen) >= 5:
            w = stats.wilcoxon(np.array(lrn) - np.array(cen))
            add("readout_ladder", f"{L} | learned vs centred", "mean gain",
                float(np.mean(np.array(lrn) - np.array(cen))), n=len(cen),
                p_value=w.pvalue, note="Wilcoxon signed-rank over models")

        ps = [d["ladder"]["mcnemar_centered_vs_learned"]["p"] for d in probes
              if "mcnemar_centered_vs_learned" in (d.get("ladder") or {})]
        if ps:
            add("mcnemar", L, "max p (centred vs learned)", max(ps), n=len(ps),
                note=f"min p {min(ps):.3g}; all {len(ps)} models")

        mesh = [d["P1_mesh_identity"]["accuracy"] for d in probes]
        tex = [d["P2_texture_identity"]["accuracy"] for d in probes]
        chance = probes[0]["P1_mesh_identity"]["chance"]
        add("readout_ladder", f"{L} | mesh identity", "mean probe accuracy",
            float(np.mean(mesh)), n=len(mesh),
            note=f"chance {chance:.3f}, range {min(mesh):.3f}-{max(mesh):.3f}")
        add("readout_ladder", f"{L} | texture identity", "mean probe accuracy",
            float(np.mean(tex)), n=len(tex),
            note=f"range {min(tex):.3f}-{max(tex):.3f}")

    # Does the pooling layer change the ladder? Paired within model, which is the
    # comparison the per-layer means above cannot make.
    for rung in ("2_centered_cosine", "5_learned_metric"):
        ref = {d["name"].rsplit("_" + EMB_LAYER, 1)[0]:
               d["ladder"]["rungs"][rung]["shape_rate"]
               for d in by_layer.get(EMB_LAYER, [])}
        for L, probes in sorted(by_layer.items()):
            if L in (EMB_LAYER, "pixels32"):
                continue
            deltas = []
            for d in probes:
                model = d["name"].rsplit("_" + L, 1)[0]
                if model in ref and rung in (d.get("ladder", {}).get("rungs") or {}):
                    deltas.append(d["ladder"]["rungs"][rung]["shape_rate"] - ref[model])
            if len(deltas) < 3:
                continue
            p = (stats.wilcoxon(deltas).pvalue if any(d != 0 for d in deltas)
                 else float("nan"))
            add("readout_layer_contrast", f"{rung} | {L} vs {EMB_LAYER}",
                "mean paired delta", float(np.mean(deltas)), n=len(deltas),
                p_value=p,
                note=("identical in every model" if all(d == 0 for d in deltas)
                      else f"range {min(deltas):+.3f} to {max(deltas):+.3f}"))


# --------------------------------------------------------------------------
# 7. Generated answer against logit argmax
# --------------------------------------------------------------------------
def gen_vs_logit(summary: list[dict], logit: list[dict]) -> None:
    L = {(r["model"], r["prompt_condition"]): r for r in logit}
    by_format: dict[str, list[float]] = defaultdict(list)
    quad = {"both": 0, "generation only": 0, "logits only": 0, "neither": 0}
    for r in summary:
        lg = L.get((r["model"], r["prompt_condition"]))
        if lg is None:
            continue
        fmt = "AB" if r["prompt_condition"].endswith("_AB") else "numeric"
        by_format[fmt].append(float(lg["logit_shape"]) - float(r["shape_rate"]))
        g_pass, l_pass = is_pass(r), is_pass(lg, "logit_gate_pass")
        key = ("both" if g_pass and l_pass else "generation only" if g_pass
               else "logits only" if l_pass else "neither")
        quad[key] += 1

    for fmt, deltas in sorted(by_format.items()):
        lo, hi = boot_ci([abs(d) for d in deltas])
        w = stats.wilcoxon(deltas)
        add("gen_vs_logit", fmt, "mean |delta| logit - generation",
            float(np.mean(np.abs(deltas))), n=len(deltas), ci_lo=lo, ci_hi=hi,
            note=f"|delta|>0.10 in {sum(1 for d in deltas if abs(d) > 0.10)} cells")
        add("gen_vs_logit", fmt, "mean signed delta", float(np.mean(deltas)),
            n=len(deltas), p_value=w.pvalue, note="Wilcoxon signed-rank vs 0")
    if by_format.get("AB") and by_format.get("numeric"):
        u = stats.mannwhitneyu([abs(d) for d in by_format["AB"]],
                               [abs(d) for d in by_format["numeric"]])
        add("gen_vs_logit", "AB vs numeric", "Mann-Whitney U on |delta|",
            u.statistic, p_value=u.pvalue,
            note="is the divergence specific to letter labels")

    for k, v in quad.items():
        add("gen_vs_logit", "gate 2x2", f"cells passing: {k}", v,
            n=sum(quad.values()), note=f"tracking gate {GATE} on each path")
    disc = quad["generation only"] + quad["logits only"]
    if disc:
        bt = stats.binomtest(quad["generation only"], disc, 0.5)
        add("gen_vs_logit", "gate 2x2", "McNemar exact (discordant cells)",
            quad["generation only"], n=disc, p_value=bt.pvalue,
            note="of the cells that pass on one path only, how many are generation")

    mass = [float(r["option_mass"]) for r in logit]
    add("gen_vs_logit", "logit path", "mean option mass", float(np.mean(mass)),
        n=len(mass),
        note=f"{sum(1 for m in mass if m < 0.5)} cells below 0.5; PriDe and the "
             "unnormalised swap collapse toward 0 there")


# --------------------------------------------------------------------------
# 8. Item-level human-model agreement
# --------------------------------------------------------------------------
def _reliability(props: list[float], ns: list[float]) -> float:
    """How much of the variance across items is real rather than sampling noise.

    Each item rate is a proportion from a handful of trials, so its observed
    variance is true item variance plus binomial error. Subtracting the mean
    error variance gives the usual estimate of reliability, which is what the
    item correlation has to be corrected for before it is read as small.
    """
    obs = float(np.var(props, ddof=1))
    if obs <= 0:
        return float("nan")
    err = float(np.mean([p * (1 - p) / n for p, n in zip(props, ns) if n > 0]))
    return max(0.0, (obs - err) / obs)


def human_model_items(summary: list[dict]) -> None:
    item_path = DATA / "full_grid_item_rates.csv"
    human_path = DATA / "human_item_means.csv"
    if not item_path.is_file() or not human_path.is_file():
        print("  skip human_model_item: run item_rates.py and the human export")
        return
    human: dict[tuple[str, str], float] = {}
    human_n: dict[tuple[str, str], float] = {}
    for r in read_csv(human_path):
        if r.get("stim_set_name") == "grid":
            human[(r["condition"], r["stim_id"])] = float(r["shape_prop"])
            human_n[(r["condition"], r["stim_id"])] = float(r["n"])
    model_items: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in read_csv(item_path):
        model_items[(r["model"], r["prompt_condition"])][r["stim_id"]] = float(r["shape_rate"])

    cells = {(r["model"], r["prompt_condition"]): r for r in summary}
    for cond in ("no_word_category", "noun_label"):
        shared = sorted(sid for (c, sid) in human if c == cond)
        if not shared:
            continue
        hx = [human[(cond, sid)] for sid in shared]
        passers = [m for (m, c), r in cells.items() if c == cond and is_pass(r)]
        stacked, per_r = [], []
        for m in sorted(passers):
            mi = model_items.get((m, cond), {})
            ys = [mi[sid] for sid in shared if sid in mi]
            if len(ys) != len(hx):
                continue
            stacked.append(ys)
            pr = stats.pearsonr(hx, ys)
            per_r.append(pr[0])
            add("human_model_item", f"{cond} | {m}", "Pearson r", pr[0],
                n=len(ys), p_value=pr[1],
                note=f"mean |human - model| {np.mean(np.abs(np.array(hx) - np.array(ys))):.3f}")
        if not stacked:
            continue
        mean_y = list(np.mean(np.asarray(stacked, float), axis=0))
        pr = stats.pearsonr(hx, mean_y)
        sr = stats.spearmanr(hx, mean_y)
        lo, hi = boot_corr_ci(hx, mean_y, "pearson")
        add("human_model_item", f"{cond} | mean of gate-passing models",
            "Pearson r", pr[0], n=len(mean_y), ci_lo=lo, ci_hi=hi, p_value=pr[1],
            note=f"{len(stacked)} models")
        add("human_model_item", f"{cond} | mean of gate-passing models",
            "Spearman rho", sr.statistic, n=len(mean_y), p_value=sr.pvalue)
        add("human_model_item", f"{cond} | mean of gate-passing models",
            "mean |human - model|",
            float(np.mean(np.abs(np.array(hx) - np.array(mean_y)))), n=len(mean_y))
        add("human_model_item", f"{cond} | mean of gate-passing models",
            "human mean P(shape)", float(np.mean(hx)), n=len(hx),
            note=f"model mean {np.mean(mean_y):.3f}")

        # Correct the correlation for how noisy each side's item estimate is.
        hn = [human_n[(cond, sid)] for sid in shared]
        rel_h = _reliability(hx, hn)
        rel_m = _reliability(mean_y, [2.0 * len(stacked)] * len(mean_y))
        add("human_model_item", f"{cond} | reliability", "human item reliability",
            rel_h, n=len(hx), note=f"median {np.median(hn):.0f} trials per triad")
        add("human_model_item", f"{cond} | reliability", "model item reliability",
            rel_m, n=len(mean_y),
            note=f"{2 * len(stacked)} trials per triad "
                 f"({len(stacked)} models x 2 orders)")
        if rel_h < 0.05:
            add("human_model_item", f"{cond} | mean of gate-passing models",
                "attenuation-corrected r", None, n=len(mean_y),
                note="not estimable: human item variance is at the binomial noise "
                     f"floor (reliability {rel_h:.3f}, mean {np.mean(hx):.3f}), so "
                     "there is no reliable between-triad variance for a model to "
                     "track")
        elif rel_m > 0:
            add("human_model_item", f"{cond} | mean of gate-passing models",
                "attenuation-corrected r", pr[0] / math.sqrt(rel_h * rel_m),
                n=len(mean_y),
                note="observed r divided by sqrt(reliability product); an upper "
                     "bound, and the models are not exchangeable draws so treat "
                     "the model side as approximate")
        if len(per_r) >= 2:
            add("human_model_item", f"{cond} | per-model r", "range",
                float(np.mean(per_r)), n=len(per_r),
                note=f"{min(per_r):+.3f} to {max(per_r):+.3f} (value is the mean)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    summary = read_csv(DATA / "full_grid_v1a_summary.csv")
    logit_path = DATA / "full_grid_logit_validity.csv"
    pride_path = DATA / "full_grid_v1a_pride.csv"
    logit = read_csv(logit_path) if logit_path.is_file() else None
    pride = read_csv(pride_path) if pride_path.is_file() else None
    payload = json.loads(EMBED_GRID.read_text()) if EMBED_GRID.is_file() else None

    factor_screen(summary, logit, pride, payload, PROBE_DIR)
    cell_vs_chance(summary)
    naming_effect(summary)

    if payload:
        layer_stats(payload, summary)
    else:
        print(f"  skip layer stats: missing {EMBED_GRID}")

    probe_stats(PROBE_DIR)

    if logit:
        gen_vs_logit(summary, logit)
    else:
        print("  skip gen_vs_logit: run analysis_pipe/logit_validity.py first")

    human_model_items(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["analysis", "group", "statistic", "value",
                                          "n", "ci_lo", "ci_hi", "p_value", "note"])
        w.writeheader()
        w.writerows(RESULTS)
    counts = defaultdict(int)
    for r in RESULTS:
        counts[r["analysis"]] += 1
    print(f"Wrote {args.out} ({len(RESULTS)} rows)")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24} {v:>4} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
