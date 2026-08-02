#!/usr/bin/env python3
"""Naming contrast restricted to gate-passing cells.

Pairs the numeric no-word similarity cell (baseline) with each numeric
noun_label cell (word) per model, per stimulus. The generation-level contrast
is interpreted only where BOTH cells pass the tracking gate (>= 0.70). The
latent contrast uses per-stimulus swap-corrected P(shape) differences with a
bootstrap CI and an exact sign test, and is reported for every cell with gate
status flagged.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
from pathlib import Path

from playground_pride_debias import DEFAULT_SESSION, parse_log

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = (
    REPO / "results" / "playground.results" / "gated_naming_contrast_2026-07-17"
)
BASELINE_CONDITION = "no_word_similarity"
NOUN_CONDITION = "noun_label"
GATE = 0.70
BOOT_N = 5000
SEED = 20260717


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def per_stim_stats(sf: dict[int, dict], tf: dict[int, dict]) -> dict[int, dict]:
    """Per-stimulus swap-corrected logit P(shape) and swap-averaged gen shape."""
    out: dict[int, dict] = {}
    for trial_id in sorted(set(sf) & set(tf)):
        s, t = sf[trial_id], tf[trial_id]
        gen = None
        if s["gen_shape"] is not None and t["gen_shape"] is not None:
            gen = 0.5 * (float(s["gen_shape"]) + float(t["gen_shape"]))
        out[trial_id] = {
            "swap_p": 0.5 * (s["p_shape"] + t["p_shape"]),
            "gen": gen,
            "tracked": (
                s["gen_shape"] == t["gen_shape"]
                if s["gen_shape"] is not None and t["gen_shape"] is not None
                else None
            ),
        }
    return out


def cell_summary(stats: dict[int, dict]) -> dict:
    tracked = [v["tracked"] for v in stats.values() if v["tracked"] is not None]
    gens = [v["gen"] for v in stats.values() if v["gen"] is not None]
    tracking = _mean([float(x) for x in tracked])
    return {
        "n": len(stats),
        "tracking": tracking,
        "gate": tracking >= GATE,
        "gen_shape": _mean(gens),
        "swap_mean_p": _mean([v["swap_p"] for v in stats.values()]),
    }


def bootstrap_ci(diffs: list[float], rng: random.Random) -> tuple[float, float]:
    means = sorted(
        _mean([rng.choice(diffs) for _ in diffs]) for _ in range(BOOT_N)
    )
    return means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N) - 1]


def sign_test_p(diffs: list[float]) -> tuple[int, int, float]:
    """Exact two-sided binomial sign test on nonzero differences."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return pos, neg, min(1.0, 2 * tail)


def contrast(base: dict[int, dict], noun: dict[int, dict], rng: random.Random) -> dict:
    ids = sorted(set(base) & set(noun))
    swap_diffs = [noun[i]["swap_p"] - base[i]["swap_p"] for i in ids]
    gen_ids = [
        i for i in ids if base[i]["gen"] is not None and noun[i]["gen"] is not None
    ]
    gen_diffs = [noun[i]["gen"] - base[i]["gen"] for i in gen_ids]
    lo, hi = bootstrap_ci(swap_diffs, rng)
    pos, neg, p = sign_test_p(swap_diffs)
    out = {
        "n_paired": len(ids),
        "swap_delta": _mean(swap_diffs),
        "swap_ci_lo": lo,
        "swap_ci_hi": hi,
        "sign_pos": pos,
        "sign_neg": neg,
        "sign_p": p,
        "gen_delta": _mean(gen_diffs) if gen_diffs else float("nan"),
    }
    if gen_diffs:
        glo, ghi = bootstrap_ci(gen_diffs, rng)
        out["gen_ci_lo"], out["gen_ci_hi"] = glo, ghi
    else:
        out["gen_ci_lo"] = out["gen_ci_hi"] = float("nan")
    return out


def collect(session: Path) -> list[dict]:
    # (condition, word) -> model -> order -> trials
    cells: dict[tuple[str, str], dict[str, dict[str, dict[int, dict]]]] = {}
    for path in sorted(session.glob("playground_smoke_30trials_*.txt")):
        parsed = parse_log(path)
        if parsed["label_set"] != "numeric":
            continue
        if parsed["condition"] not in (BASELINE_CONDITION, NOUN_CONDITION):
            continue
        key = (parsed["condition"], parsed["word"])
        for model, trials in parsed["models"].items():
            cells.setdefault(key, {}).setdefault(model, {})[parsed["order"]] = trials

    baselines = cells.get((BASELINE_CONDITION, "(none)"), {})
    rng = random.Random(SEED)
    rows = []
    for (condition, word), models in sorted(cells.items()):
        if condition != NOUN_CONDITION:
            continue
        for model, orders in sorted(models.items()):
            if len(orders) < 2 or model not in baselines:
                continue
            base_orders = baselines[model]
            if len(base_orders) < 2:
                continue
            base = per_stim_stats(
                base_orders["shape_first"], base_orders["texture_first"]
            )
            noun = per_stim_stats(orders["shape_first"], orders["texture_first"])
            base_sum = cell_summary(base)
            noun_sum = cell_summary(noun)
            rows.append(
                {
                    "model": model,
                    "word": word,
                    "base_tracking": base_sum["tracking"],
                    "base_gate": base_sum["gate"],
                    "base_gen_shape": base_sum["gen_shape"],
                    "base_swap_mean_p": base_sum["swap_mean_p"],
                    "noun_tracking": noun_sum["tracking"],
                    "noun_gate": noun_sum["gate"],
                    "noun_gen_shape": noun_sum["gen_shape"],
                    "noun_swap_mean_p": noun_sum["swap_mean_p"],
                    "both_gates_pass": base_sum["gate"] and noun_sum["gate"],
                    **contrast(base, noun, rng),
                }
            )
    return rows


def _fmt(value: float, digits: int = 2) -> str:
    return "nan" if value != value else f"{value:.{digits}f}"


def write_html(rows: list[dict], path: Path) -> None:
    def table(rows: list[dict], gen_columns: bool) -> str:
        head = (
            "<tr><th>model</th><th>word</th><th>base trk</th><th>noun trk</th>"
            + (
                "<th>gen shape: no-word</th><th>gen shape: noun</th>"
                "<th>gen &Delta; [95% CI]</th>"
                if gen_columns
                else "<th>gates</th>"
            )
            + "<th>swap P(shape): no-word</th><th>swap P(shape): noun</th>"
            "<th>swap &Delta; [95% CI]</th><th>sign +/-</th><th>sign p</th></tr>"
        )
        trs = []
        for r in rows:
            cls = "pass" if r["both_gates_pass"] else ""
            mid = (
                f'<td>{_fmt(r["base_gen_shape"])}</td>'
                f'<td>{_fmt(r["noun_gen_shape"])}</td>'
                f'<td>{_fmt(r["gen_delta"], 3)} '
                f'[{_fmt(r["gen_ci_lo"], 3)}, {_fmt(r["gen_ci_hi"], 3)}]</td>'
                if gen_columns
                else (
                    "<td>both PASS</td>"
                    if r["both_gates_pass"]
                    else f'<td>base {"PASS" if r["base_gate"] else "fail"} / '
                    f'noun {"PASS" if r["noun_gate"] else "fail"}</td>'
                )
            )
            trs.append(
                f'<tr class="{cls}"><td>{html.escape(r["model"])}</td>'
                f'<td>{html.escape(r["word"])}</td>'
                f'<td>{_fmt(r["base_tracking"])}</td>'
                f'<td>{_fmt(r["noun_tracking"])}</td>'
                f"{mid}"
                f'<td>{_fmt(r["base_swap_mean_p"])}</td>'
                f'<td>{_fmt(r["noun_swap_mean_p"])}</td>'
                f'<td>{_fmt(r["swap_delta"], 3)} '
                f'[{_fmt(r["swap_ci_lo"], 3)}, {_fmt(r["swap_ci_hi"], 3)}]</td>'
                f'<td>{r["sign_pos"]}/{r["sign_neg"]}</td>'
                f'<td>{_fmt(r["sign_p"], 4)}</td></tr>'
            )
        return f'<div class="wrap"><table>{head}{"".join(trs)}</table></div>'

    gated = [r for r in rows if r["both_gates_pass"]]
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gated naming contrast: numeric no-word similarity vs numeric noun</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#fafafa;color:#1a1a1a}}
main{{max-width:1500px;margin:auto}} .wrap{{overflow:auto;border:1px solid #ddd;margin:12px 0}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:6px 8px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
tr.pass td{{background:#e6f4ea}} code{{background:#eee;padding:1px 4px}}
.note{{border-left:4px solid #1a56b0;background:white;padding:10px 14px;margin:14px 0}}
h2{{margin-top:32px}}
</style></head><body><main>
<h1>Gated naming contrast</h1>
<p>Numeric <code>{NOUN_CONDITION}</code> (each word) against numeric
<code>{BASELINE_CONDITION}</code> (no word) in the same models, paired per
stimulus across both orders. Swap-corrected P(shape) differences use all 30
paired stimuli; bootstrap CIs use {BOOT_N} resamples; sign test is exact
binomial on nonzero differences.</p>
<div class="note"><b>Primary comparison: both cells pass the generation gate
(tracking &ge; {GATE}).</b> Only these rows support a generation-level naming
claim. The full table reports the latent (swap-corrected logit) contrast for
every cell with gate status flagged; those rows describe latent evidence
only.</div>
<h2>1 &middot; Both-gates-passing cells (generation-level contrast)</h2>
{table(gated, gen_columns=True) if gated else "<p>No cell passes both gates.</p>"}
<h2>2 &middot; All cells (latent contrast, gate status flagged)</h2>
{table(rows, gen_columns=False)}
<p>Generated by <code>playgrounds/gated_naming_contrast.py</code> from the
session logs in <code>{html.escape(str(DEFAULT_SESSION.name))}</code>.</p>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()

    rows = collect(args.session)
    if not rows:
        raise SystemExit("No paired numeric cells found")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_prefix.with_suffix(".html"))

    gated = [r for r in rows if r["both_gates_pass"]]
    print(f"cells={len(rows)} both_gates_pass={len(gated)}")
    for r in gated:
        print(
            f"  {r['model']} · {r['word']}: gen {r['base_gen_shape']:.2f}"
            f" -> {r['noun_gen_shape']:.2f} (delta {r['gen_delta']:+.3f}"
            f" [{r['gen_ci_lo']:+.3f}, {r['gen_ci_hi']:+.3f}]);"
            f" swap {r['base_swap_mean_p']:.2f} -> {r['noun_swap_mean_p']:.2f}"
            f" (delta {r['swap_delta']:+.3f}"
            f" [{r['swap_ci_lo']:+.3f}, {r['swap_ci_hi']:+.3f}],"
            f" sign {r['sign_pos']}/{r['sign_neg']}, p={r['sign_p']:.4f})"
        )
    print(f"CSV: {csv_path}")
    print(f"HTML: {args.out_prefix.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
