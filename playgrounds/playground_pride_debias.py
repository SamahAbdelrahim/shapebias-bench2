#!/usr/bin/env python3
"""Swap/full-permutation/PriDe analysis for playground text logs.

Parses saved one_pass relative probabilities from paired shape_first and
texture_first logs. PriDe follows playgrounds/pride_debias.py: estimate the
first-option prior on the first K stimuli, then evaluate each order separately
on the held-out stimuli.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from pride_debias import estimate_prior_first, fullperm_p_shape, pride_p_shape

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = (
    REPO
    / "results"
    / "playground.results"
    / "session_2026-07-17_farmshare"
)
DEFAULT_PREFIX = (
    REPO
    / "results"
    / "playground.results"
    / "prompt_pride_debias_2026-07-17"
)


def _meta(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    condition = _meta(text, "Prompt condition")
    if not condition:
        raise ValueError(f"{path}: missing Prompt condition")
    word = _meta(text, "Word") or "(none)"
    labels_text = _meta(text, "Choice labels")
    if labels_text:
        labels = tuple(part.strip() for part in labels_text.split("/"))
    else:
        labels = ("A", "B") if condition.endswith("_AB") else ("1", "2")
    order_match = re.search(r"^Trials:\s*\d+\s*\(order=([^)]+)\)", text, re.MULTILINE)
    if not order_match:
        raise ValueError(f"{path}: missing order")
    order = order_match.group(1)

    models: dict[str, dict[int, dict]] = {}
    for model_block in re.split(r"\n={72}\nMODEL: ", text)[1:]:
        model = model_block.split("\n", 1)[0].strip()
        trials: dict[int, dict] = {}
        pieces = re.split(r"(?=^Results for trial )", model_block, flags=re.MULTILINE)
        for piece in pieces[1:]:
            header = re.search(
                r"^Results for trial (\d+) \([^)]+, ground truth ([AB12]) = shape match\):",
                piece,
                re.MULTILINE,
            )
            if not header:
                continue
            trial_id = int(header.group(1))
            ground_truth = header.group(2)
            gen_match = re.search(r"^Generation pick: ([AB12])", piece, re.MULTILINE)
            one_pass = re.search(
                r"Logit scoring \[one_pass\].*?"
                r"^Relative probs:\s*([0-9.]+)\s*/\s*([0-9.]+)",
                piece,
                re.MULTILINE | re.DOTALL,
            )
            if not one_pass:
                continue
            probs = (float(one_pass.group(1)), float(one_pass.group(2)))
            shape_idx = 0 if ground_truth == labels[0] else 1
            gen_pick = gen_match.group(1) if gen_match else None
            trials[trial_id] = {
                "p_shape": probs[shape_idx],
                "p_first": probs[0],
                "gen_shape": None if gen_pick is None else gen_pick == ground_truth,
            }
        models[model] = trials
    return {
        "path": str(path),
        "condition": condition,
        "word": word,
        "label_set": "AB" if labels == ("A", "B") else "numeric",
        "order": order,
        "models": models,
    }


def _rate(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def analyze_pair(sf: dict[int, dict], tf: dict[int, dict], n_est: int) -> dict:
    trial_ids = sorted(set(sf) & set(tf))
    per_stim = [
        {
            "trial_id": trial_id,
            "sf_log_shape": sf[trial_id]["p_shape"],
            "tf_log_shape": tf[trial_id]["p_shape"],
            "sf_gen_shape": sf[trial_id]["gen_shape"],
            "tf_gen_shape": tf[trial_id]["gen_shape"],
        }
        for trial_id in trial_ids
    ]
    if len(per_stim) <= n_est:
        raise ValueError(f"Need > {n_est} paired trials; got {len(per_stim)}")

    gen_pairs = [
        (t["sf_gen_shape"], t["tf_gen_shape"])
        for t in per_stim
        if t["sf_gen_shape"] is not None and t["tf_gen_shape"] is not None
    ]
    gen_tracking = _rate(a == b for a, b in gen_pairs)
    gen_shape = _mean(
        float(value)
        for pair in gen_pairs
        for value in pair
    )
    log_tracking = _rate(
        (t["sf_log_shape"] > 0.5) == (t["tf_log_shape"] > 0.5)
        for t in per_stim
    )

    swap_ps = [
        0.5 * (t["sf_log_shape"] + t["tf_log_shape"])
        for t in per_stim
    ]
    perm_ps = [
        fullperm_p_shape(t["sf_log_shape"], t["tf_log_shape"])
        for t in per_stim
    ]

    estimate, test = per_stim[:n_est], per_stim[n_est:]
    prior_first = estimate_prior_first(estimate)
    pride_sf_ps = [
        pride_p_shape(t["sf_log_shape"], True, prior_first)
        for t in test
    ]
    pride_tf_ps = [
        pride_p_shape(1 - t["tf_log_shape"], False, prior_first)
        for t in test
    ]

    return {
        "n": len(per_stim),
        "n_est": n_est,
        "n_pride_test": len(test),
        "gen_tracking": gen_tracking,
        "gate": gen_tracking >= 0.70,
        "gen_shape": gen_shape,
        "log_tracking": log_tracking,
        "prior_first": prior_first,
        "swap_mean_p_shape": _mean(swap_ps),
        "swap_shape_rate": _rate(p > 0.5 for p in swap_ps),
        "fullperm_mean_p_shape": _mean(perm_ps),
        "fullperm_shape_rate": _rate(p > 0.5 for p in perm_ps),
        "pride_sf_mean_p_shape": _mean(pride_sf_ps),
        "pride_sf_shape_rate": _rate(p > 0.5 for p in pride_sf_ps),
        "pride_tf_mean_p_shape": _mean(pride_tf_ps),
        "pride_tf_shape_rate": _rate(p > 0.5 for p in pride_tf_ps),
    }


def collect(session: Path, n_est: int) -> list[dict]:
    grouped: dict[tuple, dict[str, dict[str, dict[int, dict]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    sources: dict[tuple, set[str]] = defaultdict(set)
    for path in sorted(session.glob("playground_smoke_30trials_*.txt")):
        parsed = parse_log(path)
        key = (parsed["condition"], parsed["word"], parsed["label_set"])
        order = parsed["order"]
        for model, trials in parsed["models"].items():
            # qwen8-only files augment the earlier six-model A/B logs.
            grouped[key][model][order] = trials
        sources[key].add(path.name)

    rows = []
    for (condition, word, label_set), models in sorted(grouped.items()):
        for model, orders in sorted(models.items()):
            if "shape_first" not in orders or "texture_first" not in orders:
                continue
            result = analyze_pair(
                orders["shape_first"],
                orders["texture_first"],
                n_est,
            )
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "word": word,
                    "label_set": label_set,
                    **result,
                    "sources": ";".join(sorted(sources[(condition, word, label_set)])),
                }
            )
    return rows


def write_html(rows: list[dict], path: Path) -> None:
    trs = []
    for row in rows:
        cls = "pass" if row["gate"] else ""
        trs.append(
            f'<tr class="{cls}"><td>{html.escape(row["model"])}</td>'
            f'<td>{html.escape(row["condition"])}</td>'
            f'<td>{html.escape(row["word"])}</td>'
            f'<td>{row["label_set"]}</td>'
            f'<td>{row["gen_tracking"]:.2f}</td>'
            f'<td>{"PASS" if row["gate"] else "fail"}</td>'
            f'<td>{row["prior_first"]:.2f}</td>'
            f'<td>{row["swap_mean_p_shape"]:.2f}</td>'
            f'<td>{row["swap_shape_rate"]:.2f}</td>'
            f'<td>{row["fullperm_mean_p_shape"]:.2f}</td>'
            f'<td>{row["fullperm_shape_rate"]:.2f}</td>'
            f'<td>{row["pride_sf_mean_p_shape"]:.2f}</td>'
            f'<td>{row["pride_sf_shape_rate"]:.2f}</td>'
            f'<td>{row["pride_tf_mean_p_shape"]:.2f}</td>'
            f'<td>{row["pride_tf_shape_rate"]:.2f}</td></tr>'
        )
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt logits: swap correction and PriDe</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#fafafa;color:#1a1a1a}}
main{{max-width:1400px;margin:auto}} .wrap{{overflow:auto;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:6px 8px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
tr.pass td{{background:#e6f4ea}} code{{background:#eee;padding:1px 4px}}
.note{{border-left:4px solid #1a56b0;background:white;padding:10px 14px;margin:14px 0}}
</style></head><body><main>
<h1>Prompt logits: swap correction, full permutation, and PriDe</h1>
<p>Saved <code>one_pass</code> probabilities; 30 paired trials per cell.
Swap/full permutation use all 30. PriDe prior uses the first 10 and reports
held-out estimates on the remaining 20.</p>
<div class="note"><b>Read mean P(shape) and shape rate together.</b>
Mean P(shape) preserves confidence. Shape rate is the fraction of stimuli above
0.5. Generation tracking still gates behavioral shape claims; corrected logits
describe latent choice evidence even when generation fails.</div>
<div class="wrap"><table><tr>
<th>model</th><th>condition</th><th>word</th><th>labels</th>
<th>gTrk</th><th>gate</th><th>priorFirst</th>
<th>swap meanP</th><th>swap rate</th>
<th>perm meanP</th><th>perm rate</th>
<th>PriSF meanP</th><th>PriSF rate</th>
<th>PriTF meanP</th><th>PriTF rate</th>
</tr>{''.join(trs)}</table></div>
<p>Generated by <code>playgrounds/playground_pride_debias.py</code>.</p>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--n-est", type=int, default=10)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()

    rows = collect(args.session, args.n_est)
    if not rows:
        raise SystemExit("No paired playground cells found")
    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_html(rows, prefix.with_suffix(".html"))

    print(f"cells={len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {prefix.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
