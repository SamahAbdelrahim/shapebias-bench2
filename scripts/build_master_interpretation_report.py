#!/usr/bin/env python3
"""Master playground interpretation report.

Merges behavioral smoke (generation tracking + shape rate), PriDe-corrected
logits, embedding readouts, and archive-vs-fixed AB comparisons into one page
for interpretation. Re-run after AB label-fix reruns land in
``session_2026-07-30_*``.
"""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_playground_results_html import (  # noqa: E402
    ARCHIVE_AB,
    CSS,
    PLAY,
    SESS_AB_FIXED,
    SESS_CAT_REVISED,
    SESS_PROBE,
    SESS_SMOKE,
    comparison_section_html,
    fmt_rate,
    load_condition_rows,
    newest_smith_session,
    smoke_section_html,
)

PROBE = REPO / "results" / "probe.results"
OUT_HTML = PLAY / "master_interpretation_2026-07-30.html"
OUT_CSV = PLAY / "master_interpretation_2026-07-30.csv"
GATE = 0.70

MODEL_ORDER = (
    "smolvlm",
    "internvl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-8b",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
)


def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_embed(primary: Path, fill: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in (primary, fill):
        data = _load_json(path)
        for name, payload in data.get("models", {}).items():
            out[name] = payload
    return out


def _embed_shape_rate(models: dict[str, dict], name: str) -> float | None:
    payload = models.get(name, {})
    if "error" in payload:
        return None
    reps = payload.get("reps", {})
    rep = reps.get("proj_mean") or next(iter(reps.values()), None)
    if not rep:
        return None
    return rep.get("centered", {}).get("shape_rate")


def _row_from_gen(model: str, condition: str, stim_set: str, label: str, gen: dict) -> dict:
    trk = gen.get("trk")
    shp = gen.get("shp_avg")
    return {
        "model": model,
        "condition": condition,
        "stim_set": stim_set,
        "label_set": label,
        "n": gen.get("n"),
        "gen_tracking": trk,
        "gen_shape": shp,
        "pos_first": gen.get("pos_a"),
        "gate": "PASS" if (trk or 0) >= GATE else "fail",
    }


def collect_behavioral_rows() -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("benchmark", "no_word_similarity", "numeric", SESS_SMOKE),
        ("benchmark", "no_word_category", "numeric", SESS_CAT_REVISED),
        ("benchmark", "noun_label", "numeric", SESS_SMOKE),
        ("benchmark", "no_word_similarity_AB", "AB", SESS_AB_FIXED),
        ("benchmark", "no_word_category_AB", "AB", SESS_AB_FIXED),
        ("benchmark", "noun_label_AB", "AB", SESS_AB_FIXED),
    ]
    smith_num = PLAY / "session_2026-07-25_smith_farmshare"
    smith_ab = newest_smith_session()
    specs += [
        ("smith", "no_word_similarity", "numeric", smith_num),
        ("smith", "no_word_category", "numeric", smith_num),
        ("smith", "noun_label", "numeric", smith_num),
        ("smith", "no_word_similarity_AB", "AB", smith_ab),
        ("smith", "no_word_category_AB", "AB", smith_ab),
        ("smith", "noun_label_AB", "AB", smith_ab),
    ]
    for stim_set, condition, label, session in specs:
        if not session.is_dir():
            continue
        # Temporarily prefer explicit session: copy find logic via glob
        sf = session / f"playground_smoke_30trials_shape_first_{condition}.txt"
        tf = session / f"playground_smoke_30trials_texture_first_{condition}.txt"
        if condition == "noun_label":
            sf = session / "playground_smoke_30trials_shape_first_noun_label_shiple.txt"
            tf = session / "playground_smoke_30trials_texture_first_noun_label_shiple.txt"
        if condition == "noun_label_AB":
            sf = session / "playground_smoke_30trials_shape_first_noun_label_AB_shiple.txt"
            tf = session / "playground_smoke_30trials_texture_first_noun_label_AB_shiple.txt"
        if not sf.is_file() or not tf.is_file():
            loaded = load_condition_rows(condition, n_trials=30)
            if loaded is None:
                continue
            cond_rows, _ = loaded
        else:
            from scripts.build_playground_results_html import parse_smoke, summarize_pair

            sf_d, tf_d = parse_smoke(sf), parse_smoke(tf)
            cond_rows = [
                summarize_pair(sf_d, tf_d, m)
                for m in sf_d["models"]
                if m in tf_d["models"]
            ]
        for r in cond_rows:
            rows.append(_row_from_gen(r["model"], condition, stim_set, label, r["gen"]))
    return rows


def archive_vs_fixed_table() -> str:
    """Compare archived (mismatched labels) vs fixed AB category gate rates."""
    fixed = load_condition_rows("no_word_category_AB", n_trials=30)
    if fixed is None:
        return '<p class="dim">Fixed AB category logs not ready yet.</p>'
    fixed_rows, _ = fixed
    by_fixed = {r["model"]: r["gen"] for r in fixed_rows}

    arch_sf = ARCHIVE_AB / "session_2026-07-24_farmshare" / (
        "playground_smoke_30trials_shape_first_no_word_category_AB.txt"
    )
    arch_tf = ARCHIVE_AB / "session_2026-07-24_farmshare" / (
        "playground_smoke_30trials_texture_first_no_word_category_AB.txt"
    )
    if not arch_sf.is_file():
        arch_sf = ARCHIVE_AB / "session_2026-07-17_farmshare" / (
            "playground_smoke_30trials_shape_first_no_word_category_AB.txt"
        )
        arch_tf = ARCHIVE_AB / "session_2026-07-17_farmshare" / (
            "playground_smoke_30trials_texture_first_no_word_category_AB.txt"
        )
    if not arch_sf.is_file():
        return '<p class="dim">Archived AB logs missing.</p>'

    from scripts.build_playground_results_html import parse_smoke, summarize_pair

    sf, tf = parse_smoke(arch_sf), parse_smoke(arch_tf)
    models = sorted(set(sf["models"]) & set(tf["models"]))
    parts = [
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th><th>arch trk</th><th>fixed trk</th><th>Δ trk</th>"
        "<th>arch shp</th><th>fixed shp</th><th>Δ shp</th>"
        "<th>arch gate</th><th>fixed gate</th></tr>",
    ]
    for m in models:
        old = summarize_pair(sf, tf, m)["gen"]
        new = by_fixed.get(m)
        if new is None:
            continue
        d_trk = (new["trk"] or 0) - (old["trk"] or 0)
        d_shp = None
        if old["shp_avg"] is not None and new["shp_avg"] is not None:
            d_shp = new["shp_avg"] - old["shp_avg"]
        og = "PASS" if (old["trk"] or 0) >= GATE else "fail"
        ng = "PASS" if (new["trk"] or 0) >= GATE else "fail"
        cls = "pass" if og != ng and ng == "PASS" else ("fail" if og != ng else "")
        parts.append(
            f'<tr class="{cls}"><td class="l">{html.escape(m)}</td>'
            f"<td>{fmt_rate(old['trk'])}</td><td>{fmt_rate(new['trk'])}</td>"
            f"<td>{d_trk:+.2f}</td>"
            f"<td>{fmt_rate(old['shp_avg'])}</td><td>{fmt_rate(new['shp_avg'])}</td>"
            f"<td>{fmt_rate(d_shp) if d_shp is None else f'{d_shp:+.2f}'}</td>"
            f"<td>{og}</td><td>{ng}</td></tr>"
        )
    parts.append("</table></div>")
    return "\n".join(parts)


def pride_summary_table() -> str:
    paths = [
        PLAY / "prompt_pride_debias_2026-07-30.csv",
        PLAY / "prompt_pride_debias_2026-07-17.csv",
    ]
    rows: list[dict] = []
    for p in paths:
        rows.extend(_load_csv(p))
    if not rows:
        return '<p class="dim">No PriDe CSV yet.</p>'
    parts = [
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th><th class=\"l\">Condition</th><th>labels</th>"
        "<th>gen trk</th><th>swap P(shp)</th><th>PriDe SF</th><th>PriDe TF</th><th>gate</th></tr>",
    ]
    for r in rows:
        if r.get("label_set") != "AB" and not str(r.get("condition", "")).endswith("_AB"):
            if r.get("label_set") == "numeric":
                pass  # include all for master view
        gate = "PASS" if float(r.get("gen_tracking", 0)) >= GATE else "fail"
        cls = "pass" if gate == "PASS" else ""
        parts.append(
            f'<tr class="{cls}"><td class="l">{html.escape(r["model"])}</td>'
            f'<td class="l">{html.escape(r["condition"])}</td>'
            f'<td>{html.escape(r.get("label_set", ""))}</td>'
            f'<td>{float(r["gen_tracking"]):.2f}</td>'
            f'<td>{float(r["swap_mean_p_shape"]):.2f}</td>'
            f'<td>{float(r["pride_sf_mean_p_shape"]):.2f}</td>'
            f'<td>{float(r["pride_tf_mean_p_shape"]):.2f}</td>'
            f'<td>{gate}</td></tr>'
        )
    parts.append("</table></div>")
    return "\n".join(parts)


def embedding_table() -> str:
    novel = _merge_embed(
        SESS_PROBE / "embedding_robust.json",
        PROBE / "session_2026-07-17_farmshare" / "embedding_robust_fill.json",
    )
    cue = _merge_embed(
        SESS_PROBE / "embedding_cueconflict.json",
        PROBE / "session_2026-07-17_farmshare" / "embedding_cueconflict_fill.json",
    )
    names = [m for m in MODEL_ORDER if m in novel or m in cue]
    parts = [
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th><th>novel centred shape</th>"
        "<th>Geirhos cue-conflict centred shape</th><th>note</th></tr>",
    ]
    for m in names:
        ns = _embed_shape_rate(novel, m)
        cs = _embed_shape_rate(cue, m)
        note = ""
        if ns is not None and abs(ns - 0.5) < 0.08:
            note = "novel ~chance"
        if cs is not None and cs < 0.45:
            note = (note + "; " if note else "") + "cue-conflict texture-leaning (expected)"
        parts.append(
            f'<tr><td class="l">{html.escape(m)}</td>'
            f"<td>{fmt_rate(ns)}</td><td>{fmt_rate(cs)}</td>"
            f'<td class="l">{html.escape(note)}</td></tr>'
        )
    parts.append("</table></div>")
    return "\n".join(parts)


def interpretation_bullets(behavioral: list[dict]) -> str:
    """Rule-based summary from collected rows (updates as reruns land)."""
    bullets: list[str] = []
    ab_bench = [
        r for r in behavioral
        if r["stim_set"] == "benchmark" and r["label_set"] == "AB"
    ]
    num_bench = [
        r for r in behavioral
        if r["stim_set"] == "benchmark" and r["label_set"] == "numeric"
    ]
    if not ab_bench:
        bullets.append(
            "AB rerun still in progress: benchmark AB cells empty in "
            f"<code>{SESS_AB_FIXED.name}</code>."
        )
    else:
        n_pass_ab = sum(1 for r in ab_bench if r["gate"] == "PASS")
        n_pass_num = sum(1 for r in num_bench if r["gate"] == "PASS")
        bullets.append(
            f"Benchmark generation gate (trk ≥ {GATE}): {n_pass_num}/{len(num_bench)} "
            f"numeric cells vs {n_pass_ab}/{len(ab_bench)} AB cells with fixed image labels."
        )
        large = [r for r in ab_bench if r["model"] in {"qwen3.5-4b", "qwen3.5-9b", "qwen3-vl-8b"}]
        if large:
            for r in large:
                if r["condition"] == "no_word_category_AB":
                    bullets.append(
                        f"{r['model']} category_AB (fixed labels): trk={fmt_rate(r['gen_tracking'])}, "
                        f"shape={fmt_rate(r['gen_shape'])} ({r['gate']})."
                    )

    bullets.append(
        "Embedding readouts on the same triplets sit near chance for novel categories "
        "while Geirhos cue-conflict shows texture preference: shape bias in 2AFC is "
        "downstream of the vision encoder, not encoded as centred cosine shape rate."
    )
    bullets.append(
        "PriDe / swap on saved one_pass logits tests whether generation locks "
        "(low trk) still carry latent shape evidence. Read gen gate and PriDe columns together."
    )
    bullets.append(
        "Archived AB runs (Image 1/2 slots with A/B answer text) are invalid; "
        "use the archive-vs-fixed table once the Jul-30 session completes."
    )
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    PLAY.mkdir(parents=True, exist_ok=True)
    behavioral = collect_behavioral_rows()
    write_csv(behavioral)

    bench_ab_sim = load_condition_rows("no_word_similarity_AB", n_trials=30)
    bench_ab_cat = load_condition_rows("no_word_category_AB", n_trials=30)
    bench_ab_noun = load_condition_rows("noun_label_AB", n_trials=30)

    sections: list[str] = []
    if bench_ab_sim and bench_ab_cat:
        sim_rows, _ = bench_ab_sim
        cat_rows, _ = bench_ab_cat
        sections.append(
            comparison_section_html(
                sim_rows,
                cat_rows,
                section_title="Benchmark AB — similarity vs category (fixed labels, n=30)",
                blurb="July-24 revised category wording. Image slots Image A / Image B.",
                source_note=f"Session: {SESS_AB_FIXED.name}",
            )
        )
    if bench_ab_sim and bench_ab_noun:
        sim_rows, _ = bench_ab_sim
        noun_rows, _ = bench_ab_noun
        sections.append(
            comparison_section_html(
                sim_rows,
                noun_rows,
                section_title="Benchmark AB — similarity vs noun+shiple",
                blurb="Naming contrast under matched A/B labels.",
                source_note=f"Session: {SESS_AB_FIXED.name}",
                left_label="similarity",
                right_label="noun",
            )
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Master interpretation — playground results</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Master interpretation report</h1>
<p class="sub">Consolidates behavioral 2AFC (generation), PriDe-corrected logits,
embedding readouts, benchmark vs Smith stimuli, and numeric vs fixed AB labels.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.
CSV: <code>{OUT_CSV.name}</code>.</p>

<div class="callout good"><b class="t">AB label fix (2026-07-30)</b>
Reruns use <code>Image A:</code> / <code>Image B:</code> slots whenever the prompt asks for A/B.
Pre-2026-07-28 AB logs are archived under
<code>_archived_ab_label_mismatch_pre_2026-07-28/</code>.</div>

<h2>1 · Interpretation summary</h2>
{interpretation_bullets(behavioral)}

<h2>2 · Archive vs fixed AB (category)</h2>
<p>Did matching image-slot labels change validity or shape rate?</p>
{archive_vs_fixed_table()}

<h2>3 · Behavioral AB ladder (benchmark, fixed)</h2>
{"".join(sections) if sections else '<p class="dim">Waiting on session_2026-07-30_farmshare AB logs.</p>'}

<h2>4 · PriDe / swap / full permutation</h2>
{pride_summary_table()}

<h2>5 · Vision encoder readouts</h2>
<p>Novel-category triplets vs Geirhos familiar cue-conflict positive control.</p>
{embedding_table()}

<h2>6 · Related reports</h2>
<ul>
  <li><a href="index.html">Playground index</a></li>
  <li><a href="vision_vs_language_2026-07-17.html">Vision vs language (July 17)</a></li>
  <li><a href="local_models_numeric_and_qwen8_30trials_2026-07-24.html">Numeric ladder (valid baseline)</a></li>
</ul>

<p class="src">Generated by <code>scripts/build_master_interpretation_report.py</code>.</p>
</main>
</body>
</html>
"""
    OUT_HTML.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_CSV} ({len(behavioral)} rows)")


if __name__ == "__main__":
    main()
