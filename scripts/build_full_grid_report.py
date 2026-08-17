#!/usr/bin/env python3
"""HTML report for the full texture-grid evaluation (v1, mode A).

Reads results/data/full_grid_v1a_summary.csv (and optional pride / embedding
artifacts) and writes a self-contained report under results/playground.results/
so it shows up next to the 30-set index.

Run:  .venv/bin/python scripts/build_full_grid_report.py
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "data" / "full_grid_v1a_summary.csv"
PRIDE = REPO / "results" / "data" / "full_grid_v1a_pride.csv"
FIG = REPO / "results" / "figures" / "full_grid"
OUT = REPO / "results" / "playground.results" / "full_grid_v1a_report.html"
INDEX = REPO / "results" / "playground.results" / "index.html"
GATE = 0.70

MODEL_ORDER = (
    "smolvlm-256m", "smolvlm",
    "internvl", "internvl-2b", "internvl-8b", "internvl-14b",
    "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b",
)
COND_ORDER = (
    "no_word_similarity", "no_word_category", "noun_label",
    "no_word_similarity_AB", "no_word_category_AB", "noun_label_AB",
)


CSS = """
:root {
  --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --card:#fff; --border:#ddd;
  --green-bg:#e6f4ea; --green-fg:#1e7d3c; --red-bg:#fdecea; --red-fg:#b3261e;
  --amber-bg:#fff4e0; --amber-fg:#8a5a00; --code-bg:#f0f0f0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6; --card:#1f2228; --border:#3a3f47;
    --green-bg:#1d3524; --green-fg:#7ddc94; --red-bg:#3b2220; --red-fg:#f28b82;
    --amber-bg:#3a2f18; --amber-fg:#fdd663; --code-bg:#2a2e35;
  }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:1.55rem; margin:0 0 4px; }
h2 { font-size:1.15rem; margin:32px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px; }
p.sub { color:var(--muted); margin-top:4px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:18px 0; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }
.tile .v { font-size:1.45rem; font-weight:700; }
.tile .l { font-size:0.8rem; color:var(--muted); }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; margin:12px 0; }
table { border-collapse:collapse; width:100%; font-size:0.84rem; background:var(--card); }
th,td { padding:6px 10px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
tr.pass td { background:var(--green-bg); }
tr.lock td { background:var(--red-bg); }
tr.fail td { background:var(--amber-bg); }
.figs { display:grid; grid-template-columns:1fr; gap:18px; margin:16px 0; }
.figs img { max-width:100%; height:auto; border:1px solid var(--border); border-radius:6px; background:var(--card); }
.callout { border-left:4px solid var(--amber-fg); padding:10px 14px; margin:14px 0; background:var(--card); border-radius:6px; }
.src { color:var(--muted); font-size:0.8rem; }
code { background:var(--code-bg); border-radius:4px; padding:1px 5px; font-size:0.85em; }
"""


def load_summary() -> list[dict]:
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: str | float, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def row_class(r: dict) -> str:
    if r["gate_pass"].lower() in ("true", "1"):
        return "pass"
    p = float(r["pos_first"])
    if p >= 0.9 or p <= 0.1:
        return "lock"
    return "fail"


def table_html(rows: list[dict]) -> str:
    by = {(r["model"], r["prompt_condition"]): r for r in rows}
    head = "<tr><th>model</th>" + "".join(
        f"<th>{html.escape(c.replace('no_word_', '').replace('_', ' '))}</th>"
        for c in COND_ORDER
    ) + "</tr>"
    body = []
    for m in MODEL_ORDER:
        cells = [f"<td>{html.escape(m)}</td>"]
        for c in COND_ORDER:
            r = by.get((m, c))
            if not r:
                cells.append("<td>—</td>")
                continue
            cls = row_class(r)
            tip = (f"trk={fmt(r['tracking'])} shp={fmt(r['shape_rate'])} "
                   f"pos1={fmt(r['pos_first'])}")
            cells.append(
                f'<td class="{cls}" title="{tip}">{fmt(r["shape_rate"])}'
                f'<br><span style="font-size:0.75em;opacity:.75">trk {fmt(r["tracking"])}</span></td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def fig_block() -> str:
    names = [
        ("fig1_validity_gates.png", "Validity gates (tracking)"),
        ("fig2_shape_bias.png", "Shape bias by framing (numeric labels; see fig4)"),
        ("fig3_naming_effect.png", "Naming effect, double-gated pairs ordered by effect size"),
        ("fig4_label_format.png", "Label format screen: 1/2 vs A/B on tracking and on P(shape)"),
        ("fig5_by_shape_texture.png", "By shape / texture"),
        ("fig6_position_bias_correction.png", "Generated answer vs logit argmax vs order-swap correction, numeric labels (similarity / category / noun)"),
        ("fig6b_position_bias_correction_ab.png", "Same three bars as fig6, A/B labels only"),
        ("fig6c_correction_by_label_format.png", "How far PriDe moves numeric vs A/B cells off the raw logit argmax"),
        ("fig6d_generation_follows_logit_ab.png", "A/B: |generation − raw logit| vs |generation − swap-corrected|; shorter bar = closer agreement"),
        ("fig7_vision_vs_behavior.png", "Vision vs behavior by framing (numeric, proj_mean)"),
        ("fig7b_sets_behavior.png", "Behavior across stimulus sets by framing (ours / Smith / cc_triads / decomposition)"),
        ("fig7b_sets_emb_vs_behavior.png", "Vision tower vs behavior across stimulus sets by framing"),
        ("fig8_embedding_layers.png", "Embedding shape rate by distinct pooling layer, and the paired difference from proj_mean"),
        ("fig9_logit_vs_generation.png", "Generated answers vs option logits: shape rate, the tracking gate on each path, and position bias"),
        ("fig9b_logit_vs_generation_pride.png", "Same as fig9 with logit P(shape) genuinely PriDe-corrected on held-out trials; panel B shows the correction move; panel C is the raw position bias"),
        ("fig10_human_model_items.png", "Item-level human-model agreement on the 114 shared grid triads"),
    ]
    parts = []
    for name, caption in names:
        path = FIG / name
        if path.is_file():
            # Relative from playground.results/
            rel = f"../figures/full_grid/{name}"
            parts.append(
                f'<figure><img src="{rel}" alt="{html.escape(caption)}">'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
        else:
            parts.append(f"<p class='src'>Pending figure: <code>{name}</code></p>")
    return '<div class="figs">' + "".join(parts) + "</div>"


def pride_note() -> str:
    if PRIDE.is_file():
        return f"<p>PriDe / order-swap table: <code>{PRIDE.name}</code>.</p>"
    return (
        '<div class="callout"><b>PriDe pending.</b> '
        "Submit <code>scripts/run_full_grid_logit_v1a.sbatch</code>, then "
        "<code>.venv/bin/python analysis_pipe/full_grid_pride.py</code>.</div>"
    )


def patch_index() -> None:
    if not INDEX.is_file():
        return
    text = INDEX.read_text(encoding="utf-8")
    marker = "full_grid_v1a_report.html"
    if marker in text:
        return
    block = (
        '<h2>2026-08-02 — Full texture grid (v1A, n=1,140)</h2>'
        f'<ul><li><a href="{marker}">Full-grid report</a> '
        '<span class="dim">(ready)</span></li>'
        '<li><a href="../data/full_grid_v1a_summary.csv">Summary CSV</a></li>'
        '<li><a href="../figures/full_grid/">Figures</a></li></ul>'
    )
    text = text.replace("<h2>2026-07-30", block + "<h2>2026-07-30", 1)
    INDEX.write_text(text, encoding="utf-8")
    print(f"  patched {INDEX}")


def main() -> int:
    if not SUMMARY.is_file():
        raise SystemExit(f"Missing {SUMMARY}; run full_grid_summary.py first")
    rows = load_summary()
    n_pass = sum(1 for r in rows if r["gate_pass"].lower() in ("true", "1"))
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full grid v1A report</title>
<style>{CSS}</style></head><body><main>
<h1>Full texture grid — v1, mode A</h1>
<p class="sub">30 shapes × 38 textures = 1,140 stimuli × 2 orders × 6 cells × 14 models
= 191,520 generation trials. Raw CSVs:
<code>results/model.results/session_full_grid_v1a/</code>.</p>
<div class="tiles">
  <div class="tile"><div class="v">{len(rows)}</div><div class="l">model × cell</div></div>
  <div class="tile"><div class="v">{n_pass}/{len(rows)}</div><div class="l">gate PASS (trk ≥ {GATE})</div></div>
  <div class="tile"><div class="v">1,140</div><div class="l">stimuli per cell</div></div>
  <div class="tile"><div class="v">2,280</div><div class="l">trials per cell</div></div>
</div>
<p>Cell values below show <b>shape rate</b> (main) and tracking (small).
Green = gate pass; amber = fail; red = position/label lock.</p>
{table_html(rows)}
{pride_note()}
<h2>Figures</h2>
{fig_block()}
<p class="src">Built by <code>scripts/build_full_grid_report.py</code>.
30-set playground reports are unchanged and still read smoke <code>.txt</code> logs.</p>
</main></body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"  wrote {OUT}")
    patch_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
