#!/usr/bin/env python3
"""HTML report for the edited Geirhos cue-conflict triad sets (n=320).

Reads results/data/cueconflict_{cc_triads,decomposition_triads}_summary.csv
and optional PriDe CSVs. Writes results/playground.results/cueconflict_triads_report.html.

Run after analysis_pipe/cueconflict_summary.py.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results" / "data"
OUT = REPO / "results" / "playground.results" / "cueconflict_triads_report.html"
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
SETS = (
    ("cc_triads", "Stylized foils (primary)"),
    ("decomposition_triads", "Source-photo foils (format confound)"),
)

CSS = """
:root {
  --bg:#fafafa; --fg:#1a1a1a; --muted:#666; --card:#fff; --border:#ddd;
  --green-bg:#e6f4ea; --green-fg:#1e7d3c; --red-bg:#fdecea; --red-fg:#b3261e;
  --amber-bg:#fff4e0; --amber-fg:#8a5a00;
}
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font: 15px/1.45 system-ui, -apple-system, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.2rem; margin: 1.6rem 0 0.5rem; }
.sub, .src, .dim { color: var(--muted); font-size: 0.92rem; }
.tiles { display:flex; flex-wrap:wrap; gap:10px; margin: 12px 0 18px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:10px 14px; min-width:110px; }
.tile .v { font-size:1.35rem; font-weight:650; }
.tile .l { color:var(--muted); font-size:0.8rem; }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:0.85rem; }
th, td { border-bottom:1px solid var(--border); padding:6px 8px; text-align:center; }
th { background:var(--card); position:sticky; top:0; }
td:first-child, th:first-child { text-align:left; white-space:nowrap; }
.pass { background:var(--green-bg); color:var(--green-fg); }
.fail { background:var(--amber-bg); color:var(--amber-fg); }
.lock { background:var(--red-bg); color:var(--red-fg); }
.callout { background:var(--amber-bg); color:var(--amber-fg); padding:10px 12px;
           border-radius:8px; margin: 10px 0 16px; }
code { font-size: 0.9em; }
"""


def load_summary(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: str) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "—"


def row_class(r: dict) -> str:
    if r.get("gate_pass", "").lower() in ("true", "1"):
        return "pass"
    try:
        if float(r.get("pos_first", 0)) >= 0.85 or float(r.get("pos_first", 1)) <= 0.15:
            return "lock"
    except ValueError:
        pass
    return "fail"


def table_html(rows: list[dict]) -> str:
    by = {(r["model"], r["prompt_condition"]): r for r in rows}
    head = "<tr><th>model</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in COND_ORDER) + "</tr>"
    body = []
    for m in MODEL_ORDER:
        cells = [f"<td>{html.escape(m)}</td>"]
        for c in COND_ORDER:
            r = by.get((m, c))
            if not r:
                cells.append("<td>—</td>")
                continue
            tip = (f"trk={fmt(r['tracking'])} shp={fmt(r['shape_rate'])} "
                   f"pos1={fmt(r['pos_first'])}")
            cells.append(
                f'<td class="{row_class(r)}" title="{tip}">{fmt(r["shape_rate"])}'
                f'<br><span style="font-size:0.75em;opacity:.75">trk {fmt(r["tracking"])}</span></td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def section_for(stim_set: str, title: str) -> str:
    summary = DATA / f"cueconflict_{stim_set}_summary.csv"
    pride = DATA / f"cueconflict_{stim_set}_pride.csv"
    if not summary.is_file():
        return f"<h2>{html.escape(title)}</h2><p class='dim'>Missing {summary.name}</p>"
    rows = load_summary(summary)
    n_pass = sum(1 for r in rows if r["gate_pass"].lower() in ("true", "1"))
    pride_note = (
        f"<p>PriDe: <code>{pride.name}</code>.</p>"
        if pride.is_file()
        else "<p class='dim'>PriDe CSV not found.</p>"
    )
    return f"""
<h2>{html.escape(title)}</h2>
<div class="tiles">
  <div class="tile"><div class="v">{len(rows)}</div><div class="l">model × cell</div></div>
  <div class="tile"><div class="v">{n_pass}/{len(rows)}</div><div class="l">gate PASS (trk ≥ {GATE})</div></div>
  <div class="tile"><div class="v">320</div><div class="l">stimuli per cell</div></div>
  <div class="tile"><div class="v">640</div><div class="l">trials per cell</div></div>
</div>
{table_html(rows)}
{pride_note}
"""


def patch_index() -> None:
    if not INDEX.is_file():
        return
    text = INDEX.read_text(encoding="utf-8")
    marker = "cueconflict_triads_report.html"
    if marker in text:
        return
    block = (
        '<h2>2026-08-08 — Cue-conflict triad sets (n=320)</h2>'
        f'<ul><li><a href="{marker}">Cue-conflict triad report</a></li>'
        '<li><a href="../data/cueconflict_cc_triads_summary.csv">cc_triads summary</a></li>'
        '<li><a href="../data/cueconflict_decomposition_triads_summary.csv">'
        "decomposition summary</a></li></ul>"
    )
    if "Full texture grid" in text:
        text = text.replace(
            "<h2>2026-08-02 — Full texture grid",
            block + "<h2>2026-08-02 — Full texture grid",
            1,
        )
    else:
        text = text.replace("<h2>2026-07-30", block + "<h2>2026-07-30", 1)
    INDEX.write_text(text, encoding="utf-8")
    print(f"  patched {INDEX}")


def main() -> int:
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue-conflict triad report</title>
<style>{CSS}</style></head><body><main>
<h1>Edited Geirhos cue-conflict triads</h1>
<p class="sub">Fixed 320-triad subset (20 per class, seed 0), 14 models × 6 cells ×
2 orders. Both sets share the same references; they differ only in the answer
options. Treat <b>cc_triads</b> as primary: decomposition foils mix isolated
objects on white with cluttered photographs, which can inflate apparent shape
bias via image format.</p>
<div class="callout"><b>Congruent trials.</b> About 5% of the subset is within-class
(e.g. dog1-dog2). Those rows stay in the CSVs and are labelled via
<code>stl_id</code> / <code>texture_set</code>; do not average them with conflict
trials without filtering.</div>
{"".join(section_for(s, t) for s, t in SETS)}
<p class="src">Built by <code>scripts/build_cueconflict_report.py</code>.
Raw CSVs: <code>results/model.results/session_cueconflict_triads/</code>.</p>
</main></body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"  wrote {OUT}")
    patch_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
