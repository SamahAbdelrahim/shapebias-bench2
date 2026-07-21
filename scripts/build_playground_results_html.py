#!/usr/bin/env python3
"""Build HTML report for local-model playground smoke + probe-era readouts.

Writes under results/playground.results/. Re-run after the AB-prompt smoke lands.
"""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAY = REPO / "results" / "playground.results"
SESS_SMOKE = PLAY / "session_2026-07-17_farmshare"
SESS_PROBE = REPO / "results" / "probe.results" / "session_2026-07-10_farmshare"
FARM_HTML = REPO / "farmshare" / "probe-experiment-results.html"

CSS = """
  :root {
    --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --card: #fff;
    --border: #ddd; --green-bg: #e6f4ea; --green-fg: #1e7d3c;
    --red-bg: #fdecea; --red-fg: #b3261e; --amber-bg: #fff4e0;
    --amber-fg: #8a5a00; --blue-bg: #e8f0fe; --blue-fg: #1a56b0;
    --code-bg: #f0f0f0;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16181c; --fg: #e8e8e8; --muted: #9aa0a6; --card: #1f2228;
      --border: #3a3f47; --green-bg: #1d3524; --green-fg: #7ddc94;
      --red-bg: #3b2220; --red-fg: #f28b82; --amber-bg: #3a2f18;
      --amber-fg: #fdd663; --blue-bg: #1c2a41; --blue-fg: #8ab4f8;
      --code-bg: #2a2e35;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  h2 { font-size: 1.2rem; margin: 36px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.02rem; margin: 20px 0 8px; }
  p.sub { color: var(--muted); margin-top: 4px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }
  .tile { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .tile .v { font-size: 1.5rem; font-weight: 700; }
  .tile .l { font-size: 0.82rem; color: var(--muted); }
  .callout { border-left: 4px solid; border-radius: 6px; padding: 12px 16px; margin: 14px 0; background: var(--card); }
  .callout.danger { border-color: var(--red-fg); }
  .callout.warn   { border-color: var(--amber-fg); }
  .callout.info   { border-color: var(--blue-fg); }
  .callout.good   { border-color: var(--green-fg); }
  .callout b.t { display: block; margin-bottom: 4px; }
  .tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; }
  table { border-collapse: collapse; width: 100%; font-size: 0.86rem; background: var(--card); }
  th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
  th { position: sticky; top: 0; background: var(--card); text-align: right; }
  th:first-child, td:first-child, th.l, td.l { text-align: left; }
  tr.pass  td { background: var(--green-bg); }
  tr.pass  td:first-child { color: var(--green-fg); font-weight: 600; }
  tr.lock  td { background: var(--red-bg); }
  tr.fail  td { background: var(--amber-bg); }
  td.dim, span.dim { color: var(--muted); }
  code { background: var(--code-bg); border-radius: 4px; padding: 1px 5px; font-size: 0.85em; }
  .src { color: var(--muted); font-size: 0.8rem; margin: 6px 0 0; }
  ul.key { font-size: 0.85rem; color: var(--muted); padding-left: 18px; }
  ul.key b { color: var(--fg); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; }
"""


def parse_smoke(path: Path) -> dict[str, dict]:
    text = path.read_text()
    meta: dict[str, str] = {}
    for key in (
        "Prompt condition:",
        "User prompt:",
        "Prompt:",
        "System prompt (LOCAL_VLM_SYSTEM_PROMPT):",
        "Trials:",
        "Paths:",
        "Timestamp:",
    ):
        m = re.search(rf"^{re.escape(key)}\s*(.*)$", text, re.M)
        if m:
            meta[key.rstrip(":")] = m.group(1).strip()
    if "User prompt" in meta and "Prompt" not in meta:
        meta["Prompt"] = meta["User prompt"]
    if "System prompt (LOCAL_VLM_SYSTEM_PROMPT)" in meta:
        meta["System prompt"] = meta["System prompt (LOCAL_VLM_SYSTEM_PROMPT)"]
    models: dict[str, dict] = {}
    for block in re.split(r"\n={72}\nMODEL: ", text)[1:]:
        name = block.split("\n", 1)[0].strip()
        gens = re.findall(r"^Generation pick: ([AB12])", block, re.M)
        two, one = [], []
        mode = None
        for line in block.splitlines():
            if "Logit scoring [two_pass]" in line:
                mode = "two"
            elif "Logit scoring [one_pass]" in line:
                mode = "one"
            elif "Logit scoring (A / B)" in line:
                mode = "two"
            elif line.startswith("Logit pick:"):
                m = re.search(r"Logit pick: ([AB12])", line)
                if m and mode == "two":
                    two.append(m.group(1))
                if m and mode == "one":
                    one.append(m.group(1))
                mode = None
        n = len(gens)
        if not one:
            one = list(two)
        models[name] = {
            "gen": gens,
            "two": two[:n] if two else [None] * n,
            "one": one[:n] if one else [None] * n,
        }
    return {"meta": meta, "models": models}


def shape_letter(order: str) -> str:
    return "A" if order == "shape_first" else "B"


def feature(pick: str | None, order: str) -> str | None:
    if pick is None:
        return None
    shape_label = (
        shape_letter(order)
        if pick in {"A", "B"}
        else ("1" if order == "shape_first" else "2")
    )
    return "shape" if pick == shape_label else "texture"


def rate(xs: list[bool]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def fmt_rate(r: float | None, digits: int = 2) -> str:
    if r is None:
        return "—"
    return f"{r:.{digits}f}"


def fmt_picks(xs: list) -> str:
    return " ".join(x if x else "?" for x in xs)


def row_class(trk: float | None, pos_a: float | None) -> str:
    if trk is None:
        return ""
    if pos_a is not None and (pos_a >= 0.9 or pos_a <= 0.1):
        return "lock"
    if trk >= 0.70:
        return "pass"
    return "fail"


def summarize_pair(sf: dict, tf: dict, model: str) -> dict:
    """sf/tf are model dicts with gen/two/one lists; order keys are shape_first / texture_first."""
    out = {"model": model}
    for path in ("gen", "two", "one"):
        sf_p = sf["models"][model][path]
        tf_p = tf["models"][model][path]
        n = min(len(sf_p), len(tf_p))
        sf_feat = [feature(sf_p[i], "shape_first") for i in range(n)]
        tf_feat = [feature(tf_p[i], "texture_first") for i in range(n)]
        shp_sf = rate([f == "shape" for f in sf_feat])
        shp_tf = rate([f == "shape" for f in tf_feat])
        shp_avg = None if shp_sf is None or shp_tf is None else 0.5 * (shp_sf + shp_tf)
        trk = rate([sf_feat[i] == tf_feat[i] for i in range(n)])
        observed = [p for p in (sf_p[:n] + tf_p[:n]) if p]
        first_label = "A" if any(p in {"A", "B"} for p in observed) else "1"
        pos_a = rate([p == first_label for p in observed])
        g2 = rate([sf_p[i] == sf["models"][model]["two"][i] for i in range(n)])
        # gen==two on texture_first alone
        g2_tf = rate(
            [
                tf_p[i] == tf["models"][model]["two"][i]
                for i in range(n)
                if tf["models"][model]["two"][i] is not None
            ]
        )
        t1 = rate(
            [
                sf["models"][model]["two"][i] == sf["models"][model]["one"][i]
                for i in range(n)
                if sf["models"][model]["one"][i] is not None
            ]
        )
        out[path] = {
            "sf_picks": sf_p[:n],
            "tf_picks": tf_p[:n],
            "shp_sf": shp_sf,
            "shp_tf": shp_tf,
            "shp_avg": shp_avg,
            "trk": trk,
            "pos_a": pos_a,
            "gen_eq_two_sf": g2,
            "gen_eq_two_tf": g2_tf,
            "two_eq_one_sf": t1,
            "n": n,
        }
    return out


def smoke_section_html(title: str, prompt_note: str, rows: list[dict], sources: str) -> str:
    tiles_trk = [r["gen"]["trk"] for r in rows if r["gen"]["trk"] is not None]
    n_pass = sum(1 for t in tiles_trk if t >= 0.70)
    n_trials = rows[0]["gen"]["n"] if rows else 0
    n_note = (
        "rates are descriptive; gate is underpowered"
        if n_trials <= 5
        else "full stimulus set per order"
    )
    parts = [
        f"<h2>{html.escape(title)}</h2>",
        f"<p>{html.escape(prompt_note)}</p>",
        '<div class="tiles">',
        f'<div class="tile"><div class="v">{len(rows)}</div><div class="l">models × {n_trials} trials × 2 orders</div></div>',
        f'<div class="tile"><div class="v">{n_pass}/{len(rows)}</div><div class="l">gen tracking ≥ 0.70 (image tracks across orders)</div></div>',
        f'<div class="tile"><div class="v">n={n_trials}</div><div class="l">per order — {n_note}</div></div>',
        "</div>",
        "<h3>Per-order picks (gen / two_pass / one_pass)</h3>",
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th><th class=\"l\">Order</th>"
        "<th class=\"l\">gen</th><th class=\"l\">two_pass</th><th class=\"l\">one_pass</th>"
        "<th>gen==two</th></tr>",
    ]
    for r in rows:
        m = r["model"]
        for order, key in (("shape_first (GT=A=shape)", "sf_picks"), ("texture_first (GT=B=shape)", "tf_picks")):
            g = r["gen"][key]
            t = r["two"][key]
            o = r["one"][key]
            eq = sum(a == b for a, b in zip(g, t) if b is not None)
            n = len(g)
            parts.append(
                f"<tr><td class=\"l\">{html.escape(m)}</td><td class=\"l\">{order}</td>"
                f"<td class=\"l mono\">{fmt_picks(g)}</td>"
                f"<td class=\"l mono\">{fmt_picks(t)}</td>"
                f"<td class=\"l mono\">{fmt_picks(o)}</td>"
                f"<td>{eq}/{n}</td></tr>"
            )
    parts.append("</table></div>")

    parts += [
        "<h3>Shape rates + validity (from generation picks unless noted)</h3>",
        "<p>Shape rate = fraction of trials choosing the shape-match image. "
        "Tracking = same feature (shape vs texture) chosen under both orders for the same trial index. "
        "PosFirst = rate of emitting the first label (A or 1) across both orders "
        "(position / label lock).</p>",
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th><th class=\"l\">path</th>"
        "<th>shp SF</th><th>shp TF</th><th>shp avg</th>"
        "<th>tracking</th><th>PosFirst</th><th>gate</th></tr>",
    ]
    for r in rows:
        for path, label in (("gen", "gen"), ("two", "two_pass"), ("one", "one_pass")):
            d = r[path]
            trk = d["trk"]
            pos = d["pos_a"]
            cls = row_class(trk, pos)
            gate = "PASS" if trk is not None and trk >= 0.70 else "fail"
            shp_cells = []
            for k in ("shp_sf", "shp_tf", "shp_avg"):
                val = fmt_rate(d[k])
                if trk is not None and trk < 0.70:
                    shp_cells.append(f'<td class="dim">({val})</td>')
                else:
                    shp_cells.append(f"<td>{val}</td>")
            parts.append(
                f'<tr class="{cls}"><td class="l">{html.escape(r["model"])}</td>'
                f'<td class="l">{label}</td>'
                + "".join(shp_cells)
                + f"<td>{fmt_rate(trk)}</td><td>{fmt_rate(pos)}</td><td>{gate}</td></tr>"
            )
    parts.append("</table></div>")
    parts.append(
        '<ul class="key">'
        "<li><b>shp SF / TF / avg</b> — shape-choice rate under shape_first, texture_first, and their mean</li>"
        "<li><b>tracking</b> — feature consistency across the two orderings (image tracking)</li>"
        "<li><b>PosFirst</b> — first-label rate (A or 1); ≥0.9 or ≤0.1 marked as hard lock (red)</li>"
        "<li><b>gate</b> — PASS when tracking ≥ 0.70 (same rule as the probe; n=5 is underpowered)</li>"
        "<li>Parenthesized shape rates = gate failed; shown for completeness only</li>"
        "</ul>"
    )
    parts.append(f'<p class="src">Sources: {html.escape(sources)}</p>')
    return "\n".join(parts)


def parse_embedding_summary(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    if not path.is_file():
        return rows
    in_sum = False
    for line in path.read_text().splitlines():
        if line.startswith("model ") and "raw" in line:
            in_sum = True
            continue
        if in_sum and line.strip() and not line.startswith("JSON") and not line.startswith("Text"):
            parts = line.split()
            if len(parts) >= 4:
                rows.append((parts[0], parts[2], parts[3]))
    return rows


def pride_table_html() -> str:
    csv_path = SESS_PROBE / "pride_debias.csv"
    if not csv_path.is_file():
        return "<p class=\"dim\">PriDe CSV not found.</p>"
    rows = list(csv.DictReader(csv_path.open()))
    # keep a readable subset: gate True or notable models
    out = [
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">source</th><th class=\"l\">model</th><th class=\"l\">cond</th>"
        "<th class=\"l\">lbl</th><th>gate</th><th>gSwap</th><th>lSwap</th>"
        "<th>lPerm</th><th>PriSF</th><th>PriTF</th><th>prior1</th></tr>",
    ]
    for r in rows:
        gate = str(r.get("gate", "")).lower() in {"true", "1", "pass"}
        cls = "pass" if gate else "fail"
        out.append(
            f'<tr class="{cls}"><td class="l">{html.escape(r["source"])}</td>'
            f'<td class="l">{html.escape(r["model"])}</td>'
            f'<td class="l">{html.escape(r["condition"])}</td>'
            f'<td class="l">{html.escape(r["label_set"])}</td>'
            f'<td>{"PASS" if gate else "fail"}</td>'
            f'<td>{float(r["gen_swap"]):.2f}</td>'
            f'<td>{float(r["log_swap"]):.2f}</td>'
            f'<td>{float(r["log_fullperm"]):.2f}</td>'
            f'<td>{float(r["log_pride_sf"]):.2f}</td>'
            f'<td>{float(r["log_pride_tf"]):.2f}</td>'
            f'<td>{float(r["prior_first"]):.2f}</td></tr>'
        )
    out.append("</table></div>")
    return "\n".join(out)


def build_similarity_report() -> str:
    rows = load_similarity_rows()

    emb = parse_embedding_summary(SESS_PROBE / "embedding_readout.txt")
    emb_rows = "".join(
        f'<tr><td class="l">{html.escape(m)}</td><td>{raw}</td><td>{cen}</td></tr>'
        for m, raw, cen in emb
    )

    smoke_html = smoke_section_html(
        "1 · Playground smoke (similarity prompt) — July 17, 2026",
        "Prompt: “You are given three images… more similar to the reference… A or B.” "
        "Job 1642256 dual-path; qwen3.5 rows use the post-fix resmoke (1642264) so "
        "two_pass matches generate’s chat-template path.",
        rows,
        "session_2026-07-17_farmshare/playground_smoke_5trials_*.txt; "
        "qwen3.5 from playground_smoke_qwen35_resmoke_*.txt",
    )

    # quick headlines from gen path
    lockish = [r for r in rows if (r["gen"]["pos_a"] or 0) >= 0.9 or (r["gen"]["pos_a"] or 1) <= 0.1]
    passish = [r for r in rows if (r["gen"]["trk"] or 0) >= 0.70]

    ab_ready = (SESS_SMOKE / "playground_smoke_5trials_shape_first_no_word_category_AB.txt").is_file() and (
        SESS_SMOKE / "playground_smoke_5trials_texture_first_no_word_category_AB.txt"
    ).is_file()
    powered_ready = (
        SESS_SMOKE / "playground_smoke_30trials_shape_first_no_word_similarity_AB.txt"
    ).is_file() and (
        SESS_SMOKE / "playground_smoke_30trials_shape_first_no_word_category_AB.txt"
    ).is_file()
    if powered_ready:
        section2 = """
<h2>2 · Powered prompt compare (n=30)</h2>
<div class="callout good"><b class="t">Ready.</b>
<a href="local_models_prompt_compare_30trials_2026-07-17.html"><code>local_models_prompt_compare_30trials_2026-07-17.html</code></a>
compares <code>no_word_similarity_AB</code> vs <code>no_word_category_AB</code> on all 30 trials.
n=5 AB smoke:
<a href="local_models_smoke_no_word_category_AB_2026-07-17.html"><code>local_models_smoke_no_word_category_AB_2026-07-17.html</code></a>.
</div>
"""
    elif ab_ready:
        section2 = """
<h2>2 · <code>no_word_category_AB</code> smoke (n=5) · n=30 pending</h2>
<div class="callout info"><b class="t">n=5 ready; n=30 in progress or queued.</b>
Companion:
<a href="local_models_smoke_no_word_category_AB_2026-07-17.html"><code>local_models_smoke_no_word_category_AB_2026-07-17.html</code></a>.
Re-run <code>scripts/build_playground_results_html.py</code> after job
<code>scripts/run_prompt_compare_30.sbatch</code> finishes to write the powered page.
</div>
"""
    else:
        section2 = """
<h2>2 · Pending — <code>no_word_category_AB</code> smoke</h2>
<div class="callout warn"><b class="t">Not in yet.</b>
Re-run this script after AB logs land. Output:
<code>local_models_smoke_no_word_category_AB_2026-07-17.html</code>.</div>
"""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local VLM playground smoke + probe readouts — 2026-07-17</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Local VLM playground smoke + probe-era readouts</h1>
<p class="sub">FarmShare · similarity-prompt smoke (July 17) plus July 10 probe / embedding / PriDe
session. Sibling page for <code>no_word_category_AB</code> builds when those logs exist.
Glossary of every column: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.
Style matches <code>farmshare/probe-experiment-results.html</code>.</p>

<div class="tiles">
  <div class="tile"><div class="v">{len(rows)}</div><div class="l">local models in smoke</div></div>
  <div class="tile"><div class="v">{len(passish)}/{len(rows)}</div><div class="l">gen tracking ≥ 0.70 on n=5×2</div></div>
  <div class="tile"><div class="v">{len(lockish)}</div><div class="l">hard PosA lock (≥0.9 or ≤0.1)</div></div>
  <div class="tile"><div class="v">~0.5</div><div class="l">embedding centred shape (July 10 ladder)</div></div>
</div>

<div class="callout info"><b class="t">How to read the smoke tables.</b>
With only 5 trials per order, tracking/gate numbers are descriptive. A model that always says “A”
gets tracking 0 and PosA 1.0. Shape rates in parentheses failed the tracking gate.</div>

{smoke_html}
{section2}
<h2>3 · July 10 probe session (full behavioral cells)</h2>
<p>The 24-cell probe + scaling ladder with swap correction and gates live in the existing report
(copied next to this file as <code>probe-experiment-results.html</code>). Open that page for
gTrk / lTrk / gShp / lShp / gPosA on 30 stimuli.</p>
<p class="src">Also: <code>results/probe.results/session_2026-07-10_farmshare/probe_experiment.{{txt,json}}</code>,
<code>probe_scaling_noun.json</code>.</p>

<h2>4 · Embedding readout (centred shape ≈ chance)</h2>
<p>Same 30 stimuli; no options / no tokens, so position bias is structurally impossible.</p>
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th>raw shape</th><th>centred shape</th></tr>
{emb_rows}
</table></div>
<p class="src">Source: <code>results/probe.results/session_2026-07-10_farmshare/embedding_readout.txt</code>.
Robustness / Geirhos positive control: <code>embedding_robust.txt</code>, <code>embedding_cueconflict.txt</code>.</p>

<h2>5 · PriDe / raw vs corrected</h2>
<p>Single-order raw rates vs swap / full permutation / PriDe on saved probe logits.</p>
{pride_table_html()}
<p class="src">Source: <code>pride_debias.{{csv,txt}}</code> in the July 10 probe session folder.</p>

<p class="src" style="margin-top:28px">Generated by <code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    return body


def load_similarity_rows() -> list[dict]:
    """Similarity-prompt smoke with qwen3.5 post-fix resmoke rows when present."""
    sf = parse_smoke(SESS_SMOKE / "playground_smoke_5trials_shape_first.txt")
    tf = parse_smoke(SESS_SMOKE / "playground_smoke_5trials_texture_first.txt")
    q_sf_p = SESS_SMOKE / "playground_smoke_qwen35_resmoke_shape_first.txt"
    q_tf_p = SESS_SMOKE / "playground_smoke_qwen35_resmoke_texture_first.txt"
    if q_sf_p.is_file() and q_tf_p.is_file():
        q_sf, q_tf = parse_smoke(q_sf_p), parse_smoke(q_tf_p)
        for m in ("qwen3.5-0.8b", "qwen3.5-4b"):
            if m in q_sf["models"]:
                sf["models"][m] = q_sf["models"][m]
            if m in q_tf["models"]:
                tf["models"][m] = q_tf["models"][m]
    models = list(sf["models"].keys())
    return [summarize_pair(sf, tf, m) for m in models]


def find_smoke_pair(prompt_condition: str, n_trials: int | None = None) -> tuple[Path, Path] | None:
    """Locate shape_first / texture_first logs for a prompt condition.

    When ``n_trials`` is set, only exact
    ``playground_smoke_{n}trials_*_{condition}.txt`` filenames match.
    Legacy similarity n=5 files omit the condition suffix (n_trials=None only).
    """
    sf_path = tf_path = None
    for sess in sorted(PLAY.glob("session_*")):
        if n_trials is not None:
            sf_c = sess / f"playground_smoke_{n_trials}trials_shape_first_{prompt_condition}.txt"
            tf_c = sess / f"playground_smoke_{n_trials}trials_texture_first_{prompt_condition}.txt"
            if sf_c.is_file() and tf_c.is_file():
                return sf_c, tf_c
            continue
        for p in sess.glob("playground_smoke_*shape_first*.txt"):
            text = p.read_text(encoding="utf-8", errors="replace")
            name_ok = prompt_condition in p.name
            meta_ok = f"Prompt condition: {prompt_condition}" in text
            legacy_sim = (
                prompt_condition == "no_word_similarity_AB"
                and p.name.endswith("shape_first.txt")
                and "Prompt condition:" not in text
                and "resmoke" not in p.name
            )
            if name_ok or meta_ok or legacy_sim:
                sf_path = p
        for p in sess.glob("playground_smoke_*texture_first*.txt"):
            text = p.read_text(encoding="utf-8", errors="replace")
            name_ok = prompt_condition in p.name
            meta_ok = f"Prompt condition: {prompt_condition}" in text
            legacy_sim = (
                prompt_condition == "no_word_similarity_AB"
                and p.name.endswith("texture_first.txt")
                and "Prompt condition:" not in text
                and "resmoke" not in p.name
            )
            if name_ok or meta_ok or legacy_sim:
                tf_path = p
    if sf_path and tf_path:
        return sf_path, tf_path
    return None


def load_condition_rows(
    prompt_condition: str,
    n_trials: int | None = None,
    *,
    merge_qwen35_resmoke: bool = False,
) -> tuple[list[dict], tuple[Path, Path]] | None:
    pair = find_smoke_pair(prompt_condition, n_trials=n_trials)
    if pair is None and n_trials is not None:
        pair = find_smoke_pair(prompt_condition, n_trials=None)
    if pair is None:
        return None
    sf_path, tf_path = pair
    sf, tf = parse_smoke(sf_path), parse_smoke(tf_path)
    if merge_qwen35_resmoke:
        q_sf_p = SESS_SMOKE / "playground_smoke_qwen35_resmoke_shape_first.txt"
        q_tf_p = SESS_SMOKE / "playground_smoke_qwen35_resmoke_texture_first.txt"
        if q_sf_p.is_file() and q_tf_p.is_file():
            q_sf, q_tf = parse_smoke(q_sf_p), parse_smoke(q_tf_p)
            for m in ("qwen3.5-0.8b", "qwen3.5-4b"):
                if m in q_sf["models"]:
                    sf["models"][m] = q_sf["models"][m]
                if m in q_tf["models"]:
                    tf["models"][m] = q_tf["models"][m]
    models = [m for m in sf["models"] if m in tf["models"]]
    return [summarize_pair(sf, tf, m) for m in models], (sf_path, tf_path)


def comparison_section_html(
    sim_rows: list[dict],
    ab_rows: list[dict],
    *,
    section_title: str,
    blurb: str,
    source_note: str,
    left_label: str = "sim",
    right_label: str = "AB",
) -> str:
    """Side-by-side generation metrics for two prompt conditions."""
    by_sim = {r["model"]: r for r in sim_rows}
    by_ab = {r["model"]: r for r in ab_rows}
    models = [m for m in by_sim if m in by_ab]
    n = sim_rows[0]["gen"]["n"] if models else 0
    parts = [
        f"<h2>{html.escape(section_title)}</h2>",
        f"<p>{blurb}</p>",
        '<div class="tablewrap"><table>',
        "<tr><th class=\"l\">Model</th>"
        f"<th>{html.escape(left_label)} shp avg</th>"
        f"<th>{html.escape(right_label)} shp avg</th><th>Δ avg</th>"
        f"<th>{html.escape(left_label)} trk</th>"
        f"<th>{html.escape(right_label)} trk</th><th>Δ trk</th>"
        f"<th>{html.escape(left_label)} PosFirst</th>"
        f"<th>{html.escape(right_label)} PosFirst</th>"
        f"<th>{html.escape(left_label)} gate</th>"
        f"<th>{html.escape(right_label)} gate</th>"
        "<th class=\"l\">pick changes (SF / TF)</th></tr>",
    ]
    n_gate_flip = 0
    n_any_pick_change = 0
    for m in models:
        s, a = by_sim[m]["gen"], by_ab[m]["gen"]
        d_avg = None if s["shp_avg"] is None or a["shp_avg"] is None else a["shp_avg"] - s["shp_avg"]
        d_trk = None if s["trk"] is None or a["trk"] is None else a["trk"] - s["trk"]
        s_gate = "PASS" if (s["trk"] or 0) >= 0.70 else "fail"
        a_gate = "PASS" if (a["trk"] or 0) >= 0.70 else "fail"
        if s_gate != a_gate:
            n_gate_flip += 1
        sf_chg = sum(x != y for x, y in zip(s["sf_picks"], a["sf_picks"]))
        tf_chg = sum(x != y for x, y in zip(s["tf_picks"], a["tf_picks"]))
        if sf_chg or tf_chg:
            n_any_pick_change += 1
        cls = ""
        if s_gate != a_gate:
            cls = "fail" if a_gate == "fail" else "pass"
        parts.append(
            f'<tr class="{cls}"><td class="l">{html.escape(m)}</td>'
            f"<td>{fmt_rate(s['shp_avg'])}</td><td>{fmt_rate(a['shp_avg'])}</td>"
            f"<td>{fmt_rate(d_avg) if d_avg is None else f'{d_avg:+.2f}'}</td>"
            f"<td>{fmt_rate(s['trk'])}</td><td>{fmt_rate(a['trk'])}</td>"
            f"<td>{fmt_rate(d_trk) if d_trk is None else f'{d_trk:+.2f}'}</td>"
            f"<td>{fmt_rate(s['pos_a'])}</td><td>{fmt_rate(a['pos_a'])}</td>"
            f"<td>{s_gate}</td><td>{a_gate}</td>"
            f"<td class=\"l\">{sf_chg}/{len(s['sf_picks'])} SF · {tf_chg}/{len(s['tf_picks'])} TF</td></tr>"
        )
    parts.append("</table></div>")

    if n_any_pick_change == 0 and n_gate_flip == 0:
        call = (
            f'<div class="callout good"><b class="t">No material change on n={n}.</b> '
            "Every model kept the same generation picks under both prompts, so shape rates, "
            "tracking, PosA, and gate labels match.</div>"
        )
    elif n_gate_flip == 0:
        call = (
            f'<div class="callout info"><b class="t">Some picks moved; gate labels did not.</b> '
            f"{n_any_pick_change}/{len(models)} models changed at least one generation pick. "
            "Tracking ≥ 0.70 status is unchanged for every model.</div>"
        )
    else:
        call = (
            f'<div class="callout warn"><b class="t">Gate labels flipped for {n_gate_flip} model(s).</b> '
            f"{n_any_pick_change}/{len(models)} models changed at least one generation pick. "
            "Read the Δ columns before treating either prompt as interchangeable.</div>"
        )
    parts.insert(2, call)
    parts.append(f'<p class="src">{source_note}</p>')
    return "\n".join(parts)


def try_build_powered_prompt_compare() -> Path | None:
    """Build the n=30 prompt comparison; add fixed-word rows when available."""
    sim = load_condition_rows("no_word_similarity_AB", n_trials=30)
    ab = load_condition_rows("no_word_category_AB", n_trials=30)
    if sim is None or ab is None:
        return None
    sim_rows, (sim_sf, sim_tf) = sim
    ab_rows, (ab_sf, ab_tf) = ab
    # Prefer matching n if one side still fell back
    n_sim = sim_rows[0]["gen"]["n"] if sim_rows else 0
    n_ab = ab_rows[0]["gen"]["n"] if ab_rows else 0
    if n_sim < 30 or n_ab < 30:
        return None

    pass_sim = sum(1 for r in sim_rows if (r["gen"]["trk"] or 0) >= 0.70)
    pass_ab = sum(1 for r in ab_rows if (r["gen"]["trk"] or 0) >= 0.70)
    cmp_no_word = comparison_section_html(
        sim_rows,
        ab_rows,
        section_title="Comparison — no_word_similarity_AB vs no_word_category_AB (n=30)",
        blurb=(
            "Same 6 local models, full 30-trial set, both orderings, dual scoring paths. "
            "Shared system prompt: <code>LOCAL_VLM_SYSTEM_PROMPT</code> "
            "(\"Answer concisely. Do not explain your reasoning.\"). "
            "Δ = category_AB − similarity on the generation path."
        ),
        source_note=(
            f"Similarity: <code>{sim_sf.name}</code>, <code>{sim_tf.name}</code>. "
            f"AB: <code>{ab_sf.name}</code>, <code>{ab_tf.name}</code>."
        ),
    )

    noun_sf = SESS_SMOKE / "playground_smoke_30trials_shape_first_noun_label_AB_shiple.txt"
    noun_tf = SESS_SMOKE / "playground_smoke_30trials_texture_first_noun_label_AB_shiple.txt"
    noun_html = ""
    noun_comparisons = ""
    noun_tile = '<div class="tile"><div class="v">pending</div><div class="l">shiple noun-label run</div></div>'
    prompt_count = "2 no-word AB prompts"
    if noun_sf.is_file() and noun_tf.is_file():
        noun_sf_data, noun_tf_data = parse_smoke(noun_sf), parse_smoke(noun_tf)
        noun_models = [m for m in noun_sf_data["models"] if m in noun_tf_data["models"]]
        noun_rows = [
            summarize_pair(noun_sf_data, noun_tf_data, m)
            for m in noun_models
        ]
        if noun_rows and noun_rows[0]["gen"]["n"] >= 30:
            pass_noun = sum(
                1 for r in noun_rows if (r["gen"]["trk"] or 0) >= 0.70
            )
            noun_tile = (
                f'<div class="tile"><div class="v">{pass_noun}/{len(noun_rows)}</div>'
                '<div class="l">noun_label_AB + shiple gen gate PASS</div></div>'
            )
            prompt_count = "3 AB prompts"
            noun_html = smoke_section_html(
                "3 · noun_label_AB + fixed sudo word “shiple” (n=30)",
                (
                    "User prompt: The first image is a shiple… which A or B is "
                    "also a shiple? Same six models, trials, orders, and scoring paths."
                ),
                noun_rows,
                f"{noun_sf.parent.name}/{noun_sf.name}; {noun_tf.name}",
            )
            noun_comparisons = (
                comparison_section_html(
                    sim_rows,
                    noun_rows,
                    section_title=(
                        "Comparison — no_word_similarity_AB vs "
                        "noun_label_AB + shiple"
                    ),
                    blurb=(
                        "Δ = noun_label_AB + shiple − no_word_similarity_AB "
                        "on the generation path."
                    ),
                    source_note=(
                        f"Similarity: <code>{sim_sf.name}</code>, "
                        f"<code>{sim_tf.name}</code>. Shiple: "
                        f"<code>{noun_sf.name}</code>, <code>{noun_tf.name}</code>."
                    ),
                    left_label="similarity",
                    right_label="shiple",
                )
                + comparison_section_html(
                    ab_rows,
                    noun_rows,
                    section_title=(
                        "Comparison — no_word_category_AB vs "
                        "noun_label_AB + shiple"
                    ),
                    blurb=(
                        "Δ = noun_label_AB + shiple − no_word_category_AB "
                        "on the generation path."
                    ),
                    source_note=(
                        f"No-word category: <code>{ab_sf.name}</code>, "
                        f"<code>{ab_tf.name}</code>. Shiple: "
                        f"<code>{noun_sf.name}</code>, <code>{noun_tf.name}</code>."
                    ),
                    left_label="no-word AB",
                    right_label="shiple",
                )
            )
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local VLM prompt compare — 30 trials</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Local VLM prompt compare — 30 trials × {prompt_count}</h1>
<p class="sub">Powered follow-up to the n=5 smoke. Sibling n=5 pages:
<a href="local_models_smoke_similarity_2026-07-17.html">similarity</a> ·
<a href="local_models_smoke_no_word_category_AB_2026-07-17.html">no_word_category_AB</a>.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.</p>

<div class="tiles">
  <div class="tile"><div class="v">{len(sim_rows)}</div><div class="l">models</div></div>
  <div class="tile"><div class="v">30×2</div><div class="l">trials × orders</div></div>
  <div class="tile"><div class="v">{pass_sim}/{len(sim_rows)}</div><div class="l">similarity gen gate PASS</div></div>
  <div class="tile"><div class="v">{pass_ab}/{len(ab_rows)}</div><div class="l">category_AB gen gate PASS</div></div>
  {noun_tile}
</div>

<div class="callout info"><b class="t">Prompt contract.</b>
All conditions use A/B labels. The first two omit a novel word; the third uses
the fixed sudo word <code>shiple</code>.
System message is identical across SmolVLM / InternVL / Qwen3-VL / Qwen3.5
(<code>LOCAL_VLM_SYSTEM_PROMPT</code>) for generate and score_choices.</div>

{smoke_section_html(
    "1 · no_word_similarity_AB (n=30)",
    "User prompt: three images / more similar to the reference / A or B.",
    sim_rows,
    f"{sim_sf.parent.name}/{sim_sf.name}; {sim_tf.name}",
)}
{smoke_section_html(
    "2 · no_word_category_AB (n=30)",
    "User prompt: See this object… find another one of the two (A or B).",
    ab_rows,
    f"{ab_sf.parent.name}/{ab_sf.name}; {ab_tf.name}",
)}
{noun_html}
{cmp_no_word}
{noun_comparisons}
<p class="src" style="margin-top:28px">Generated by <code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "local_models_prompt_compare_30trials_2026-07-17.html"
    out.write_text(body)
    return out


def try_build_ab_report() -> Path | None:
    """If AB-prompt smoke logs exist, write a companion HTML with vs-similarity comparison."""
    ab = load_condition_rows("no_word_category_AB", n_trials=5)
    sim = load_condition_rows("no_word_similarity_AB", n_trials=5, merge_qwen35_resmoke=True)
    if ab is None:
        return None
    rows, (sf_path, tf_path) = ab
    prompt = parse_smoke(sf_path)["meta"].get("Prompt", "no_word_category_AB")
    cmp_html = ""
    if sim is not None:
        sim_rows, _ = sim
        cmp_html = comparison_section_html(
            sim_rows,
            rows,
            section_title="2 · Comparison — similarity vs <code>no_word_category_AB</code> (n=5 smoke)",
            blurb=(
                "Same models, n=5 per order. Similarity columns use the July 17 similarity smoke "
                "(qwen3.5 from post-fix resmoke when available). For the powered n=30 comparison see "
                "<a href=\"local_models_prompt_compare_30trials_2026-07-17.html\">"
                "<code>local_models_prompt_compare_30trials_2026-07-17.html</code></a> when ready."
            ),
            source_note=(
                f"AB: <code>{sf_path.name}</code>; <code>{tf_path.name}</code>."
            ),
        )
    passish = [r for r in rows if (r["gen"]["trk"] or 0) >= 0.70]
    lockish = [r for r in rows if (r["gen"]["pos_a"] or 0) >= 0.9 or (r["gen"]["pos_a"] or 1) <= 0.1]
    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local VLM smoke — no_word_category_AB</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Local VLM smoke — <code>no_word_category_AB</code></h1>
<p class="sub">Prompt from <code>eval_core.PROMPT_TEMPLATES</code>.
Sibling similarity report:
<a href="local_models_smoke_similarity_2026-07-17.html"><code>local_models_smoke_similarity_2026-07-17.html</code></a>.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.</p>

<div class="tiles">
  <div class="tile"><div class="v">{len(rows)}</div><div class="l">local models in smoke</div></div>
  <div class="tile"><div class="v">{len(passish)}/{len(rows)}</div><div class="l">gen tracking ≥ 0.70 on n=5×2</div></div>
  <div class="tile"><div class="v">{len(lockish)}</div><div class="l">hard PosA lock (≥0.9 or ≤0.1)</div></div>
  <div class="tile"><div class="v">AB</div><div class="l">no_word_category_AB wording</div></div>
</div>

{smoke_section_html(
    "1 · Dual-path smoke under no_word_category_AB",
    f"Prompt: {prompt}",
    rows,
    f"{sf_path.parent.name}/{sf_path.name}; {tf_path.name}",
)}
{cmp_html}
<p class="src" style="margin-top:28px">Generated by <code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "local_models_smoke_no_word_category_AB_2026-07-17.html"
    out.write_text(html_body)
    return out


def _rows_from_paths(sf_path: Path, tf_path: Path) -> list[dict]:
    sf, tf = parse_smoke(sf_path), parse_smoke(tf_path)
    return [
        summarize_pair(sf, tf, model)
        for model in sf["models"]
        if model in tf["models"]
    ]


def _append_rows(base: list[dict], extra: list[dict]) -> list[dict]:
    by_model = {row["model"]: row for row in base}
    for row in extra:
        by_model[row["model"]] = row
    return list(by_model.values())


def try_build_numeric_qwen8_report() -> Path | None:
    """Build numeric-label and Qwen3-VL-8B comparisons when all logs exist."""
    def pair(condition: str, suffix: str = "") -> tuple[Path, Path]:
        return (
            SESS_SMOKE
            / f"playground_smoke_30trials_shape_first_{condition}{suffix}.txt",
            SESS_SMOKE
            / f"playground_smoke_30trials_texture_first_{condition}{suffix}.txt",
        )

    numeric_paths = {
        "similarity": pair("no_word_similarity"),
        "category": pair("no_word_category"),
        "noun": pair("noun_label", "_shiple"),
    }
    q8_ab_paths = {
        "similarity": pair("no_word_similarity_AB", "_qwen3-vl-8b"),
        "category": pair("no_word_category_AB", "_qwen3-vl-8b"),
        "noun": pair("noun_label_AB", "_shiple_qwen3-vl-8b"),
    }
    if not all(path.is_file() for paths in numeric_paths.values() for path in paths):
        return None
    if not all(path.is_file() for paths in q8_ab_paths.values() for path in paths):
        return None

    numeric = {
        key: _rows_from_paths(*paths) for key, paths in numeric_paths.items()
    }

    existing_ab = {
        "similarity": _rows_from_paths(
            SESS_SMOKE / "playground_smoke_30trials_shape_first_no_word_similarity_AB.txt",
            SESS_SMOKE / "playground_smoke_30trials_texture_first_no_word_similarity_AB.txt",
        ),
        "category": _rows_from_paths(
            SESS_SMOKE / "playground_smoke_30trials_shape_first_no_word_category_AB.txt",
            SESS_SMOKE / "playground_smoke_30trials_texture_first_no_word_category_AB.txt",
        ),
        "noun": _rows_from_paths(
            SESS_SMOKE / "playground_smoke_30trials_shape_first_noun_label_AB_shiple.txt",
            SESS_SMOKE / "playground_smoke_30trials_texture_first_noun_label_AB_shiple.txt",
        ),
    }
    ab = {
        key: _append_rows(existing_ab[key], _rows_from_paths(*q8_ab_paths[key]))
        for key in existing_ab
    }

    numeric_sections = "\n".join(
        [
            smoke_section_html(
                "1 · Numeric similarity (1/2)",
                "No word; which candidate is more similar to the reference?",
                numeric["similarity"],
                f"{numeric_paths['similarity'][0].name}; {numeric_paths['similarity'][1].name}",
            ),
            smoke_section_html(
                "2 · Numeric category, no word (1/2)",
                "See this object; find another one of the two.",
                numeric["category"],
                f"{numeric_paths['category'][0].name}; {numeric_paths['category'][1].name}",
            ),
            smoke_section_html(
                "3 · Numeric noun label + shiple (1/2)",
                "The first image is a shiple; which candidate is also a shiple?",
                numeric["noun"],
                f"{numeric_paths['noun'][0].name}; {numeric_paths['noun'][1].name}",
            ),
        ]
    )

    within_numeric = "\n".join(
        [
            comparison_section_html(
                numeric["similarity"],
                numeric["category"],
                section_title="Numeric wording effect — similarity vs category",
                blurb="Δ = no-word category − no-word similarity on generation.",
                source_note="Both conditions use 1/2 and no novel word.",
                left_label="similarity",
                right_label="category",
            ),
            comparison_section_html(
                numeric["category"],
                numeric["noun"],
                section_title="Numeric naming effect — no word vs shiple",
                blurb=(
                    "Δ = noun_label + shiple − no_word_category on generation. "
                    "Interpret shape-rate differences only where both tracking gates pass."
                ),
                source_note="Both conditions use 1/2; only the noun condition adds shiple.",
                left_label="no word",
                right_label="shiple",
            ),
        ]
    )

    label_effects = "\n".join(
        comparison_section_html(
            ab[key],
            numeric[key],
            section_title=f"Label-set effect — {key}: A/B vs 1/2",
            blurb=(
                "Same framing and images. Δ = numeric 1/2 − letter A/B on generation."
            ),
            source_note=(
                "A/B rows combine the six-model July 17 runs with the new "
                "qwen3-vl-8b-only logs."
            ),
            left_label="A/B",
            right_label="1/2",
        )
        for key in ("similarity", "category", "noun")
    )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Numeric labels + Qwen3-VL-8B — 30 trials</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Numeric labels + Qwen3-VL-8B</h1>
<p class="sub">30 trials × 2 orders · 7 local models · three framings · A/B vs 1/2.
Companion A/B report:
<a href="local_models_prompt_compare_30trials_2026-07-17.html">three-prompt comparison</a>.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.</p>

<div class="callout info"><b class="t">Design.</b>
The numeric runs use the same playground model order, images, generation call, two_pass,
one_pass, and shared local system prompt as the completed A/B runs. Qwen3-VL-8B was added
to all six prompt × label-set cells; the other six models were rerun only for the three
new numeric cells.</div>

{numeric_sections}
{within_numeric}
{label_effects}
<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "local_models_numeric_and_qwen8_30trials_2026-07-17.html"
    out.write_text(body)
    return out


def try_build_word_generality_report() -> Path | None:
    """Five-sudo-word noun comparison under numeric and A/B labels."""
    words = ("shiple", "clapher", "plailass", "procation", "adinefults")
    cells: dict[tuple[str, str], list[dict]] = {}

    for word in words:
        numeric_paths = (
            SESS_SMOKE
            / f"playground_smoke_30trials_shape_first_noun_label_{word}.txt",
            SESS_SMOKE
            / f"playground_smoke_30trials_texture_first_noun_label_{word}.txt",
        )
        if not all(path.is_file() for path in numeric_paths):
            return None
        cells[(word, "1/2")] = _rows_from_paths(*numeric_paths)

        ab_paths = (
            SESS_SMOKE
            / f"playground_smoke_30trials_shape_first_noun_label_AB_{word}.txt",
            SESS_SMOKE
            / f"playground_smoke_30trials_texture_first_noun_label_AB_{word}.txt",
        )
        if not all(path.is_file() for path in ab_paths):
            return None
        ab_rows = _rows_from_paths(*ab_paths)
        if word == "shiple":
            q8_paths = (
                SESS_SMOKE
                / "playground_smoke_30trials_shape_first_noun_label_AB_shiple_qwen3-vl-8b.txt",
                SESS_SMOKE
                / "playground_smoke_30trials_texture_first_noun_label_AB_shiple_qwen3-vl-8b.txt",
            )
            if not all(path.is_file() for path in q8_paths):
                return None
            ab_rows = _append_rows(ab_rows, _rows_from_paths(*q8_paths))
        cells[(word, "A/B")] = ab_rows

    detail_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for (word, labels), rows in cells.items():
        for row in rows:
            gen = row["gen"]
            gate = (gen["trk"] or 0) >= 0.70
            cls = "pass" if gate else ("lock" if gen["pos_a"] is not None and (
                gen["pos_a"] >= 0.9 or gen["pos_a"] <= 0.1
            ) else "")
            detail_rows.append(
                f'<tr class="{cls}"><td>{html.escape(row["model"])}</td>'
                f"<td>{word}</td><td>{labels}</td>"
                f"<td>{fmt_rate(gen['shp_sf'])}</td>"
                f"<td>{fmt_rate(gen['shp_tf'])}</td>"
                f"<td>{fmt_rate(gen['shp_avg'])}</td>"
                f"<td>{fmt_rate(gen['trk'])}</td>"
                f"<td>{fmt_rate(gen['pos_a'])}</td>"
                f"<td>{'PASS' if gate else 'fail'}</td></tr>"
            )
            grouped.setdefault((row["model"], labels), []).append(
                {
                    "word": word,
                    "tracking": gen["trk"],
                    "shape": gen["shp_avg"],
                    "gate": gate,
                }
            )

    summary_rows = []
    for (model, labels), values in grouped.items():
        trackings = [v["tracking"] for v in values if v["tracking"] is not None]
        passes = [v for v in values if v["gate"]]
        pass_words = ", ".join(v["word"] for v in passes) or "none"
        mean_shape_pass = (
            sum(v["shape"] for v in passes) / len(passes)
            if passes
            else None
        )
        summary_rows.append(
            f"<tr><td>{html.escape(model)}</td><td>{labels}</td>"
            f"<td>{sum(v['gate'] for v in values)}/5</td>"
            f"<td>{sum(trackings) / len(trackings):.2f}</td>"
            f"<td>{min(trackings):.2f}–{max(trackings):.2f}</td>"
            f"<td>{fmt_rate(mean_shape_pass)}</td>"
            f"<td>{html.escape(pass_words)}</td></tr>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sudo-word generality — 30 trials</title><style>{CSS}</style></head>
<body><main>
<h1>Sudo-word generality</h1>
<p class="sub">Five curated sudo words × 2 label sets × 7 models ×
30 trials × 2 orders. Same <code>noun_label</code> framing and shared local
system prompt throughout.</p>
<div class="callout info"><b class="t">Question.</b>
Does the noun-related tracking pattern generalize beyond <code>shiple</code>,
or does it depend on the phonological form of one pseudo-word?</div>

<h2>Across-word summary</h2>
<div class="tablewrap"><table><tr><th>model</th><th>labels</th>
<th>gate passes</th><th>mean tracking</th><th>tracking range</th>
<th>mean shape among passes</th><th>passing words</th></tr>
{''.join(summary_rows)}</table></div>

<h2>Every model × word × label cell</h2>
<div class="tablewrap"><table><tr><th>model</th><th>word</th><th>labels</th>
<th>shp SF</th><th>shp TF</th><th>shp avg</th><th>tracking</th>
<th>PosFirst</th><th>gate</th></tr>
{''.join(detail_rows)}</table></div>
<p class="src">Green = tracking ≥ 0.70. Red = hard first/second-label lock.
Generated by <code>scripts/build_playground_results_html.py</code>.</p>
</main></body></html>"""
    out = PLAY / "local_models_sudo_word_generality_30trials_2026-07-17.html"
    out.write_text(body)
    return out


def main() -> None:
    PLAY.mkdir(parents=True, exist_ok=True)
    out = PLAY / "local_models_smoke_similarity_2026-07-17.html"
    out.write_text(build_similarity_report())
    print(f"wrote {out}")

    if FARM_HTML.is_file():
        dest = PLAY / "probe-experiment-results.html"
        dest.write_text(FARM_HTML.read_text())
        print(f"copied {dest}")

    ab = try_build_ab_report()
    if ab:
        print(f"wrote {ab}")
    else:
        print("AB-prompt smoke not ready yet")

    powered = try_build_powered_prompt_compare()
    if powered:
        print(f"wrote {powered}")
    else:
        print("n=30 prompt compare not ready yet (waiting on both conditions)")

    numeric_q8 = try_build_numeric_qwen8_report()
    if numeric_q8:
        print(f"wrote {numeric_q8}")
    else:
        print("numeric + qwen3-vl-8b report not ready yet")

    word_report = try_build_word_generality_report()
    if word_report:
        print(f"wrote {word_report}")
    else:
        print("sudo-word generality report not ready yet")


if __name__ == "__main__":
    main()
