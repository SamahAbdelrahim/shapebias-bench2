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
SESS_AB_FIXED = PLAY / "session_2026-07-30_farmshare"
# Revised no_word_category / _AB wording (July 24). Prefer these over July-17
# old-prompt logs when rebuilding canonical HTML. Old logs stay on disk as baseline.
SESS_CAT_REVISED = PLAY / "session_2026-07-24_farmshare"
SESS_SMITH = PLAY / "session_2026-07-25_smith_farmshare"
ARCHIVE_AB = PLAY / "_archived_ab_label_mismatch_pre_2026-07-28"


def newest_smith_session() -> Path:
    """Prefer the newest Smith session folder (post label-fix rerun when present)."""
    candidates = sorted(PLAY.glob("session_*smith*"), reverse=True)
    return candidates[0] if candidates else SESS_SMITH
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
<a href="local_models_prompt_compare_30trials_2026-07-24.html"><code>local_models_prompt_compare_30trials_2026-07-24.html</code></a>
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

    Prefers the newest session folder (e.g. revised July-24 category over
    July-17 old-prompt category) when both exist.
    """
    sf_path = tf_path = None
    # Newest session first so revised-prompt logs win over older baselines.
    for sess in sorted(PLAY.glob("session_*"), reverse=True):
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
            if (name_ok or meta_ok or legacy_sim) and sf_path is None:
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
            if (name_ok or meta_ok or legacy_sim) and tf_path is None:
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
            "Same models, full 30-trial set, both orderings, dual scoring paths. "
            "Category uses the <b>July-24 revised</b> wording; similarity remains July-17. "
            "Shared system prompt: <code>LOCAL_VLM_SYSTEM_PROMPT</code> "
            "(\"Answer concisely. Do not explain your reasoning.\"). "
            "Δ = category_AB − similarity on the generation path."
        ),
        source_note=(
            f"Similarity: <code>{sim_sf.parent.name}/{sim_sf.name}</code>, "
            f"<code>{sim_tf.name}</code>. "
            f"AB category (revised): <code>{ab_sf.parent.name}/{ab_sf.name}</code>, "
            f"<code>{ab_tf.name}</code>."
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
All conditions use A/B labels. Similarity remains the July-17 wording; category
uses the <b>July-24 revised</b> no_word_category_AB wording. The third condition
uses the fixed sudo word <code>shiple</code>.
System message is identical across SmolVLM / InternVL / Qwen3-VL / Qwen3.5
(<code>LOCAL_VLM_SYSTEM_PROMPT</code>) for generate and score_choices.
Old-vs-new category numbers:
<a href="no_word_category_prompt_revision_2026-07-24.csv"><code>no_word_category_prompt_revision_2026-07-24.csv</code></a>
·
<a href="no_word_category_prompt_revision_2026-07-24.html">full audit page</a>.
<b>Only category changed;</b> similarity and noun sections match the pre-revision report.</div>

{category_revision_audit_html()}

{smoke_section_html(
    "1 · no_word_similarity_AB (n=30)",
    "User prompt: three images / more similar to the reference / A or B.",
    sim_rows,
    f"{sim_sf.parent.name}/{sim_sf.name}; {sim_tf.name}",
)}
{smoke_section_html(
    "2 · no_word_category_AB (n=30) — revised July 24 wording",
    (
        "User prompt: This first image is an object… which of the following two "
        "images (A or B) is another one? Replaces the July-17 “See this object… "
        "find another one of the two” wording."
    ),
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
    out = PLAY / "local_models_prompt_compare_30trials_2026-07-24.html"
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
                "<a href=\"local_models_prompt_compare_30trials_2026-07-24.html\">"
                "<code>local_models_prompt_compare_30trials_2026-07-24.html</code></a> when ready."
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


def _load_category_pair(session: Path, ab: bool = False) -> list[dict] | None:
    """Load no_word_category rows from a specific session folder."""
    cond = "no_word_category_AB" if ab else "no_word_category"
    sf = session / f"playground_smoke_30trials_shape_first_{cond}.txt"
    tf = session / f"playground_smoke_30trials_texture_first_{cond}.txt"
    if not sf.is_file() or not tf.is_file():
        return None
    return _rows_from_paths(sf, tf)


def category_revision_audit_html() -> str:
    """Old (July-17) vs new (July-24) category prompt — all models, both label sets."""
    old_num = _load_category_pair(SESS_SMOKE, ab=False) or []
    new_num = _load_category_pair(SESS_CAT_REVISED, ab=False) or []
    old_ab = _load_category_pair(SESS_SMOKE, ab=True) or []
    new_ab = _load_category_pair(SESS_CAT_REVISED, ab=True) or []

    def gate(trk: float | None) -> str:
        return "PASS" if (trk or 0) >= 0.70 else "fail"

    blocks = []
    for label, old_rows, new_rows in (
        ("Numeric 1/2 (no_word_category)", old_num, new_num),
        ("A/B (no_word_category_AB)", old_ab, new_ab),
    ):
        old_by = {r["model"]: r for r in old_rows}
        new_by = {r["model"]: r for r in new_rows}
        models = sorted(set(old_by) | set(new_by))
        rows_html = []
        n_flip = 0
        for m in models:
            og = old_by[m]["gen"] if m in old_by else None
            ng = new_by[m]["gen"] if m in new_by else None
            if og is None:
                rows_html.append(
                    f'<tr><td class="l">{html.escape(m)}</td>'
                    f'<td colspan="3" class="dim">not in old log</td>'
                    f'<td>{ng["trk"]:.2f}</td><td>{ng["shp_avg"]:.2f}</td>'
                    f'<td>{gate(ng["trk"])}</td><td>new only</td></tr>'
                )
                continue
            if ng is None:
                rows_html.append(
                    f'<tr><td class="l">{html.escape(m)}</td>'
                    f'<td>{og["trk"]:.2f}</td><td>{og["shp_avg"]:.2f}</td>'
                    f'<td>{gate(og["trk"])}</td>'
                    f'<td colspan="3" class="dim">not in new log</td><td>—</td></tr>'
                )
                continue
            og_gate, ng_gate = gate(og["trk"]), gate(ng["trk"])
            cls = ""
            change = "same"
            if og_gate != ng_gate:
                n_flip += 1
                change = f"{og_gate}->{ng_gate}"
                cls = "pass" if ng_gate == "PASS" else "fail"
            rows_html.append(
                f'<tr class="{cls}"><td class="l">{html.escape(m)}</td>'
                f'<td>{og["trk"]:.2f}</td><td>{og["shp_avg"]:.2f}</td><td>{og_gate}</td>'
                f'<td>{ng["trk"]:.2f}</td><td>{ng["shp_avg"]:.2f}</td><td>{ng_gate}</td>'
                f'<td>{change}</td></tr>'
            )
        blocks.append(
            f"<h3>{html.escape(label)}</h3>"
            f'<p class="src">Old: <code>{SESS_SMOKE.name}</code> · '
            f'New: <code>{SESS_CAT_REVISED.name}</code> · '
            f"Gate flips: {n_flip}/{len(models)}</p>"
            '<div class="tablewrap"><table>'
            "<tr><th class=\"l\">Model</th>"
            "<th>old trk</th><th>old shp</th><th>old gate</th>"
            "<th>new trk</th><th>new shp</th><th>new gate</th>"
            "<th>gate change</th></tr>"
            + "".join(rows_html)
            + "</table></div>"
        )

    q35_note = (
        "<p><b>Qwen3.5 family (gate ≥ 0.70):</b> "
        "Similarity and noun cells were <em>not</em> rerun and are unchanged from July-17. "
        "Under the <b>old</b> category wording, 4b failed both category cells (numeric 0.13, "
        "A/B 0.03), 9b passed numeric category but failed A/B similarity and A/B noun, "
        "and 27b failed both category cells. "
        "Under the <b>new</b> wording, the only Qwen3.5 gate flip is "
        "<code>qwen3.5-4b</code> numeric category (0.13→0.93 PASS). "
        "9b and 27b category tracking improved but 27b stays under the gate (0.63 numeric, "
        "0.60 A/B). qwen3.5-27b is 4-bit; other Qwen3.5 rungs are bf16.</p>"
    )

    return (
        '<h2 id="category-revision">Category prompt revision audit (July-17 → July-24)</h2>'
        '<div class="callout good"><b class="t">Pipeline check.</b> '
        "Only <code>no_word_category</code> and <code>no_word_category_AB</code> were rerun "
        "with the revised prompt. Every other condition (similarity, noun, sudo-word runs) "
        "still reads July-17 logs; verified byte-for-byte on tracking/shape for non-category cells. "
        "No accidental cross-session contamination.</div>"
        "<p><b>Prompt change:</b> old — "
        "“See this object in the first image. Can you find another one of the two …”; "
        "new — “You are given three images. This first image is an object. "
        "Which of the following two images … is another one?”</p>"
        + q35_note
        + "".join(blocks)
        + '<p class="src">CSV: '
        '<a href="no_word_category_prompt_revision_2026-07-24.csv">'
        "<code>no_word_category_prompt_revision_2026-07-24.csv</code></a></p>"
    )


def try_build_category_revision_report() -> Path | None:
    """Standalone page documenting old vs new category prompt impact."""
    if not SESS_CAT_REVISED.is_dir():
        return None
    if _load_category_pair(SESS_CAT_REVISED) is None:
        return None
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Category prompt revision audit — 2026-07-24</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Category prompt revision audit</h1>
<p class="sub">Old vs new <code>no_word_category</code> wording only. Canonical reports
(<a href="local_models_numeric_and_qwen8_30trials_2026-07-24.html">numeric</a>,
<a href="local_models_prompt_compare_30trials_2026-07-24.html">A/B compare</a>)
embed this section and use July-24 category logs everywhere else unchanged.</p>
{category_revision_audit_html()}
<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "no_word_category_prompt_revision_2026-07-24.html"
    out.write_text(body)
    return out


def try_build_numeric_qwen8_report() -> Path | None:
    """Build numeric-label and Qwen3-VL-8B comparisons when all logs exist."""
    def pair(session: Path, condition: str, suffix: str = "") -> tuple[Path, Path]:
        return (
            session
            / f"playground_smoke_30trials_shape_first_{condition}{suffix}.txt",
            session
            / f"playground_smoke_30trials_texture_first_{condition}{suffix}.txt",
        )

    # Category uses the July-24 revised-prompt session; similarity/noun stay July-17.
    cat_sess = SESS_CAT_REVISED if SESS_CAT_REVISED.is_dir() else SESS_SMOKE
    numeric_paths = {
        "similarity": pair(SESS_SMOKE, "no_word_similarity"),
        "category": pair(cat_sess, "no_word_category"),
        "noun": pair(SESS_SMOKE, "noun_label", "_shiple"),
    }
    # July-24 category_AB already includes the full ladder (incl. 8b/9b/27b).
    # Similarity/noun A/B still need the July-17 base + qwen3-vl-8b fill files.
    q8_ab_paths = {
        "similarity": pair(SESS_SMOKE, "no_word_similarity_AB", "_qwen3-vl-8b"),
        "noun": pair(SESS_SMOKE, "noun_label_AB", "_shiple_qwen3-vl-8b"),
    }
    if not all(path.is_file() for paths in numeric_paths.values() for path in paths):
        return None
    if not all(path.is_file() for paths in q8_ab_paths.values() for path in paths):
        return None
    cat_ab_paths = pair(cat_sess, "no_word_category_AB")
    if not all(path.is_file() for path in cat_ab_paths):
        return None

    numeric = {
        key: _rows_from_paths(*paths) for key, paths in numeric_paths.items()
    }

    existing_ab = {
        "similarity": _rows_from_paths(
            SESS_SMOKE / "playground_smoke_30trials_shape_first_no_word_similarity_AB.txt",
            SESS_SMOKE / "playground_smoke_30trials_texture_first_no_word_similarity_AB.txt",
        ),
        "category": _rows_from_paths(*cat_ab_paths),
        "noun": _rows_from_paths(
            SESS_SMOKE / "playground_smoke_30trials_shape_first_noun_label_AB_shiple.txt",
            SESS_SMOKE / "playground_smoke_30trials_texture_first_noun_label_AB_shiple.txt",
        ),
    }
    ab = {
        "similarity": _append_rows(
            existing_ab["similarity"], _rows_from_paths(*q8_ab_paths["similarity"])
        ),
        "category": existing_ab["category"],
        "noun": _append_rows(
            existing_ab["noun"], _rows_from_paths(*q8_ab_paths["noun"])
        ),
    }

    cat_src = f"{cat_sess.name}/{numeric_paths['category'][0].name}"
    numeric_sections = "\n".join(
        [
            smoke_section_html(
                "1 · Numeric similarity (1/2)",
                "No word; which candidate is more similar to the reference?",
                numeric["similarity"],
                f"{numeric_paths['similarity'][0].name}; {numeric_paths['similarity'][1].name}",
            ),
            smoke_section_html(
                "2 · Numeric category, no word (1/2) — revised July 24 wording",
                (
                    "This first image is an object; which of the following two images "
                    "is another one? (Replaces the July-17 “See this object… find "
                    "another one of the two” wording.)"
                ),
                numeric["category"],
                f"{cat_src}; {numeric_paths['category'][1].name}",
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
                blurb=(
                    "Δ = revised no-word category − no-word similarity on generation. "
                    "Category uses the July-24 prompt revision."
                ),
                source_note=(
                    f"Similarity: July-17 session. Category: {cat_sess.name}."
                ),
                left_label="similarity",
                right_label="category",
            ),
            comparison_section_html(
                numeric["category"],
                numeric["noun"],
                section_title="Numeric naming effect — no word vs shiple",
                blurb=(
                    "Δ = noun_label + shiple − revised no_word_category on generation. "
                    "Interpret shape-rate differences only where both tracking gates pass."
                ),
                source_note=(
                    f"Category: {cat_sess.name}. Noun: July-17 session."
                ),
                left_label="no word",
                right_label="shiple",
            ),
        ]
    )

    label_effect_notes = {
        "similarity": (
            "A/B rows combine the six-model July 17 runs with the "
            "qwen3-vl-8b-only fill logs."
        ),
        "category": (
            f"Both A/B and 1/2 category use the revised July-24 wording "
            f"({cat_sess.name}); full ladder in each log."
        ),
        "noun": (
            "A/B rows combine the six-model July 17 runs with the "
            "qwen3-vl-8b-only fill logs."
        ),
    }
    label_effects = "\n".join(
        comparison_section_html(
            ab[key],
            numeric[key],
            section_title=f"Label-set effect — {key}: A/B vs 1/2",
            blurb=(
                "Same framing and images. Δ = numeric 1/2 − letter A/B on generation."
            ),
            source_note=label_effect_notes[key],
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
<p class="sub">30 trials × 2 orders · local ladder · three framings · A/B vs 1/2.
Category cells use the <b>July-24 revised</b> no_word_category wording; similarity and
noun remain July-17. Companion A/B report:
<a href="local_models_prompt_compare_30trials_2026-07-24.html">three-prompt comparison</a>.
Old-vs-new category CSV:
<a href="no_word_category_prompt_revision_2026-07-24.csv"><code>no_word_category_prompt_revision_2026-07-24.csv</code></a>.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.</p>

<div class="callout info"><b class="t">Design.</b>
Similarity and noun use the July-17 playground protocol. Category was rerun on
2026-07-24 with the revised wording (“This first image is an object… which …
is another one?”) after the July-17 “See this object… find another one of the two”
framing suppressed tracking in large Qwen models. qwen3.5-27b is 4-bit (nf4);
other Qwen3.5 rungs are bf16. <b>Only category cells changed;</b> similarity and
noun numbers are identical to the pre-revision report.</div>

{category_revision_audit_html()}

{numeric_sections}
{within_numeric}
{label_effects}
<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "local_models_numeric_and_qwen8_30trials_2026-07-24.html"
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


def _pair(session: Path, condition: str, suffix: str = "") -> tuple[Path, Path]:
    return (
        session / f"playground_smoke_30trials_shape_first_{condition}{suffix}.txt",
        session / f"playground_smoke_30trials_texture_first_{condition}{suffix}.txt",
    )


def _load_ladder_from_session(
    session: Path,
    *,
    ab_session: Path | None = None,
) -> tuple[dict[str, list], dict[str, list]] | None:
    """Load similarity/category/noun cells; numeric and AB may live in different sessions."""
    ab_sess = ab_session or session
    numeric_paths = {
        "similarity": _pair(session, "no_word_similarity"),
        "category": _pair(session, "no_word_category"),
        "noun": _pair(session, "noun_label", "_shiple"),
    }
    ab_paths = {
        "similarity": _pair(ab_sess, "no_word_similarity_AB"),
        "category": _pair(ab_sess, "no_word_category_AB"),
        "noun": _pair(ab_sess, "noun_label_AB", "_shiple"),
    }
    if not all(path.is_file() for paths in numeric_paths.values() for path in paths):
        return None
    if not all(path.is_file() for paths in ab_paths.values() for path in paths):
        return None
    numeric = {key: _rows_from_paths(*paths) for key, paths in numeric_paths.items()}
    ab = {key: _rows_from_paths(*paths) for key, paths in ab_paths.items()}
    return numeric, ab


def try_build_smith_numeric_report() -> Path | None:
    """Smith stimuli: full numeric + A/B ladder (same columns as local report)."""
    ab_sess = newest_smith_session()
    loaded = _load_ladder_from_session(SESS_SMITH, ab_session=ab_sess)
    if loaded is None:
        return None
    numeric, ab = loaded
    numeric_paths = {
        "similarity": _pair(SESS_SMITH, "no_word_similarity"),
        "category": _pair(SESS_SMITH, "no_word_category"),
        "noun": _pair(SESS_SMITH, "noun_label", "_shiple"),
    }
    ab_paths = {
        "similarity": _pair(ab_sess, "no_word_similarity_AB"),
        "category": _pair(ab_sess, "no_word_category_AB"),
        "noun": _pair(ab_sess, "noun_label_AB", "_shiple"),
    }

    numeric_sections = "\n".join(
        [
            smoke_section_html(
                "1 · Numeric similarity (1/2) — Smith stimuli",
                "No word; which candidate is more similar to the reference?",
                numeric["similarity"],
                f"{numeric_paths['similarity'][0].name}; {numeric_paths['similarity'][1].name}",
            ),
            smoke_section_html(
                "2 · Numeric category (1/2) — Smith stimuli",
                "Revised July-24 no_word_category wording on Linda Smith probe triplets.",
                numeric["category"],
                f"{numeric_paths['category'][0].name}; {numeric_paths['category'][1].name}",
            ),
            smoke_section_html(
                "3 · Numeric noun + shiple (1/2) — Smith stimuli",
                "First image is a shiple; which candidate is also a shiple?",
                numeric["noun"],
                f"{numeric_paths['noun'][0].name}; {numeric_paths['noun'][1].name}",
            ),
        ]
    )

    within_numeric = comparison_section_html(
        numeric["similarity"],
        numeric["category"],
        section_title="Within numeric — similarity vs category (Smith)",
        blurb="Δ = category − similarity on generation picks.",
        source_note=f"Session: {SESS_SMITH.name}",
        left_label="similarity",
        right_label="category",
    )
    within_numeric += comparison_section_html(
        numeric["category"],
        numeric["noun"],
        section_title="Within numeric — category vs noun + shiple",
        blurb="Δ = noun − category on generation picks.",
        source_note=f"Session: {SESS_SMITH.name}",
        left_label="category",
        right_label="noun",
    )

    label_effects = "\n".join(
        comparison_section_html(
            ab[key],
            numeric[key],
            section_title=f"Label-set effect — {key}: A/B vs 1/2 (Smith)",
            blurb="Same framing and Smith images. Δ = numeric 1/2 − letter A/B on generation.",
            source_note=f"Numeric: {SESS_SMITH.name}; A/B: {ab_sess.name}",
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
<title>Smith stimuli — numeric ladder + Qwen8 — 30 trials</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Smith stimuli — full numeric ladder</h1>
<p class="sub">30 Smith probe triplets × 2 orders · 9 local models · three framings · A/B vs 1/2.
Local benchmark counterpart:
<a href="local_models_numeric_and_qwen8_30trials_2026-07-24.html">our stimuli report</a>.
PriDe/swap:
<a href="smith_prompt_pride_debias_2026-07-25.html"><code>smith_prompt_pride_debias_2026-07-25.html</code></a>.
Glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.</p>

<div class="callout info"><b class="t">Stimuli.</b>
Linda Smith probe set (<code>previous-lit-stimuli/smith_stimuli/</code>): each trial is
probe + shape_match + color_match (color_match = texture distractor). Same prompts and
validity gates as the local July-24 ladder.</div>

{numeric_sections}
{within_numeric}
{label_effects}
<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "smith_numeric_and_qwen8_30trials_2026-07-25.html"
    out.write_text(body)
    return out


def try_build_smith_prompt_compare() -> Path | None:
    """Smith stimuli: A/B three-prompt comparison (n=30)."""
    ab_sess = newest_smith_session()
    ab_paths = {
        "similarity": _pair(ab_sess, "no_word_similarity_AB"),
        "category": _pair(ab_sess, "no_word_category_AB"),
        "noun": _pair(ab_sess, "noun_label_AB", "_shiple"),
    }
    if not all(path.is_file() for paths in ab_paths.values() for path in paths):
        return None
    ab = {key: _rows_from_paths(*paths) for key, paths in ab_paths.items()}
    sim_rows = ab["similarity"]
    cat_rows = ab["category"]
    noun_rows = ab["noun"]
    if not sim_rows or sim_rows[0]["gen"]["n"] < 30:
        return None

    pass_sim = sum(1 for r in sim_rows if (r["gen"]["trk"] or 0) >= 0.70)
    pass_cat = sum(1 for r in cat_rows if (r["gen"]["trk"] or 0) >= 0.70)
    pass_noun = sum(1 for r in noun_rows if (r["gen"]["trk"] or 0) >= 0.70)

    cmp_no_word = comparison_section_html(
        sim_rows,
        cat_rows,
        section_title="Comparison — similarity vs category (Smith, n=30)",
        blurb="Δ = category_AB − similarity_AB on generation.",
        source_note=f"Session: {ab_sess.name}",
    )
    noun_comparisons = comparison_section_html(
        sim_rows,
        noun_rows,
        section_title="Comparison — similarity vs noun + shiple (Smith)",
        blurb="Δ = noun − similarity on generation.",
        source_note=f"Session: {ab_sess.name}",
        left_label="similarity",
        right_label="shiple",
    ) + comparison_section_html(
        cat_rows,
        noun_rows,
        section_title="Comparison — category vs noun + shiple (Smith)",
        blurb="Δ = noun − category on generation.",
        source_note=f"Session: {ab_sess.name}",
        left_label="category",
        right_label="shiple",
    )

    date_tag = ab_sess.name.replace("session_", "").replace("_farmshare", "").replace("_smith", "")
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smith stimuli — A/B prompt compare — 30 trials</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Smith stimuli — A/B prompt compare</h1>
<p class="sub">30 trials × 3 AB prompts × 9 models. Numeric ladder:
<a href="smith_numeric_and_qwen8_30trials_2026-07-25.html">Smith numeric report</a>.
Local counterpart:
<a href="local_models_prompt_compare_30trials_2026-07-24.html">our stimuli A/B compare</a>.</p>

<div class="tiles">
  <div class="tile"><div class="v">{len(sim_rows)}</div><div class="l">models</div></div>
  <div class="tile"><div class="v">{pass_sim}/{len(sim_rows)}</div><div class="l">similarity gen PASS</div></div>
  <div class="tile"><div class="v">{pass_cat}/{len(cat_rows)}</div><div class="l">category gen PASS</div></div>
  <div class="tile"><div class="v">{pass_noun}/{len(noun_rows)}</div><div class="l">noun gen PASS</div></div>
</div>

{smoke_section_html(
    "1 · no_word_similarity_AB (Smith)",
    "Three images; more similar to the reference; A or B.",
    sim_rows,
    f"{ab_paths['similarity'][0].name}; {ab_paths['similarity'][1].name}",
)}
{smoke_section_html(
    "2 · no_word_category_AB (Smith)",
    "Revised July-24 category wording on Smith probe triplets.",
    cat_rows,
    f"{ab_paths['category'][0].name}; {ab_paths['category'][1].name}",
)}
{smoke_section_html(
    "3 · noun_label_AB + shiple (Smith)",
    "First image is a shiple; which A or B is also a shiple?",
    noun_rows,
    f"{ab_paths['noun'][0].name}; {ab_paths['noun'][1].name}",
)}
{cmp_no_word}
{noun_comparisons}
<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / f"smith_prompt_compare_30trials_{date_tag}.html"
    out.write_text(body)
    return out


def write_index_html() -> Path:
    """Landing page grouping playground HTML reports by date/stimuli set."""
    reports = [
        ("2026-07-30 — AB label fix (Image A/B slots)", [
            ("Master interpretation report", "master_interpretation_2026-07-30.html"),
            ("Master CSV", "master_interpretation_2026-07-30.csv"),
        ]),
        ("2026-07-24 — local (revised category)", [
            ("Numeric ladder + Qwen8", "local_models_numeric_and_qwen8_30trials_2026-07-24.html"),
            ("A/B prompt compare (n=30)", "local_models_prompt_compare_30trials_2026-07-24.html"),
            ("Category revision audit", "no_word_category_prompt_revision_2026-07-24.html"),
            ("Category revision CSV", "no_word_category_prompt_revision_2026-07-24.csv"),
        ]),
        ("2026-07-25 — Smith stimuli", [
            ("Numeric ladder", "smith_numeric_and_qwen8_30trials_2026-07-25.html"),
            ("A/B prompt compare", "smith_prompt_compare_30trials_2026-07-25.html"),
            ("PriDe / swap / full-perm", "smith_prompt_pride_debias_2026-07-25.html"),
        ]),
        ("2026-07-17 — local baseline", [
            ("Similarity smoke (n=5) + probe readouts", "local_models_smoke_similarity_2026-07-17.html"),
            ("no_word_category_AB smoke (n=5)", "local_models_smoke_no_word_category_AB_2026-07-17.html"),
            ("Prompt wording interpretation", "prompt_wording_interpretation_2026-07-17.html"),
            ("Sudo-word generality", "local_models_sudo_word_generality_30trials_2026-07-17.html"),
            ("Gated naming contrast", "gated_naming_contrast_2026-07-17.html"),
            ("Prompt PriDe", "prompt_pride_debias_2026-07-17.html"),
            ("Vision vs language", "vision_vs_language_2026-07-17.html"),
            ("July 10 probe (copy)", "probe-experiment-results.html"),
        ]),
    ]
    sections = []
    for heading, links in reports:
        items = []
        for label, href in links:
            path = PLAY / href
            status = "ready" if path.is_file() else "pending"
            items.append(
                f'<li><a href="{html.escape(href)}">{html.escape(label)}</a> '
                f'<span class="dim">({status})</span></li>'
            )
        sections.append(f"<h2>{html.escape(heading)}</h2><ul>{''.join(items)}</ul>")
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Playground results index</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Playground results</h1>
<p class="sub">FarmShare local VLM smoke + probe readouts.
Column glossary: <a href="REPORT_GLOSSARY.md"><code>REPORT_GLOSSARY.md</code></a>.
Session logs: <code>session_*_farmshare/</code> · job stdout: <code>jobs/</code>.</p>
{''.join(sections)}
<p class="src">Generated by <code>scripts/build_playground_results_html.py</code>.</p>
</main>
</body>
</html>
"""
    out = PLAY / "index.html"
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

    cat_audit = try_build_category_revision_report()
    if cat_audit:
        print(f"wrote {cat_audit}")
    else:
        print("category revision audit not ready yet")

    smith_num = try_build_smith_numeric_report()
    if smith_num:
        print(f"wrote {smith_num}")
    else:
        print("Smith numeric ladder not ready yet")

    smith_ab = try_build_smith_prompt_compare()
    if smith_ab:
        print(f"wrote {smith_ab}")
    else:
        print("Smith A/B prompt compare not ready yet")

    idx = write_index_html()
    print(f"wrote {idx}")


if __name__ == "__main__":
    main()
