#!/usr/bin/env python3
"""Vision-tower vs language-side report for the July 17 playground session.

Merges July 10 embedding robust/cue-conflict JSON with the July 17 fill for
smolvlm / qwen3.5-0.8b / qwen3.5-4b, then summarizes today's gated naming and
PriDe language-side results. The claim to evaluate: behavioral shape bias in
these VLMs is produced downstream of the vision encoder.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "results" / "probe.results"
PLAY = REPO / "results" / "playground.results"
JUL10 = PROBE / "session_2026-07-10_farmshare"
JUL17 = PROBE / "session_2026-07-17_farmshare"
OUT = PLAY / "vision_vs_language_2026-07-17.html"

REP_ORDER = ("proj_mean", "vit_last_mean", "vit_penult_mean", "vit_pooler")
MODEL_ORDER = (
    "smolvlm",
    "qwen3.5-0.8b",
    "qwen3-vl-2b",
    "internvl",
    "qwen3-vl-4b",
    "qwen3.5-4b",
    "qwen3-vl-8b",
    "internvl-2b",
    "internvl-8b",
    "internvl-14b",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_embedding(primary: Path, fill: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in (primary, fill):
        data = _load_json(path)
        for name, payload in data.get("models", {}).items():
            out[name] = payload
    return out


def _fmt(x: float | None, digits: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.{digits}f}"


def embedding_rows(models: dict[str, dict], preferred_rep: str = "proj_mean") -> list[dict]:
    rows = []
    names = [m for m in MODEL_ORDER if m in models] + [
        m for m in sorted(models) if m not in MODEL_ORDER
    ]
    for name in names:
        payload = models[name]
        if "error" in payload:
            rows.append(
                {
                    "model": name,
                    "rep": "—",
                    "shape": None,
                    "ci": None,
                    "retr_s": None,
                    "retr_t": None,
                    "error": payload["error"],
                }
            )
            continue
        reps = payload.get("reps", {})
        rep = preferred_rep if preferred_rep in reps else next(iter(reps), None)
        if rep is None:
            continue
        e = reps[rep]
        c = e["centered"]
        rows.append(
            {
                "model": name,
                "rep": rep,
                "shape": c["shape_rate"],
                "ci": c["ci95"],
                "retr_s": e.get("retrieval_shape_at1"),
                "retr_t": e.get("retrieval_texture_at1"),
                "error": None,
                "all_reps": {
                    r: {
                        "shape": reps[r]["centered"]["shape_rate"],
                        "ci": reps[r]["centered"]["ci95"],
                        "retr_s": reps[r].get("retrieval_shape_at1"),
                        "retr_t": reps[r].get("retrieval_texture_at1"),
                    }
                    for r in REP_ORDER
                    if r in reps
                },
            }
        )
    return rows


def table_embed(rows: list[dict], title_note: str) -> str:
    trs = []
    for r in rows:
        if r.get("error"):
            trs.append(
                f'<tr class="fail"><td class="l">{html.escape(r["model"])}</td>'
                f'<td class="l" colspan="5">ERROR: {html.escape(r["error"])}</td></tr>'
            )
            continue
        shape = r["shape"]
        cls = ""
        if shape is not None and 0.40 <= shape <= 0.60:
            cls = "pass"  # chance band: expected under the null
        elif shape is not None and shape < 0.40:
            cls = "fail"  # texture side
        ci = r["ci"]
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        trs.append(
            f'<tr class="{cls}"><td class="l">{html.escape(r["model"])}</td>'
            f'<td class="l">{html.escape(r["rep"])}</td>'
            f"<td>{_fmt(shape)}</td><td>{ci_s}</td>"
            f'<td>{_fmt(r["retr_s"])}</td><td>{_fmt(r["retr_t"])}</td></tr>'
        )
    return f"""
<p>{title_note}</p>
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th class="l">read-out</th>
<th>centred shape</th><th>95% CI</th><th>retr@1 shape</th><th>retr@1 texture</th></tr>
{''.join(trs)}</table></div>
"""


def robust_detail_table(models: dict[str, dict], focus: tuple[str, ...]) -> str:
    """Full four-readout table for the newly filled models."""
    trs = []
    for name in focus:
        payload = models.get(name)
        if not payload or "reps" not in payload:
            trs.append(
                f'<tr class="fail"><td class="l">{html.escape(name)}</td>'
                f'<td class="l" colspan="5">missing</td></tr>'
            )
            continue
        reps = payload["reps"]
        first = True
        for rep in REP_ORDER:
            if rep not in reps:
                continue
            e = reps[rep]
            c = e["centered"]
            ci = c["ci95"]
            trs.append(
                f'<tr><td class="l">{html.escape(name) if first else ""}</td>'
                f'<td class="l">{html.escape(rep)}</td>'
                f'<td>{c["shape_rate"]:.2f}</td>'
                f"<td>[{ci[0]:.2f}, {ci[1]:.2f}]</td>"
                f'<td>{e.get("retrieval_shape_at1", float("nan")):.2f}</td>'
                f'<td>{e.get("retrieval_texture_at1", float("nan")):.2f}</td></tr>'
            )
            first = False
    return f"""
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th class="l">read-out</th>
<th>centred shape</th><th>95% CI</th><th>retr@1 shape</th><th>retr@1 texture</th></tr>
{''.join(trs)}</table></div>
"""


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def language_word_summary(rows: list[dict]) -> str:
    """Across-word gate summary for numeric noun cells."""
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        if r["condition"] != "noun_label" or r["label_set"] != "numeric":
            continue
        by[r["model"]].append(r)
    trs = []
    for model in MODEL_ORDER:
        cells = by.get(model)
        if not cells:
            continue
        passes = [c for c in cells if c["gate"] == "True"]
        trks = [float(c["gen_tracking"]) for c in cells]
        shps = [float(c["gen_shape"]) for c in passes]
        words = ", ".join(c["word"] for c in passes) or "none"
        mean_shp = f"{sum(shps)/len(shps):.2f}" if shps else "—"
        trs.append(
            f'<tr class="{"pass" if passes else ""}">'
            f'<td class="l">{html.escape(model)}</td>'
            f"<td>{len(passes)}/{len(cells)}</td>"
            f"<td>{sum(trks)/len(trks):.2f}</td>"
            f"<td>{min(trks):.2f}–{max(trks):.2f}</td>"
            f"<td>{mean_shp}</td>"
            f'<td class="l">{html.escape(words)}</td></tr>'
        )
    return f"""
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th>gate passes</th><th>mean tracking</th>
<th>tracking range</th><th>mean shape among passes</th><th class="l">passing words</th></tr>
{''.join(trs)}</table></div>
"""


def language_gated_contrast(rows: list[dict]) -> str:
    gated = [r for r in rows if r["both_gates_pass"] == "True"]
    if not gated:
        return "<p>No both-gates-passing cells.</p>"
    trs = []
    for r in gated:
        trs.append(
            f'<tr class="pass"><td class="l">{html.escape(r["model"])}</td>'
            f'<td class="l">{html.escape(r["word"])}</td>'
            f'<td>{float(r["base_gen_shape"]):.2f} → {float(r["noun_gen_shape"]):.2f}</td>'
            f'<td>{float(r["gen_delta"]):+.3f} '
            f'[{float(r["gen_ci_lo"]):+.3f}, {float(r["gen_ci_hi"]):+.3f}]</td>'
            f'<td>{float(r["base_swap_mean_p"]):.2f} → {float(r["noun_swap_mean_p"]):.2f}</td>'
            f'<td>{float(r["swap_delta"]):+.3f} '
            f'[{float(r["swap_ci_lo"]):+.3f}, {float(r["swap_ci_hi"]):+.3f}]</td>'
            f'<td>{r["sign_pos"]}/{r["sign_neg"]}</td>'
            f'<td>{float(r["sign_p"]):.4f}</td></tr>'
        )
    return f"""
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th class="l">word</th>
<th>gen shape: no-word → noun</th><th>gen Δ [95% CI]</th>
<th>swap P(shape): no-word → noun</th><th>swap Δ [95% CI]</th>
<th>sign +/-</th><th>sign p</th></tr>
{''.join(trs)}</table></div>
"""


def language_pride_highlights(rows: list[dict]) -> str:
    """Key latent cells for models that matter."""
    want = {
        ("qwen3.5-4b", "no_word_similarity", "(none)", "numeric"),
        ("qwen3.5-4b", "noun_label", "shiple", "numeric"),
        ("qwen3.5-4b", "noun_label", "clapher", "numeric"),
        ("qwen3.5-4b", "no_word_category", "(none)", "numeric"),
        ("qwen3-vl-8b", "noun_label", "shiple", "numeric"),
        ("qwen3-vl-8b", "noun_label", "clapher", "numeric"),
        ("qwen3-vl-8b", "noun_label_AB", "shiple", "AB"),
        ("qwen3.5-4b", "noun_label_AB", "shiple", "AB"),
    }
    trs = []
    for r in rows:
        key = (r["model"], r["condition"], r["word"], r["label_set"])
        if key not in want:
            continue
        cls = "pass" if r["gate"] == "True" else ""
        trs.append(
            f'<tr class="{cls}"><td class="l">{html.escape(r["model"])}</td>'
            f'<td class="l">{html.escape(r["condition"])}</td>'
            f'<td class="l">{html.escape(r["word"])}</td>'
            f'<td>{r["label_set"]}</td>'
            f'<td>{float(r["gen_tracking"]):.2f}</td>'
            f'<td>{"PASS" if r["gate"] == "True" else "fail"}</td>'
            f'<td>{float(r["prior_first"]):.2f}</td>'
            f'<td>{float(r["swap_mean_p_shape"]):.2f}</td>'
            f'<td>{float(r["fullperm_mean_p_shape"]):.2f}</td>'
            f'<td>{float(r["pride_sf_mean_p_shape"]):.2f} / '
            f'{float(r["pride_tf_mean_p_shape"]):.2f}</td></tr>'
        )
    return f"""
<div class="tablewrap"><table>
<tr><th class="l">Model</th><th class="l">condition</th><th class="l">word</th>
<th>labels</th><th>gTrk</th><th>gate</th><th>priorFirst</th>
<th>swap meanP</th><th>perm meanP</th><th>PriDe SF / TF</th></tr>
{''.join(trs)}</table></div>
"""


def build() -> Path:
    novel = merge_embedding(
        JUL10 / "embedding_robust.json",
        JUL17 / "embedding_robust_fill.json",
    )
    cue = merge_embedding(
        JUL10 / "embedding_cueconflict.json",
        JUL17 / "embedding_cueconflict_fill.json",
    )
    fill_names = ("smolvlm", "qwen3.5-0.8b", "qwen3.5-4b")
    fill_ready = all(
        name in novel and "reps" in novel[name] for name in fill_names
    )

    pride = load_csv(PLAY / "prompt_pride_debias_2026-07-17.csv")
    gated = load_csv(PLAY / "gated_naming_contrast_2026-07-17.csv")
    word_gen = load_csv(PLAY / "prompt_pride_debias_2026-07-17.csv")

    novel_rows = embedding_rows(novel)
    cue_rows = embedding_rows(cue)

    status = (
        "July 17 fill complete for smolvlm, qwen3.5-0.8b, and qwen3.5-4b."
        if fill_ready
        else (
            "July 17 fill not yet available. Tables below show the July 10 ladder "
            "only; re-run this builder after "
            "<code>session_2026-07-17_farmshare/embedding_*_fill.json</code> lands."
        )
    )

    q35 = None
    if "qwen3.5-4b" in novel and "reps" in novel["qwen3.5-4b"]:
        q35 = novel["qwen3.5-4b"]["reps"].get("proj_mean", {}).get("centered", {}).get(
            "shape_rate"
        )
    tile_encoder = _fmt(q35) if q35 is not None else "≈ 0.5"

    # Downstream claim: models with behavioral gate pass vs their embed shape
    claim_trs = []
    behav = {
        "qwen3-vl-8b": (
            "0.80–0.83 (July 10 noun 1/2; playground numeric noun ~0.83–0.92)",
            0.83,
        ),
        "qwen3.5-4b": (
            "0.82–0.95 among gate-passing numeric noun words today",
            0.88,
        ),
    }
    for model, (behav_note, behav_val) in behav.items():
        emb = None
        if model in novel and "reps" in novel[model]:
            emb = novel[model]["reps"].get("proj_mean", {}).get("centered", {}).get(
                "shape_rate"
            )
        claim_trs.append(
            f'<tr class="pass"><td class="l">{html.escape(model)}</td>'
            f'<td class="l">{html.escape(behav_note)}</td>'
            f"<td>{_fmt(emb)}</td>"
            f"<td>{_fmt(behav_val - emb if emb is not None else None)}</td></tr>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision tower vs language side — 2026-07-17</title>
<style>
  :root {{
    --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --card: #fff;
    --border: #ddd; --green-bg: #e6f4ea; --green-fg: #1e7d3c;
    --red-bg: #fdecea; --red-fg: #b3261e; --amber-bg: #fff4e0;
    --amber-fg: #8a5a00; --blue-bg: #e8f0fe; --blue-fg: #1a56b0;
    --code-bg: #f0f0f0;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16181c; --fg: #e8e8e8; --muted: #9aa0a6; --card: #1f2228;
      --border: #3a3f47; --green-bg: #1d3524; --green-fg: #7ddc94;
      --red-bg: #3b2220; --red-fg: #f28b82; --amber-bg: #3a2f18;
      --amber-fg: #fdd663; --blue-bg: #1c2a41; --blue-fg: #8ab4f8;
      --code-bg: #2a2e35;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.2rem; margin: 36px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ font-size: 1.02rem; margin: 20px 0 8px; }}
  p.sub {{ color: var(--muted); margin-top: 4px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
  .tile {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .tile .v {{ font-size: 1.5rem; font-weight: 700; }}
  .tile .l {{ font-size: 0.82rem; color: var(--muted); }}
  .callout {{ border-left: 4px solid; border-radius: 6px; padding: 12px 16px; margin: 14px 0; background: var(--card); }}
  .callout.danger {{ border-color: var(--red-fg); }}
  .callout.warn   {{ border-color: var(--amber-fg); }}
  .callout.info   {{ border-color: var(--blue-fg); }}
  .callout.good   {{ border-color: var(--green-fg); }}
  .callout b.t {{ display: block; margin-bottom: 4px; }}
  .tablewrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; background: var(--card); }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }}
  th {{ position: sticky; top: 0; background: var(--card); }}
  th:first-child, td:first-child, th.l, td.l {{ text-align: left; }}
  tr.pass td {{ background: var(--green-bg); }}
  tr.pass td:first-child {{ color: var(--green-fg); font-weight: 600; }}
  tr.lock td {{ background: var(--red-bg); }}
  tr.fail td {{ background: var(--amber-bg); }}
  code {{ background: var(--code-bg); border-radius: 4px; padding: 1px 5px; font-size: 0.85em; }}
  .src {{ color: var(--muted); font-size: 0.8rem; margin: 6px 0 0; }}
  ul.key {{ font-size: 0.9rem; color: var(--muted); padding-left: 18px; }}
  ul.key b {{ color: var(--fg); }}
</style></head><body><main>
<h1>Vision tower vs language side</h1>
<p class="sub">Same 30 novel stimuli. Vision-tower read-outs have no prompt, no
choice labels, and no pseudo-word. Language-side results are today's playground
2AFC (generation + swap/PriDe-corrected logits). {status}</p>

<div class="tiles">
  <div class="tile"><div class="v">{tile_encoder}</div><div class="l">qwen3.5-4b encoder centred shape (proj_mean) on the same novel stimuli</div></div>
  <div class="tile"><div class="v">0.82–0.95</div><div class="l">generation shape among today's gate-passing numeric-noun cells</div></div>
  <div class="tile"><div class="v">4 / 35</div><div class="l">both-gates-passing naming contrasts (all qwen3.5-4b)</div></div>
  <div class="tile"><div class="v">downstream</div><div class="l">behavioral bias appears after the vision encoder</div></div>
</div>

<div class="callout good"><b class="t">Claim.</b>
The behavioral shape bias measured in positioned 2AFC is not present in the
vision encoder's geometry on these stimuli. Where generation tracks the images
and prefers shape, the same model's centred embedding shape rate sits near
chance. That places the bias in the language / decision stage.</div>

<h2>A · Vision tower (encoder / representation locus)</h2>
<p>Tartaglini-style score: for each stimulus,
<code>1[cos(ref, shape_match) &gt; cos(ref, texture_match)]</code>, after
mean-centering across the 30-stimulus set to remove feature anisotropy.
Retrieval@1 is the positive control that the read-out resolves object identity
(chance = 1/30 ≈ 0.03).</p>

<div class="callout info"><b class="t">Why this locus is clean for the encoder claim.</b>
No positioned options, so letter/position priors cannot contaminate the score.
The live confound for embedding work is feature anisotropy (raw cosines near 1);
centering addresses that. Literature at this locus (Ritter 2017; Tartaglini,
Vong &amp; Lake 2022; Muttenthaler et al. 2025) is measuring a different
construct than positioned word extension.</div>

<h3>A1 · Novel stimuli (same set as the playground)</h3>
{table_embed(novel_rows, "Primary read-out shown: <code>proj_mean</code> (LM-projected image tokens). Green = chance band [0.40, 0.60].")}

<h3>A2 · July 17 fill — all four read-outs for the missing models</h3>
<p>These three models carried today's playground results but had no July 10
encoder probe. Same battery as the ladder. Notes: SmolVLM's vision-tower
hidden-state path failed (<code>vit__err</code>: unpack error); only
<code>proj_mean</code> is available, and it leans texture (0.10) with
usable retrieval (0.53). qwen3.5-0.8b also leans texture (0.27) with weak
proj retrieval. qwen3.5-4b is the critical cell: centred shape 0.53
(CI spans 0.5) with penultimate retrieval@1 = 1.00.</p>
{robust_detail_table(novel, fill_names) if fill_ready else "<p class='src'>Fill JSON not present yet.</p>"}

<h3>A3 · Geirhos familiar-category cue-conflict (sensitivity control)</h3>
{table_embed(cue_rows, "Same read-out. Texture preference (shape rate well below 0.5) is the literature-expected direction for these encoders on familiar cue-conflict images.")}

<p class="src">Sources: <code>session_2026-07-10_farmshare/embedding_{{robust,cueconflict}}.json</code>
and <code>session_2026-07-17_farmshare/embedding_*_fill.json</code>.</p>

<h2>B · Language side (positioned 2AFC locus) — today's results</h2>
<p>Prompt, choice labels, and novel word all live here. This is the locus that
matches the developmental word-extension task and that carries option-symbol,
option-position, and image-order artifacts.</p>

<h3>B1 · Sudo-word generality (numeric noun)</h3>
<p>Five curated words × 7 models. The shiple pattern is word-general: A/B
collapses everywhere; numeric rescues the larger Qwens; among gate-passing
cells, shape preference is high and stable across words.</p>
{language_word_summary(word_gen)}

<h3>B2 · Gated naming contrast (numeric no-word similarity vs numeric noun)</h3>
<p>Strict generation-level comparison: both cells must pass tracking ≥ 0.70.
Only qwen3.5-4b qualifies. Against the best label-free framing, adding the noun
lowers swap-corrected P(shape); generation is flat or slightly lower. The
earlier “label raises latent shape” reading held only against the depressed
category baseline.</p>
{language_gated_contrast(gated)}

<h3>B3 · Latent evidence (swap / full permutation / PriDe)</h3>
<p>Selected cells. Generation gate still bounds behavioral claims; corrected
logits describe latent choice evidence even when generation locks.</p>
{language_pride_highlights(pride)}

<p class="src">Sources:
<code>gated_naming_contrast_2026-07-17.{{csv,html}}</code>,
<code>prompt_pride_debias_2026-07-17.{{csv,html}}</code>,
<code>local_models_sudo_word_generality_30trials_2026-07-17.html</code>.</p>

<h2>C · Is the bias produced downstream?</h2>
<div class="callout good"><b class="t">Yes, on the present evidence.</b>
For every model that shows a gate-passing behavioral shape preference, the
encoder centred shape rate on the same stimuli remains near chance. The gap is
large and estimator-invariant on the language side (swap, full permutation,
PriDe agree when priors are not extreme).</div>

<div class="tablewrap"><table>
<tr><th class="l">Model</th><th class="l">Language-side shape (gate-passing)</th>
<th>Encoder centred shape (proj_mean)</th><th>gap</th></tr>
{''.join(claim_trs)}
</table></div>

<ul class="key">
  <li><b>What “downstream” means here.</b> The vision tower does not prefer the
  shape match over the texture match in cosine geometry. The preference appears
  when the language model is asked to choose among positioned options under a
  prompt. The prompt and the pseudo-word never enter the encoder, so condition
  effects (similarity vs category vs noun; A/B vs 1/2; word identity) are
  language-side by construction.</li>
  <li><b>What the encoder null is not.</b> It is not a blind probe. Retrieval@1
  for shape-match identity is high on the sensitive read-outs (often 0.77–1.00
  vs chance 0.03), and the Geirhos control recovers texture preference. The
  null is about shape-vs-texture organization of these novel stimuli, not about
  whether the tower sees the images.</li>
  <li><b>How to read today's language evidence with this dissociation.</b>
  (1) Validity (tracking) is language-side: letter locks and category-frame
  locks are option priors, not encoder failures. (2) Shape preference among
  valid cells is language-side: high generation and corrected-logit shape rates
  coexist with chance encoder geometry. (3) Naming effects are language-side
  and baseline-dependent: relative to similarity, the noun slightly reduces
  latent shape in qwen3.5-4b; relative to category, it raises it. Neither
  pattern can be attributed to a change in the vision tower.</li>
  <li><b>Open limit.</b> This report still measures the encoder with a single
  image at a time. It does not probe LM hidden states after the prompt has
  fused with image tokens. That would localize <em>where</em> in the language
  stack the preference forms; it is not required for the encoder-vs-downstream
  claim.</li>
</ul>

<p class="src" style="margin-top:28px">Generated by
<code>scripts/build_vision_vs_language_report.py</code>.</p>
</main></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    return OUT


def main() -> int:
    out = build()
    print(f"Wrote {out}")
    fill = JUL17 / "embedding_robust_fill.json"
    print(f"fill_present={fill.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
