#!/usr/bin/env python3
"""Pre-batch probe implementing P0/P1 recommendations.

Design (per model):
  conditions   : noun_label, no_word_category
  label sets   : numeric (1/2), ab (A/B)          <- P1 letter-prior ablation
  orders       : shape_first, texture_first       <- P0 both orders
  measures     : generation (behavioral DV) + next-token logits (internal)  <- P2
  stimuli      : up to N (default 30)              <- P1 don't trust ranks < 30

Reported per (model, condition, label_set):
  - parse_rate (generation)
  - image tracking gate: fraction of stimuli whose SHAPE/TEXTURE choice is
    stable across the A/B position swap (>= 0.70 = pass)                     <- P0 gate
  - position bias: fraction choosing the FIRST option (A or 1)
  - swap-corrected shape preference: per-stimulus mean of P(shape) across the
    two orders, which cancels side bias by construction                      <- P0 swap correction
  - shape rate is only interpreted when the tracking gate passes.

Prompts and word list come from evaluation_pipe.eval_core (the real pipeline).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from evaluation_pipe.data import load_trials
from evaluation_pipe.eval_core import PROMPT_TEMPLATES
from evaluation_pipe.models import create_model, list_models

NOTEBOOK_MODELS = [
    "smolvlm",
    "internvl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
]

CONDITIONS = ["noun_label", "no_word_category"]
LABEL_SETS = {
    "numeric": {"suffix": "", "choices": ("1", "2"), "parse": r"[12]", "first": "1"},
    "ab": {"suffix": "_AB", "choices": ("A", "B"), "parse": r"[ABab]", "first": "A"},
}
TRACKING_GATE = 0.70
DEFAULT_WORD = "shiple"  # pseudo-word from WORD_PAIRS


def build_prompt(condition: str, label_set: str, word: str) -> str:
    key = condition + LABEL_SETS[label_set]["suffix"]
    tmpl = PROMPT_TEMPLATES[key]
    return tmpl.format(word=word) if "{word}" in tmpl else tmpl


def parse_choice(text: str, label_set: str) -> str | None:
    pat = LABEL_SETS[label_set]["parse"]
    hits = re.findall(pat, text or "")
    if not hits:
        return None
    tok = hits[-1].upper()
    if label_set == "numeric":
        return "first" if tok == "1" else "second"
    return "first" if tok == "A" else "second"


def relative_probs(logits: list[float]) -> tuple[float, float]:
    m = max(logits)
    e0, e1 = math.exp(logits[0] - m), math.exp(logits[1] - m)
    z = e0 + e1
    return e0 / z, e1 / z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=NOTEBOOK_MODELS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--n-stimuli", type=int, default=30)
    ap.add_argument("--word", default=DEFAULT_WORD)
    ap.add_argument("--out-prefix", default="results/probe_experiment")
    args = ap.parse_args()

    conditions = args.conditions

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run inside an srun GPU allocation.")
        return 1

    out_txt = REPO_ROOT / f"{args.out_prefix}.txt"
    out_json = REPO_ROOT / f"{args.out_prefix}.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *s):
            self.s = s

        def write(self, d):
            for x in self.s:
                x.write(d)
                x.flush()

        def flush(self):
            for x in self.s:
                x.flush()

    fh = open(out_txt, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)

    device = "cuda"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Models: {args.models}")
    print(f"Conditions: {conditions}")
    print(f"Available: {list_models()}")

    dataset = Path(os.environ["IMAGE_DATASET"])
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset

    sf = {t.trial_id: t for t in load_trials(dataset, order="shape_first")}
    tf = {t.trial_id: t for t in load_trials(dataset, order="texture_first")}
    stim_ids = sorted(set(sf) & set(tf))[: args.n_stimuli]
    print(f"Stimuli: {len(stim_ids)} (word={args.word!r})")

    imgs = {}
    for sid in stim_ids:
        imgs[(sid, "sf")] = sf[sid].load_images()
        imgs[(sid, "tf")] = tf[sid].load_images()

    all_results: dict = {"config": vars(args), "models": {}}

    for name in args.models:
        print("\n" + "#" * 72)
        print(f"MODEL: {name}")
        print("#" * 72)
        model = None
        model_out: dict = {}
        try:
            model = create_model(name, device=device)
            print(f"Loaded: {model.name} on {next(model._model.parameters()).device}")

            for condition in conditions:
                for lset in LABEL_SETS:
                    prompt = build_prompt(condition, lset, args.word)
                    choices = LABEL_SETS[lset]["choices"]

                    per_stim = []
                    parse_ok = 0
                    parse_total = 0
                    for sid in stim_ids:
                        rec = {"stim": sid}
                        for order_key, order_map in (("sf", sf), ("tf", tf)):
                            ref, a, b = imgs[(sid, order_key)]
                            shape_is_first = order_key == "sf"

                            gen = model.generate(images=[ref, a, b], prompt=prompt)
                            pick = parse_choice(gen.raw_text, lset)
                            parse_total += 1
                            if pick is not None:
                                parse_ok += 1
                            gen_first = pick == "first"
                            gen_shape = (
                                None if pick is None else (gen_first == shape_is_first)
                            )

                            log_shape = None
                            log_first = None
                            try:
                                sc = model.score_choices(
                                    images=[ref, a, b],
                                    prompt=prompt,
                                    choice_texts=choices,
                                )
                                p_first, p_second = relative_probs(sc["choice_logits"])
                                log_first = p_first
                                log_shape = p_first if shape_is_first else p_second
                            except Exception as exc:
                                rec[f"{order_key}_logit_err"] = f"{type(exc).__name__}: {exc}"

                            rec[f"{order_key}_gen_pick"] = pick
                            rec[f"{order_key}_gen_shape"] = gen_shape
                            rec[f"{order_key}_gen_first"] = gen_first if pick else None
                            rec[f"{order_key}_log_shape"] = log_shape
                            rec[f"{order_key}_log_first"] = log_first
                        per_stim.append(rec)

                    metrics = _aggregate(per_stim)
                    metrics["parse_rate"] = parse_ok / max(parse_total, 1)
                    key = f"{condition}|{lset}"
                    model_out[key] = {"metrics": metrics, "per_stim": per_stim}
                    _print_block(name, condition, lset, prompt, metrics)

        except Exception as exc:
            import traceback

            print(f"MODEL FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            sys.__stderr__.flush()
            model_out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    pass
                del model
            gc.collect()
            torch.cuda.empty_cache()
        all_results["models"][name] = model_out

    _print_final_table(all_results)

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)
    print(f"\nJSON: {out_json}")
    print(f"Text: {out_txt}")

    sys.stdout = sys.__stdout__
    fh.close()
    return 0


def _aggregate(per_stim: list[dict]) -> dict:
    n = len(per_stim)

    def frac(pred):
        vals = [pred(r) for r in per_stim]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else float("nan")

    gen_track_vals = []
    for r in per_stim:
        s_sf, s_tf = r["sf_gen_shape"], r["tf_gen_shape"]
        if s_sf is not None and s_tf is not None:
            gen_track_vals.append(1.0 if s_sf == s_tf else 0.0)
    gen_tracking = (sum(gen_track_vals) / len(gen_track_vals)) if gen_track_vals else float("nan")

    log_track_vals = []
    for r in per_stim:
        ls_sf, ls_tf = r["sf_log_shape"], r["tf_log_shape"]
        if ls_sf is not None and ls_tf is not None:
            log_track_vals.append(1.0 if (ls_sf > 0.5) == (ls_tf > 0.5) else 0.0)
    log_tracking = (sum(log_track_vals) / len(log_track_vals)) if log_track_vals else float("nan")

    swap_shape_vals = []
    for r in per_stim:
        ls_sf, ls_tf = r["sf_log_shape"], r["tf_log_shape"]
        if ls_sf is not None and ls_tf is not None:
            swap_shape_vals.append((ls_sf + ls_tf) / 2)
    swap_shape_logit = (sum(swap_shape_vals) / len(swap_shape_vals)) if swap_shape_vals else float("nan")

    gen_shape_all = []
    for r in per_stim:
        for k in ("sf_gen_shape", "tf_gen_shape"):
            if r[k] is not None:
                gen_shape_all.append(1.0 if r[k] else 0.0)
    gen_shape_rate = (sum(gen_shape_all) / len(gen_shape_all)) if gen_shape_all else float("nan")

    gen_first_rate = frac(lambda r: (
        None if r["sf_gen_first"] is None and r["tf_gen_first"] is None
        else _avg_bools(r["sf_gen_first"], r["tf_gen_first"])
    ))
    log_first_rate = frac(lambda r: (
        None if r["sf_log_first"] is None or r["tf_log_first"] is None
        else (r["sf_log_first"] + r["tf_log_first"]) / 2
    ))

    return {
        "n": n,
        "gen_tracking": gen_tracking,
        "log_tracking": log_tracking,
        "gen_shape_rate": gen_shape_rate,
        "swap_shape_logit": swap_shape_logit,
        "gen_first_rate": gen_first_rate,
        "log_first_rate": log_first_rate,
        "gate_pass": (not math.isnan(gen_tracking)) and gen_tracking >= TRACKING_GATE,
    }


def _avg_bools(*vals):
    xs = [1.0 if v else 0.0 for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(x):
    return "  nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:5.2f}"


def _print_block(model, condition, lset, prompt, m):
    print(f"\n--- {model} | {condition} | {lset} ---")
    print(f"prompt: {prompt}")
    gate = "PASS" if m["gate_pass"] else "FAIL"
    print(
        f"parse={_fmt(m['parse_rate'])}  gen_track={_fmt(m['gen_tracking'])} [{gate}]  "
        f"log_track={_fmt(m['log_tracking'])}"
    )
    print(
        f"first-option bias  gen={_fmt(m['gen_first_rate'])}  log={_fmt(m['log_first_rate'])}"
    )
    if m["gate_pass"]:
        print(
            f"SHAPE preference   gen={_fmt(m['gen_shape_rate'])}  "
            f"logit(swap-corrected)={_fmt(m['swap_shape_logit'])}"
        )
    else:
        print(
            f"SHAPE preference   [gated out: tracking<{TRACKING_GATE}]  "
            f"(ungated gen={_fmt(m['gen_shape_rate'])}, logit_swap={_fmt(m['swap_shape_logit'])})"
        )


def _print_final_table(res):
    print("\n" + "=" * 72)
    print("SUMMARY (swap-corrected; shape pref only meaningful when gate=PASS)")
    print("=" * 72)
    hdr = (
        f"{'model':13} {'condition':16} {'lbl':4} {'parse':>5} {'gTrk':>5} "
        f"{'lTrk':>5} {'gate':>4} {'gShp':>5} {'lShp':>5} {'gPosA':>5}"
    )
    print(hdr)
    for name, mo in res["models"].items():
        if "error" in mo:
            print(f"{name:13} ERROR: {mo['error']}")
            continue
        for key, blk in mo.items():
            m = blk["metrics"]
            cond, lset = key.split("|")
            print(
                f"{name:13} {cond:16} {lset:4} {_fmt(m['parse_rate'])} "
                f"{_fmt(m['gen_tracking'])} {_fmt(m['log_tracking'])} "
                f"{'P' if m['gate_pass'] else 'F':>4} {_fmt(m['gen_shape_rate'])} "
                f"{_fmt(m['swap_shape_logit'])} {_fmt(m['gen_first_rate'])}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
