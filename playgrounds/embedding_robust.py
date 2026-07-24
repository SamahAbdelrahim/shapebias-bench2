#!/usr/bin/env python3
"""Point 2 robustness battery — hardens the representation-vs-behavior dissociation.

For each VLM we extract SEVERAL representations of every image on the same 30
stimuli and, for each, score the Tartaglini shape preference
    shape_pref = 1[ cos(ref, shape_match) > cos(ref, texture_match) ].

Representations (each tried independently; failures are recorded, not fatal):
  proj_mean       : get_image_features() token features, mean-pooled (LM space; baseline)
  vit_last_mean   : vision-tower last_hidden_state, mean-pooled
  vit_penult_mean : vision-tower penultimate hidden state, mean-pooled (Tartaglini locus)
  vit_pooler      : vision-tower pooler_output when present (CLS-style summary)

For every representation we report, with and without mean-centering:
  - shape rate with a bootstrap 95% CI over stimuli (n=30)
  - a POSITIVE CONTROL: object-identity retrieval. For each reference, rank all
    shape-match images by cosine; is its own stimulus top-1? (chance = 1/30).
    Same for texture matches. High retrieval => the read-out resolves these
    images, so a ~0.5 shape-vs-texture rate is a real null, not a blind probe.

Centering: deep pooled features are anisotropic (raw cosines ~1.0); we mean-center
across the stimulus set before cosine. Both raw and centered are reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from evaluation_pipe.models import create_model

LADDER_MODELS = [
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-8b",
    "internvl",
    "internvl-2b",
    "internvl-8b",
    "internvl-14b",
]


def _pool_mean(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    if t.dim() == 1:
        return t
    return t.reshape(-1, t.shape[-1]).mean(dim=0)


def _find_tower(model):
    core = getattr(model, "model", model)
    for attr in ("visual", "vision_tower", "vision_model"):
        tower = getattr(core, attr, None) or getattr(model, attr, None)
        if tower is not None:
            return tower
    return None


def extract_reps(wrapper, img: Image.Image) -> dict[str, torch.Tensor]:
    """Return {rep_name: pooled embedding [D] on CPU}. Missing reps are omitted."""
    model = wrapper._model
    processor = wrapper._processor
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    vi = processor.image_processor(images=[img], return_tensors="pt")
    pv = vi["pixel_values"].to(device=device, dtype=dtype)
    grid = vi["image_grid_thw"].to(device) if "image_grid_thw" in vi else None

    reps: dict[str, torch.Tensor] = {}

    # --- baseline: get_image_features (LM-projected token features) ---
    try:
        with torch.inference_mode():
            feats = model.get_image_features(
                pixel_values=pv, **({"image_grid_thw": grid} if grid is not None else {})
            )
        hs = feats.last_hidden_state if hasattr(feats, "last_hidden_state") else feats
        if isinstance(hs, (list, tuple)):
            hs = hs[0]
        reps["proj_mean"] = _pool_mean(hs).cpu()
    except Exception as exc:  # noqa: BLE001
        reps["proj_mean__err"] = f"{type(exc).__name__}: {exc}"  # type: ignore

    # --- vision tower hidden states ---
    tower = _find_tower(model)
    if tower is not None:
        kw = {"grid_thw": grid} if grid is not None else {}
        out = None
        try:
            with torch.inference_mode():
                out = tower(pv, output_hidden_states=True, **kw)
        except TypeError:
            try:
                with torch.inference_mode():
                    out = tower(pv, **kw)
            except Exception as exc:  # noqa: BLE001
                reps["vit__err"] = f"call: {type(exc).__name__}: {exc}"  # type: ignore
        except Exception as exc:  # noqa: BLE001
            reps["vit__err"] = f"call: {type(exc).__name__}: {exc}"  # type: ignore

        if out is not None:
            last = getattr(out, "last_hidden_state", None)
            if last is not None:
                reps["vit_last_mean"] = _pool_mean(last).cpu()
            hidden = getattr(out, "hidden_states", None)
            if hidden is not None and len(hidden) >= 2:
                reps["vit_penult_mean"] = _pool_mean(hidden[-2]).cpu()
            pooler = getattr(out, "pooler_output", None)
            if pooler is not None:
                reps["vit_pooler"] = _pool_mean(pooler).cpu()

    return reps


def _shape_rate(ref, shp, tex, center: bool):
    """Per-stimulus shape preferences (list of 0/1) for one representation."""
    R = torch.stack(ref)
    S = torch.stack(shp)
    T = torch.stack(tex)
    if center:
        mu = torch.cat([R, S, T]).mean(dim=0)
        R, S, T = R - mu, S - mu, T - mu
    cs = F.cosine_similarity(R, S, dim=1)
    ct = F.cosine_similarity(R, T, dim=1)
    prefs = (cs > ct).float().tolist()
    margin = (cs - ct).mean().item()
    return prefs, margin, R, S, T


def _bootstrap_ci(prefs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(prefs)
    means = []
    for _ in range(n_boot):
        s = [prefs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def _retrieval_at1(R, X):
    """Positive control: for each i, is argmax_j cos(R_i, X_j) == i? Return accuracy."""
    Rn = F.normalize(R, dim=1)
    Xn = F.normalize(X, dim=1)
    sim = Rn @ Xn.t()  # [n, n]
    top1 = sim.argmax(dim=1)
    correct = (top1 == torch.arange(len(R))).float().mean().item()
    return correct


_CC_PAT = re.compile(r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$")


def build_cueconflict_triplets(root: Path, n: int, seed: int):
    """Familiar-category positive control from the Geirhos cue-conflict set.

    Mirrors the novel-object 2AFC exactly:
      ref          = cue-conflict image (shape class S exemplar se, texture class T exemplar te)
      shape match  = SAME shape exemplar (S, se) restyled with a texture class not in {S, T}
      texture match= SAME texture exemplar (T, te) applied to a shape class not in {S, T}
    Same-class images (S == T) are excluded, following Geirhos et al. Triplets are
    balanced across the 16 shape classes.
    """
    items = []  # (path, S, se, T, te)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            m = _CC_PAT.match(f.name)
            if not m:
                continue
            S, se, T, te = m.groups()
            if S == T:
                continue
            items.append((f, S, se, T, te))

    by_shape_ex: dict[tuple, list] = {}
    by_tex_ex: dict[tuple, list] = {}
    for it in items:
        by_shape_ex.setdefault((it[1], it[2]), []).append(it)
        by_tex_ex.setdefault((it[3], it[4]), []).append(it)

    rng = random.Random(seed)
    rng.shuffle(items)
    per_class_cap = max(1, -(-n // 16))  # ceil(n/16), balances shape classes
    used: dict[str, int] = {}
    triplets = []
    for f, S, se, T, te in items:
        if used.get(S, 0) >= per_class_cap:
            continue
        shape_cands = [x for x in by_shape_ex[(S, se)] if x[0] != f and x[3] not in (S, T)]
        tex_cands = [x for x in by_tex_ex[(T, te)] if x[0] != f and x[1] not in (S, T)]
        if not shape_cands or not tex_cands:
            continue
        sc = rng.choice(shape_cands)
        tc = rng.choice(tex_cands)
        triplets.append(
            (
                f"{f.stem}|s:{sc[0].stem}|t:{tc[0].stem}",
                Image.open(f).convert("RGB"),
                Image.open(sc[0]).convert("RGB"),
                Image.open(tc[0]).convert("RGB"),
            )
        )
        used[S] = used.get(S, 0) + 1
        if len(triplets) >= n:
            break
    return triplets

_SMITH_PAT = re.compile(
    r"^smith_bg(\d+)_(\d+)(probe|color_match|shape_match)\.jpg$"
)

def build_smith_probe_triplets(root: Path, n: int, seed: int):
    """Build triplets from the Smith probe set.

    Each triplet uses the same background and trial:
      ref          = smith_bgX_Yprobe.jpg
      shape match  = smith_bgX_Yshape_match.jpg
      color match  = smith_bgX_Ycolor_match.jpg

    The evaluation tests whether the probe representation is closer to the
    shape-preserving or color-preserving match.
    """
    items: dict[tuple[str, str], dict[str, Path]] = {}

    for f in sorted(root.iterdir()):
        m = _SMITH_PAT.match(f.name)
        if not m:
            continue

        bg, trial, kind = m.groups()
        items.setdefault((bg, trial), {})[kind] = f

    rng = random.Random(seed)
    keys = list(items.keys())
    rng.shuffle(keys)

    triplets = []
    for bg, trial in keys:
        imgs = items[(bg, trial)]

        if not all(k in imgs for k in ("probe", "shape_match", "color_match")):
            continue

        triplets.append(
            (
                f"bg:{bg}|trial:{trial}",
                Image.open(imgs["probe"]).convert("RGB"),
                Image.open(imgs["shape_match"]).convert("RGB"),
                Image.open(imgs["color_match"]).convert("RGB"),
            )
        )

        if len(triplets) >= n:
            break

    return triplets

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LADDER_MODELS)
    ap.add_argument("--n-stimuli", type=int, default=30)
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix (default: results/probe.results/session_*/embedding_robust)",
    )
    ap.add_argument(
        "--cue-conflict",
        default=None,
        help="Path to Geirhos cue-conflict dir; if set, run the familiar-category "
        "positive control on these stimuli instead of IMAGE_DATASET.",
    )
    ap.add_argument(
        "--smith-probe",
        default=None,
        help="Path to Linda Smith probe-shapematch-colormatch dataset; if set, run the familiar-category "
        "positive control on these stimuli instead of IMAGE_DATASET.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out_prefix is None:
        from evaluation_pipe.eval_core import default_session_results_dir

        if args.cue_conflict:
            name = "embedding_cueconflict"
        elif args.smith_probe:
            name = "embedding_smith_probe"
        else:
            name = "embedding_robust"
        args.out_prefix = str(default_session_results_dir("probe") / name)

    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("No GPU available")
        return 1

    out_txt = Path(f"{args.out_prefix}.txt")
    out_json = Path(f"{args.out_prefix}.json")
    if not out_txt.is_absolute():
        out_txt = REPO_ROOT / out_txt
        out_json = REPO_ROOT / out_json
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

    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Device: {device}")
    print(f"Models: {args.models}")

    if args.cue_conflict:
        cc_root = Path(args.cue_conflict)
        triplets = build_cueconflict_triplets(cc_root, args.n_stimuli, args.seed)
        print(f"POSITIVE CONTROL: {len(triplets)} familiar-category cue-conflict triplets from {cc_root}")
        for tid, *_ in triplets[:5]:
            print(f"  e.g. {tid}")
    elif args.smith_probe:
        triplets = build_smith_probe_triplets(
            Path(args.smith_probe),
            args.n_stimuli,
            args.seed,
        )
    else:
        dataset = Path(os.environ["IMAGE_DATASET"])
        if not dataset.is_absolute():
            dataset = REPO_ROOT / dataset

        trial_dirs = sorted(
            (d for d in dataset.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )[: args.n_stimuli]
        print(f"Stimuli: {len(trial_dirs)} from {dataset}")

        triplets = [
            (
                int(d.name),
                Image.open(d / "reference.png").convert("RGB"),
                Image.open(d / "shape_match.png").convert("RGB"),
                Image.open(d / "texture_match.png").convert("RGB"),
            )
            for d in trial_dirs
        ]

    all_results: dict = {"config": vars(args), "models": {}}

    for name in args.models:
        print("\n" + "#" * 72)
        print(f"MODEL: {name}")
        print("#" * 72)
        wrapper = None
        model_out: dict = {}
        try:
            device = "cuda" if torch.cuda.is_available() else "mps"
            wrapper = create_model(name, device=device)

            # Collect per-image reps: rep_name -> {'ref':[...],'shape':[...],'texture':[...]}
            by_rep: dict[str, dict[str, list]] = {}
            errs: dict[str, str] = {}
            for sid, ref_img, shape_img, tex_img in triplets:
                r_ref = extract_reps(wrapper, ref_img)
                r_shp = extract_reps(wrapper, shape_img)
                r_tex = extract_reps(wrapper, tex_img)
                for k, v in r_ref.items():
                    if k.endswith("__err"):
                        errs[k] = v
                        continue
                    if k not in r_shp or k not in r_tex:
                        continue
                    by_rep.setdefault(k, {"ref": [], "shape": [], "texture": []})
                    by_rep[k]["ref"].append(v)
                    by_rep[k]["shape"].append(r_shp[k])
                    by_rep[k]["texture"].append(r_tex[k])

            rep_results = {}
            for rep, data in by_rep.items():
                dim = int(data["ref"][0].shape[-1])
                entry = {"dim": dim, "n": len(data["ref"])}
                for center in (False, True):
                    prefs, margin, R, S, T = _shape_rate(
                        data["ref"], data["shape"], data["texture"], center
                    )
                    rate = sum(prefs) / len(prefs)
                    lo, hi = _bootstrap_ci(prefs)
                    tag = "centered" if center else "raw"
                    entry[tag] = {
                        "shape_rate": rate,
                        "ci95": [lo, hi],
                        "mean_margin": margin,
                    }
                    if center:
                        entry["retrieval_shape_at1"] = _retrieval_at1(R, S)
                        entry["retrieval_texture_at1"] = _retrieval_at1(R, T)
                rep_results[rep] = entry

            model_out = {"reps": rep_results, "errors": errs}
            _print_model(name, rep_results, errs)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"MODEL FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            sys.__stderr__.flush()
            model_out = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if wrapper is not None:
                try:
                    wrapper.unload()
                except Exception:
                    pass
                del wrapper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results["models"][name] = model_out

    _print_summary(all_results)

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)
    print(f"\nJSON: {out_json}")
    print(f"Text: {out_txt}")

    sys.stdout = sys.__stdout__
    fh.close()
    return 0


def _print_model(name, rep_results, errs):
    for rep, e in rep_results.items():
        c = e["centered"]
        print(
            f"  {rep:16} dim={e['dim']:5}  shape(centered)={c['shape_rate']:.2f} "
            f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]  margin={c['mean_margin']:+.4f}  "
            f"retr@1 shape={e['retrieval_shape_at1']:.2f} tex={e['retrieval_texture_at1']:.2f}"
        )
    for k, v in errs.items():
        print(f"  [{k}] {v}")


def _print_summary(res):
    print("\n" + "=" * 78)
    print("SUMMARY — embedding shape rate (centered, 95% CI) across read-outs")
    print("retr@1 = object-identity retrieval (chance=1/30=0.03); high => probe is sensitive")
    print("=" * 78)
    print(f"{'model':14} {'readout':16} {'shape':>5} {'ci95':>13} {'retrS':>6} {'retrT':>6}")
    for name, mo in res["models"].items():
        if "error" in mo:
            print(f"{name:14} ERROR: {mo['error']}")
            continue
        for rep, e in mo["reps"].items():
            c = e["centered"]
            print(
                f"{name:14} {rep:16} {c['shape_rate']:5.2f} "
                f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] "
                f"{e['retrieval_shape_at1']:6.2f} {e['retrieval_texture_at1']:6.2f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
