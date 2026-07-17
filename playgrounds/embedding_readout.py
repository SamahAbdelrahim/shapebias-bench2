#!/usr/bin/env python3
"""Point 2: Tartaglini-style representational read-out on the same stimuli.

For each VLM, embed reference / shape_match / texture_match images through the
model's own vision tower (no options, no answer tokens -- position and label
bias are structurally impossible), then score

    shape_pref = 1[ cos(ref, shape_match) > cos(ref, texture_match) ]

per stimulus (Tartaglini, Vong & Lake, 2022, adapted to VLM vision encoders).
Crossing this with the behavioral 2AFC results (probe_scaling_noun) gives the
representation-vs-behavior dissociation table.

Embedding extraction: tries `model.get_image_features(...)` first (standard
transformers VLM API), falling back to common vision-tower attributes. The
patch/token embeddings are mean-pooled to a single vector per image.
"""

from __future__ import annotations

import argparse
import json
import gc
import os
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


def _pool(t: torch.Tensor) -> torch.Tensor:
    """Mean-pool arbitrary [.., D] patch embeddings to a single [D] vector."""
    t = t.float()
    if t.dim() == 1:
        return t
    return t.reshape(-1, t.shape[-1]).mean(dim=0)


def _first_tensor(obj) -> torch.Tensor:
    """Unwrap tuples/lists/model-outputs down to one tensor of embeddings."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        parts = [_first_tensor(o) for o in obj if o is not None]
        if len(parts) == 1:
            return parts[0]
        return torch.cat([p.reshape(-1, p.shape[-1]) for p in parts], dim=0)
    if hasattr(obj, "last_hidden_state"):
        return obj.last_hidden_state
    raise TypeError(f"Cannot extract tensor from {type(obj)}")


def embed_image(wrapper, img: Image.Image) -> tuple[torch.Tensor, str]:
    """Return (pooled embedding [D] on CPU, extraction-path label)."""
    model = wrapper._model
    processor = wrapper._processor
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    vis_inputs = processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = vis_inputs["pixel_values"].to(device=device, dtype=dtype)
    extra = {}
    if "image_grid_thw" in vis_inputs:
        extra["image_grid_thw"] = vis_inputs["image_grid_thw"].to(device)

    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(pixel_values=pixel_values, **extra)
            return _pool(_first_tensor(feats)).cpu(), "get_image_features"

        core = getattr(model, "model", model)
        for attr in ("visual", "vision_tower", "vision_model"):
            tower = getattr(core, attr, None)
            if tower is not None:
                if extra:
                    feats = tower(pixel_values, grid_thw=extra["image_grid_thw"])
                else:
                    feats = tower(pixel_values)
                return _pool(_first_tensor(feats)).cpu(), f"tower:{attr}"

    raise RuntimeError(f"No vision feature path found for {wrapper.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LADDER_MODELS)
    ap.add_argument("--n-stimuli", type=int, default=30)
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix (default: results/probe.results/session_*/embedding_readout)",
    )
    args = ap.parse_args()

    if args.out_prefix is None:
        from evaluation_pipe.eval_core import default_session_results_dir

        args.out_prefix = str(
            default_session_results_dir("probe") / "embedding_readout"
        )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run inside an srun GPU allocation.")
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

    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Models: {args.models}")

    dataset = Path(os.environ["IMAGE_DATASET"])
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset

    trial_dirs = sorted(
        (d for d in dataset.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )[: args.n_stimuli]
    print(f"Stimuli: {len(trial_dirs)} from {dataset}")

    triplets = []
    for d in trial_dirs:
        triplets.append(
            (
                int(d.name),
                Image.open(d / "reference.png").convert("RGB"),
                Image.open(d / "shape_match.png").convert("RGB"),
                Image.open(d / "texture_match.png").convert("RGB"),
            )
        )

    all_results: dict = {"config": vars(args), "models": {}}

    for name in args.models:
        print("\n" + "#" * 72)
        print(f"MODEL: {name}")
        print("#" * 72)
        wrapper = None
        model_out: dict = {}
        try:
            wrapper = create_model(name, device="cuda")
            print(f"Loaded: {wrapper.name}")

            # Pass 1: collect pooled embeddings for every image.
            embs: dict[tuple[int, str], torch.Tensor] = {}
            path_label = None
            for sid, ref_img, shape_img, tex_img in triplets:
                embs[(sid, "ref")], path_label = embed_image(wrapper, ref_img)
                embs[(sid, "shape")], _ = embed_image(wrapper, shape_img)
                embs[(sid, "tex")], _ = embed_image(wrapper, tex_img)

            # Centering: deep pooled features are highly anisotropic (a shared
            # dominant direction makes all raw cosines ~1.0). Subtracting the
            # mean embedding across the whole stimulus set removes it.
            mean_vec = torch.stack(list(embs.values())).mean(dim=0)

            def _cos(a: torch.Tensor, b: torch.Tensor, center: bool) -> float:
                if center:
                    a, b = a - mean_vec, b - mean_vec
                return F.cosine_similarity(a, b, dim=0).item()

            per_stim = []
            for sid, _, _, _ in triplets:
                e_ref, e_shape, e_tex = embs[(sid, "ref")], embs[(sid, "shape")], embs[(sid, "tex")]
                raw_s, raw_t = _cos(e_ref, e_shape, False), _cos(e_ref, e_tex, False)
                cen_s, cen_t = _cos(e_ref, e_shape, True), _cos(e_ref, e_tex, True)
                per_stim.append(
                    {
                        "stim": sid,
                        "cos_shape_raw": raw_s,
                        "cos_texture_raw": raw_t,
                        "margin_raw": raw_s - raw_t,
                        "shape_pref_raw": raw_s > raw_t,
                        "cos_shape_centered": cen_s,
                        "cos_texture_centered": cen_t,
                        "margin_centered": cen_s - cen_t,
                        "shape_pref_centered": cen_s > cen_t,
                    }
                )

            n = len(per_stim)
            rate_raw = sum(r["shape_pref_raw"] for r in per_stim) / n
            rate_cen = sum(r["shape_pref_centered"] for r in per_stim) / n
            margin_raw = sum(r["margin_raw"] for r in per_stim) / n
            margin_cen = sum(r["margin_centered"] for r in per_stim) / n
            dim = int(mean_vec.shape[-1])
            model_out = {
                "extraction": path_label,
                "embed_dim": dim,
                "n": n,
                "embed_shape_rate_raw": rate_raw,
                "embed_shape_rate_centered": rate_cen,
                "mean_margin_raw": margin_raw,
                "mean_margin_centered": margin_cen,
                "per_stim": per_stim,
                "embeddings": {
                    f"{sid}/{kind}": v.tolist() for (sid, kind), v in embs.items()
                },
            }
            print(
                f"extraction={path_label}  dim={dim}\n"
                f"EMBEDDING shape rate  raw = {rate_raw:.2f} (margin {margin_raw:+.5f})   "
                f"centered = {rate_cen:.2f} (margin {margin_cen:+.5f})   n={n}"
            )
        except Exception as exc:
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
            torch.cuda.empty_cache()
        all_results["models"][name] = model_out

    print("\n" + "=" * 60)
    print("SUMMARY — embedding-similarity shape preference (no options,")
    print("no tokens: position/label bias structurally impossible)")
    print("=" * 60)
    print(f"{'model':14} {'extraction':20} {'raw':>5} {'centered':>8} {'cenMargin':>10}")
    for name, mo in all_results["models"].items():
        if "error" in mo:
            print(f"{name:14} ERROR: {mo['error']}")
        else:
            print(
                f"{name:14} {mo['extraction']:20} "
                f"{mo['embed_shape_rate_raw']:5.2f} {mo['embed_shape_rate_centered']:8.2f} "
                f"{mo['mean_margin_centered']:+10.5f}"
            )

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)
    print(f"\nJSON: {out_json}")
    print(f"Text: {out_txt}")

    sys.stdout = sys.__stdout__
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
