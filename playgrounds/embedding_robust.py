#!/usr/bin/env python3
"""Point 2 robustness battery — hardens the representation-vs-behavior dissociation.

For each VLM we extract SEVERAL representations of every image on the same 30
stimuli and, for each, score the Tartaglini shape preference
    shape_pref = 1[ sim(ref, shape_match) > sim(ref, texture_match) ].

Representations (each tried independently; failures are recorded, not fatal):
  proj       : get_image_features() token features (LM space; baseline)
  vit_last   : vision-tower last_hidden_state
  vit_penult : vision-tower penultimate hidden state (Tartaglini locus)
  vit_pooler      : vision-tower pooler_output when present (CLS-style summary)

--metric controls how sim(A, B) is computed between two images' token sets:
  mean     (default) mean-pool tokens to one vector per image, then cosine
  chamfer  keep tokens as a set; cosine each token to its best match on the
           other side, both directions, and average (works with ragged token
           counts across images, e.g. different tiling)
  sinkhorn like chamfer but with an entropy-regularized optimal-transport
           match instead of greedy best-match; slower, tune via
           --sinkhorn-reg / --sinkhorn-max-iter

For every representation we report, with and without mean-centering:
  - shape rate with a bootstrap 95% CI over stimuli
  - a POSITIVE CONTROL: object-identity retrieval. For each reference, rank all
    shape-match images by cosine (on mean-pooled vectors, regardless of
    --metric); is its own stimulus top-1? (chance = 1/n_stimuli). Same for
    texture matches. High retrieval => the read-out resolves these images, so
    a ~0.5 shape-vs-texture rate is a real null, not a blind probe.

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

import ot
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from evaluation_pipe.models import create_model

LADDER_MODELS = [
    "smolvlm",
    "internvl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3-vl-8b",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
]


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """Collapse a feature tensor to [n_tokens, D], batch dim included."""
    t = t.float()
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t.reshape(-1, t.shape[-1])


def _pool_mean(t: torch.Tensor) -> torch.Tensor:
    return _flatten(t).mean(dim=0)


def _chamfer_sim(A: torch.Tensor, B: torch.Tensor) -> float:
    """Symmetric best-match cosine similarity between two token sets."""
    An = F.normalize(A, dim=-1)
    Bn = F.normalize(B, dim=-1)
    sim = An @ Bn.t()  # [n_A, n_B]
    a_to_b = sim.max(dim=1).values.mean()
    b_to_a = sim.max(dim=0).values.mean()
    return ((a_to_b + b_to_a) / 2).item()


def _sinkhorn_sim(A: torch.Tensor, B: torch.Tensor, reg: float = 0.075, num_iter_max: int = 1000) -> float:
    """Entropy-regularized optimal-transport similarity between two token sets.

    Uniform weight per token; cost is 1 - cosine sim. reg=0.075 is a balanced
    default (lower = sharper matching but less numerically stable at small n).
    """
    An = F.normalize(A, dim=-1)
    Bn = F.normalize(B, dim=-1)
    sim = An @ Bn.t()
    cost = 1 - sim

    n_a, n_b = A.shape[0], B.shape[0]
    a = torch.full((n_a,), 1.0 / n_a, dtype=torch.double)
    b = torch.full((n_b,), 1.0 / n_b, dtype=torch.double)

    plan = ot.sinkhorn(a.numpy(), b.numpy(), cost.double().numpy(), reg, numItermax=num_iter_max)
    return float(1 - (plan * cost.double().numpy()).sum())


def _find_tower(model):
    core = getattr(model, "model", model)
    for attr in ("visual", "vision_tower", "vision_model"):
        tower = getattr(core, attr, None) or getattr(model, attr, None)
        if tower is not None:
            return tower
    return None


def extract_reps(wrapper, img: Image.Image) -> dict[str, torch.Tensor]:
    """Return {rep_name: token tensor [n_tokens, D] on CPU}. Missing reps are omitted."""
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
        hs = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        if isinstance(hs, (list, tuple)):
            hs = hs[0]
        reps["proj"] = _flatten(hs).cpu()
    except Exception as exc:  # noqa: BLE001
        reps["proj__err"] = f"{type(exc).__name__}: {exc}"  # type: ignore

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
                reps["vit_last"] = _flatten(last).cpu()
            hidden = getattr(out, "hidden_states", None)
            if hidden is not None and len(hidden) >= 2:
                reps["vit_penult"] = _flatten(hidden[-2]).cpu()
            pooler = getattr(out, "pooler_output", None)
            if pooler is not None:
                reps["vit_pooler"] = _flatten(pooler).cpu()

    return reps


def _shape_rate(ref, shp, tex, center: bool, metric: str = "mean", sinkhorn_reg: float = 0.075, sinkhorn_max_iter: int = 1000):
    """Per-stimulus shape preferences (list of 0/1) for one representation.

    ref/shp/tex are lists of [n_tokens, D] tensors, one per stimulus. Token
    counts can differ across images (e.g. different tiling).
    """
    if metric in ("chamfer", "sinkhorn"):
        if center:
            mu = torch.cat(ref + shp + tex, dim=0).mean(dim=0)
            ref = [t - mu for t in ref]
            shp = [t - mu for t in shp]
            tex = [t - mu for t in tex]
        if metric == "chamfer":
            cs = torch.tensor([_chamfer_sim(r, s) for r, s in zip(ref, shp)])
            ct = torch.tensor([_chamfer_sim(r, t) for r, t in zip(ref, tex)])
        else:
            cs = torch.tensor([_sinkhorn_sim(r, s, reg=sinkhorn_reg, num_iter_max=sinkhorn_max_iter) for r, s in zip(ref, shp)])
            ct = torch.tensor([_sinkhorn_sim(r, t, reg=sinkhorn_reg, num_iter_max=sinkhorn_max_iter) for r, t in zip(ref, tex)])
        R = torch.stack([_pool_mean(t) for t in ref])
        S = torch.stack([_pool_mean(t) for t in shp])
        T = torch.stack([_pool_mean(t) for t in tex])
    else:
        R = torch.stack([_pool_mean(t) for t in ref])
        S = torch.stack([_pool_mean(t) for t in shp])
        T = torch.stack([_pool_mean(t) for t in tex])
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


def _retrieval_by_label(R, X, lab_r, lab_x):
    """Label-level retrieval: does the top-1 neighbour carry the same label?

    ``_retrieval_at1`` demands the *exact* stimulus index. That is the right
    question only when each label occurs once, which holds for the 30-stimulus
    sets but not for the texture grid: there each mesh recurs across many
    textures, so a same-mesh neighbour is scored as an error and the control
    reads far too low (0.05-0.15 on the 114-triad sample, against 0.77-1.00 on
    the n=30 set). Asking whether the neighbour shares the reference's mesh is
    the question the sensitivity control was meant to ask, and its chance level
    is 1/n_labels rather than 1/n_stimuli.
    """
    Rn = F.normalize(R, dim=1)
    Xn = F.normalize(X, dim=1)
    top1 = (Rn @ Xn.t()).argmax(dim=1).tolist()
    hits = [1.0 if lab_x[j] == lab_r[i] else 0.0 for i, j in enumerate(top1)]
    return sum(hits) / len(hits) if hits else float("nan")

def build_default_triplets(root: Path, n: int, seed: int):
    if not root.is_absolute():
        root = REPO_ROOT / root

    trial_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "reference.png").exists()
        and (d / "shape_match.png").exists()
        and (d / "texture_match.png").exists()
    )

    rng = random.Random(seed)
    rng.shuffle(trial_dirs)
    trial_dirs = trial_dirs[:n]
    print(f"Stimuli: {len(trial_dirs)} randomly sampled from {root} (seed={seed})")
    triplets = [
        (
            d.name,
            Image.open(d / "reference.png").convert("RGB"),
            Image.open(d / "shape_match.png").convert("RGB"),
            Image.open(d / "texture_match.png").convert("RGB"),
        )
        for d in trial_dirs
    ]
    return triplets


def build_grid_triplets(stim_pkg: str, stim_set: str, n: int, seed: int):
    """Sample from a full texture-grid package (manifest-driven nested layout).

    Returns ``(triplets, meta)``, where ``meta[i]`` carries the mesh and texture
    labels of triplet ``i``. The labels are needed because the grid reuses each
    mesh across many textures, which makes exact-stimulus retrieval the wrong
    sensitivity control (see :func:`_retrieval_by_label`).
    """
    from evaluation_pipe.eval_core import load_stimuli_grid, materialize_stimulus

    records = load_stimuli_grid(stim_pkg, stim_set)
    rng = random.Random(seed)
    # Stratify by shape so the sample covers the 30 STLs when n is large enough.
    by_shape: dict[str, list] = {}
    for rec in records:
        by_shape.setdefault(rec["stl_id"], []).append(rec)
    shapes = sorted(by_shape, key=lambda s: int(s))
    picked = []
    # Round-robin across shapes until we have n.
    idxs = {s: 0 for s in shapes}
    for s in shapes:
        rng.shuffle(by_shape[s])
    while len(picked) < n and any(idxs[s] < len(by_shape[s]) for s in shapes):
        for s in shapes:
            i = idxs[s]
            if i < len(by_shape[s]):
                picked.append(by_shape[s][i])
                idxs[s] = i + 1
            if len(picked) >= n:
                break
    print(
        f"Stimuli: {len(picked)} stratified from grid {stim_pkg}/{stim_set} "
        f"({len(shapes)} shapes, seed={seed})"
    )
    import hashlib
    triplets = []
    meta = []
    for rec in picked:
        stim = materialize_stimulus(rec)
        sid = int(hashlib.md5(rec["stim_id"].encode()).hexdigest()[:8], 16)
        triplets.append(
            (sid, stim["reference"], stim["shape_match"], stim["texture_match"])
        )
        # kept alongside the triplets so retrieval can be scored at the mesh and
        # texture level, not only at the exact-stimulus level
        meta.append({"stl_id": str(rec["stl_id"]), "texture_set": str(rec["texture_set"])})
    return triplets, meta

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

    print(f"Stimuli: {len(triplets)} balanced-sampled from {root} (seed={seed})")
    
    return triplets

_GEIRHOS_PAT = re.compile(r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$")

def build_geirhos_unaltered_triplets(
    root: Path,
    n: int,
    seed: int,
):
    cue_root = root / "cue_conflict"
    original_root = root / "original"
    texture_root = root / "texture"

    items = []

    for shape_dir in sorted(p for p in cue_root.iterdir() if p.is_dir()):
        for f in sorted(shape_dir.iterdir()):
            m = _GEIRHOS_PAT.match(f.name)
            if not m:
                continue

            shape, shape_id, tex, tex_id = m.groups()
            items.append((f, shape, shape_id, tex, tex_id))

    rng = random.Random(seed)
    rng.shuffle(items)

    per_class_cap = max(1, -(-n // 16))
    used = {}

    triplets = []

    for ref_path, shape, shape_id, tex, tex_id in items:
        if used.get(shape, 0) >= per_class_cap:
            continue

        shape_path = original_root / shape / f"{shape}{shape_id}.png"
        texture_path = texture_root / tex / f"{tex}{tex_id}.png"

        if not shape_path.exists() or not texture_path.exists():
            continue

        triplets.append(
            (
                ref_path.stem,
                Image.open(ref_path).convert("RGB"),
                Image.open(shape_path).convert("RGB"),
                Image.open(texture_path).convert("RGB"),
            )
        )

        used[shape] = used.get(shape, 0) + 1

        if len(triplets) >= n:
            break

    print(f"Stimuli: {len(triplets)} sampled from {root} (seed={seed})")

    return triplets

_SMITH_PAT = re.compile(r"^smith_bg(\d+)_(\d+)(probe|color_match|shape_match)\.jpg$")

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
    
    print(f"Stimuli: {len(triplets)} randomly sampled from {root} (seed={seed})")

    return triplets

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LADDER_MODELS)
    ap.add_argument("--n-stimuli", type=int, default=30)
    ap.add_argument(
        "--metric",
        choices=["mean", "chamfer", "sinkhorn"],
        default="mean",
        help="How to compare two images' token sets: mean-pool then cosine "
             "(default), chamfer (best-match cosine per token, both directions), "
             "or sinkhorn (entropy-regularized optimal transport; slower, spot-check "
             "on small --n-stimuli before running the full ladder).",
    )
    ap.add_argument(
        "--sinkhorn-reg",
        type=float,
        default=0.075,
        help="Entropy regularization for --metric sinkhorn (lower = sharper "
             "matching, less stable; higher = smoother, faster).",
    )
    ap.add_argument(
        "--sinkhorn-max-iter",
        type=int,
        default=1000,
        help="Max Sinkhorn iterations before giving up (ot.sinkhorn's own default is 1000).",
    )
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
        "--geirhos-unaltered",
        default=None,
        help="Path to geirhos_unaltered dataset (cue_conflict/original/texture folders).",
    )
    ap.add_argument(
        "--smith-probe",
        default=None,
        help="Path to Linda Smith probe-shapematch-colormatch dataset; if set, run the familiar-category "
        "positive control on these stimuli instead of IMAGE_DATASET.",
    )
    ap.add_argument(
        "--grid-pkg",
        default=None,
        help="Full texture-grid package under stimuli_pipe/ (e.g. stimuli_texture_grid_v1_scratch). "
             "Samples stratified by shape from its manifest.",
    )
    ap.add_argument(
        "--stim-set",
        default=None,
        help="Stimulus set name under --grid-pkg (default: stimuli_A_auto_contrast).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out_prefix is None:
        from evaluation_pipe.eval_core import default_session_results_dir

        if args.cue_conflict:
            name = "embedding_cueconflict"
        elif args.geirhos_unaltered:
            name = "embedding_geirhos_unaltered"
        elif args.smith_probe:
            name = "embedding_smith_probe"
        elif args.grid_pkg:
            name = "embedding_grid"
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
    print(f"Metric: {args.metric}")
    if args.metric == "sinkhorn" and args.n_stimuli > 50:
        print(f"  [note] sinkhorn is ~4-20x slower than chamfer; running at "
              f"n_stimuli={args.n_stimuli}. Consider a smaller --n-stimuli for a spot-check.")

    grid_meta = None  # mesh/texture labels; only the grid builder supplies them
    if args.cue_conflict:
        cc_root = Path(args.cue_conflict)
        triplets = build_cueconflict_triplets(cc_root, args.n_stimuli, args.seed)
        print(f"POSITIVE CONTROL: {len(triplets)} familiar-category cue-conflict triplets from {cc_root}")
        for tid, *_ in triplets[:5]:
            print(f"  e.g. {tid}")
    elif args.geirhos_unaltered:
        geirhos_root = Path(args.geirhos_unaltered)
        triplets = build_geirhos_unaltered_triplets(
            geirhos_root,
            args.n_stimuli,
            args.seed,
        )
        print(f"GEIRHOS UNALTERED: {len(triplets)} triplets from {geirhos_root}")
        for tid, *_ in triplets[:5]:
            print(f"  e.g. {tid}")
    elif args.smith_probe:
        smith_root = Path(args.smith_probe)
        triplets = build_smith_probe_triplets(smith_root, args.n_stimuli, args.seed)
        print(f"SMITH PROBE: {len(triplets)} triplets from {smith_root}")
        for tid, *_ in triplets[:5]:
            print(f"  e.g. {tid}")
    elif args.grid_pkg:
        stim_set = args.stim_set or "stimuli_A_auto_contrast"
        triplets, grid_meta = build_grid_triplets(
            args.grid_pkg, stim_set, args.n_stimuli, args.seed
        )
        print(f"FULL GRID: {len(triplets)} stratified triplets from {args.grid_pkg}/{stim_set}")
    else:
        triplets = build_default_triplets(Path(os.environ["IMAGE_DATASET"]), args.n_stimuli, args.seed)
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
            n_triplets = len(triplets)
            for i, (sid, ref_img, shape_img, tex_img) in enumerate(triplets, start=1):
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
                print(f"  [{i}/{n_triplets}] {sid}", flush=True)

            rep_results = {}
            for rep, data in by_rep.items():
                dim = int(data["ref"][0].shape[-1])
                entry = {"dim": dim, "n": len(data["ref"])}
                for center in (False, True):
                    prefs, margin, R, S, T = _shape_rate(
                        data["ref"], data["shape"], data["texture"], center,
                        metric=args.metric, sinkhorn_reg=args.sinkhorn_reg,
                        sinkhorn_max_iter=args.sinkhorn_max_iter,
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
                        # Only safe when this read-out produced a vector for every
                        # triplet; a skipped image would misalign labels and rows.
                        if grid_meta and len(grid_meta) == entry["n"]:
                            # Label-level control. The shape-match gallery shares
                            # the reference's mesh; the texture-match gallery
                            # shares its texture set.
                            mesh_lab = [m["stl_id"] for m in grid_meta]
                            tex_lab = [m["texture_set"] for m in grid_meta]
                            entry["retrieval_mesh_at1"] = _retrieval_by_label(
                                R, S, mesh_lab, mesh_lab
                            )
                            entry["retrieval_texset_at1"] = _retrieval_by_label(
                                R, T, tex_lab, tex_lab
                            )
                            entry["retrieval_mesh_chance"] = 1.0 / len(set(mesh_lab))
                            entry["retrieval_texset_chance"] = 1.0 / len(set(tex_lab))
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
        line = (
            f"  {rep:16} dim={e['dim']:5}  shape(centered)={c['shape_rate']:.2f} "
            f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]  margin={c['mean_margin']:+.4f}  "
            f"retr@1 shape={e['retrieval_shape_at1']:.2f} tex={e['retrieval_texture_at1']:.2f}"
        )
        if "retrieval_mesh_at1" in e:
            line += (
                f"  mesh@1={e['retrieval_mesh_at1']:.2f} "
                f"texset@1={e['retrieval_texset_at1']:.2f}"
            )
        print(line)
    for k, v in errs.items():
        print(f"  [{k}] {v}")


def _print_summary(res):
    n = res["config"]["n_stimuli"]
    print("\n" + "=" * 78)
    print("SUMMARY — embedding shape rate (centered, 95% CI) across read-outs")
    print(f"Metric: {res['config']['metric']}")
    print(
        f"retr@1 = matching-stimulus retrieval (chance=1/{n}={1/n:.2f}); "
        "high => probe is sensitive"
    )   
    print("=" * 78)
    has_labels = any(
        "retrieval_mesh_at1" in e
        for mo in res["models"].values()
        for e in (mo.get("reps") or {}).values()
    )
    if has_labels:
        print(
            "mesh@1/texset@1 = label-level retrieval; on the grid a mesh recurs "
            "across textures, so exact-stimulus retr@1 understates sensitivity"
        )
    head = f"{'model':14} {'readout':16} {'shape':>5} {'ci95':>13} {'retrS':>6} {'retrT':>6}"
    if has_labels:
        head += f" {'mesh':>6} {'texset':>6}"
    print(head)
    for name, mo in res["models"].items():
        if "error" in mo:
            print(f"{name:14} ERROR: {mo['error']}")
            continue
        for rep, e in mo["reps"].items():
            c = e["centered"]
            row = (
                f"{name:14} {rep:16} {c['shape_rate']:5.2f} "
                f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] "
                f"{e['retrieval_shape_at1']:6.2f} {e['retrieval_texture_at1']:6.2f}"
            )
            if has_labels:
                row += (
                    f" {e.get('retrieval_mesh_at1', float('nan')):6.2f}"
                    f" {e.get('retrieval_texset_at1', float('nan')):6.2f}"
                )
            print(row)


if __name__ == "__main__":
    raise SystemExit(main())