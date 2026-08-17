#!/usr/bin/env python3
"""Concept erasure and injection at the LM-facing projector output.

The linkage analyses can only show correlation between the encoder's shape
subspace and behavior. This module makes the causal test: remove (or amplify)
the mesh-identity subspace from the projected image tokens at generation time
and measure the change in behavioral shape rate.

Method: LEACE-style closed-form linear erasure (Belrose et al. 2023). With
feature covariance S, cross-covariance S_xz to one-hot labels, and whitening
W = S^{-1/2}, the least-squares eraser removes the span of W S_xz in whitened
space:

    U  = orthonormal basis of W S_xz          (rank <= n_classes - 1)
    A  = S^{1/2} U U^T S^{-1/2}
    x' = x + (alpha - 1) A (x - mu)           alpha=0 erase, alpha>1 amplify

Fitting is fold-aware: one eraser per mesh fold (from linear_probe.build_splits),
fit on the tokens of the meshes NOT in that fold, so the eraser applied to a
stimulus never saw that stimulus' mesh. Controls: a texture-identity eraser
(same construction on texture labels) and rank-matched random subspaces.

Usage
-----
    python playgrounds/intervene.py --self-test
    python playgrounds/intervene.py --fit \
        results/.../embeddings_grid/qwen3.5-4b_proj_lm_tokens.npz \
        --target mesh --out results/.../erasers/qwen3.5-4b_mesh_bank.npz

At generation time, scripts/run_local.py --intervene-bank ... attaches the
forward hook via :func:`attach` and selects the fold eraser per stimulus with
:meth:`Intervention.set_stimulus`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# fitting (numpy, offline)
# --------------------------------------------------------------------------


def _sqrt_and_inv_sqrt(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 1e-10)
    return (
        vecs @ np.diag(np.sqrt(vals)) @ vecs.T,
        vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T,
    )


def fit_eraser(X: np.ndarray, labels: np.ndarray, shrink: float = 0.05,
               rank: int | None = None) -> dict:
    """Closed-form LEACE eraser for one label set. Returns {mu, A, rank}."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    n, D = Xc.shape
    S = (Xc.T @ Xc) / n
    S += shrink * np.trace(S) / D * np.eye(D)
    S_half, S_ihalf = _sqrt_and_inv_sqrt(S)

    classes = sorted(set(labels.tolist()))
    Z = np.zeros((n, len(classes)))
    for j, c in enumerate(classes):
        Z[labels == c, j] = 1.0
    Zc = Z - Z.mean(axis=0)
    Sxz = (Xc.T @ Zc) / n  # [D, C]

    M = S_ihalf @ Sxz
    Um, sv, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sv > sv.max() * 1e-6).sum()) if sv.size else 0
    if rank is not None:
        r = min(r, rank)
    U = Um[:, :r]
    A = S_half @ U @ U.T @ S_ihalf
    return {"mu": mu.astype(np.float32), "A": A.astype(np.float32), "rank": r}


def random_eraser(X: np.ndarray, rank: int, seed: int,
                  shrink: float = 0.05) -> dict:
    """Rank-matched control: erase a random subspace in the same whitened space."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    n, D = Xc.shape
    S = (Xc.T @ Xc) / n
    S += shrink * np.trace(S) / D * np.eye(D)
    S_half, S_ihalf = _sqrt_and_inv_sqrt(S)
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((D, rank)))
    A = S_half @ U @ U.T @ S_ihalf
    return {"mu": mu.astype(np.float32), "A": A.astype(np.float32), "rank": rank}


def fit_bank(tokens_npz: Path, target: str, out_path: Path,
             random_seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Fold-aware eraser bank from a per-token export.

    For the erased label set (mesh or texture), one eraser per fold, fit
    without that fold's labels. Also writes rank-matched random erasers fit on
    all tokens (the random subspace has no fold structure to leak).
    """
    from linear_probe import GridEmbeddings, build_splits

    z = np.load(tokens_npz, allow_pickle=False)
    X = z["X"].astype(np.float32)
    stl = z["stl_id"].astype(int)
    tex = z["texture_set"].astype(str)
    print(f"{tokens_npz.name}: X={X.shape}, {len(set(stl))} meshes, "
          f"{len(set(tex))} textures")

    # build_splits only needs the label universe; reuse the pooled NPZ layout
    pooled = GridEmbeddings(
        X=np.zeros((len(stl), 1), dtype=np.float32),
        role=np.array(["reference"] * len(stl)),
        stl_id=stl, texture_set=tex,
    )
    splits = build_splits(pooled)

    labels = stl if target == "mesh" else tex
    folds = splits["mesh_folds"] if target == "mesh" else splits["tex_pair_folds"]

    payload: dict[str, np.ndarray] = {}
    fold_of: dict[str, int] = {}
    ranks = []
    for i, fold in enumerate(folds):
        fold_set = {str(x) for x in fold}
        for lab in fold:
            fold_of[str(lab)] = i
        tr = np.array([str(l) not in fold_set for l in labels])
        er = fit_eraser(X[tr], labels[tr])
        payload[f"mu_{i}"] = er["mu"]
        payload[f"A_{i}"] = er["A"]
        ranks.append(er["rank"])
        print(f"  fold {i}: fit on {int(tr.sum())} tokens, rank={er['rank']}")

    for s in random_seeds:
        er = random_eraser(X, rank=max(ranks), seed=s)
        payload[f"mu_rand{s}"] = er["mu"]
        payload[f"A_rand{s}"] = er["A"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **payload,
        meta=np.array(json.dumps({
            "target": target,
            "n_folds": len(folds),
            "fold_of": fold_of,
            "ranks": ranks,
            "random_seeds": list(random_seeds),
            "source": str(tokens_npz),
        })),
    )
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# runtime (torch, generation-time hook)
# --------------------------------------------------------------------------


class Intervention:
    """Holds an eraser bank on device and transforms projector outputs.

    arm: "target" (the erased concept, fold-aware), or "random<seed>".
    alpha: 0.0 erases the subspace, values > 1 amplify it.
    """

    def __init__(self, bank_path: Path, arm: str = "target", alpha: float = 0.0):
        z = np.load(bank_path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.arm = arm
        self.alpha = float(alpha)
        self._np = {k: z[k] for k in z.files if k != "meta"}
        self._torch: dict[str, object] = {}
        self._current = "0"
        self.n_calls = 0

    def set_stimulus(self, stl_id=None, texture_set=None) -> None:
        if self.arm.startswith("random"):
            self._current = f"rand{self.arm[len('random'):] or 0}"
            return
        lab = str(stl_id) if self.meta["target"] == "mesh" else str(texture_set)
        self._current = str(self.meta["fold_of"].get(lab, 0))

    def _tensors(self, key: str, device, dtype):
        import torch

        ck = f"{key}|{device}"
        if ck not in self._torch:
            self._torch[ck] = (
                torch.from_numpy(self._np[f"mu_{key}"]).to(device=device,
                                                           dtype=torch.float32),
                torch.from_numpy(self._np[f"A_{key}"]).to(device=device,
                                                          dtype=torch.float32),
            )
        return self._torch[ck]

    def transform(self, t):
        import torch

        orig_dtype = t.dtype
        mu, A = self._tensors(self._current, t.device, t.dtype)
        x = t.float()
        y = x + (self.alpha - 1.0) * (x - mu) @ A.T
        self.n_calls += 1
        return y.to(orig_dtype)


def _resolve_projector(model):
    """The module whose output is the LM-facing projected token stream."""
    core = getattr(model, "model", model)
    for attr in ("visual", "vision_tower", "vision_model"):
        tower = getattr(core, attr, None) or getattr(model, attr, None)
        if tower is not None and hasattr(tower, "merger"):
            return tower.merger  # Qwen3.5 / Qwen3-VL patch merger
    for attr in ("multi_modal_projector", "connector"):
        mod = getattr(core, attr, None) or getattr(model, attr, None)
        if mod is not None:
            return mod  # InternVL projector / SmolVLM connector
    raise RuntimeError("no projector module found to hook")


def attach(model, itv: Intervention):
    """Register the forward hook; returns the handle for removal."""
    module = _resolve_projector(model)

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (itv.transform(output[0]), *output[1:])
        return itv.transform(output)

    handle = module.register_forward_hook(_hook)
    print(f"intervention attached at {type(module).__name__} "
          f"(arm={itv.arm}, alpha={itv.alpha}, target={itv.meta['target']})")
    return handle


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def run_self_test() -> int:
    """Erasure must kill a linear probe on held-out data; the rank-matched
    random control must not; injection must scale the class signal."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n_cls, per, D, r = 20, 40, 128, 12
    means = np.zeros((n_cls, D))
    means[:, :r] = rng.standard_normal((n_cls, r)) * 3.0
    y = np.repeat(np.arange(n_cls), per)
    X = means[y] + rng.standard_normal((n_cls * per, D))
    # nuisance structure everywhere else
    X[:, r:] += rng.standard_normal((n_cls * per, D - r)) * 2.0

    half = np.tile(np.arange(per) < per // 2, n_cls)
    er = fit_eraser(X[half], y[half])
    rc = random_eraser(X[half], rank=er["rank"], seed=0)

    def probe_acc(Z):
        clf = LogisticRegression(max_iter=2000).fit(Z[half], y[half])
        return float((clf.predict(Z[~half]) == y[~half]).mean())

    def apply(e, Z, alpha=0.0):
        return Z + (alpha - 1.0) * (Z - e["mu"]) @ e["A"].T

    base = probe_acc(X)
    erased = probe_acc(apply(er, X))
    rand = probe_acc(apply(rc, X))
    amp = apply(er, X, alpha=3.0)
    sig = np.linalg.norm((amp - X.mean(0))[:, :r].mean(0))

    print(f"probe accuracy  base={base:.3f}  erased={erased:.3f}  "
          f"random-erased={rand:.3f}  (chance={1 / n_cls:.3f})")
    ok = True
    if not base > 0.9:
        print("FAIL: base probe should be near ceiling")
        ok = False
    if not erased < 0.15:
        print("FAIL: erasure should push the probe to chance")
        ok = False
    if not rand > 0.8:
        print("FAIL: rank-matched random erasure should barely hurt")
        ok = False
    if ok:
        print("PASS")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, help="per-token NPZ from embedding_export")
    ap.add_argument("--target", choices=["mesh", "texture"], default="mesh")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.fit:
        ap.error("one of --fit or --self-test is required")
    out = args.out or args.fit.with_name(
        args.fit.stem.replace("_proj_lm_tokens", "") + f"_{args.target}_bank.npz")
    fit_bank(args.fit, args.target, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
