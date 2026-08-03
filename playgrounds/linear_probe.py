#!/usr/bin/env python3
"""Linear-probe and read-out-power analysis of vision-encoder embeddings.

Motivation
----------
The published encoder measure is mean-centred cosine similarity between pooled
embeddings, scored as ``1[cos(ref, shape_match) > cos(ref, texture_match)]``
(``playgrounds/embedding_robust.py``). Cosine weights every dimension equally and
is not fit to the task. If mesh identity occupies a small subspace while surface
texture dominates the variance, cosine reports chance even when the information is
present and a downstream reader could recover it.

This script separates three hypotheses that the cosine measure cannot:

  H1  shape is not linearly available in the encoder at all
  H2  shape is present but low-weighted in the unweighted cosine geometry
  H3  shape is present and strongly weighted, yet the texture-sharing candidate is
      still nearer than the shape-sharing candidate in the directed comparison

Analyses
--------
  P1  mesh identity, 30-way, held-out textures            (chance 1/30)
  P2  texture identity, 38-way, held-out meshes           (chance 1/38)
  P3  the same two probes on raw downsampled pixels       (baseline)
  L   read-out power ladder: the *same* held-out triads scored under read-outs of
      increasing power, reporting the same dependent variable (shape-match rate)

Held-out design
---------------
The grid is a full 30 mesh x 38 texture crossing. Foils are deterministic offsets,
not random: the texture-match foil is mesh ``s+15 (mod 30)`` and the shape-match
foil is texture ``t+19 (mod 38)``. Both offsets are involutions, so meshes fall
into 15 pairs and textures into 19 pairs. Splits are built from whole pairs, so
that a triad's foils cannot straddle the train/test boundary.

For the ladder, block (i, j) trains on ``M_-i x T_-j`` and tests triads drawn
entirely from ``M_i x T_j``: test meshes and test textures were both never seen.

Note on the reviewer's phrasing: holding out *meshes* is not well defined for a
30-way mesh-identity classifier, since a class never trained on cannot be
predicted. The request is only coherent in the metric-learning formulation, which
is what the ladder's top rung implements.

Usage
-----
    python playgrounds/linear_probe.py --self-test
    python playgrounds/linear_probe.py --emb results/.../embeddings/qwen3.5-9b_proj_mean.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]

N_MESH_FOLDS = 5
N_TEX_FOLDS = 5
MESH_OFFSET = 15  # texture_match_stl_id = ((stl_id - 1 + 15) mod 30) + 1
TEX_OFFSET = 19  # shape_match_texture_set = texture +19 in the sorted list
DEFAULT_C_GRID = (1e-3, 1e-2, 1e-1, 1.0)


# --------------------------------------------------------------------------
# data container
# --------------------------------------------------------------------------


@dataclass
class GridEmbeddings:
    """Embeddings for every (role, mesh, texture) cell of the stimulus grid."""

    X: np.ndarray  # [N, D] float32
    role: np.ndarray  # [N] str, one of reference/shape_match/texture_match
    stl_id: np.ndarray  # [N] int
    texture_set: np.ndarray  # [N] str
    name: str = "unnamed"

    meshes: np.ndarray = field(init=False)
    textures: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.meshes = np.array(sorted(set(int(s) for s in self.stl_id)))
        self.textures = np.array(sorted(set(str(t) for t in self.texture_set)))
        self._index: dict[tuple[str, int, str], int] = {}
        for i, (r, s, t) in enumerate(zip(self.role, self.stl_id, self.texture_set)):
            self._index[(str(r), int(s), str(t))] = i

    def row(self, role: str, mesh: int, texture: str) -> int | None:
        return self._index.get((role, int(mesh), str(texture)))

    def reference_block(
        self, meshes, textures
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference-role rows for the given mesh x texture block."""
        mset, tset = {int(m) for m in meshes}, {str(t) for t in textures}
        idx = [
            i
            for i, (r, s, t) in enumerate(
                zip(self.role, self.stl_id, self.texture_set)
            )
            if r == "reference" and int(s) in mset and str(t) in tset
        ]
        idx = np.array(idx, dtype=int)
        return (
            self.X[idx],
            np.array([int(s) for s in self.stl_id[idx]]),
            np.array([str(t) for t in self.texture_set[idx]]),
        )

    @classmethod
    def load(cls, path: Path) -> "GridEmbeddings":
        z = np.load(path, allow_pickle=False)
        return cls(
            X=z["X"].astype(np.float32),
            role=z["role"].astype(str),
            stl_id=z["stl_id"].astype(int),
            texture_set=z["texture_set"].astype(str),
            name=path.stem,
        )


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def texture_family(name: str) -> str:
    """Carpet008_1K-JPG -> Carpet. Used to keep folds family-balanced."""
    m = re.match(r"^([A-Za-z]+)", str(name))
    return m.group(1) if m else str(name)


def involution_pairs(items: list, offset: int) -> list[tuple]:
    """Pair item k with item k+offset. With len == 2*offset this is an involution."""
    n = len(items)
    seen, pairs = set(), []
    for k in range(n):
        j = (k + offset) % n
        key = tuple(sorted((k, j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((items[k], items[j]) if k != j else (items[k],))
    return pairs


def family_balanced_folds(textures: list[str], n_folds: int) -> list[list[str]]:
    """Round-robin textures to folds within each family, so families spread evenly."""
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    by_family: dict[str, list[str]] = {}
    for t in sorted(textures):
        by_family.setdefault(texture_family(t), []).append(t)
    cursor = 0
    for fam in sorted(by_family):
        for t in by_family[fam]:
            folds[cursor % n_folds].append(t)
            cursor += 1
    return folds


def pair_folds(pairs: list[tuple], n_folds: int) -> list[list]:
    """Round-robin whole involution pairs to folds, then flatten."""
    folds: list[list] = [[] for _ in range(n_folds)]
    for k, pair in enumerate(pairs):
        folds[k % n_folds].extend(pair)
    return folds


def build_splits(emb: GridEmbeddings) -> dict:
    meshes = [int(m) for m in emb.meshes]
    textures = [str(t) for t in emb.textures]
    mesh_pairs = involution_pairs(meshes, MESH_OFFSET)
    tex_pairs = involution_pairs(textures, TEX_OFFSET)
    return {
        "mesh_folds": pair_folds(mesh_pairs, N_MESH_FOLDS),
        "tex_pair_folds": pair_folds(tex_pairs, N_TEX_FOLDS),
        "tex_family_folds": family_balanced_folds(textures, N_TEX_FOLDS),
        "mesh_pairs": mesh_pairs,
        "tex_pairs": tex_pairs,
    }


def assert_disjoint(splits: dict, emb: GridEmbeddings) -> None:
    """Fail loudly if any split leaks. Cheap insurance against a silent bug."""
    meshes, textures = set(int(m) for m in emb.meshes), set(
        str(t) for t in emb.textures
    )
    for key, folds, universe in (
        ("mesh_folds", splits["mesh_folds"], meshes),
        ("tex_pair_folds", splits["tex_pair_folds"], textures),
        ("tex_family_folds", splits["tex_family_folds"], textures),
    ):
        flat = [x for f in folds for x in f]
        assert len(flat) == len(set(flat)), f"{key}: duplicate across folds"
        assert set(flat) == universe, f"{key}: does not cover the universe"

    # The load-bearing one: a test block must be closed under both foil offsets.
    mlist, tlist = sorted(meshes), sorted(textures)
    for fold in splits["mesh_folds"]:
        for m in fold:
            partner = mlist[(mlist.index(m) + MESH_OFFSET) % len(mlist)]
            assert partner in fold, (
                f"mesh fold not closed under the +{MESH_OFFSET} foil offset: "
                f"{m} in fold but partner {partner} is not"
            )
    for fold in splits["tex_pair_folds"]:
        for t in fold:
            partner = tlist[(tlist.index(t) + TEX_OFFSET) % len(tlist)]
            assert partner in fold, (
                f"texture fold not closed under the +{TEX_OFFSET} foil offset: "
                f"{t} in fold but partner {partner} is not"
            )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_ci(values, n_boot=2000, seed=0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on paired binary decisions (same triads, two read-outs)."""
    from scipy.stats import binomtest

    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p": float(p)}


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def _fit_logistic(Xtr, ytr, C: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(Xtr)
    # sklearn >= 1.7 dropped multi_class; lbfgs is multinomial by default.
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def _tune_C(Xtr, ytr, groups, c_grid) -> float:
    """Inner CV on the training fold only. Tuned once, then reused (documented)."""
    uniq = sorted(set(groups))
    if len(uniq) < 2 or len(c_grid) == 1:
        return c_grid[len(c_grid) // 2]
    inner = [g for g in uniq[: max(2, len(uniq) // 3)]]
    tr = np.array([g not in inner for g in groups])
    te = ~tr
    if tr.sum() == 0 or te.sum() == 0 or len(set(ytr[tr])) < 2:
        return c_grid[len(c_grid) // 2]
    best, best_acc = c_grid[0], -1.0
    for C in c_grid:
        scaler, clf = _fit_logistic(Xtr[tr], ytr[tr], C)
        acc = float((clf.predict(scaler.transform(Xtr[te])) == ytr[te]).mean())
        if acc > best_acc:
            best, best_acc = C, acc
    return best


def identity_probe(
    emb: GridEmbeddings,
    target: str,
    folds: list[list],
    fold_axis: str,
    c_grid=DEFAULT_C_GRID,
    n_perm: int = 0,
    seed: int = 0,
) -> dict:
    """Decode ``target`` ('mesh'|'texture') with held-out ``fold_axis`` groups."""
    all_m = [int(m) for m in emb.meshes]
    all_t = [str(t) for t in emb.textures]

    correct, fold_acc, C_used = [], [], None
    for fold in folds:
        if fold_axis == "texture":
            tr_t = [t for t in all_t if t not in set(fold)]
            Xtr, mtr, ttr = emb.reference_block(all_m, tr_t)
            Xte, mte, tte = emb.reference_block(all_m, fold)
            gtr = ttr
        else:
            tr_m = [m for m in all_m if m not in set(fold)]
            Xtr, mtr, ttr = emb.reference_block(tr_m, all_t)
            Xte, mte, tte = emb.reference_block(fold, all_t)
            gtr = mtr

        ytr = mtr if target == "mesh" else ttr
        yte = mte if target == "mesh" else tte
        if len(set(ytr)) < 2 or len(Xte) == 0:
            continue
        if C_used is None:
            C_used = _tune_C(Xtr, ytr, gtr, c_grid)
        scaler, clf = _fit_logistic(Xtr, ytr, C_used)
        hit = (clf.predict(scaler.transform(Xte)) == yte).astype(int)
        correct.extend(hit.tolist())
        fold_acc.append(float(hit.mean()))

    acc = float(np.mean(correct)) if correct else float("nan")
    lo, hi = bootstrap_ci(correct, seed=seed)
    n_classes = len(all_m) if target == "mesh" else len(all_t)
    out = {
        "target": target,
        "held_out": fold_axis,
        "n_test": len(correct),
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "accuracy": acc,
        "ci95": [lo, hi],
        "per_fold": fold_acc,
        "C": C_used,
    }

    if n_perm:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            perm_hits = []
            for fold in folds:
                if fold_axis == "texture":
                    tr_t = [t for t in all_t if t not in set(fold)]
                    Xtr, mtr, ttr = emb.reference_block(all_m, tr_t)
                    Xte, mte, tte = emb.reference_block(all_m, fold)
                else:
                    tr_m = [m for m in all_m if m not in set(fold)]
                    Xtr, mtr, ttr = emb.reference_block(tr_m, all_t)
                    Xte, mte, tte = emb.reference_block(fold, all_t)
                ytr = (mtr if target == "mesh" else ttr).copy()
                yte = mte if target == "mesh" else tte
                rng.shuffle(ytr)
                scaler, clf = _fit_logistic(Xtr, ytr, C_used)
                perm_hits.extend(
                    (clf.predict(scaler.transform(Xte)) == yte).astype(int).tolist()
                )
            null.append(float(np.mean(perm_hits)))
        null_arr = np.array(null)
        out["perm_null_mean"] = float(null_arr.mean())
        out["perm_null_p95"] = float(np.percentile(null_arr, 95))
        out["perm_p"] = float((null_arr >= acc).mean())

    return out


# --------------------------------------------------------------------------
# read-out power ladder
# --------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def _prefs(R, S, T) -> np.ndarray:
    return (_cos(R, S) > _cos(R, T)).astype(int)


def _zca(Xtr: np.ndarray, shrink: bool = True):
    """Unsupervised reweighting: whitening fit on the training block only."""
    mu = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - mu
    if shrink:
        cov = LedoitWolf(assume_centered=True).fit(Xc).covariance_
    else:
        cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mu, W.astype(np.float32)


def readout_ladder(
    emb: GridEmbeddings,
    splits: dict,
    c_grid=DEFAULT_C_GRID,
    pca_ks=(8, 32, 128),
    seed: int = 0,
) -> dict:
    """Score the same held-out triads under read-outs of increasing power."""
    all_m = [int(m) for m in emb.meshes]
    all_t = [str(t) for t in emb.textures]
    rungs: dict[str, list[int]] = {}
    n_blocks, C_used = 0, None

    def add(name, prefs):
        rungs.setdefault(name, []).extend(np.asarray(prefs).astype(int).tolist())

    for mesh_fold in splits["mesh_folds"]:
        for tex_fold in splits["tex_pair_folds"]:
            triads = [
                (s, t, emb.row("reference", s, t), emb.row("shape_match", s, t),
                 emb.row("texture_match", s, t))
                for s in mesh_fold
                for t in tex_fold
            ]
            triads = [x for x in triads if None not in x[2:]]
            if not triads:
                continue
            ri = np.array([x[2] for x in triads])
            si = np.array([x[3] for x in triads])
            ti = np.array([x[4] for x in triads])
            R, S, T = emb.X[ri], emb.X[si], emb.X[ti]

            tr_m = [m for m in all_m if m not in set(mesh_fold)]
            tr_t = [t for t in all_t if t not in set(tex_fold)]
            Xtr, mtr, _ = emb.reference_block(tr_m, tr_t)
            if len(Xtr) < 10:
                continue
            n_blocks += 1

            # rung 1 - raw cosine
            add("1_raw_cosine", _prefs(R, S, T))

            # rung 2 - mean-centred cosine, centred over the evaluated set.
            # This is the published measure; kept identical for comparability.
            mu_eval = np.concatenate([R, S, T]).mean(axis=0, keepdims=True)
            add("2_centered_cosine", _prefs(R - mu_eval, S - mu_eval, T - mu_eval))

            # rung 2b - centred on train statistics only (leak-free variant)
            mu_tr = Xtr.mean(axis=0, keepdims=True)
            add("2b_centered_train", _prefs(R - mu_tr, S - mu_tr, T - mu_tr))

            # rung 3 - ZCA whitening, unsupervised, fit on train block
            mu_z, Wz = _zca(Xtr)
            add(
                "3_zca_cosine",
                _prefs((R - mu_z) @ Wz, (S - mu_z) @ Wz, (T - mu_z) @ Wz),
            )

            # rung 4 - PCA truncation, unsupervised, fit on train block
            for k in pca_ks:
                if k >= min(Xtr.shape):
                    continue
                pca = PCA(n_components=k, random_state=seed).fit(Xtr)
                add(
                    f"4_pca{k}_cosine",
                    _prefs(pca.transform(R), pca.transform(S), pca.transform(T)),
                )

            # rung 5 - learned linear metric. Mesh-discriminative subspace fit on
            # train meshes, applied to held-out meshes: a generalisation test, not
            # a lookup.
            if C_used is None:
                C_used = _tune_C(Xtr, mtr, mtr, c_grid)
            scaler, clf = _fit_logistic(Xtr, mtr, C_used)
            Wl = clf.coef_.T.astype(np.float32)
            proj = lambda Z: scaler.transform(Z) @ Wl  # noqa: E731
            add("5_learned_metric", _prefs(proj(R), proj(S), proj(T)))

            # rung 6 - ceiling. Deliberately leaky: the metric sees the test
            # meshes. An upper bound on what any linear read-out could do.
            Xall, mall, _ = emb.reference_block(all_m, all_t)
            sc2, clf2 = _fit_logistic(Xall, mall, C_used)
            W2 = clf2.coef_.T.astype(np.float32)
            proj2 = lambda Z: sc2.transform(Z) @ W2  # noqa: E731
            add("6_ceiling_leaky", _prefs(proj2(R), proj2(S), proj2(T)))

    out = {"n_blocks": n_blocks, "rungs": {}}
    for name in sorted(rungs):
        v = rungs[name]
        lo, hi = bootstrap_ci(v, seed=seed)
        out["rungs"][name] = {
            "shape_rate": float(np.mean(v)),
            "ci95": [lo, hi],
            "n": len(v),
        }
    if "2_centered_cosine" in rungs and "5_learned_metric" in rungs:
        out["mcnemar_centered_vs_learned"] = mcnemar(
            np.array(rungs["2_centered_cosine"]), np.array(rungs["5_learned_metric"])
        )
    out["_per_triad"] = {k: v for k, v in rungs.items()}
    return out


def retrieval_metrics(emb: GridEmbeddings) -> dict:
    """Mesh-level and texture-level retrieval, the metric the grid run should use.

    ``_retrieval_at1`` in embedding_robust.py demands the *exact* stimulus index.
    On the grid each mesh recurs across many textures, so a same-mesh neighbour
    counts as an error and the control reads far too low. Asking instead whether
    the nearest neighbour shares the reference's mesh is the correct question,
    and is the unsupervised sibling of the mesh-identity probe.
    """
    all_m = [int(m) for m in emb.meshes]
    all_t = [str(t) for t in emb.textures]
    Xr, mr, tr = emb.reference_block(all_m, all_t)
    idx_s = [emb.row("shape_match", s, t) for s, t in zip(mr, tr)]
    keep = [i for i, v in enumerate(idx_s) if v is not None]
    if not keep:
        return {}
    Xr, mr, tr = Xr[keep], mr[keep], tr[keep]
    Xs = emb.X[np.array([idx_s[i] for i in keep])]

    mu = np.concatenate([Xr, Xs]).mean(axis=0, keepdims=True)
    A = Xr - mu
    B = Xs - mu
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    top1 = np.argmax(A @ B.T, axis=1)

    n = len(mr)
    exact = float(np.mean(top1 == np.arange(n)))
    mesh_hit = float(np.mean(mr[top1] == mr))
    # the shape-match image of cell (s,t) carries texture t+19, so "same texture"
    # is evaluated against the shape-match image's own texture label
    ts = tr[top1]
    tex_hit = float(np.mean(ts == tr))
    return {
        "exact_at1": exact,
        "exact_chance": 1.0 / n,
        "mesh_at1": mesh_hit,
        "mesh_chance": 1.0 / len(all_m),
        "texture_at1": tex_hit,
        "texture_chance": 1.0 / len(all_t),
        "n": n,
    }


# --------------------------------------------------------------------------
# synthetic self-test
# --------------------------------------------------------------------------


def synthetic(
    n_mesh=30, n_tex=38, dim=1152, mesh_dims=20, tex_scale=6.0, noise=0.35, seed=0
) -> GridEmbeddings:
    """The reviewer's hypothetical, built to spec.

    Mesh identity lives in ``mesh_dims`` of ``dim`` dimensions; surface texture
    fills the rest with much larger variance. Unweighted cosine should therefore
    be swamped and read at or below chance, while a linear probe that can
    reweight should recover mesh identity near ceiling. If this test does not
    behave that way, the analysis is broken, not the encoder.
    """
    rng = np.random.default_rng(seed)
    meshes = list(range(1, n_mesh + 1))
    textures = [f"Tex{k:03d}_1K-JPG" for k in range(n_tex)]
    M = rng.normal(size=(n_mesh, mesh_dims))
    T = rng.normal(size=(n_tex, dim - mesh_dims)) * tex_scale

    rows, X = [], []
    for si, s in enumerate(meshes):
        for ti, t in enumerate(textures):
            sm_t = (ti + TEX_OFFSET) % n_tex  # shape_match: same mesh, texture +19
            tm_s = (si + MESH_OFFSET) % n_mesh  # texture_match: mesh +15, same texture
            for role, mi, tj in (
                ("reference", si, ti),
                ("shape_match", si, sm_t),
                ("texture_match", tm_s, ti),
            ):
                vec = np.concatenate([M[mi], T[tj]]) + rng.normal(size=dim) * noise
                X.append(vec)
                rows.append((role, s, t))

    return GridEmbeddings(
        X=np.array(X, dtype=np.float32),
        role=np.array([r[0] for r in rows]),
        stl_id=np.array([r[1] for r in rows]),
        texture_set=np.array([r[2] for r in rows]),
        name="synthetic",
    )


def run_self_test() -> int:
    print("Synthetic self-test: mesh identity in 20 of 1152 dims, texture in 1132.")
    print("Expectation: centred cosine at or below chance; mesh probe near ceiling.\n")
    emb = synthetic()
    splits = build_splits(emb)
    assert_disjoint(splits, emb)
    print("splits: disjoint and closed under both foil offsets  [ok]")
    print(
        f"  mesh folds {[len(f) for f in splits['mesh_folds']]}, "
        f"texture folds {[len(f) for f in splits['tex_pair_folds']]}"
    )

    ret = retrieval_metrics(emb)
    print(
        f"\nretrieval  exact@1={ret['exact_at1']:.3f} (chance {ret['exact_chance']:.3f})"
        f"  mesh@1={ret['mesh_at1']:.3f} (chance {ret['mesh_chance']:.3f})"
    )

    p1 = identity_probe(
        emb, "mesh", splits["tex_family_folds"], "texture", c_grid=(0.01,)
    )
    print(
        f"P1 mesh identity   acc={p1['accuracy']:.3f} "
        f"[{p1['ci95'][0]:.3f},{p1['ci95'][1]:.3f}]  chance={p1['chance']:.3f}"
    )

    lad = readout_ladder(emb, splits, c_grid=(0.01,), pca_ks=(32,))
    print("\nread-out power ladder (shape-match rate on held-out triads):")
    for k, v in lad["rungs"].items():
        print(
            f"  {k:22s} {v['shape_rate']:.3f} "
            f"[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]  n={v['n']}"
        )

    centered = lad["rungs"]["2_centered_cosine"]["shape_rate"]
    learned = lad["rungs"]["5_learned_metric"]["shape_rate"]
    ok = True
    if not p1["accuracy"] > 0.8:
        print(f"\nFAIL: mesh probe should be near ceiling, got {p1['accuracy']:.3f}")
        ok = False
    if not centered < 0.35:
        print(f"FAIL: centred cosine should be swamped, got {centered:.3f}")
        ok = False
    if not learned > 0.8:
        print(f"FAIL: learned metric should recover shape, got {learned:.3f}")
        ok = False
    if ok:
        print(
            f"\nPASS. Centred cosine {centered:.2f} vs learned metric {learned:.2f} "
            f"on identical triads.\nThe pipeline detects information that cosine "
            f"cannot see, which is the discrimination the reviewer asks for."
        )
    return 0 if ok else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", type=Path, help="embeddings .npz from embedding_export.py")
    ap.add_argument("--self-test", action="store_true", help="synthetic sanity check")
    ap.add_argument("--out", type=Path, help="write results JSON here")
    ap.add_argument("--n-perm", type=int, default=0, help="label permutations for P1/P2")
    ap.add_argument("--pixels", type=Path, help="pixel-baseline .npz for P3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.emb:
        ap.error("one of --emb or --self-test is required")

    emb = GridEmbeddings.load(args.emb)
    print(f"{emb.name}: X={emb.X.shape}, {len(emb.meshes)} meshes, "
          f"{len(emb.textures)} textures")
    splits = build_splits(emb)
    assert_disjoint(splits, emb)

    results = {
        "source": str(args.emb),
        "name": emb.name,
        "dim": int(emb.X.shape[1]),
        "n_meshes": len(emb.meshes),
        "n_textures": len(emb.textures),
        "retrieval": retrieval_metrics(emb),
        "P1_mesh_identity": identity_probe(
            emb, "mesh", splits["tex_family_folds"], "texture",
            n_perm=args.n_perm, seed=args.seed,
        ),
        "P2_texture_identity": identity_probe(
            emb, "texture", splits["mesh_folds"], "mesh",
            n_perm=args.n_perm, seed=args.seed,
        ),
    }
    lad = readout_ladder(emb, splits, seed=args.seed)
    per_triad = lad.pop("_per_triad")
    results["ladder"] = lad

    if args.pixels and args.pixels.exists():
        pix = GridEmbeddings.load(args.pixels)
        psplits = build_splits(pix)
        results["P3_pixel_mesh"] = identity_probe(
            pix, "mesh", psplits["tex_family_folds"], "texture", seed=args.seed
        )
        results["P3_pixel_texture"] = identity_probe(
            pix, "texture", psplits["mesh_folds"], "mesh", seed=args.seed
        )

    _print_report(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        np.savez_compressed(
            args.out.with_suffix(".pertriad.npz"),
            **{k: np.array(v, dtype=np.int8) for k, v in per_triad.items()},
        )
        print(f"\nJSON: {args.out}")
    return 0


def _print_report(r: dict) -> None:
    ret = r.get("retrieval") or {}
    if ret:
        print(
            f"\nretrieval  exact@1={ret['exact_at1']:.3f} (chance {ret['exact_chance']:.4f})"
            f"  mesh@1={ret['mesh_at1']:.3f} (chance {ret['mesh_chance']:.3f})"
            f"  tex@1={ret['texture_at1']:.3f} (chance {ret['texture_chance']:.3f})"
        )
    for key in ("P1_mesh_identity", "P2_texture_identity", "P3_pixel_mesh",
                "P3_pixel_texture"):
        p = r.get(key)
        if not p:
            continue
        line = (
            f"{key:22s} acc={p['accuracy']:.3f} "
            f"[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] chance={p['chance']:.3f}"
        )
        if "perm_p" in p:
            line += f"  perm_p={p['perm_p']:.4f}"
        print(line)
    print("\nread-out power ladder (shape-match rate, held-out meshes and textures):")
    for k, v in (r.get("ladder", {}).get("rungs") or {}).items():
        print(
            f"  {k:22s} {v['shape_rate']:.3f} "
            f"[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]  n={v['n']}"
        )
    mc = r.get("ladder", {}).get("mcnemar_centered_vs_learned")
    if mc:
        print(f"  McNemar centred vs learned: p={mc['p']:.4g} "
              f"(n01={mc['n01']}, n10={mc['n10']})")


if __name__ == "__main__":
    sys.exit(main())
