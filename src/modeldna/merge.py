"""Merge decomposition: estimate the parent mixture of a merged model (F5).

mergekit's mainstream methods — linear, slerp, task arithmetic, TIES/DARE —
all yield per-tensor weights that are (near-)linear combinations of the
parent tensors:  W_merge ~= sum_i alpha_i * W_i.  PCS sample positions are
pure functions of (seed, role, layer, tensor size), so the identity
survives sampling: the merged model's PCS vector is the same combination of
the parents' PCS vectors, and the mixture weights can be recovered by least
squares on fingerprints alone — no full weight download.

Merge parents are almost always fine-tunes of one shared base, which makes
their raw PCS vectors nearly collinear (cosine 0.99+, all dominated by the
base). The fit is therefore constrained to sum(alpha) = 1, which is
algebraically a regression in task-vector (delta) space, where fine-tune
directions are nearly orthogonal and the system is well-conditioned.
Passing the shared base adds a column that absorbs un-finetuned mass, so
task-arithmetic merges (weights not summing to 1 over the parents)
decompose cleanly too.

Depth-changing merges (passthrough / frankenmerges) are out of scope: they
break per-layer alignment and are rejected up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import Fingerprint, extract_fingerprint
from modeldna.fingerprint.methods import cosine
from modeldna.io.source import open_source

#: alpha below this is treated as "did not contribute"
MATERIAL_ALPHA = 0.05
#: residual shrink vs the best single parent needed to call MERGE
MERGE_GAIN = 0.30
#: raw-space reconstruction cosine below this means the parents don't explain the model
EXPLAINED_COS = 0.90

SUMMARY_DESCRIPTIONS: dict[str, str] = {
    "MERGE": "weights are consistent with a multi-parent merge of the given parents",
    "SINGLE_PARENT": "one candidate explains the weights; no evidence the others contributed",
    "AMBIGUOUS": "the mixture improves the fit only marginally; evidence is "
    "suggestive, not conclusive (a shared base can look like a weak mixture — "
    "try passing --base)",
    "UNEXPLAINED": "the given parents do not explain this model's weights",
}


@dataclass
class MergeDecomposition:
    target_id: str
    parent_ids: list[str]
    base_id: str | None
    n_samples: int

    #: pooled sum-to-one least-squares weights, keyed by parent/base id
    alphas: dict[str, float] = field(default_factory=dict)
    #: std of the per-role fits — real heterogeneity, not a formal SE
    alpha_spread: dict[str, float] = field(default_factory=dict)
    per_role: dict[str, dict[str, float]] = field(default_factory=dict)
    #: id -> alpha per layer (empty when layers can't be split back out)
    per_layer: dict[str, list[float]] = field(default_factory=dict)

    recon_cos: float = 0.0  # cosine(target samples, fitted mixture)
    best_single: str = ""
    best_single_cos: float = 0.0
    #: 1 - rss(mixture) / rss(best single candidate); the merge-evidence stat
    gain_vs_single: float = 0.0
    #: independent cross-check fitted on 1-D norm/bias vectors (F2)
    f2_alphas: dict[str, float] = field(default_factory=dict)

    summary: str = "AMBIGUOUS"
    notes: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return SUMMARY_DESCRIPTIONS[self.summary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "parent_ids": self.parent_ids,
            "base_id": self.base_id,
            "n_samples": self.n_samples,
            "alphas": self.alphas,
            "alpha_spread": self.alpha_spread,
            "per_role": self.per_role,
            "per_layer": self.per_layer,
            "recon_cos": self.recon_cos,
            "best_single": self.best_single,
            "best_single_cos": self.best_single_cos,
            "gain_vs_single": self.gain_vs_single,
            "f2_alphas": self.f2_alphas,
            "summary": self.summary,
            "description": self.description,
            "notes": self.notes,
        }


# -- fitting ---------------------------------------------------------------------


def _affine_fit(y: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, float, int]:
    """min ||y - cols @ a||^2  s.t.  sum(a) = 1.

    Substituting a_k = 1 - sum(others) reduces this to unconstrained least
    squares on deltas relative to the last column (the pivot) — the
    well-conditioned task-vector form. Returns (a, rss, rank).
    """
    n, k = cols.shape
    pivot = cols[:, -1]
    if k == 1:
        r = y - pivot
        return np.array([1.0]), float(r @ r), 1
    d = cols[:, :-1] - pivot[:, None]
    b, _, rank, _ = np.linalg.lstsq(d, y - pivot, rcond=None)
    a = np.append(b, 1.0 - b.sum())
    r = y - cols @ a
    return a, float(r @ r), int(rank)


def _collinearity_notes(cols: np.ndarray, ids: list[str]) -> list[str]:
    """Warn when candidate task vectors are unidentifiable from each other."""
    notes: list[str] = []
    pivot = cols[:, -1]
    d = cols[:, :-1] - pivot[:, None]
    norms = np.linalg.norm(d, axis=0)
    ref = max(float(np.max(norms)), 1e-30)
    live: list[int] = []
    for i, nm in enumerate(norms):
        if nm < 1e-6 * ref:
            notes.append(
                f"{ids[i]} is numerically identical to {ids[-1]}; "
                "their weights are interchangeable"
            )
        else:
            live.append(i)
    if len(live) >= 2:
        dn = d[:, live] / norms[live]
        corr = dn.T @ dn
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                if abs(corr[a, b]) > 0.95:
                    notes.append(
                        f"task vectors of {ids[live[a]]} and {ids[live[b]]} are "
                        f"nearly parallel (r={corr[a, b]:.3f}); their split is "
                        "poorly identified"
                    )
    return notes


# -- sample assembly ---------------------------------------------------------------


def _shared_equal_roles(
    target: dict[str, list[float]], columns: list[dict[str, list[float]]]
) -> list[str]:
    """Roles present in the target and every column with identical sample sizes."""
    roles = set(target)
    for c in columns:
        roles &= set(c)
    return sorted(
        r for r in roles if all(len(c[r]) == len(target[r]) for c in columns)
    )


def _stack(
    role_dicts: list[dict[str, list[float]]], roles: list[str]
) -> np.ndarray:
    """(n_samples, n_models) matrix of concatenated role samples."""
    return np.column_stack(
        [
            np.concatenate([np.asarray(d[r], dtype=np.float64) for r in roles])
            for d in role_dicts
        ]
    )


# -- decomposition ------------------------------------------------------------------


def _validate(target: Fingerprint, parents: list[Fingerprint], base: Fingerprint | None) -> None:
    if len(parents) < 2 and base is None:
        raise ValueError(
            "merge decomposition needs at least two candidate parents "
            "(for a single hypothesis pair use `modeldna compare`)"
        )
    columns = parents + ([base] if base else [])
    ids = [c.model_id for c in columns]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids")
    if target.model_id in ids:
        raise ValueError("target cannot also be a candidate parent")
    for c in columns:
        if c.seed != target.seed:
            raise ValueError(
                f"sampling seeds differ ({target.model_id}: {target.seed}, "
                f"{c.model_id}: {c.seed}); PCS samples are not comparable"
            )
        if c.arch.n_layers != target.arch.n_layers:
            raise ValueError(
                f"layer count differs ({target.model_id}: {target.arch.n_layers}, "
                f"{c.model_id}: {c.arch.n_layers}); depth-changing merges "
                "(passthrough / frankenmerge) are not decomposable in v1"
            )
        if not target.arch.shape_compatible(c.arch):
            raise ValueError(
                f"tensor shapes of {c.model_id} are incompatible with "
                f"{target.model_id}; linear merge decomposition needs "
                "matching shapes"
            )


def decompose_fingerprints(
    target: Fingerprint,
    parents: list[Fingerprint],
    base: Fingerprint | None = None,
) -> MergeDecomposition:
    """Fit the target's PCS samples as a sum-to-one mixture of the candidates."""
    _validate(target, parents, base)
    columns = parents + ([base] if base else [])  # base last -> delta pivot
    ids = [c.model_id for c in columns]

    roles = _shared_equal_roles(target.pcs_samples, [c.pcs_samples for c in columns])
    if not roles:
        raise ValueError("no shared PCS roles with matching sample sizes")

    y = _stack([target.pcs_samples], roles)[:, 0]
    x = _stack([c.pcs_samples for c in columns], roles)

    dec = MergeDecomposition(
        target_id=target.model_id,
        parent_ids=[p.model_id for p in parents],
        base_id=base.model_id if base else None,
        n_samples=int(y.size),
    )
    dec.notes.extend(_collinearity_notes(x, ids))

    a, rss, rank = _affine_fit(y, x)
    if rank < len(ids) - 1:
        dec.notes.append(
            "candidate task vectors are rank-deficient; weights are not "
            "uniquely determined"
        )
    dec.alphas = {i: float(v) for i, v in zip(ids, a)}
    dec.recon_cos = cosine(y, x @ a)

    # the merge-evidence statistic: how much of the residual left by the best
    # single candidate does the mixture remove?
    single_rss = {i: float(np.sum((y - x[:, j]) ** 2)) for j, i in enumerate(ids)}
    dec.best_single = min(single_rss, key=single_rss.get)
    dec.best_single_cos = cosine(y, x[:, ids.index(dec.best_single)])
    best_rss = single_rss[dec.best_single]
    dec.gain_vs_single = 1.0 - rss / best_rss if best_rss > 1e-30 else 0.0

    _fit_per_role(dec, target, columns, ids, roles)
    _fit_per_layer(dec, target, columns, ids, roles)
    _fit_f2(dec, target, columns, ids)

    dec.summary = _summarize(dec)
    _add_notes(dec)
    return dec


def _fit_per_role(
    dec: MergeDecomposition,
    target: Fingerprint,
    columns: list[Fingerprint],
    ids: list[str],
    roles: list[str],
) -> None:
    per_id: dict[str, list[float]] = {i: [] for i in ids}
    for role in roles:
        yr = np.asarray(target.pcs_samples[role], dtype=np.float64)
        xr = np.column_stack(
            [np.asarray(c.pcs_samples[role], dtype=np.float64) for c in columns]
        )
        ar, _, _ = _affine_fit(yr, xr)
        dec.per_role[role] = {i: float(v) for i, v in zip(ids, ar)}
        for i, v in zip(ids, ar):
            per_id[i].append(float(v))
    if len(roles) >= 2:
        dec.alpha_spread = {i: float(np.std(v)) for i, v in per_id.items()}


def _fit_per_layer(
    dec: MergeDecomposition,
    target: Fingerprint,
    columns: list[Fingerprint],
    ids: list[str],
    roles: list[str],
) -> None:
    """Split each role's samples back into equal per-layer chunks and refit.

    Within a role every layer has the same tensor shape, so the extractor
    contributed equal-length chunks in layer order; a role whose length isn't
    divisible by n_layers (shape varies with depth) is skipped.
    """
    n_layers = target.arch.n_layers
    if n_layers < 2:
        return
    split_roles = [r for r in roles if len(target.pcs_samples[r]) % n_layers == 0]
    if not split_roles:
        dec.notes.append("PCS samples can't be split by layer; per-layer fit skipped")
        return

    profile: dict[str, list[float]] = {i: [] for i in ids}
    for layer in range(n_layers):
        parts_y, parts_x = [], []
        for role in split_roles:
            chunk = len(target.pcs_samples[role]) // n_layers
            s = slice(layer * chunk, (layer + 1) * chunk)
            parts_y.append(np.asarray(target.pcs_samples[role][s], dtype=np.float64))
            parts_x.append(
                np.column_stack(
                    [np.asarray(c.pcs_samples[role][s], dtype=np.float64) for c in columns]
                )
            )
        al, _, _ = _affine_fit(np.concatenate(parts_y), np.vstack(parts_x))
        for i, v in zip(ids, al):
            profile[i].append(float(v))
    dec.per_layer = profile


def _fit_f2(
    dec: MergeDecomposition,
    target: Fingerprint,
    columns: list[Fingerprint],
    ids: list[str],
) -> None:
    """Independent cross-check on the 1-D norm/bias vectors, which merge
    linearly too. Low-powered (norm gains barely move under light SFT) but
    it is a different signal family entirely."""
    roles = _shared_equal_roles(
        target.vector_samples, [c.vector_samples for c in columns]
    )
    if not roles:
        return
    y = _stack([target.vector_samples], roles)[:, 0]
    x = _stack([c.vector_samples for c in columns], roles)
    if y.size < 16 * len(ids):
        return
    a, _, rank = _affine_fit(y, x)
    if rank < len(ids) - 1:
        return
    dec.f2_alphas = {i: float(v) for i, v in zip(ids, a)}


def _summarize(dec: MergeDecomposition) -> str:
    if dec.recon_cos < EXPLAINED_COS:
        return "UNEXPLAINED"
    material = [p for p in dec.parent_ids if dec.alphas[p] >= MATERIAL_ALPHA]
    if len(material) <= 1:
        return "SINGLE_PARENT"
    if dec.gain_vs_single >= MERGE_GAIN:
        return "MERGE"
    return "AMBIGUOUS"


def _add_notes(dec: MergeDecomposition) -> None:
    if dec.summary == "MERGE" and dec.gain_vs_single < 0.98:
        dec.notes.append(
            "a residual remains after the mixture — consistent with "
            "post-merge fine-tuning, TIES/DARE sparsification, or an "
            "unlisted parent"
        )
    if dec.per_layer:
        max_layer_std = max(
            float(np.std(v)) for p, v in dec.per_layer.items() if p in dec.parent_ids
        )
        if max_layer_std > 0.10 and dec.summary == "MERGE":
            dec.notes.append(
                "mixture weights vary across layers "
                f"(max per-parent std {max_layer_std:.2f}); consistent with a "
                "gradient/slerp-style merge"
            )
    if dec.f2_alphas and dec.summary == "MERGE":
        diff = max(
            abs(dec.f2_alphas.get(p, 0.0) - dec.alphas[p]) for p in dec.parent_ids
        )
        if diff > 0.15:
            dec.notes.append(
                "norm-vector cross-check (F2) diverges from the F3 fit "
                f"(max deviation {diff:.2f}); treat exact weights with caution"
            )
    negatives = [i for i, v in dec.alphas.items() if v < -MATERIAL_ALPHA]
    if negatives:
        dec.notes.append(
            f"negative weight for {', '.join(negatives)} — task-vector "
            "subtraction is a real mergekit technique, but check "
            "identifiability warnings first"
        )


# -- orchestration -------------------------------------------------------------------


def _resolve_fingerprint(
    ident: str, db: ReferenceDB | None, mode: str, revision: str
) -> Fingerprint:
    """Load from the reference DB when indexed there, else fetch and extract."""
    if db is not None and db.get(ident) is not None:
        return db.load_fingerprint(ident)
    src = open_source(ident, revision=revision)
    return extract_fingerprint(src, mode=mode, model_id=ident, revision=revision)


def decompose_targets(
    target: str,
    parents: list[str],
    base: str | None = None,
    db: ReferenceDB | None = None,
    mode: str = "fast",
    revision: str = "main",
) -> MergeDecomposition:
    """Resolve ids to fingerprints (DB first) and decompose."""
    fp_t = _resolve_fingerprint(target, db, mode, revision)
    fp_parents = [_resolve_fingerprint(p, db, mode, revision) for p in parents]
    fp_base = _resolve_fingerprint(base, db, mode, revision) if base else None
    return decompose_fingerprints(fp_t, fp_parents, base=fp_base)
