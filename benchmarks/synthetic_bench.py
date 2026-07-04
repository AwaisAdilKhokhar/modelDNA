"""Synthetic lineage benchmark: the LineageBench scaffolding, run on generated
models (real-model LineageBench needs Hub access; this validates the pipeline
and the calibrator end to end and runs in CI).

Population:
  positives  - fine-tunes and heavy continuations of each base at several
               noise scales, plus a float16 re-cast ("quantized copy" analog)
  negatives  - all cross-base same-architecture pairs (the hard negative:
               independent training with identical shapes)

Metrics: AUROC, TPR at the FPR<=1% operating point, false positives on hard
negatives at the default 0.9 reporting threshold, and max calibration gap.

Usage: python benchmarks/synthetic_bench.py [--bases 6] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from modeldna.calibration import DEFAULT_CALIBRATOR
from modeldna.compare import compare_fingerprints
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import LocalSource
from modeldna.testing import make_tiny_llama, write_safetensors

DIMS = dict(n_layers=8, hidden=64, n_heads=8, n_kv_heads=4, intermediate=128)
NOISE_LEVELS = [0.0005, 0.002, 0.01, 0.04, 0.1, 0.2]


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney)."""
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # midranks for ties
    for s in np.unique(scores):
        mask = scores == s
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(pos: list[float], neg: list[float], max_fpr: float = 0.01) -> float:
    thresh = np.quantile(neg, 1 - max_fpr) if neg else 1.0
    return float(np.mean([p > thresh for p in pos]))


def max_calibration_gap(pos: list[float], neg: list[float], bins: int = 5) -> float:
    """Max |mean predicted - empirical positive rate| over populated bins."""
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    edges = np.linspace(0, 1, bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if mask.sum() >= 5:
            worst = max(worst, abs(scores[mask].mean() - labels[mask].mean()))
    return worst


def run(n_bases: int, workdir: Path) -> dict:
    fps, weights = {}, {}
    for s in range(n_bases):
        root = workdir / f"base{s}"
        weights[s] = make_tiny_llama(root, seed=s, **DIMS)
        fps[f"base{s}"] = extract_fingerprint(LocalSource(root), model_id=f"base{s}")

    pos_scores: list[float] = []
    pos_by_kind: dict[str, list[float]] = {}
    for s in range(n_bases):
        for j, noise in enumerate(NOISE_LEVELS):
            root = workdir / f"d{s}_{j}"
            make_tiny_llama(root, seed=1000 + s * 100 + j, base_weights=weights[s],
                            noise=noise, **DIMS)
            fp = extract_fingerprint(LocalSource(root))
            ev = compare_fingerprints(fp, fps[f"base{s}"])
            p = DEFAULT_CALIBRATOR.predict(ev.features())
            pos_scores.append(p)
            kind = "fine-tune" if noise <= 0.01 else "continued-pretrain"
            pos_by_kind.setdefault(kind, []).append(p)

        # float16 re-cast of the base (quantized-copy analog)
        qroot = workdir / f"q{s}"
        shutil.copytree(workdir / f"base{s}", qroot)
        write_safetensors(
            qroot / "model.safetensors",
            {k: v.astype(np.float16) for k, v in weights[s].items()},
        )
        fp = extract_fingerprint(LocalSource(qroot))
        p = DEFAULT_CALIBRATOR.predict(compare_fingerprints(fp, fps[f"base{s}"]).features())
        pos_scores.append(p)
        pos_by_kind.setdefault("fp16-recast", []).append(p)

    neg_scores: list[float] = []
    for i in range(n_bases):
        for j in range(n_bases):
            if i != j:
                ev = compare_fingerprints(fps[f"base{i}"], fps[f"base{j}"])
                neg_scores.append(DEFAULT_CALIBRATOR.predict(ev.features()))

    results = {
        "n_positive_pairs": len(pos_scores),
        "n_hard_negative_pairs": len(neg_scores),
        "auroc": round(auroc(pos_scores, neg_scores), 6),
        "tpr_at_fpr_1pct": round(tpr_at_fpr(pos_scores, neg_scores), 4),
        "false_positives_at_0.9": int(np.sum(np.array(neg_scores) >= 0.9)),
        "min_positive_p": round(min(pos_scores), 4),
        "max_negative_p": round(max(neg_scores), 4),
        "max_calibration_gap": round(max_calibration_gap(pos_scores, neg_scores), 4),
        "by_kind_min_p": {k: round(min(v), 4) for k, v in pos_by_kind.items()},
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bases", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="modeldna-bench-"))
    try:
        results = run(args.bases, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps(results, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))

    # ship gates (synthetic analog of PRD 10.2)
    ok = (
        results["auroc"] >= 0.99
        and results["tpr_at_fpr_1pct"] >= 0.95
        and results["false_positives_at_0.9"] == 0
    )
    print("\nship gates:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
