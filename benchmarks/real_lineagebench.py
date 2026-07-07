"""Real-model LineageBench: the synthetic_bench.py ship gates, run on real Hub
weights instead of generated models (PRD section 10.1-10.3).

Ground truth lives in ``lineagebench_pairs.json``: a set of foundation bases
(already in the seed reference DB) and a set of suspect models whose
parentage is corroborated by the publishing org's own documentation, not
just the self-reported ``base_model`` tag. This is a deliberately smaller,
honestly-scoped slice of the PRD's >=300-pair target -- see README.

Fingerprints are cached in ``lineagebench_cache/`` (committed to the repo, a
few MB total) so the benchmark is reproducible without re-hitting the Hub.
Foundation parents are copied out of the local reference DB when present
(``modeldna db build`` output) instead of re-fetched; only the suspect
models actually need a Hub round-trip. Interrupted runs resume: anything
already in the cache is skipped.

Usage: python benchmarks/real_lineagebench.py [--out results.json] [--no-fetch]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modeldna.calibration import DEFAULT_CALIBRATOR
from modeldna.compare import compare_fingerprints
from modeldna.db.families import guess_family
from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import open_source
from modeldna.verdict import Thresholds, judge

HERE = Path(__file__).parent
MANIFEST_PATH = HERE / "lineagebench_pairs.json"
CACHE_DIR = HERE / "lineagebench_cache"


# -- metrics (mirrors synthetic_bench.py; kept standalone on purpose) --------


def auroc(pos: list[float], neg: list[float]) -> float:
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
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
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    edges = np.linspace(0, 1, bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if mask.sum() >= 5:
            worst = max(worst, abs(scores[mask].mean() - labels[mask].mean()))
    return worst


# -- cache -------------------------------------------------------------------


def ensure_cached(model_id: str, cache: ReferenceDB, prod: ReferenceDB) -> None:
    if cache.get(model_id) is not None:
        return
    entry = prod.get(model_id)
    if entry is not None:
        fp = prod.load_fingerprint(model_id)
        cache.add(fp, family=entry.family, meta={"source": "copied-from-local-db"})
        print(f"  cached {model_id} (copied from local reference DB)")
        return
    print(f"  fetching {model_id} from the Hub ...")
    src = open_source(model_id)
    fp = extract_fingerprint(src, mode="fast", model_id=model_id)
    cache.add(fp, family=guess_family(model_id), meta={"source": "hub-fast-fetch"})
    print(f"    ok ({src.bytes_read / 1e6:.0f} MB read)")


def build_cache(manifest: dict, cache: ReferenceDB) -> None:
    prod = ReferenceDB()
    for p in manifest["parents"]:
        ensure_cached(p, cache, prod)
    for pair in manifest["pairs"]:
        ensure_cached(pair["suspect"], cache, prod)


# -- evaluation ---------------------------------------------------------------


def run(manifest: dict, cache: ReferenceDB) -> dict:
    parents = manifest["parents"]
    parent_family = {p: guess_family(p) for p in parents}
    fps = {m: cache.load_fingerprint(m) for m in parents}
    for pair in manifest["pairs"]:
        fps[pair["suspect"]] = cache.load_fingerprint(pair["suspect"])

    pos_scores: list[float] = []
    pos_by_kind: dict[str, list[float]] = {}
    neg_scores: list[float] = []
    attribution: list[dict] = []
    limitation_cases: list[dict] = []

    for pair in manifest["pairs"]:
        suspect, true_parent, kind = pair["suspect"], pair["parent"], pair["kind"]
        fp_s = fps[suspect]

        evidences = []
        for p in parents:
            ev = compare_fingerprints(fp_s, fps[p])
            evidences.append((ev, parent_family[p]))
        verdict = judge(
            evidences,
            suspect_dtype=fp_s.arch.torch_dtype,
            candidate_dtypes={p: fps[p].arch.torch_dtype for p in parents},
            thresholds=Thresholds(),
        )
        top1_correct = verdict.best is not None and verdict.best.candidate_id == true_parent

        ev_true = compare_fingerprints(fp_s, fps[true_parent])
        score = DEFAULT_CALIBRATOR.predict(ev_true.features())

        record = {
            "suspect": suspect,
            "true_parent": true_parent,
            "kind": kind,
            "score_vs_true_parent": round(score, 4),
            "verdict": verdict.verdict.value,
            "verdict_best_candidate": verdict.best.candidate_id if verdict.best else None,
            "top1_correct": top1_correct,
        }
        attribution.append(record)

        if pair["in_positive_pool"]:
            pos_scores.append(score)
            pos_by_kind.setdefault(kind, []).append(score)
            # wrong-parent comparisons: this suspect against every foundation
            # model that ISN'T its true parent and isn't in the true parent's
            # family -- the false-positive-risk pairing a real scan would face
            true_fam = parent_family[true_parent]
            for p in parents:
                if p != true_parent and parent_family[p] != true_fam:
                    ev_wrong = compare_fingerprints(fp_s, fps[p])
                    neg_scores.append(DEFAULT_CALIBRATOR.predict(ev_wrong.features()))
        else:
            limitation_cases.append(record)

    # parent-vs-parent cross-family negatives
    for i, a in enumerate(parents):
        for b in parents[i + 1 :]:
            if parent_family[a] == parent_family[b]:
                continue
            ev = compare_fingerprints(fps[a], fps[b])
            neg_scores.append(DEFAULT_CALIBRATOR.predict(ev.features()))

    results = {
        "n_positive_pairs": len(pos_scores),
        "n_hard_negative_pairs": len(neg_scores),
        "n_limitation_cases": len(limitation_cases),
        "auroc": round(auroc(pos_scores, neg_scores), 6) if neg_scores else None,
        "tpr_at_fpr_1pct": round(tpr_at_fpr(pos_scores, neg_scores), 4) if neg_scores else None,
        "false_positives_at_0.9": int(np.sum(np.array(neg_scores) >= 0.9)) if neg_scores else 0,
        "min_positive_p": round(min(pos_scores), 4) if pos_scores else None,
        "max_negative_p": round(max(neg_scores), 4) if neg_scores else None,
        "max_calibration_gap": (
            round(max_calibration_gap(pos_scores, neg_scores), 4) if neg_scores else None
        ),
        "by_kind_min_p": {k: round(min(v), 4) for k, v in pos_by_kind.items()},
        "top1_attribution_accuracy": round(
            np.mean([r["top1_correct"] for r in attribution]), 4
        ),
        "attribution": attribution,
        "limitation_cases": limitation_cases,
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--no-fetch", action="store_true",
        help="skip cache building; fail if any manifest model is missing from cache",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    cache = ReferenceDB(CACHE_DIR)

    if not args.no_fetch:
        build_cache(manifest, cache)

    results = run(manifest, cache)

    summary = {k: v for k, v in results.items() if k not in ("attribution", "limitation_cases")}
    print(json.dumps(summary, indent=2))
    print("\nlimitation cases (excluded from the ship-gate pool by design):")
    for r in results["limitation_cases"]:
        print(f"  {r['suspect']} ({r['kind']}): score={r['score_vs_true_parent']} "
              f"verdict={r['verdict']} best_candidate={r['verdict_best_candidate']}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))

    ok = (
        results["auroc"] is not None
        and results["auroc"] >= 0.99
        and results["tpr_at_fpr_1pct"] >= 0.95
        and results["false_positives_at_0.9"] == 0
    )
    print("\nship gates:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
