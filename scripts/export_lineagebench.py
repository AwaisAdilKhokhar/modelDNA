"""Publish LineageBench as a named public Hugging Face dataset.

LineageBench is the real-model lineage benchmark in ``benchmarks/``: a set of
suspect models whose parentage is corroborated by the publishing org's own
documentation (never the self-reported ``base_model`` tag), plus modelDNA's
per-pair results as the reference implementation. This script packages it as a
standalone, named dataset artifact — distinct from the modeldna-atlas dataset —
so the benchmark can be cited and loaded on its own:

    README.md                dataset card (from lineagebench_card_template.md)
    ground_truth.jsonl       one row per suspect: parent, kind, citation
    reference_results.jsonl  modelDNA's calibrated score + verdict per suspect
    parents.json             the candidate foundation parents
    metrics.json             headline ship-gate metrics

Usage:  python scripts/export_lineagebench.py [--out DIR] [--push] [--repo-id ID]

--push uploads the directory to the Hub (dataset repo, created if missing)
with whatever HF credentials are ambient (stored login or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from modeldna import __version__
from modeldna.db.families import guess_family

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"
MANIFEST = BENCH / "lineagebench_pairs.json"
RESULTS = BENCH / "lineagebench_results.json"
TEMPLATE = Path(__file__).parent / "lineagebench_card_template.md"
DEFAULT_REPO_ID = "AwaisAdilKhokhar/lineagebench"

# citation notes in the manifest sometimes append a modeldna-specific gloss after
# a " -- "; the dataset ground truth keeps only the sourcing sentence.
_CITATION_SPLIT = " -- "


def jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def build(out: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not RESULTS.exists():
        raise SystemExit(
            "benchmarks/lineagebench_results.json is missing — run "
            "`python benchmarks/real_lineagebench.py --no-fetch --out "
            "benchmarks/lineagebench_results.json` first."
        )
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    attribution = {r["suspect"]: r for r in results.get("attribution", [])}

    parents = manifest["parents"]
    pairs = manifest["pairs"]
    families = sorted({guess_family(p) for p in parents})

    # top-1 attribution over the *positive pool* only (the limitation cases are
    # scored separately, so the repo-wide accuracy in results.json understates
    # the in-scope figure).
    pos_ids = {p["suspect"] for p in pairs if p["in_positive_pool"]}
    pos_top1 = sum(
        1 for r in attribution.values() if r["suspect"] in pos_ids and r["top1_correct"]
    )
    n_pos = results["n_positive_pairs"]

    out.mkdir(parents=True, exist_ok=True)

    ground_truth = [
        {
            "suspect": p["suspect"],
            "parent": p["parent"],
            "kind": p["kind"],
            "in_positive_pool": p["in_positive_pool"],
            "citation": p["citation"].split(_CITATION_SPLIT, 1)[0].strip(),
        }
        for p in pairs
    ]
    reference_results = [
        {
            "suspect": r["suspect"],
            "true_parent": r["true_parent"],
            "kind": r["kind"],
            "score_vs_true_parent": r["score_vs_true_parent"],
            "verdict": r["verdict"],
            "verdict_best_candidate": r["verdict_best_candidate"],
            "top1_correct": r["top1_correct"],
        }
        for r in (attribution[p["suspect"]] for p in pairs if p["suspect"] in attribution)
    ]

    metrics = {
        k: results[k]
        for k in (
            "n_positive_pairs", "n_hard_negative_pairs", "n_limitation_cases",
            "auroc", "tpr_at_fpr_1pct", "false_positives_at_0.9",
            "min_positive_p", "max_negative_p", "by_kind_min_p",
            "top1_attribution_accuracy",
        )
        if k in results
    }
    metrics["reference_implementation"] = f"modeldna=={__version__}"
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    jsonl(out / "ground_truth.jsonl", ground_truth)
    jsonl(out / "reference_results.jsonl", reference_results)
    (out / "parents.json").write_text(
        json.dumps(parents, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    stats = {
        "n_pairs": len(pairs),
        "n_parents": len(parents),
        "n_families": len(families),
        "n_positives": results["n_positive_pairs"],
        "n_negatives": results["n_hard_negative_pairs"],
        "auroc": results["auroc"],
        "tpr": results["tpr_at_fpr_1pct"],
        "fp_at_09": results["false_positives_at_0.9"],
        "min_pos": results["min_positive_p"],
        "max_neg": results["max_negative_p"],
        "top1": f"{pos_top1}/{n_pos}",
        "modeldna_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    card = TEMPLATE.read_text(encoding="utf-8")
    for key, val in stats.items():
        card = card.replace(f"__{key.upper()}__", str(val))
    (out / "README.md").write_text(card, encoding="utf-8", newline="\n")
    return stats


def push(out: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out,
        commit_message=f"lineagebench refresh {datetime.now(timezone.utc).date()}",
        delete_patterns=["*.jsonl", "*.json"],  # drop rows removed upstream
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "lineagebench")
    ap.add_argument("--push", action="store_true", help="upload to the Hub after building")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = ap.parse_args()

    stats = build(args.out)
    print(json.dumps(stats, indent=2))
    print(f"-> {args.out}")
    if args.push:
        print("uploaded:", push(args.out, args.repo_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
