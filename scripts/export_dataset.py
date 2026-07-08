"""Export the fingerprint DB + inferred lineage graph as a Hugging Face dataset.

Builds a dataset directory from the same three fingerprint sources as the
Atlas (reference DB, LineageBench cache, mergebench cache):

    README.md                dataset card (generated from dataset_card_template.md)
    models.jsonl             one row per fingerprinted model
    pairs.jsonl              every depth-compatible pairwise comparison
    edges.jsonl              the inferred graph: detected / documented / merge edges
    fingerprints/<id>.json.gz  the raw fingerprints themselves
    modeldna-refdb.tar.gz    bases-only reference DB (`modeldna db pull --url ...`)

Usage:  python scripts/export_dataset.py [--out DIR] [--push] [--repo-id ID]

--push uploads the directory to the Hub (dataset repo, created if missing)
with whatever HF credentials are ambient (stored login or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas import (  # noqa: E402
    EDGE_P,
    MERGE_RESULTS,
    PAIRS,
    RESULTS,
    group_of,
    load_fingerprints,
)

from modeldna import __version__  # noqa: E402
from modeldna.calibration import DEFAULT_CALIBRATOR  # noqa: E402
from modeldna.compare import compare_fingerprints  # noqa: E402
from modeldna.db.store import ReferenceDB, _safe_name, default_db_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).parent / "dataset_card_template.md"
DEFAULT_REPO_ID = "AwaisAdilKhokhar/modeldna-atlas"

ARCH_FIELDS = (
    "model_type", "hidden_size", "n_heads", "n_kv_heads",
    "intermediate_size", "vocab_size", "n_params", "torch_dtype",
)


def jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def merge_edges() -> dict[tuple[str, str], float]:
    """(target, parent) -> fitted alpha, for the validated decompositions."""
    out: dict[tuple[str, str], float] = {}
    if MERGE_RESULTS.exists():
        mres = json.loads(MERGE_RESULTS.read_text())
        for r in mres.values():
            if isinstance(r, dict) and "target" in r and "alphas" in r:
                for pid, w in r["alphas"].items():
                    if w >= 0.05:  # base columns and distractors sit at ~0
                        out[(r["target"], pid)] = round(float(w), 3)
    return out


def build(out: Path) -> dict:
    fps, families, ref_ids = load_fingerprints()
    manifest = json.loads(PAIRS.read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    documented = {(p["suspect"], p["parent"]): p for p in manifest["pairs"]}
    verdicts = {r["suspect"]: r for r in results.get("attribution", [])}
    merges = merge_edges()

    ids = sorted(fps, key=str.lower)
    out.mkdir(parents=True, exist_ok=True)
    fp_dir = out / "fingerprints"
    fp_dir.mkdir(exist_ok=True)

    models = []
    for m in ids:
        fp = fps[m]
        v = verdicts.get(m)
        fname = _safe_name(m) + ".json.gz"
        fp.save(fp_dir / fname)
        row = {
            "model_id": m,
            "family": families[m],
            "group": group_of(families[m]),
            "n_layers": fp.arch.n_layers,
            **{k: getattr(fp.arch, k) for k in ARCH_FIELDS},
            "revision": fp.revision,
            "fingerprint_mode": fp.mode,
            "fingerprint_created_at": fp.created_at,
            "fingerprint_file": f"fingerprints/{fname}",
            "in_reference_db": m in ref_ids,
            "documented_parent": v["true_parent"] if v else None,
            "lineagebench_verdict": v["verdict"] if v else None,
        }
        models.append(row)

    print(f"comparing {len(ids)} models pairwise ...")
    pairs, edges = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if fps[a].arch.n_layers != fps[b].arch.n_layers:
                continue
            ev = compare_fingerprints(fps[a], fps[b])
            feats = ev.features()
            p = DEFAULT_CALIBRATOR.predict(feats)
            if p is None:
                continue
            p = round(float(p), 4)
            doc = documented.get((a, b)) or documented.get((b, a))
            alpha = merges.get((a, b)) or merges.get((b, a))
            child = None
            if doc:
                child = doc["suspect"]
            elif alpha:
                child = a if (a, b) in merges else b
            row = {
                "model_a": a, "model_b": b, "p_derived": p,
                **{k: (round(v, 4) if v is not None else None) for k, v in feats.items()},
                "detected": p >= EDGE_P,
                "documented_kind": doc["kind"] if doc else None,
                "child": child,
                "merge_alpha": alpha,
            }
            pairs.append(row)
            if row["detected"] or doc or alpha:
                edges.append(row)

    # documented pairs the tool abstains on (depth/quantization mismatch):
    # never compared above, but they belong in the graph with p_derived null
    compared = {frozenset((r["model_a"], r["model_b"])) for r in pairs}
    for (suspect, parent), doc in documented.items():
        if suspect in fps and parent in fps and frozenset((suspect, parent)) not in compared:
            edges.append({
                "model_a": suspect, "model_b": parent, "p_derived": None,
                "sigma_r": None, "vector_cos": None, "pcs_cos": None, "spectra_r": None,
                "detected": False, "documented_kind": doc["kind"],
                "child": suspect, "merge_alpha": None,
            })

    jsonl(out / "models.jsonl", models)
    jsonl(out / "pairs.jsonl", pairs)
    jsonl(out / "edges.jsonl", edges)

    # bases-only reference DB, same artifact `modeldna db pull` installs
    refdb_size = 0.0
    if default_db_path().exists():
        arc = ReferenceDB().export_archive(out / "modeldna-refdb.tar.gz")
        refdb_size = arc.stat().st_size / 1e6

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modeldna_version": __version__,
        "n_models": len(models),
        "n_reference": len(ref_ids & set(ids)),
        "n_pairs": len(pairs),
        "n_edges": len(edges),
        "n_detected": sum(1 for e in edges if e["detected"]),
        "n_documented": sum(1 for e in edges if e["documented_kind"]),
        "refdb_mb": round(refdb_size, 1),
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
        commit_message=f"refresh {datetime.now(timezone.utc).date()}",
        delete_patterns=["fingerprints/*", "*.jsonl"],  # drop rows/models removed upstream
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "hf-dataset")
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
