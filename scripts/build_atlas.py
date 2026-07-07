"""Build the modelDNA Atlas — docs/index.html.

Loads every fingerprint available locally (the reference DB plus the
committed LineageBench cache), compares all depth-compatible pairs with the
real evidence pipeline, and bakes one self-contained HTML page: an
interactive weight-space family graph, the pairwise similarity matrix, and
the LineageBench scoreboard. No server, no CDN — the page is a single file
suitable for GitHub Pages.

Usage:  python scripts/build_atlas.py [--out docs/index.html]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from modeldna.calibration import DEFAULT_CALIBRATOR
from modeldna.compare import compare_fingerprints
from modeldna.db.families import guess_family
from modeldna.db.store import ReferenceDB, default_db_path

REPO = Path(__file__).resolve().parents[1]
BENCH_CACHE = REPO / "benchmarks" / "lineagebench_cache"
PAIRS = REPO / "benchmarks" / "lineagebench_pairs.json"
RESULTS = REPO / "benchmarks" / "lineagebench_results.json"
TEMPLATE = Path(__file__).parent / "atlas_template.html"

EDGE_P = 0.9  # graph edges: calibrated P(derived) at/above the reporting threshold

#: visual grouping only (7 groups + gray "Other") — never itself a lineage claim
GROUPS = {
    "llama": "Llama", "codellama": "Llama",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "phi": "Phi",
    "deepseek": "DeepSeek",
    "pythia": "Pythia",
}


def group_of(family: str) -> str:
    for prefix, group in GROUPS.items():
        if family.startswith(prefix):
            return group
    return "Other"


def load_fingerprints() -> tuple[dict, dict, set[str]]:
    """(fingerprints, family by id, ids present in the reference DB)."""
    fps: dict = {}
    families: dict[str, str] = {}
    ref_ids: set[str] = set()
    sources = []
    if default_db_path().exists():
        sources.append(("ref", ReferenceDB()))
    sources.append(("bench", ReferenceDB(BENCH_CACHE)))
    for kind, db in sources:
        for e in db.entries():
            if kind == "ref":
                ref_ids.add(e.model_id)
            if e.model_id not in fps:
                fps[e.model_id] = db.load_fingerprint(e.model_id)
                families[e.model_id] = e.family or guess_family(e.model_id)
    return fps, families, ref_ids


def force_layout(n: int, edges: list[dict], groups: list[int], iters: int = 1800) -> np.ndarray:
    """Small deterministic 3-D force layout, seeded by group direction."""
    rng = np.random.default_rng(7)
    n_groups = max(groups) + 1
    angle = np.array([2 * math.pi * g / n_groups for g in groups])
    lift = np.array([(g % 3 - 1) * 0.9 for g in groups])  # spread groups off-plane
    pos = np.stack([np.cos(angle), lift, np.sin(angle)], axis=1) * 10
    pos += rng.normal(0, 1.5, (n, 3))
    ei = np.array([[e["a"], e["b"]] for e in edges]) if edges else np.zeros((0, 2), int)
    ew = np.array([e["p"] for e in edges]) if edges else np.zeros(0)
    for it in range(iters):
        temp = 0.12 * (1 - it / iters) + 0.005
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d**2).sum(-1) + 1e-6
        rep = (d / dist2[..., None]).sum(1) * 16.0  # inverse-distance repulsion
        force = rep - pos * 0.035  # weak centering
        if len(ei):
            delta = pos[ei[:, 0]] - pos[ei[:, 1]]
            dl = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6
            # generous rest lengths so tight families stay legible clusters,
            # not a single overlapping blob
            rest = 3.4 - 1.6 * ew[:, None]
            f = (dl - rest) * delta / dl * 0.8
            np.add.at(force, ei[:, 0], -f)
            np.add.at(force, ei[:, 1], f)
        pos += np.clip(force, -3, 3) * temp
    pos -= pos.mean(0)
    pos /= np.abs(pos).max() + 1e-9  # normalize to [-1, 1]^3
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "site" / "index.html")
    args = ap.parse_args()

    fps, families, ref_ids = load_fingerprints()
    manifest = json.loads(PAIRS.read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    verified = {(p["suspect"], p["parent"]): p for p in manifest["pairs"]}
    verdicts = {r["suspect"]: r for r in results.get("attribution", [])}

    # bench suspects inherit their documented parent's family, so e.g.
    # zephyr colors as Mistral instead of falling into gray "Other"
    for (suspect, parent) in verified:
        if suspect in families and parent in families:
            if group_of(families[suspect]) == "Other":
                families[suspect] = families[parent]

    # order nodes by group -> family -> name so the matrix shows family blocks
    ids = sorted(fps, key=lambda m: (group_of(families[m]), families[m], m.lower()))
    idx = {m: i for i, m in enumerate(ids)}
    group_names = sorted({group_of(families[m]) for m in ids} - {"Other"}) + ["Other"]

    nodes = []
    for m in ids:
        arch = fps[m].arch
        v = verdicts.get(m)
        nodes.append({
            "id": m,
            "family": families[m],
            "group": group_names.index(group_of(families[m])),
            "layers": arch.n_layers,
            "params": round(arch.n_params / 1e9, 2),
            "type": arch.model_type,
            "ref": m in ref_ids,
            "bench": m in verdicts,
            "verdict": v["verdict"] if v else None,
            "kind": verified.get((m, v["true_parent"]))["kind"] if v else None,
        })

    print(f"comparing {len(ids)} models pairwise ...")
    matrix = [[None] * len(ids) for _ in ids]
    edges = []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if fps[a].arch.n_layers != fps[b].arch.n_layers:
                continue
            ev = compare_fingerprints(fps[a], fps[b])
            p = DEFAULT_CALIBRATOR.predict(ev.features())
            if p is None:
                continue
            p = round(float(p), 4)
            matrix[i][j] = matrix[j][i] = p
            ver = verified.get((a, b)) or verified.get((b, a))
            if p >= EDGE_P or ver:
                f = ev.features()
                edges.append({
                    "a": i, "b": j, "p": p,
                    "verified": bool(ver),
                    "kind": ver["kind"] if ver else None,
                    # direction: child listed first in the manifest
                    "child": idx[ver["suspect"]] if ver else None,
                    "feat": {k: (round(v, 3) if v is not None else None)
                             for k, v in f.items()},
                })

    # documented pairs the tool abstains on (depth/quantization mismatch)
    for (suspect, parent), pairdoc in verified.items():
        if suspect in idx and parent in idx:
            if not any({e["a"], e["b"]} == {idx[suspect], idx[parent]} for e in edges):
                edges.append({
                    "a": idx[suspect], "b": idx[parent], "p": None,
                    "verified": True, "kind": pairdoc["kind"],
                    "child": idx[suspect], "feat": None,
                })

    strong = [e for e in edges if e["p"] is not None and e["p"] >= EDGE_P]
    pos = force_layout(len(ids), strong, [n["group"] for n in nodes])
    for n, (x, y, z) in zip(nodes, pos):
        n["x"], n["y"], n["z"] = round(float(x), 4), round(float(y), 4), round(float(z), 4)

    summary = {k: v for k, v in results.items()
               if k not in ("attribution", "limitation_cases")}
    data = {
        "generated": manifest.get("generated", ""),
        "groups": group_names,
        "nodes": nodes,
        "edges": edges,
        "matrix": matrix,
        "bench": {"summary": summary,
                  "attribution": results.get("attribution", [])},
    }

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__ATLAS_DATA__*/", json.dumps(data, separators=(",", ":"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"{len(nodes)} models · {len(strong)} detected edges · "
          f"{sum(1 for e in edges if e['verified'])} verified pairs "
          f"-> {args.out} ({args.out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
