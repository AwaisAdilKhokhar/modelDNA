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
MERGE_CACHE = REPO / "benchmarks" / "mergebench_cache"
GROWTH_CACHE = REPO / "benchmarks" / "atlas_growth_cache"  # trending models, grown weekly
PAIRS = REPO / "benchmarks" / "lineagebench_pairs.json"
RESULTS = REPO / "benchmarks" / "lineagebench_results.json"
MERGE_RESULTS = REPO / "benchmarks" / "merge_decompose_results.json"
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
    if MERGE_CACHE.exists():
        sources.append(("mergebench", ReferenceDB(MERGE_CACHE)))
    if GROWTH_CACHE.exists():
        sources.append(("growth", ReferenceDB(GROWTH_CACHE)))
    for kind, db in sources:
        for e in db.entries():
            if kind == "ref":
                ref_ids.add(e.model_id)
            if e.model_id not in fps:
                fps[e.model_id] = db.load_fingerprint(e.model_id)
                families[e.model_id] = e.family or guess_family(e.model_id)
    return fps, families, ref_ids


def force_layout(n: int, edges: list[dict], groups: list[int], iters: int = 2400) -> np.ndarray:
    """Small deterministic 3-D force layout.

    Groups are seeded on a wide ring and held in their sector by anchor
    gravity while foreign groups repel each other harder than siblings, so
    families read as separate constellations. A final collision pass
    guarantees a minimum node separation inside each cluster.
    """
    rng = np.random.default_rng(7)
    g = np.asarray(groups)
    n_groups = g.max() + 1
    angle = 2 * math.pi * g / n_groups
    lift = (g % 3 - 1) * 3.2  # spread groups across three vertical layers
    anchors = np.stack([np.cos(angle) * 11.0, lift, np.sin(angle) * 11.0], axis=1)
    pos = anchors + rng.normal(0, 1.2, (n, 3))
    same = g[:, None] == g[None, :]
    rep_k = np.where(same, 14.0, 44.0)  # foreign clusters repel much harder
    ei = np.array([[e["a"], e["b"]] for e in edges]) if edges else np.zeros((0, 2), int)
    ew = np.array([e["p"] for e in edges]) if edges else np.zeros(0)
    for it in range(iters):
        temp = 0.12 * (1 - it / iters) + 0.004
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d**2).sum(-1) + 1e-6
        rep = (d * (rep_k / dist2)[..., None]).sum(1)
        force = rep + (anchors - pos) * 0.05  # sector gravity doubles as centering
        if len(ei):
            delta = pos[ei[:, 0]] - pos[ei[:, 1]]
            dl = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-6
            # generous rest lengths so tight families stay legible clusters,
            # not a single overlapping blob
            rest = 4.0 - 1.8 * ew[:, None]
            f = (dl - rest) * delta / dl * 0.8
            np.add.at(force, ei[:, 0], -f)
            np.add.at(force, ei[:, 1], f)
        pos += np.clip(force, -3, 3) * temp
    min_sep = 1.5
    for _ in range(80):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d**2).sum(-1) + 1e-9)
        np.fill_diagonal(dist, np.inf)
        ii, jj = np.where((dist < min_sep) & (np.arange(n)[:, None] < np.arange(n)))
        if not len(ii):
            break
        dvec = pos[ii] - pos[jj]
        dl = np.linalg.norm(dvec, axis=1, keepdims=True) + 1e-9
        shift = dvec / dl * (min_sep - dl) * 0.5
        np.add.at(pos, ii, shift)
        np.add.at(pos, jj, -shift)
    pos -= pos.mean(0)
    # robust scale: one far-flung node must not squash every cluster together
    pos /= np.quantile(np.abs(pos), 0.96) + 1e-9
    return np.clip(pos, -1.3, 1.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "site" / "index.html")
    args = ap.parse_args()

    fps, families, ref_ids = load_fingerprints()
    manifest = json.loads(PAIRS.read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    verified = {(p["suspect"], p["parent"]): p for p in manifest["pairs"]}
    verdicts = {r["suspect"]: r for r in results.get("attribution", [])}

    # merge decompositions validated against published mergekit configs
    # (benchmarks/merge_decompose_bench.py): (target, parent) -> fitted alpha
    merge_w: dict[tuple[str, str], float] = {}
    merge_targets: dict[str, str] = {}  # target -> label for the node card
    merge_cards: list[dict] = []  # the "recipes recovered" table on the page
    if MERGE_RESULTS.exists():
        mres = json.loads(MERGE_RESULTS.read_text())
        for key, label in (("neuralpipe_slerp", "slerp merge · decomposed"),
                           ("monarch_dare_ties", "dare_ties merge · decomposed")):
            r = mres.get(key)
            if not r or "target" not in r:
                continue
            merge_targets[r["target"]] = label
            for pid, w in r["alphas"].items():
                if w >= 0.05:  # base columns and distractors sit at ~0
                    merge_w[(r["target"], pid)] = round(float(w), 3)
        np_r = mres.get("neuralpipe_slerp")
        if np_r and "target" in np_r:
            merge_cards.append({
                "target": np_r["target"],
                "method": "slerp · layer-varying t",
                "summary": np_r["summary"],
                "rows": [
                    {"id": pid, "alpha": round(w, 3), "published": None, "base": False}
                    for pid, w in sorted(np_r["alphas"].items(), key=lambda kv: -kv[1])
                ],
                "note": (
                    "the model card's opposite attention/MLP t-curves are "
                    "recovered layer by layer at r="
                    f"{np_r['attn_t_corr_vs_config']:.3f} (attention) and "
                    f"r={np_r['mlp_t_corr_vs_config']:.2f} (MLP); the global "
                    "weights above are the depth average"
                ),
            })
        mo = mres.get("monarch_dare_ties")
        if mo and "target" in mo:
            pub = mo.get("published_weights", {})
            merge_cards.append({
                "target": mo["target"],
                "method": "dare_ties",
                "summary": mo["summary"],
                "rows": [
                    {"id": pid, "alpha": round(w, 3),
                     "published": pub.get(pid), "base": pid not in pub}
                    for pid, w in sorted(mo["alphas"].items(), key=lambda kv: -kv[1])
                ],
                "note": (
                    "max deviation from the published recipe "
                    f"{mo['max_abs_error']:.3f}; the verdict hedges to "
                    f"{mo['summary']} because the Beagle parents are themselves "
                    "overlapping merges — the weights still land on the card"
                ),
            })

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
            "kind": merge_targets.get(m)
            or (verified.get((m, v["true_parent"]))["kind"] if v else None),
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
            mw = merge_w.get((a, b)) or merge_w.get((b, a))
            if p >= EDGE_P or ver or mw:
                f = ev.features()
                child = idx[ver["suspect"]] if ver else None
                if mw:
                    child = idx[a if (a, b) in merge_w else b]
                edges.append({
                    "a": i, "b": j, "p": p,
                    "verified": bool(ver) or bool(mw),
                    "kind": (f"merge α={mw:.2f}" if mw
                             else ver["kind"] if ver else None),
                    "merge": mw,
                    # direction: child listed first in the manifest
                    "child": child,
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
        "merges": merge_cards,
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
