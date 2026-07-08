"""MergeBench: validate `modeldna decompose` against real published mergekit configs.

Two real Hub merges whose exact mergekit YAML is on the model card serve as
ground truth (fingerprints cached in mergebench_cache/, ~300 MB of Hub reads
each to regenerate with `modeldna fingerprint`):

1. mlabonne/NeuralPipe-7B-slerp — slerp of OpenPipe/mistral-ft-optimized-1218
   and mlabonne/NeuralHermes-2.5-Mistral-7B with OPPOSITE layer-varying
   t-curves: self_attn t anchors [0, 0.5, 0.3, 0.7, 1], mlp [1, 0.5, 0.7,
   0.3, 0]. Tests per-role per-layer recovery of the interpolation curve.

2. mlabonne/Monarch-7B — dare_ties of OmniTruthyBeagle-7B-v0 (weight 0.36),
   NeuBeagle-7B (0.34), NeuralOmniBeagle-7B (0.30) over base
   mistralai/Mistral-7B-v0.1. Tests global weight recovery under DARE
   sparsification (densities 0.6-0.65). The parents are themselves
   overlapping Beagle merges (task vectors r=0.987), so the summary label is
   expected to hedge to AMBIGUOUS while the weights land on the config —
   that abstention is by design, not a miss.

3. Chain closure: mlabonne/AlphaMonarch-7B (DPO descendant of Monarch via
   NeuralMonarch; fingerprint in lineagebench_cache) must put alpha ~1 on
   Monarch-7B once Monarch is in the candidate list.

Run:  python benchmarks/merge_decompose_bench.py
Exits non-zero if any gate fails; writes merge_decompose_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from modeldna.fingerprint.extract import Fingerprint
from modeldna.merge import _affine_fit, decompose_fingerprints

HERE = Path(__file__).parent
CACHE = HERE / "mergebench_cache" / "fingerprints"
LINEAGE_CACHE = HERE / "lineagebench_cache" / "fingerprints"


def fp(name: str, cache: Path = CACHE) -> Fingerprint:
    return Fingerprint.load(cache / (name + ".json.gz"))


def slerp_t_curves(n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """mergekit interpolates the 5 t anchors evenly across the layer range."""
    anchors = np.linspace(0, n_layers - 1, 5)
    t_attn = np.interp(np.arange(n_layers), anchors, [0, 0.5, 0.3, 0.7, 1])
    t_mlp = np.interp(np.arange(n_layers), anchors, [1, 0.5, 0.7, 0.3, 0])
    return t_attn, t_mlp


def per_role_layer_alpha(
    target: Fingerprint, p1: Fingerprint, p2: Fingerprint, role: str
) -> np.ndarray:
    """Fitted alpha of p2 per layer for one role (slerp t recovery)."""
    n_layers = target.arch.n_layers
    chunk = len(target.pcs_samples[role]) // n_layers
    out = []
    for layer in range(n_layers):
        s = slice(layer * chunk, (layer + 1) * chunk)
        y = np.asarray(target.pcs_samples[role][s])
        x = np.column_stack(
            [np.asarray(p1.pcs_samples[role][s]), np.asarray(p2.pcs_samples[role][s])]
        )
        a, _, _ = _affine_fit(y, x)
        out.append(float(a[1]))
    return np.array(out)


def bench_slerp(results: dict, failures: list[str]) -> None:
    tgt = fp("mlabonne__NeuralPipe-7B-slerp")
    p1 = fp("OpenPipe__mistral-ft-optimized-1218")
    p2 = fp("mlabonne__NeuralHermes-2.5-Mistral-7B")

    dec = decompose_fingerprints(tgt, [p1, p2])
    t_attn, t_mlp = slerp_t_curves(tgt.arch.n_layers)
    attn_fit = np.mean(
        [per_role_layer_alpha(tgt, p1, p2, r) for r in dec.per_role if r.startswith("attn")],
        axis=0,
    )
    mlp_fit = per_role_layer_alpha(tgt, p1, p2, "mlp.down")
    attn_r = float(np.corrcoef(attn_fit, t_attn)[0, 1])
    mlp_r = float(np.corrcoef(mlp_fit, t_mlp)[0, 1])

    results["neuralpipe_slerp"] = {
        "target": tgt.model_id,
        "summary": dec.summary,
        "alphas": dec.alphas,
        "recon_cos": dec.recon_cos,
        "gain_vs_single": dec.gain_vs_single,
        "attn_t_corr_vs_config": attn_r,
        "mlp_t_corr_vs_config": mlp_r,
        "attn_t_fitted": [round(v, 3) for v in attn_fit],
        "mlp_t_fitted": [round(v, 3) for v in mlp_fit],
        "notes": dec.notes,
    }
    print(f"NeuralPipe-7B-slerp: {dec.summary}  "
          f"attn t-curve corr {attn_r:.4f}  mlp {mlp_r:.4f}  "
          f"gain {dec.gain_vs_single:.3f}")

    if dec.summary != "MERGE":
        failures.append(f"slerp: expected MERGE, got {dec.summary}")
    if attn_r < 0.99:
        failures.append(f"slerp: attn t-curve corr {attn_r:.4f} < 0.99")
    if mlp_r < 0.95:
        failures.append(f"slerp: mlp t-curve corr {mlp_r:.4f} < 0.95")
    if not any("vary across layers" in n for n in dec.notes):
        failures.append("slerp: layer-variation note did not fire")


#: published dare_ties weights from the Monarch-7B model card
MONARCH_TRUTH = {
    "mlabonne/OmniTruthyBeagle-7B-v0": 0.36,
    "mlabonne/NeuBeagle-7B": 0.34,
    "mlabonne/NeuralOmniBeagle-7B": 0.30,
}


def bench_monarch(results: dict, failures: list[str]) -> None:
    tgt = fp("mlabonne__Monarch-7B")
    parents = [
        fp("mlabonne__OmniTruthyBeagle-7B-v0"),
        fp("mlabonne__NeuBeagle-7B"),
        fp("mlabonne__NeuralOmniBeagle-7B"),
    ]
    base = fp("mistralai__Mistral-7B-v0.1", LINEAGE_CACHE)

    dec = decompose_fingerprints(tgt, parents, base=base)
    errs = {p: dec.alphas[p] - w for p, w in MONARCH_TRUTH.items()}
    max_err = max(abs(v) for v in errs.values())

    results["monarch_dare_ties"] = {
        "target": tgt.model_id,
        "summary": dec.summary,
        "alphas": dec.alphas,
        "published_weights": MONARCH_TRUTH,
        "max_abs_error": max_err,
        "alpha_spread": dec.alpha_spread,
        "recon_cos": dec.recon_cos,
        "gain_vs_single": dec.gain_vs_single,
        "notes": dec.notes,
    }
    fitted = "/".join(f"{dec.alphas[p]:.3f}" for p in MONARCH_TRUTH)
    print(f"Monarch-7B dare_ties: {dec.summary}  fitted {fitted} "
          f"vs published 0.36/0.34/0.30  max err {max_err:.3f}")

    if max_err > 0.03:
        failures.append(f"monarch: max weight error {max_err:.3f} > 0.03")
    if abs(dec.alphas[base.model_id]) > 0.05:
        failures.append(f"monarch: base alpha {dec.alphas[base.model_id]:.3f}, expected ~0")
    if dec.summary not in ("MERGE", "AMBIGUOUS"):
        failures.append(f"monarch: expected MERGE/AMBIGUOUS, got {dec.summary}")
    if not any("nearly parallel" in n for n in dec.notes):
        failures.append("monarch: collinearity warning did not fire "
                        "(the Beagle parents share ancestry)")


def bench_alphamonarch_chain(results: dict, failures: list[str]) -> None:
    am = fp("mlabonne__AlphaMonarch-7B", LINEAGE_CACHE)
    monarch = fp("mlabonne__Monarch-7B")
    decoy = fp("teknium__OpenHermes-2.5-Mistral-7B", LINEAGE_CACHE)
    base = fp("mistralai__Mistral-7B-v0.1", LINEAGE_CACHE)

    dec = decompose_fingerprints(am, [monarch, decoy], base=base)
    results["alphamonarch_chain"] = {
        "target": am.model_id,
        "summary": dec.summary,
        "alphas": dec.alphas,
        "gain_vs_single": dec.gain_vs_single,
    }
    print(f"AlphaMonarch chain: {dec.summary}  "
          f"alpha(Monarch) {dec.alphas[monarch.model_id]:.3f}")

    if dec.summary != "SINGLE_PARENT":
        failures.append(f"alphamonarch: expected SINGLE_PARENT, got {dec.summary}")
    if dec.alphas[monarch.model_id] < 0.95:
        failures.append(
            f"alphamonarch: alpha(Monarch) {dec.alphas[monarch.model_id]:.3f} < 0.95"
        )


def main() -> int:
    results: dict = {}
    failures: list[str] = []
    bench_slerp(results, failures)
    bench_monarch(results, failures)
    bench_alphamonarch_chain(results, failures)

    out = HERE / "merge_decompose_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out}")
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
