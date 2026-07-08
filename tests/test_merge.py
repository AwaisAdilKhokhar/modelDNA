"""Merge decomposition: recover mixture weights of synthetic merges."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import make_tiny_llama
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import open_source
from modeldna.merge import decompose_fingerprints

DIMS = dict(n_layers=8, hidden=64, n_heads=8, n_kv_heads=4, intermediate=128)


def _fp(root: Path, model_id: str):
    return extract_fingerprint(open_source(str(root)), mode="fast", model_id=model_id)


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    """Base + three sibling fine-tunes, fingerprinted once for the module."""
    tmp = tmp_path_factory.mktemp("merge-zoo")
    base_w = make_tiny_llama(tmp / "base", seed=0, **DIMS)
    p1_w = make_tiny_llama(tmp / "p1", seed=1, base_weights=base_w, noise=0.004, **DIMS)
    p2_w = make_tiny_llama(tmp / "p2", seed=2, base_weights=base_w, noise=0.004, **DIMS)
    p3_w = make_tiny_llama(tmp / "p3", seed=3, base_weights=base_w, noise=0.004, **DIMS)
    weights = {"base": base_w, "p1": p1_w, "p2": p2_w, "p3": p3_w}
    fps = {name: _fp(tmp / name, f"local/{name}") for name in weights}
    return tmp, weights, fps


def _merge_model(tmp: Path, name: str, mixes: dict[str, np.ndarray], noise: float = 0.0):
    root = tmp / name
    make_tiny_llama(root, seed=99, base_weights=mixes, noise=noise, **DIMS)
    return _fp(root, f"local/{name}")


def test_linear_merge_recovers_weights(zoo):
    tmp, w, fps = zoo
    merged = {k: 0.6 * w["p1"][k] + 0.4 * w["p2"][k] for k in w["p1"]}
    fp_m = _merge_model(tmp, "m-60-40", merged)

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"]])
    assert dec.summary == "MERGE"
    assert abs(dec.alphas["local/p1"] - 0.6) < 0.02
    assert abs(dec.alphas["local/p2"] - 0.4) < 0.02
    assert dec.recon_cos > 0.999
    assert dec.gain_vs_single > 0.9
    # per-layer profile is flat for a uniform linear merge
    assert max(np.std(v) for v in dec.per_layer.values()) < 0.05
    # F2 cross-check agrees
    assert abs(dec.f2_alphas["local/p1"] - 0.6) < 0.05


def test_distractor_parent_gets_zero(zoo):
    tmp, w, fps = zoo
    merged = {k: 0.5 * w["p1"][k] + 0.5 * w["p2"][k] for k in w["p1"]}
    fp_m = _merge_model(tmp, "m-distract", merged)

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"], fps["p3"]])
    assert dec.summary == "MERGE"
    assert abs(dec.alphas["local/p3"]) < 0.05
    assert abs(dec.alphas["local/p1"] - 0.5) < 0.05


def test_single_parent_is_not_called_a_merge(zoo):
    tmp, w, fps = zoo
    ft = {k: v + np.random.default_rng(7).normal(0, 0.001, v.shape).astype(np.float32)
          for k, v in w["p1"].items()}
    fp_t = _merge_model(tmp, "ft-of-p1", ft)

    dec = decompose_fingerprints(fp_t, [fps["p1"], fps["p2"]])
    assert dec.summary == "SINGLE_PARENT"
    assert dec.alphas["local/p1"] > 0.9
    assert dec.best_single == "local/p1"


def test_sibling_finetune_is_not_a_merge(zoo):
    # an independent fine-tune of the same base shares the base direction with
    # every parent; the fit must not upgrade that to MERGE
    tmp, w, fps = zoo
    dec = decompose_fingerprints(fps["p3"], [fps["p1"], fps["p2"]])
    assert dec.summary != "MERGE"


def test_unrelated_target_is_unexplained(zoo, tmp_path):
    tmp, w, fps = zoo
    make_tiny_llama(tmp_path / "alien", seed=42, **DIMS)
    fp_alien = _fp(tmp_path / "alien", "local/alien")
    dec = decompose_fingerprints(fp_alien, [fps["p1"], fps["p2"]])
    assert dec.summary == "UNEXPLAINED"


def test_task_arithmetic_with_base_column(zoo):
    tmp, w, fps = zoo
    merged = {
        k: w["base"][k] + 0.5 * (w["p1"][k] - w["base"][k]) + 0.3 * (w["p2"][k] - w["base"][k])
        for k in w["p1"]
    }
    fp_m = _merge_model(tmp, "m-task-arith", merged)

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"]], base=fps["base"])
    assert dec.summary == "MERGE"
    assert abs(dec.alphas["local/p1"] - 0.5) < 0.03
    assert abs(dec.alphas["local/p2"] - 0.3) < 0.03
    assert abs(dec.alphas["local/base"] - 0.2) < 0.03


def test_layer_gradient_merge_shows_in_profile(zoo):
    tmp, w, fps = zoo
    n = DIMS["n_layers"]

    def coeff(key: str) -> float:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) / (n - 1) if m else 0.5

    merged = {k: (1 - coeff(k)) * w["p1"][k] + coeff(k) * w["p2"][k] for k in w["p1"]}
    fp_m = _merge_model(tmp, "m-gradient", merged)

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"]])
    assert dec.summary == "MERGE"
    prof = dec.per_layer["local/p2"]
    assert prof[0] < 0.15 and prof[-1] > 0.85
    assert any("vary across layers" in note for note in dec.notes)


def test_post_merge_finetune_still_decomposes(zoo):
    tmp, w, fps = zoo
    merged = {k: 0.7 * w["p1"][k] + 0.3 * w["p2"][k] for k in w["p1"]}
    fp_m = _merge_model(tmp, "m-then-ft", merged, noise=0.002)

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"]])
    assert dec.summary == "MERGE"
    assert abs(dec.alphas["local/p1"] - 0.7) < 0.1
    assert any("residual" in note for note in dec.notes)


def test_validation_errors(zoo, tmp_path):
    tmp, w, fps = zoo
    with pytest.raises(ValueError, match="at least two"):
        decompose_fingerprints(fps["p3"], [fps["p1"]])
    with pytest.raises(ValueError, match="duplicate"):
        decompose_fingerprints(fps["p3"], [fps["p1"], fps["p1"]])
    with pytest.raises(ValueError, match="cannot also be"):
        decompose_fingerprints(fps["p1"], [fps["p1"], fps["p2"]])

    shallow_dims = dict(DIMS, n_layers=4)
    make_tiny_llama(tmp_path / "shallow", seed=5, **shallow_dims)
    fp_shallow = _fp(tmp_path / "shallow", "local/shallow")
    with pytest.raises(ValueError, match="layer count"):
        decompose_fingerprints(fp_shallow, [fps["p1"], fps["p2"]])


def test_identical_parents_flagged(zoo):
    tmp, w, fps = zoo
    merged = {k: 0.5 * w["p1"][k] + 0.5 * w["p2"][k] for k in w["p1"]}
    fp_m = _merge_model(tmp, "m-ident", merged)
    fp_p1_copy = _fp(tmp / "p1", "local/p1-mirror")

    dec = decompose_fingerprints(fp_m, [fps["p1"], fps["p2"], fp_p1_copy])
    assert any("interchangeable" in note or "nearly parallel" in note
               for note in dec.notes)
    # the pair splits arbitrarily but their sum is still ~the true weight
    combined = dec.alphas["local/p1"] + dec.alphas["local/p1-mirror"]
    assert abs(combined - 0.5) < 0.05
