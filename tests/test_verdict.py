import numpy as np
import pytest

from conftest import make_tiny_llama
from modeldna.calibration import DEFAULT_CALIBRATOR, LogisticCalibrator
from modeldna.compare import Evidence, compare_fingerprints
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import LocalSource
from modeldna.verdict import Thresholds, Verdict, VerdictClass, judge


def fake_evidence(
    candidate="org/base",
    sigma_r=1.0,
    vector_cos=1.0,
    pcs_cos=1.0,
    spectra_r=1.0,
    **kw,
) -> Evidence:
    ev = Evidence(suspect_id="org/suspect", candidate_id=candidate)
    ev.shape_compatible = kw.get("shape_compatible", True)
    ev.layers_match = kw.get("layers_match", True)
    ev.inventory_match = kw.get("inventory_match", False)
    if sigma_r is not None:
        ev.sigma_r = {"attn.q": sigma_r}
    if vector_cos is not None:
        ev.vector_cos = {"norm.in": vector_cos}
    if pcs_cos is not None:
        ev.pcs_cos = {"attn.q": pcs_cos}
    if spectra_r is not None:
        ev.spectra_r = {"attn.q": spectra_r}
    return ev


# -- calibration ------------------------------------------------------------


def test_calibrator_missing_features_neutral():
    p_missing = DEFAULT_CALIBRATOR.predict({})
    assert p_missing < 0.5  # absence of evidence is not evidence of lineage


def test_calibrator_fit_nonnegative():
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(200):
        derived = rng.random() < 0.5
        X.append(
            {
                "sigma_r": (0.98 if derived else rng.uniform(-0.5, 0.5)) + rng.normal(0, 0.01),
                "vector_cos": (1.0 if derived else 0.9) + rng.normal(0, 0.005),
                "pcs_cos": (0.9 if derived else 0.0) + rng.normal(0, 0.05),
                "spectra_r": 0.99 + rng.normal(0, 0.005),
            }
        )
        y.append(int(derived))
    cal = LogisticCalibrator.fit(X, y)
    assert all(w >= 0 for w in cal.coef.values())
    assert cal.predict(X[y.index(1)]) > 0.5


def test_calibrator_roundtrip(tmp_path):
    p = tmp_path / "cal.json"
    DEFAULT_CALIBRATOR.save(p)
    cal = LogisticCalibrator.load(p)
    assert cal.coef == DEFAULT_CALIBRATOR.coef
    assert cal.predict({"sigma_r": 1.0, "vector_cos": 1.0, "pcs_cos": 1.0, "spectra_r": 1.0}) \
        == pytest.approx(DEFAULT_CALIBRATOR.predict(
            {"sigma_r": 1.0, "vector_cos": 1.0, "pcs_cos": 1.0, "spectra_r": 1.0}))


# -- judge class logic (fabricated evidence) ----------------------------------


def test_judge_fine_tune():
    v = judge([(fake_evidence(pcs_cos=0.995), "llama")])
    assert v.verdict == VerdictClass.FINE_TUNE
    assert v.probability > 0.9


def test_judge_same_lineage():
    # continued-pretraining regime: lineage signals strong, parameter delta big
    v = judge([(fake_evidence(pcs_cos=0.45, sigma_r=0.97), "qwen")])
    assert v.verdict == VerdictClass.SAME_LINEAGE


def test_judge_exact_copy():
    ev = fake_evidence(pcs_cos=1.0, sigma_r=1.0, inventory_match=True)
    v = judge([(ev, "llama")])
    assert v.verdict == VerdictClass.EXACT_COPY


def test_judge_renamed_mirror_is_quantized_or_exact():
    # same values, renamed tensors -> inventory hash differs
    ev = fake_evidence(pcs_cos=1.0, sigma_r=1.0, inventory_match=False)
    v = judge([(ev, "llama")])
    assert v.verdict == VerdictClass.QUANTIZED_COPY


def test_judge_quantized_copy_dtype():
    ev = fake_evidence(pcs_cos=0.9993, sigma_r=0.9995)
    v = judge(
        [(ev, "llama")],
        suspect_dtype="float16",
        candidate_dtypes={"org/base": "bfloat16"},
    )
    assert v.verdict == VerdictClass.QUANTIZED_COPY


def test_judge_no_match():
    v = judge([(fake_evidence(sigma_r=0.2, vector_cos=0.9, pcs_cos=0.0, spectra_r=0.95), "phi")])
    assert v.verdict == VerdictClass.NO_MATCH
    assert v.probability < 0.5


def test_judge_abstention_band():
    ev = fake_evidence(sigma_r=0.85, vector_cos=0.95, pcs_cos=0.15, spectra_r=0.97)
    v = judge([(ev, "gemma")])
    assert v.verdict == VerdictClass.SAME_FAMILY_UNRESOLVED
    assert 0.5 <= v.probability < 0.9


def test_judge_family_tie():
    a = fake_evidence(candidate="qwen/base", pcs_cos=0.992)
    b = fake_evidence(candidate="qwen/base-instruct", pcs_cos=0.990)
    v = judge([(a, "qwen2.5"), (b, "qwen2.5")])
    assert v.verdict == VerdictClass.SAME_FAMILY_UNRESOLVED
    assert {c.candidate_id for c in v.candidates[:2]} == {"qwen/base", "qwen/base-instruct"}


def test_judge_likely_merge():
    a = fake_evidence(candidate="llama/base", pcs_cos=0.7)
    b = fake_evidence(candidate="mistral/base", pcs_cos=0.68)
    v = judge([(a, "llama"), (b, "mistral")])
    assert v.verdict == VerdictClass.LIKELY_MERGE


def test_judge_clear_winner_within_family():
    a = fake_evidence(candidate="qwen/base", pcs_cos=0.999)
    b = fake_evidence(candidate="qwen/other", pcs_cos=0.4, sigma_r=0.9)
    v = judge([(a, "qwen2.5"), (b, "qwen2.5")])
    assert v.verdict == VerdictClass.FINE_TUNE
    assert v.best.candidate_id == "qwen/base"


def test_judge_ruled_out_all():
    ev = fake_evidence(layers_match=False, sigma_r=None, vector_cos=None,
                       pcs_cos=None, spectra_r=None)
    v = judge([(ev, "llama")])
    assert v.verdict == VerdictClass.NO_MATCH
    assert v.candidates[0].ruled_out == "layer count mismatch"


def test_judge_no_candidates():
    v = judge([])
    assert v.verdict == VerdictClass.INSUFFICIENT


def test_verdict_serializable():
    v = judge([(fake_evidence(pcs_cos=0.99), "llama")])
    d = v.to_dict()
    assert d["verdict"] == "FINE_TUNE"
    assert d["candidates"][0]["evidence"]["pcs_cos_mean"] == 0.99


# -- end-to-end on synthetic models -------------------------------------------


@pytest.fixture(scope="module")
def model_zoo(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zoo")
    base_w = make_tiny_llama(tmp / "base", n_layers=8, hidden=64, n_heads=8,
                             n_kv_heads=4, intermediate=128, seed=0)
    make_tiny_llama(tmp / "ft", n_layers=8, hidden=64, n_heads=8,
                    n_kv_heads=4, intermediate=128, seed=50,
                    base_weights=base_w, noise=0.002)
    make_tiny_llama(tmp / "heavy", n_layers=8, hidden=64, n_heads=8,
                    n_kv_heads=4, intermediate=128, seed=51,
                    base_weights=base_w, noise=0.15)
    make_tiny_llama(tmp / "other", n_layers=8, hidden=64, n_heads=8,
                    n_kv_heads=4, intermediate=128, seed=9)
    fps = {
        name: extract_fingerprint(LocalSource(tmp / name), model_id=name)
        for name in ("base", "ft", "heavy", "other")
    }
    return fps


def test_e2e_fine_tune(model_zoo):
    ev = compare_fingerprints(model_zoo["ft"], model_zoo["base"])
    v = judge([(ev, "base-family")])
    assert v.verdict == VerdictClass.FINE_TUNE
    assert v.probability > 0.95


def test_e2e_heavy_continuation(model_zoo):
    ev = compare_fingerprints(model_zoo["heavy"], model_zoo["base"])
    v = judge([(ev, "base-family")])
    assert v.verdict == VerdictClass.SAME_LINEAGE
    assert v.probability > 0.9


def test_e2e_independent_same_arch(model_zoo):
    """The defamation-risk surface: independent same-shape model must not match."""
    ev = compare_fingerprints(model_zoo["other"], model_zoo["base"])
    v = judge([(ev, "base-family")])
    assert v.verdict == VerdictClass.NO_MATCH
    assert v.probability < 0.5


def test_e2e_self_is_exact_copy(model_zoo):
    ev = compare_fingerprints(model_zoo["base"], model_zoo["base"])
    v = judge([(ev, "base-family")])
    assert v.verdict == VerdictClass.EXACT_COPY


def test_e2e_ranking(model_zoo):
    evs = [
        (compare_fingerprints(model_zoo["ft"], model_zoo["base"]), "base-family"),
        (compare_fingerprints(model_zoo["ft"], model_zoo["other"]), "other-family"),
    ]
    v = judge(evs)
    assert v.best.candidate_id == "base"
    assert v.candidates[0].probability > v.candidates[1].probability
