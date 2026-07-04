import numpy as np
import pytest

from conftest import make_tiny_llama
from modeldna.fingerprint.extract import Fingerprint, extract_fingerprint
from modeldna.fingerprint.methods import (
    cosine,
    fnv1a,
    log_spectrum_distance,
    pearson,
    randomized_svals,
    zscore,
)
from modeldna.io.source import LocalSource


# -- numerics ---------------------------------------------------------------


def test_fnv1a_stable():
    # frozen: sampling seeds derive from these values
    assert fnv1a(20260704, "attn.q", 0) == fnv1a(20260704, "attn.q", 0)
    assert fnv1a("a") != fnv1a("b")
    assert fnv1a(1, "x") != fnv1a("1x")


def test_pearson_basics():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, a) == pytest.approx(1.0)
    assert pearson(a, -a) == pytest.approx(-1.0)
    assert pearson(a, np.ones(4)) == 0.0  # degenerate side
    assert pearson(a, np.array([1.0, 2.0])) == 0.0  # length mismatch


def test_cosine_basics():
    a = np.array([1.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(a, np.zeros(2)) == 0.0


def test_zscore_degenerate():
    assert np.all(zscore(np.ones(5)) == 0)


def test_randomized_svals_close_to_exact():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(300, 200)).astype(np.float32)
    exact = np.linalg.svd(mat, compute_uv=False)[:16]
    approx = randomized_svals(mat, 16, seed=1)
    np.testing.assert_allclose(approx, exact, rtol=0.05)


def test_log_spectrum_distance():
    s = np.array([10.0, 5.0, 1.0])
    assert log_spectrum_distance(s, s) == 0.0
    assert log_spectrum_distance(s, s * 2) > 0


# -- extraction ---------------------------------------------------------------


def test_extract_basic(tiny_model):
    root, _ = tiny_model
    fp = extract_fingerprint(LocalSource(root), mode="fast")
    assert fp.arch.n_layers == 4
    assert set(fp.sigma_curves) >= {"attn.q", "attn.k", "attn.v", "attn.o"}
    assert len(fp.sigma_curves["attn.q"]) == 4
    assert "attn.q" in fp.pcs_samples
    assert "norm.in" in fp.vector_norms
    assert "attn.q" in fp.spectra_sketch
    assert fp.bytes_read > 0


def test_fast_full_sample_identity(tiny_model):
    """PCS sample positions must be identical across modes."""
    root, _ = tiny_model
    fast = extract_fingerprint(LocalSource(root), mode="fast")
    full = extract_fingerprint(LocalSource(root), mode="full")
    np.testing.assert_allclose(
        fast.pcs_samples["attn.q"], full.pcs_samples["attn.q"], rtol=1e-6
    )
    # full mode also carries exact spectra
    assert full.spectra_exact and not fast.spectra_exact


def test_finetune_vs_unrelated(tmp_path):
    base_dir = tmp_path / "base"
    base_weights = make_tiny_llama(base_dir, seed=0)
    ft_dir = tmp_path / "ft"
    make_tiny_llama(ft_dir, seed=99, base_weights=base_weights, noise=0.001)
    other_dir = tmp_path / "other"
    make_tiny_llama(other_dir, seed=7)

    fp_base = extract_fingerprint(LocalSource(base_dir))
    fp_ft = extract_fingerprint(LocalSource(ft_dir))
    fp_other = extract_fingerprint(LocalSource(other_dir))

    # PCS cosine: fine-tune ~1, independent ~0
    cos_ft = cosine(np.array(fp_base.pcs_samples["attn.q"]), np.array(fp_ft.pcs_samples["attn.q"]))
    cos_other = cosine(
        np.array(fp_base.pcs_samples["attn.q"]), np.array(fp_other.pcs_samples["attn.q"])
    )
    assert cos_ft > 0.99
    assert abs(cos_other) < 0.2

    # sigma curves: fine-tune correlates, independent much less
    r_ft = pearson(
        zscore(np.array(fp_base.sigma_curves["attn.q"])),
        zscore(np.array(fp_ft.sigma_curves["attn.q"])),
    )
    assert r_ft > 0.99


def test_fingerprint_roundtrip(tiny_model, tmp_path):
    root, _ = tiny_model
    fp = extract_fingerprint(LocalSource(root))
    for suffix in (".json", ".json.gz"):
        p = tmp_path / f"fp{suffix}"
        fp.save(p)
        fp2 = Fingerprint.load(p)
        assert fp2.model_id == fp.model_id
        assert fp2.arch == fp.arch
        np.testing.assert_allclose(fp2.pcs_samples["attn.q"], fp.pcs_samples["attn.q"])


def test_bad_mode(tiny_model):
    root, _ = tiny_model
    with pytest.raises(ValueError):
        extract_fingerprint(LocalSource(root), mode="turbo")
