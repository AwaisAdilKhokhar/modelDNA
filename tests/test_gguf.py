"""GGUF dequantize-and-sample: quantized copies must compare, not abstain.

The invariant under test: a GGUF file is sampled at the *same logical
element positions* as the safetensors checkpoint it was converted from —
same canonical roles, same LCG offsets, converter row permutation undone —
so the only difference between the two fingerprints is quantization noise.
Files are written with the official ``gguf`` package (an implementation
independent of our parser).
"""

import numpy as np
import pytest

from conftest import make_tiny_llama
from modeldna.compare import compare_fingerprints
from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.gguf import GGUFWeightIndex, discover_gguf_weights, parse_gguf
from modeldna.io.source import LocalSource
from modeldna.io.weights import WeightIndex, _lcg_offsets, open_weight_index
from modeldna.scan import scan
from modeldna.testing import write_gguf_llama
from modeldna.verdict import VerdictClass

DIMS = dict(n_layers=4, hidden=64, n_heads=8, n_kv_heads=4, intermediate=128, vocab=256)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """One model in both formats: safetensors original, Q8_0 GGUF conversion."""
    tmp = tmp_path_factory.mktemp("pair")
    tensors = make_tiny_llama(tmp / "st", seed=11, **DIMS)
    write_gguf_llama(
        tmp / "gg", tensors, n_heads=DIMS["n_heads"], n_kv_heads=DIMS["n_kv_heads"],
        vocab=DIMS["vocab"], quant="Q8_0",
    )
    return tmp, tensors


# -- parsing -------------------------------------------------------------------


def test_parse_header(pair):
    tmp, tensors = pair
    src = LocalSource(tmp / "gg")
    g = parse_gguf(src, discover_gguf_weights(src.list_files())[0])
    assert g.meta["general.architecture"] == "llama"
    assert g.meta["llama.block_count"] == DIMS["n_layers"]
    by_name = {t.name: t for t in g.tensors}
    # shapes come back in numpy/HF order, not ggml's reversed ne
    assert by_name["token_embd.weight"].shape == (DIMS["vocab"], DIMS["hidden"])
    assert by_name["blk.0.ffn_down.weight"].shape == (DIMS["hidden"], DIMS["intermediate"])
    assert by_name["blk.0.attn_q.weight"].dtype == "Q8_0"
    assert by_name["blk.0.attn_norm.weight"].dtype == "F32"


def test_signature_from_gguf_metadata(pair):
    tmp, _ = pair
    fp = extract_fingerprint(LocalSource(tmp / "gg"), model_id="gg")
    sig = fp.arch
    assert sig.model_type == "llama"
    assert sig.core_shape() == (
        DIMS["hidden"], DIMS["n_layers"], DIMS["n_heads"], DIMS["n_kv_heads"],
        DIMS["intermediate"],
    )
    assert sig.torch_dtype == "q8_0"
    assert sig.weights_format.startswith("gguf:")
    assert sig.canonical_coverage == 1.0


# -- reads ---------------------------------------------------------------------


def test_read_tensor_unpermutes_qk(pair):
    """The converter's llama q/k row rewrite must be undone on read."""
    tmp, tensors = pair
    idx = GGUFWeightIndex(LocalSource(tmp / "gg"))
    for hf, gg in [
        ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("model.layers.2.self_attn.k_proj.weight", "blk.2.attn_k.weight"),
        ("model.layers.1.mlp.down_proj.weight", "blk.1.ffn_down.weight"),
        ("model.layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"),
    ]:
        got = idx.read_tensor(gg)
        want = tensors[hf]
        assert got.shape == want.shape
        # Q8_0 noise only — row order must match exactly, so per-row cosine
        # against the *unpermuted* original is ~1 for every row
        cos = (got * want).sum(axis=-1) / (
            np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1)
        )
        assert cos.min() > 0.999


@pytest.mark.parametrize("quant", ["F16", "Q8_0", "Q4_0", "Q5_1"])
def test_sampled_reads_match_full_tensor(tmp_path, quant):
    """Block-aligned partial reads must equal slices of the full dequant."""
    rng = np.random.default_rng(3)
    tensors = {
        "model.norm.weight": (1 + rng.normal(0, 0.3, 64)).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight":
            rng.normal(0, 0.02, (1024, 768)).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight":
            rng.normal(0, 0.02, (256, 768)).astype(np.float32),
        "model.layers.0.mlp.up_proj.weight":
            rng.normal(0, 0.02, (2048, 768)).astype(np.float32),
    }
    write_gguf_llama(tmp_path, tensors, n_heads=8, n_kv_heads=2, quant=quant)
    idx = GGUFWeightIndex(LocalSource(tmp_path))
    for name in ("blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.ffn_up.weight"):
        full = idx.read_tensor(name).reshape(-1)
        np.testing.assert_array_equal(
            idx.read_flat_slice(name, 1000, 5000), full[1000:6000]
        )
        got = idx.read_sample(name, seed=123, n_blocks=8, block_len=1000)
        offs = _lcg_offsets(idx.info(name).numel, 8, 1000, 123)
        np.testing.assert_array_equal(
            got, np.concatenate([full[o : o + 1000] for o in offs])
        )


def test_kquant_block_mapping(tmp_path):
    """K-quants (256-element superblocks) go through the same range math.

    gguf-py has no Q4_K/Q6_K quantizer, so the blocks are random bytes —
    the point is that partial reads agree with the full dequantization,
    whatever the values decode to.
    """
    from gguf import GGUFWriter
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    rng = np.random.default_rng(9)
    w = GGUFWriter(str(tmp_path / "k.gguf"), arch="llama")
    w.add_block_count(1)
    for name, qt in [("blk.0.ffn_up.weight", GGMLQuantizationType.Q4_K),
                     ("blk.0.ffn_down.weight", GGMLQuantizationType.Q6_K)]:
        _, type_size = GGML_QUANT_SIZES[qt]
        raw = rng.integers(0, 256, size=(64, 2 * type_size), dtype=np.uint8)
        w.add_tensor(name, raw, raw_dtype=qt)  # 64 rows x 512 elements
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    idx = GGUFWeightIndex(LocalSource(tmp_path))
    for name in ("blk.0.ffn_up.weight", "blk.0.ffn_down.weight"):
        assert idx.info(name).shape == (64, 512)
        full = idx.read_tensor(name).reshape(-1)
        got = idx.read_sample(name, seed=5, n_blocks=6, block_len=300)
        offs = _lcg_offsets(full.size, 6, 300, 5)
        want = np.concatenate([full[o : o + 300] for o in offs])
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(idx.read_flat_slice(name, 100, 700), full[100:800])


# -- fingerprint parity across formats -------------------------------------------


def test_f32_gguf_fingerprint_identical(tmp_path):
    tensors = make_tiny_llama(tmp_path / "st", seed=21, **DIMS)
    write_gguf_llama(tmp_path / "gg", tensors, n_heads=DIMS["n_heads"],
                     n_kv_heads=DIMS["n_kv_heads"], vocab=DIMS["vocab"], quant="F32")
    fp_st = extract_fingerprint(LocalSource(tmp_path / "st"), model_id="m")
    fp_gg = extract_fingerprint(LocalSource(tmp_path / "gg"), model_id="m")
    ev = compare_fingerprints(fp_gg, fp_st)
    assert ev.shape_compatible and ev.layers_match
    assert ev.pcs_cos_mean == pytest.approx(1.0, abs=1e-6)
    assert ev.sigma_r_mean == pytest.approx(1.0, abs=1e-6)
    assert ev.spectra_r_mean == pytest.approx(1.0, abs=1e-4)
    assert ev.vector_cos_mean == pytest.approx(1.0, abs=1e-6)


def test_quantized_fingerprint_close(pair):
    tmp, _ = pair
    fp_st = extract_fingerprint(LocalSource(tmp / "st"), model_id="st")
    fp_gg = extract_fingerprint(LocalSource(tmp / "gg"), model_id="gg")
    ev = compare_fingerprints(fp_gg, fp_st)
    assert ev.pcs_cos_mean > 0.999  # Q8_0 is nearly lossless
    assert ev.sigma_r_mean > 0.999
    assert ev.vector_cos_mean > 0.999  # norms are stored unquantized


def test_gguf_prefetch_covers_extraction(pair):
    """Fast-mode GGUF extraction over a 'remote' source reads via the plan only."""
    from test_prefetch import CountingSource

    from modeldna.fingerprint.extract import FingerprintExtractor

    tmp, _ = pair
    src = CountingSource(tmp / "gg")
    ex = FingerprintExtractor(src, mode="fast")  # header/config reads happen here
    src.direct_reads = 0
    lazy = extract_fingerprint(LocalSource(tmp / "gg"), model_id="gg").to_dict()
    fast = ex.run().to_dict()
    assert src.batch_calls >= 1
    assert src.direct_reads == 0
    for volatile in ("created_at", "bytes_read", "model_id"):
        lazy.pop(volatile), fast.pop(volatile)
    assert fast == lazy


# -- scan integration --------------------------------------------------------------


def test_scan_gguf_quant_of_indexed_base(pair, tmp_path):
    """The headline case: a Q8_0 GGUF of an indexed base is a QUANTIZED_COPY."""
    tmp, _ = pair
    db = ReferenceDB(tmp_path / "refdb")
    fp = extract_fingerprint(LocalSource(tmp / "st"), model_id="org/base")
    db.add(fp, family="family-a")
    res = scan(str(tmp / "gg"), db=db)
    assert res.verdict.verdict == VerdictClass.QUANTIZED_COPY
    assert res.verdict.best.candidate_id == "org/base"
    assert res.verdict.probability > 0.9


def test_scan_deep_quant_notes_ambiguity(pair, tmp_path):
    """Q4 noise overlaps the light-SFT band: lineage called, class hedged."""
    tmp, tensors = pair
    write_gguf_llama(tmp_path / "q4", tensors, n_heads=DIMS["n_heads"],
                     n_kv_heads=DIMS["n_kv_heads"], vocab=DIMS["vocab"], quant="Q4_0")
    db = ReferenceDB(tmp_path / "refdb")
    fp = extract_fingerprint(LocalSource(tmp / "st"), model_id="org/base")
    db.add(fp, family="family-a")
    res = scan(str(tmp_path / "q4"), db=db)
    assert res.verdict.verdict == VerdictClass.FINE_TUNE
    assert res.verdict.best.candidate_id == "org/base"
    assert any("quantized copy" in n for n in res.verdict.notes)


# -- discovery ----------------------------------------------------------------------


def test_discovery_prefers_highest_fidelity():
    files = [
        "m-Q2_K.gguf", "m-Q4_K_M.gguf", "m-Q6_K.gguf", "mmproj-f16.gguf",
        "readme.md",
    ]
    assert discover_gguf_weights(files) == ["m-Q6_K.gguf"]


def test_discovery_groups_split_files():
    files = [
        "big-Q4_K_M-00002-of-00002.gguf", "big-Q4_K_M-00001-of-00002.gguf",
        "big-Q2_K.gguf",
    ]
    assert discover_gguf_weights(files) == [
        "big-Q4_K_M-00001-of-00002.gguf", "big-Q4_K_M-00002-of-00002.gguf",
    ]


def test_multipart_gguf_index(tmp_path):
    tensors = make_tiny_llama(tmp_path / "st", seed=31, **DIMS)
    write_gguf_llama(tmp_path / "gg", tensors, n_heads=DIMS["n_heads"],
                     n_kv_heads=DIMS["n_kv_heads"], vocab=DIMS["vocab"],
                     quant="F32", parts=3)
    idx = GGUFWeightIndex(LocalSource(tmp_path / "gg"))
    assert len(idx.gguf_files) == 3
    assert len(idx.tensors) == len(tensors)
    got = idx.read_tensor("blk.0.attn_q.weight")
    np.testing.assert_allclose(
        got, tensors["model.layers.0.self_attn.q_proj.weight"], rtol=1e-6
    )


def test_open_weight_index_prefers_safetensors(tmp_path):
    tensors = make_tiny_llama(tmp_path, seed=41, **DIMS)
    write_gguf_llama(tmp_path, tensors, n_heads=DIMS["n_heads"],
                     n_kv_heads=DIMS["n_kv_heads"], vocab=DIMS["vocab"], quant="Q8_0")
    idx = open_weight_index(LocalSource(tmp_path))
    assert isinstance(idx, WeightIndex)  # full precision wins when both exist
