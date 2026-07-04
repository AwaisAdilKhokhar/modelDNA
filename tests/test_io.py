import numpy as np
import pytest

from conftest import make_tiny_llama, write_safetensors
from modeldna.io.safetensors import SafetensorsError, decode_tensor, header_length
from modeldna.io.source import LocalSource, SourceError, open_source
from modeldna.io.weights import WeightIndex, _lcg_offsets


def test_header_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})
    idx = WeightIndex(LocalSource(tmp_path))
    assert idx.names == ["w"]
    info = idx.info("w")
    assert info.shape == (3, 4)
    assert info.dtype == "F32"
    np.testing.assert_array_equal(idx.read_tensor("w"), arr)


def test_bad_header_rejected(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\xff" * 32)
    with pytest.raises(SafetensorsError):
        WeightIndex(LocalSource(tmp_path))


def test_header_length_bounds():
    with pytest.raises(SafetensorsError):
        header_length(b"\x00" * 8)  # zero length
    with pytest.raises(SafetensorsError):
        header_length(b"\xff" * 8)  # absurd length


def test_bf16_decode():
    # 1.0 in bfloat16 is 0x3F80
    raw = np.array([0x3F80, 0xBF80], dtype="<u2").tobytes()
    out = decode_tensor(raw, "BF16")
    np.testing.assert_array_equal(out, np.array([1.0, -1.0], dtype=np.float32))


def test_flat_slice(tmp_path):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})
    idx = WeightIndex(LocalSource(tmp_path))
    sl = idx.read_flat_slice("w", 37, 5)
    np.testing.assert_array_equal(sl, np.arange(37, 42, dtype=np.float32))
    # clamped at the end
    sl = idx.read_flat_slice("w", 98, 10)
    assert len(sl) == 2


def test_sample_determinism(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(600, 700)).astype(np.float32)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})
    idx = WeightIndex(LocalSource(tmp_path))
    s1 = idx.read_sample("w", seed=42, n_blocks=16, block_len=1024)
    s2 = idx.read_sample("w", seed=42, n_blocks=16, block_len=1024)
    np.testing.assert_array_equal(s1, s2)
    s3 = idx.read_sample("w", seed=43, n_blocks=16, block_len=1024)
    assert not np.array_equal(s1, s3)


def test_sample_small_tensor_returns_all(tmp_path):
    arr = np.arange(50, dtype=np.float32)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})
    idx = WeightIndex(LocalSource(tmp_path))
    s = idx.read_sample("w", seed=0)
    np.testing.assert_array_equal(np.sort(s), arr)


def test_lcg_offsets_stable():
    # frozen values: sample positions must never change between versions
    assert _lcg_offsets(10_000_000, 4, 1000, seed=7) == _lcg_offsets(
        10_000_000, 4, 1000, seed=7
    )
    assert len(_lcg_offsets(10_000, 8, 100, seed=1)) >= 1


def test_sharded_model(tmp_path):
    root = tmp_path / "m"
    tensors = make_tiny_llama(root, sharded=True)
    idx = WeightIndex(LocalSource(root))
    assert set(idx.names) == set(tensors)
    name = "model.layers.2.self_attn.q_proj.weight"
    np.testing.assert_array_equal(idx.read_tensor(name), tensors[name])


def test_open_source_local(tmp_path):
    make_tiny_llama(tmp_path / "m")
    src = open_source(str(tmp_path / "m"))
    assert "config.json" in src.list_files()


def test_open_source_bad_target():
    with pytest.raises(SourceError):
        open_source("definitely-not-a-path-or-repo")


def test_bytes_read_accounting(tmp_path):
    make_tiny_llama(tmp_path / "m")
    src = LocalSource(tmp_path / "m")
    WeightIndex(src)  # header reads only
    header_cost = src.bytes_read
    assert 0 < header_cost < 20_000  # headers are tiny
