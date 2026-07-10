"""GGUF support: dequantize-and-sample fingerprinting for quantized models.

GGUF (llama.cpp / ollama / LM Studio) is a huge fraction of what people
actually download, and its tensors are block-quantized: every run of 32 or
256 elements packs scales and quantized values into a fixed number of
bytes. Fixed-size blocks mean an element range maps to a byte range exactly
like it does for safetensors — fetch the covering blocks, dequantize, trim.
So the same seeded fingerprint positions can be sampled from a Q4_K_M file
over HTTP range requests, and a quantized copy compares against its fp16
parent instead of abstaining.

We parse the container ourselves (the header is kilobytes to a few MB, most
of it the embedded tokenizer) and delegate the per-type bit-twiddling to
the official ``gguf`` package's numpy dequantizers, which track llama.cpp.

One conversion quirk matters for sampling parity: ``convert_hf_to_gguf.py``
permutes attention q/k rows for llama-family models (HF interleaved RoPE ->
ggml layout). The index un-permutes on read, so element positions line up
with safetensors fingerprints of the same weights.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from modeldna.io.source import ModelSource
from modeldna.io.weights import (
    BaseWeightIndex,
    WeightIndexError,
    _lcg_offsets,
)

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types -> struct format (scalars only)
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}
_T_STRING = 8
_T_ARRAY = 9

#: arrays longer than this are consumed but not kept (tokenizer vocab etc.)
_KEEP_ARRAY_MAX = 4096


class GGUFError(WeightIndexError):
    """Raised when a file does not parse as valid GGUF."""


def _ggml_types():
    """(GGMLQuantizationType, GGML_QUANT_SIZES) from the gguf package."""
    try:
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise GGUFError(
            "reading GGUF files requires the 'gguf' package (pip install gguf)"
        ) from e
    return GGMLQuantizationType, GGML_QUANT_SIZES


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Location, type, and logical layout of one tensor inside a GGUF file."""

    name: str
    dtype: str  # ggml type name ("F16", "Q4_K", ...)
    shape: tuple[int, ...]  # numpy/HF order (rows, cols) — ne reversed
    start: int  # absolute byte offset within the file
    end: int
    shard: str  # gguf filename
    ggml_type: int
    block_elems: int  # elements per quantization block
    block_bytes: int  # stored bytes per block
    #: >0: rows were permuted by convert_hf_to_gguf's llama q/k rewrite,
    #: using this head count; reads un-permute to the HF row order
    perm_heads: int = 0

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class _Cursor:
    """Sequential reader over one file, fetched in coarse range requests."""

    CHUNK = 1 << 20

    def __init__(self, source: ModelSource, filename: str):
        self.source = source
        self.filename = filename
        self.size = source.size(filename)
        self.buf = b""
        self.base = 0  # absolute offset of buf[0]
        self.pos = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > self.size:
            raise GGUFError(f"truncated GGUF header in {self.filename}")
        if end > self.base + len(self.buf):
            fetch_end = min(self.size, max(end, self.pos + self.CHUNK))
            self.buf = self.source.read_range(self.filename, self.pos, fetch_end)
            self.base = self.pos
        off = self.pos - self.base
        self.pos = end
        return self.buf[off : off + n]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", errors="replace")


@dataclass
class GGUFFile:
    """Parsed header of one GGUF file (metadata + tensor catalog)."""

    filename: str
    version: int
    meta: dict[str, Any]
    array_lens: dict[str, int]  # lengths of arrays too big to keep
    tensors: list[GGUFTensorInfo]
    tokenizer_hash: str = ""


def _read_value(cur: _Cursor, vtype: int, key: str, out: GGUFFile) -> Any:
    if vtype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[vtype]
        v = struct.unpack(fmt, cur.take(struct.calcsize(fmt)))[0]
        return bool(v) if vtype == 7 else v
    if vtype == _T_STRING:
        return cur.string()
    if vtype != _T_ARRAY:
        raise GGUFError(f"unknown GGUF metadata type {vtype} for key {key!r}")

    etype = cur.u32()
    count = cur.u64()
    keep = count <= _KEEP_ARRAY_MAX
    # hash the token vocabulary while consuming it — it doubles as the
    # tokenizer identity when the repo ships no tokenizer.json
    hasher = hashlib.sha256() if key == "tokenizer.ggml.tokens" else None
    values: list[Any] = []
    if etype == _T_STRING:
        for _ in range(count):
            b = cur.take(cur.u64())
            if hasher is not None:
                hasher.update(struct.pack("<Q", len(b)))
                hasher.update(b)
            if keep:
                values.append(b.decode("utf-8", errors="replace"))
    elif etype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[etype]
        width = struct.calcsize(fmt)
        raw = cur.take(width * count)
        if hasher is not None:
            hasher.update(raw)
        if keep:
            values = list(np.frombuffer(raw, dtype=fmt).tolist())
    else:
        raise GGUFError(f"unsupported GGUF array element type {etype} for key {key!r}")
    if hasher is not None:
        out.tokenizer_hash = hasher.hexdigest()
    if not keep:
        out.array_lens[key] = count
        return None
    return values


def parse_gguf(source: ModelSource, filename: str) -> GGUFFile:
    """Parse header, metadata KVs, and tensor catalog of one GGUF file."""
    QT, sizes = _ggml_types()
    cur = _Cursor(source, filename)
    if cur.take(4) != GGUF_MAGIC:
        raise GGUFError(f"{filename} is not a GGUF file (bad magic)")
    version = cur.u32()
    if version not in (2, 3):
        raise GGUFError(f"unsupported GGUF version {version} in {filename}")
    n_tensors = cur.u64()
    n_kv = cur.u64()

    out = GGUFFile(filename=filename, version=version, meta={}, array_lens={}, tensors=[])
    for _ in range(n_kv):
        key = cur.string()
        vtype = cur.u32()
        v = _read_value(cur, vtype, key, out)
        if v is not None:
            out.meta[key] = v

    alignment = int(out.meta.get("general.alignment", DEFAULT_ALIGNMENT))
    raw_infos: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.u32()
        ne = tuple(cur.u64() for _ in range(n_dims))
        ggml_type = cur.u32()
        offset = cur.u64()
        raw_infos.append((name, ne, ggml_type, offset))

    data_start = -(-cur.pos // alignment) * alignment
    for name, ne, ggml_type, offset in raw_infos:
        try:
            qt = QT(ggml_type)
            block_elems, block_bytes = sizes[qt]
        except (ValueError, KeyError):
            raise GGUFError(
                f"unsupported ggml tensor type {ggml_type} for {name!r} in {filename}"
            ) from None
        numel = 1
        for d in ne:
            numel *= d
        if numel % block_elems:
            raise GGUFError(f"tensor {name!r} size not a multiple of its block size")
        start = data_start + offset
        out.tensors.append(
            GGUFTensorInfo(
                name=name,
                dtype=qt.name,
                shape=tuple(reversed(ne)),  # ne is fastest-first; numpy is row-major
                start=start,
                end=start + numel // block_elems * block_bytes,
                shard=filename,
                ggml_type=ggml_type,
                block_elems=block_elems,
                block_bytes=block_bytes,
            )
        )
    return out


# -- quant-file selection --------------------------------------------------

#: quant tags in descending fidelity — the fingerprint wants the least noise
_QUANT_RANK = [
    "f32", "bf16", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_1", "q5_0",
    "q4_k_m", "q4_k_s", "iq4_xs", "iq4_nl", "q4_1", "q4_0", "q3_k_l", "q3_k_m",
    "q3_k_s", "iq3_m", "iq3_s", "iq3_xxs", "q2_k", "iq2_m", "iq2_s", "iq2_xs",
    "iq2_xxs", "iq1_m", "iq1_s",
]

_PART_RX = re.compile(r"^(?P<base>.+)-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def _quant_rank(name: str) -> int:
    low = name.lower()
    for i, tag in enumerate(_QUANT_RANK):
        if tag in low:
            return i
    return len(_QUANT_RANK)


def discover_gguf_weights(files: list[str]) -> list[str]:
    """Pick the highest-fidelity GGUF (all parts, if split) from a file list."""
    groups: dict[str, list[str]] = {}
    for f in files:
        base = PurePosixPath(f).name
        if not base.lower().endswith(".gguf") or base.lower().startswith("mmproj"):
            continue
        m = _PART_RX.match(f)
        groups.setdefault(m.group("base") + ".gguf" if m else f, []).append(f)
    if not groups:
        return []
    best = min(groups, key=lambda g: (_quant_rank(PurePosixPath(g).name), g))
    return sorted(groups[best])


# -- the index ---------------------------------------------------------------


def _decode(raw: bytes, t: GGUFTensorInfo) -> np.ndarray:
    """Whole quantization blocks -> flat float array."""
    if t.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if t.dtype == "F64":
        return np.frombuffer(raw, dtype="<f8")
    if t.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if t.dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2")
        return (u.astype(np.uint32) << 16).view(np.float32)
    if t.dtype in ("I8", "I16", "I32", "I64"):
        return np.frombuffer(raw, dtype=f"<i{t.block_bytes}").astype(np.float32)
    from gguf import quants

    QT, _ = _ggml_types()
    return quants.dequantize(np.frombuffer(raw, dtype=np.uint8), QT(t.ggml_type)).reshape(-1)


def _llama_row_perm(rows: int, n_heads: int) -> np.ndarray:
    """logical HF row -> stored ggml row, for convert_hf_to_gguf's q/k permute.

    The converter runs ``w.reshape(H, 2, hd//2, ...).swapaxes(1, 2).reshape``,
    which interleaves each head's two halves; this is its inverse as an
    index map (logical[l] = stored[perm[l]]).
    """
    hd = rows // n_heads
    half = hd // 2
    logical = np.arange(rows)
    h, r = np.divmod(logical, hd)
    a, b = np.divmod(r, half)
    return h * hd + 2 * b + a


class GGUFWeightIndex(BaseWeightIndex):
    """WeightIndex-compatible view over a GGUF file (or split-file set).

    All reads are block-aligned: an element range is widened to whole
    quantization blocks, fetched, dequantized with llama.cpp's reference
    kernels, and trimmed — so sampled positions match the safetensors path
    element-for-element while values carry only quantization noise.
    """

    def __init__(self, source: ModelSource, files: list[str] | None = None):
        super().__init__(source)
        self.tensors: dict[str, GGUFTensorInfo] = {}
        selected = files or discover_gguf_weights(source.list_files())
        if not selected:
            raise WeightIndexError(f"no GGUF weights found in {source.name}")
        self.gguf_files = selected
        self.weights_format = "gguf:" + PurePosixPath(selected[0]).name
        self.meta: dict[str, Any] = {}
        self.array_lens: dict[str, int] = {}
        self.tokenizer_hash = ""
        for f in selected:
            g = parse_gguf(source, f)
            self.meta.update(g.meta)
            self.array_lens.update(g.array_lens)
            if g.tokenizer_hash:
                self.tokenizer_hash = g.tokenizer_hash
            for t in g.tensors:
                if t.name in self.tensors:
                    raise GGUFError(f"tensor {t.name!r} appears in multiple GGUF parts")
                self.tensors[t.name] = t
        self._mark_permuted()

    def _meta_int(self, key: str, default: int = 0) -> int:
        v = self.meta.get(key, default)
        if isinstance(v, list):  # some archs store per-layer arrays
            v = v[0] if v else default
        return int(v)

    def _mark_permuted(self) -> None:
        """Flag llama-family q/k tensors whose rows the converter permuted."""
        if self.meta.get("general.architecture") != "llama":
            return
        n_heads = self._meta_int("llama.attention.head_count")
        n_kv = self._meta_int("llama.attention.head_count_kv", n_heads)
        for name, t in self.tensors.items():
            m = re.match(r"blk\.\d+\.(attn_q|attn_k)\.", name)
            if not m:
                continue
            heads = n_heads if m.group(1) == "attn_q" else n_kv
            rows = t.shape[0]
            if heads and rows % heads == 0 and (rows // heads) % 2 == 0:
                self.tensors[name] = replace(t, perm_heads=heads)

    # -- element-range plumbing ------------------------------------------------

    def _stored_segments(self, t: GGUFTensorInfo, e0: int, e1: int) -> list[tuple[int, int]]:
        """Map a logical flat element range to stored flat element ranges."""
        if not t.perm_heads:
            return [(e0, e1)]
        cols = t.shape[1] if len(t.shape) > 1 else 1
        perm = _llama_row_perm(t.shape[0], t.perm_heads)
        out: list[tuple[int, int]] = []
        while e0 < e1:
            row, col = divmod(e0, cols)
            n = min(e1 - e0, cols - col)
            s0 = int(perm[row]) * cols + col
            out.append((s0, s0 + n))
            e0 += n
        return out

    def _block_span(self, t: GGUFTensorInfo, s0: int, s1: int) -> tuple[int, int, int]:
        """(byte start, byte end, first covered element) of the covering blocks."""
        b0 = s0 // t.block_elems
        b1 = -(-s1 // t.block_elems)
        return (t.start + b0 * t.block_bytes, t.start + b1 * t.block_bytes,
                b0 * t.block_elems)

    def _read_stored(self, t: GGUFTensorInfo, s0: int, s1: int) -> np.ndarray:
        bs, be, first = self._block_span(t, s0, s1)
        flat = _decode(self._read(t.shard, bs, be), t)
        return flat[s0 - first : s1 - first]

    # -- reads -------------------------------------------------------------------

    def read_tensor(self, name: str) -> np.ndarray:
        t = self.info(name)
        arr = _decode(self._read(t.shard, t.start, t.end), t).reshape(t.shape)
        if t.perm_heads:
            arr = arr[_llama_row_perm(t.shape[0], t.perm_heads)]
        return arr

    def read_flat_slice(self, name: str, start_elem: int, n_elem: int) -> np.ndarray:
        """Read a contiguous run of elements from the flattened logical tensor."""
        t = self.info(name)
        start_elem = max(0, min(start_elem, t.numel))
        n_elem = min(n_elem, t.numel - start_elem)
        segs = self._stored_segments(t, start_elem, start_elem + n_elem)
        parts = [self._read_stored(t, s0, s1) for s0, s1 in segs]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def plan_flat_slice(self, name: str, start_elem: int, n_elem: int) -> list[tuple[int, int]]:
        """Byte ranges read_flat_slice() with the same arguments will need."""
        t = self.info(name)
        start_elem = max(0, min(start_elem, t.numel))
        n_elem = min(n_elem, t.numel - start_elem)
        return [
            self._block_span(t, s0, s1)[:2]
            for s0, s1 in self._stored_segments(t, start_elem, start_elem + n_elem)
        ]

    def plan_sample(
        self, name: str, seed: int, n_blocks: int = 64, block_len: int = 4096
    ) -> list[tuple[int, int]]:
        """Byte ranges read_sample() with the same arguments will need."""
        t = self.info(name)
        if t.numel <= n_blocks * block_len:
            return [(t.start, t.end)]
        ranges: list[tuple[int, int]] = []
        for o in _lcg_offsets(t.numel, n_blocks, block_len, seed):
            ranges.extend(self.plan_flat_slice(name, o, block_len))
        return ranges

    def read_sample(
        self, name: str, seed: int, n_blocks: int = 64, block_len: int = 4096
    ) -> np.ndarray:
        """Dequantized sample at the same element positions as safetensors.

        Positions depend only on (numel, seed, n_blocks, block_len) — the
        very same LCG as the safetensors index — so a GGUF quant and its
        fp16 original are sampled at identical logical elements.
        """
        t = self.info(name)
        if t.numel <= n_blocks * block_len:
            return self.read_tensor(name).reshape(-1)
        offsets = _lcg_offsets(t.numel, n_blocks, block_len, seed)
        return np.concatenate(
            [self.read_flat_slice(name, o, block_len) for o in offsets]
        )

    # -- arch metadata ------------------------------------------------------------

    def quant_name(self) -> str:
        """Human-readable quantization label ("q4_k_m", "f16", ...)."""
        ftype = self.meta.get("general.file_type")
        if ftype is not None:
            try:
                from gguf.constants import LlamaFileType

                name = LlamaFileType(int(ftype)).name
                return name.removeprefix("MOSTLY_").removeprefix("ALL_").lower()
            except (ValueError, ImportError):
                pass
        counts: dict[str, int] = {}
        for t in self.tensors.values():
            if len(t.shape) == 2:
                counts[t.dtype] = counts.get(t.dtype, 0) + 1
        return max(counts, key=counts.get).lower() if counts else "gguf"

    def hf_config(self) -> dict[str, Any]:
        """config.json-shaped dict synthesized from GGUF metadata."""
        arch = self.meta.get("general.architecture", "")

        def g(key: str, default: Any = None) -> Any:
            return self.meta.get(f"{arch}.{key}", default)

        heads = self._meta_int(f"{arch}.attention.head_count")
        vocab = self._meta_int(f"{arch}.vocab_size") or self.array_lens.get(
            "tokenizer.ggml.tokens", 0
        )
        if not vocab and "token_embd.weight" in self.tensors:
            vocab = self.tensors["token_embd.weight"].shape[0]
        return {
            "model_type": arch,
            "hidden_size": self._meta_int(f"{arch}.embedding_length"),
            "num_hidden_layers": self._meta_int(f"{arch}.block_count"),
            "num_attention_heads": heads,
            "num_key_value_heads": self._meta_int(f"{arch}.attention.head_count_kv", heads),
            "intermediate_size": self._meta_int(f"{arch}.feed_forward_length"),
            "vocab_size": vocab,
            "rope_theta": g("rope.freq_base"),
            "rms_norm_eps": g(
                "attention.layer_norm_rms_epsilon", g("attention.layer_norm_epsilon")
            ),
            "torch_dtype": self.quant_name(),
        }
