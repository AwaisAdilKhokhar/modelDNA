"""Minimal safetensors format support.

We parse the format ourselves instead of depending on the `safetensors`
package because the whole point of modeldna is *partial* reads: the header
(a JSON blob at the start of the file) tells us the byte range of every
tensor, and from there we can pull individual tensors — or slices of them —
over HTTP range requests without downloading the shard.

Format reference: an 8-byte little-endian u64 header length N, followed by
N bytes of JSON mapping tensor names to {dtype, shape, data_offsets}, then
the raw tensor data region.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

HEADER_SIZE_BYTES = 8
# Sanity cap: a header bigger than 100 MB means a corrupt or malicious file.
MAX_HEADER_LEN = 100_000_000

# safetensors dtype string -> (numpy dtype used for raw decode, bytes per element)
_DTYPES: dict[str, tuple[str, int]] = {
    "F64": ("<f8", 8),
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),  # decoded manually, numpy has no native bfloat16
    "I64": ("<i8", 8),
    "I32": ("<i4", 4),
    "I16": ("<i2", 2),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "BOOL": ("u1", 1),
    "F8_E4M3": ("u1", 1),
    "F8_E5M2": ("u1", 1),
}

FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}


class SafetensorsError(ValueError):
    """Raised when a file does not parse as valid safetensors."""


@dataclass(frozen=True)
class TensorInfo:
    """Location and type of one tensor inside one shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset within the shard file
    end: int
    shard: str  # shard filename

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        return _DTYPES[self.dtype][1]


def header_length(first8: bytes) -> int:
    """Decode the 8-byte length prefix."""
    if len(first8) < HEADER_SIZE_BYTES:
        raise SafetensorsError("file too short to be safetensors")
    (n,) = struct.unpack("<Q", first8[:HEADER_SIZE_BYTES])
    if n == 0 or n > MAX_HEADER_LEN:
        raise SafetensorsError(f"implausible safetensors header length: {n}")
    return n


def parse_header(header_bytes: bytes, shard: str) -> dict[str, TensorInfo]:
    """Parse the JSON header into TensorInfo entries with absolute offsets."""
    try:
        raw = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafetensorsError(f"bad safetensors header in {shard}: {e}") from e

    data_start = HEADER_SIZE_BYTES + len(header_bytes)
    tensors: dict[str, TensorInfo] = {}
    for name, entry in raw.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in _DTYPES:
            raise SafetensorsError(f"unsupported dtype {dtype!r} for tensor {name!r}")
        begin, end = entry["data_offsets"]
        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(entry["shape"]),
            start=data_start + begin,
            end=data_start + end,
            shard=shard,
        )
    return tensors


def decode_tensor(raw: bytes, dtype: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Decode raw tensor bytes to a numpy array.

    Float types come back as float32 (float64 stays float64) so downstream
    statistics never have to care about storage precision. BF16 is widened
    by placing the stored 16 bits into the top half of a float32.
    """
    np_dtype, _ = _DTYPES[dtype]
    arr = np.frombuffer(raw, dtype=np_dtype)
    if dtype == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        arr = arr.astype(np.float32)
    elif dtype == "BOOL":
        arr = arr.astype(bool)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr
