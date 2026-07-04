"""Tensor-level view over a model's safetensors shards.

Builds an index of every tensor (name, dtype, shape, byte range, shard) from
the shard headers alone — kilobytes of traffic — and then serves full
tensors, contiguous slices, or seeded sample blocks on demand.
"""

from __future__ import annotations

import numpy as np

from modeldna.io.safetensors import (
    HEADER_SIZE_BYTES,
    SafetensorsError,
    TensorInfo,
    decode_tensor,
    header_length,
    parse_header,
)
from modeldna.io.source import ModelSource

# Gaps smaller than this get merged into one range request: the extra bytes
# are cheaper than another HTTP round trip.
COALESCE_GAP = 65_536


class WeightIndexError(RuntimeError):
    pass


def _lcg_offsets(numel: int, n_blocks: int, block_len: int, seed: int) -> list[int]:
    """Deterministic pseudo-random block offsets.

    Hand-rolled LCG instead of numpy's Generator so the sample positions are
    reproducible forever, independent of numpy version. The reference DB
    stores samples taken at these exact positions, so drift here would
    silently break every comparison.
    """
    span = numel - block_len
    if span <= 0:
        return [0]
    mask = (1 << 64) - 1
    state = (seed ^ 0x9E3779B97F4A7C15) & mask
    offsets: set[int] = set()
    for _ in range(n_blocks):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        offsets.add((state >> 11) % span)
    return sorted(offsets)


class WeightIndex:
    """Index of all tensors across a model's safetensors shards."""

    def __init__(self, source: ModelSource):
        self.source = source
        self.tensors: dict[str, TensorInfo] = {}
        self._load()

    def _load(self) -> None:
        files = self.source.list_files()
        shards = self._discover_shards(files)
        if not shards:
            raise WeightIndexError(
                f"no safetensors weights found in {self.source.name}"
            )
        for shard in shards:
            first8 = self.source.read_range(shard, 0, HEADER_SIZE_BYTES)
            hlen = header_length(first8)
            header = self.source.read_range(shard, HEADER_SIZE_BYTES, HEADER_SIZE_BYTES + hlen)
            for name, info in parse_header(header, shard).items():
                if name in self.tensors:
                    raise SafetensorsError(f"tensor {name!r} appears in multiple shards")
                self.tensors[name] = info

    def _discover_shards(self, files: list[str]) -> list[str]:
        import json

        # Sharded checkpoints ship an index file mapping tensors to shards;
        # trust it when present so we skip stray safetensors files.
        if "model.safetensors.index.json" in files:
            idx = json.loads(self.source.read_text("model.safetensors.index.json"))
            return sorted(set(idx.get("weight_map", {}).values()))
        candidates = [f for f in files if f.endswith(".safetensors") and "/" not in f]
        # Adapters live alongside merged weights in some repos; skip them.
        return sorted(f for f in candidates if not f.startswith("adapter"))

    # -- queries ------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(self.tensors.keys())

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise WeightIndexError(f"no tensor named {name!r}") from None

    def total_weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    # -- reads --------------------------------------------------------------

    def read_tensor(self, name: str) -> np.ndarray:
        t = self.info(name)
        raw = self.source.read_range(t.shard, t.start, t.end)
        return decode_tensor(raw, t.dtype, t.shape)

    def read_flat_slice(self, name: str, start_elem: int, n_elem: int) -> np.ndarray:
        """Read a contiguous run of elements from the flattened tensor."""
        t = self.info(name)
        start_elem = max(0, min(start_elem, t.numel))
        n_elem = min(n_elem, t.numel - start_elem)
        b0 = t.start + start_elem * t.itemsize
        raw = self.source.read_range(t.shard, b0, b0 + n_elem * t.itemsize)
        return decode_tensor(raw, t.dtype)

    def read_sample(
        self, name: str, seed: int, n_blocks: int = 64, block_len: int = 4096
    ) -> np.ndarray:
        """Read a deterministic pseudo-random sample of a tensor.

        Sample positions depend only on (numel, seed, n_blocks, block_len),
        so two models with matching shapes are sampled at identical element
        positions — which is what makes sampled cosines meaningful.
        """
        t = self.info(name)
        if t.numel <= n_blocks * block_len:
            return self.read_tensor(name).reshape(-1)
        offsets = _lcg_offsets(t.numel, n_blocks, block_len, seed)
        ranges = [
            (t.start + o * t.itemsize, t.start + (o + block_len) * t.itemsize)
            for o in offsets
        ]
        chunks = self._read_coalesced(t, ranges)
        return np.concatenate(chunks)

    def _read_coalesced(
        self, t: TensorInfo, ranges: list[tuple[int, int]]
    ) -> list[np.ndarray]:
        """Fetch byte ranges, merging near-adjacent ones into single reads."""
        merged: list[list[int]] = []
        for s, e in ranges:
            if merged and s - merged[-1][1] <= COALESCE_GAP:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        blobs = {(s, e): self.source.read_range(t.shard, s, e) for s, e in map(tuple, merged)}
        out: list[np.ndarray] = []
        for s, e in ranges:
            for (ms, me), blob in blobs.items():
                if ms <= s and e <= me:
                    raw = blob[s - ms : e - ms]
                    out.append(decode_tensor(raw, t.dtype))
                    break
        return out
