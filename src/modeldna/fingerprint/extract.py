"""Fingerprint extraction: turn a model into a small, comparable signature.

A fingerprint bundles the four MVP signals:

    F1  sigma_curves     per-layer std of each attention/MLP projection
    F2  vector_norms/samples   1-D tensors (norm gains, attention biases)
    F3  pcs_samples      seeded element samples of the 2-D weights
    F4  spectra          top-k singular values per layer (sketch + exact)

Fast mode reads seeded slices (~100-300 MB for a 7B); full mode reads the
complete attention/MLP tensors. Sample positions are pure functions of
(seed, role, layer, tensor size), so any two models with matching shapes are
sampled at identical element positions — the property that makes sampled
cosines between them meaningful.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from modeldna.arch.canonical import (
    ATTN_ROLES,
    MLP_ROLES,
    NORM_ROLES,
    CanonicalMap,
    canonicalize,
)
from modeldna.arch.signature import ArchSignature, read_signature
from modeldna.io.source import ModelSource
from modeldna.io.weights import WeightIndex, _lcg_offsets
from modeldna.fingerprint.methods import fnv1a, randomized_svals

FP_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260704

# Frozen sampling parameters. Changing any of these invalidates every stored
# fingerprint, so they are versioned via FP_SCHEMA_VERSION.
SAMPLE_BLOCKS = 32
SAMPLE_BLOCK_LEN = 16_384  # -> up to 512k elements per matrix
PCS_PER_ROLE = 98_304  # total PCS elements per role, split across layers
SKETCH_ROWS = 192
SVD_TOP_K = 32
SPECTRA_STRIDE_FAST = 4

#: 2-D roles that get PCS samples (attention + one MLP anchor)
PCS_ROLES = ATTN_ROLES + ("mlp.down", "mlp.fc2")


@dataclass
class Fingerprint:
    model_id: str
    arch: ArchSignature
    mode: str = "fast"
    seed: int = DEFAULT_SEED
    revision: str = ""
    created_at: str = ""
    schema_version: int = FP_SCHEMA_VERSION

    # F1: role -> per-layer raw sigma (normalized at compare time)
    sigma_curves: dict[str, list[float]] = field(default_factory=dict)
    # F2: 1-D role -> per-layer L2 norms, and concatenated even subsamples
    vector_norms: dict[str, list[float]] = field(default_factory=dict)
    vector_samples: dict[str, list[float]] = field(default_factory=dict)
    # F3: 2-D role -> concatenated seeded element samples across layers
    pcs_samples: dict[str, list[float]] = field(default_factory=dict)
    # F4: role -> {layer: top-k singular values}
    spectra_sketch: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    spectra_exact: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    bytes_read: int = 0

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "mode": self.mode,
            "seed": self.seed,
            "created_at": self.created_at,
            "bytes_read": self.bytes_read,
            "arch": self.arch.to_dict(),
            "sigma_curves": self.sigma_curves,
            "vector_norms": self.vector_norms,
            "vector_samples": self.vector_samples,
            "pcs_samples": self.pcs_samples,
            "spectra_sketch": self.spectra_sketch,
            "spectra_exact": self.spectra_exact,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fingerprint":
        return cls(
            model_id=d["model_id"],
            arch=ArchSignature.from_dict(d["arch"]),
            mode=d.get("mode", "fast"),
            seed=d.get("seed", DEFAULT_SEED),
            revision=d.get("revision", ""),
            created_at=d.get("created_at", ""),
            schema_version=d.get("schema_version", FP_SCHEMA_VERSION),
            sigma_curves=d.get("sigma_curves", {}),
            vector_norms=d.get("vector_norms", {}),
            vector_samples=d.get("vector_samples", {}),
            pcs_samples=d.get("pcs_samples", {}),
            spectra_sketch=d.get("spectra_sketch", {}),
            spectra_exact=d.get("spectra_exact", {}),
            bytes_read=d.get("bytes_read", 0),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = json.dumps(self.to_dict(), separators=(",", ":")).encode()
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(payload))
        else:
            path.write_bytes(payload)

    @classmethod
    def load(cls, path: str | Path) -> "Fingerprint":
        path = Path(path)
        raw = path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return cls.from_dict(json.loads(raw))


def _fl(a: np.ndarray) -> list[float]:
    """float32-rounded python list (keeps JSON size sane)."""
    return [float(x) for x in np.asarray(a, dtype=np.float32)]


def _sample_positions(numel: int, seed: int, n_blocks: int, block_len: int) -> np.ndarray | None:
    """Flat indices read_sample() would fetch; None means 'whole tensor'."""
    if numel <= n_blocks * block_len:
        return None
    offsets = _lcg_offsets(numel, n_blocks, block_len, seed)
    return np.concatenate([np.arange(o, o + block_len) for o in offsets])


def _even_subsample(v: np.ndarray, cap: int = 256) -> np.ndarray:
    if v.size <= cap:
        return v
    idx = np.linspace(0, v.size - 1, cap).astype(np.int64)
    return v[idx]


def _sketch_band(rows: int, tseed: int) -> tuple[int, int]:
    """(first row, row count) of the seeded row band the SVD sketch reads."""
    r = min(rows, SKETCH_ROWS)
    row0 = tseed % (rows - r + 1) if rows > r else 0
    return row0, r


class FingerprintExtractor:
    def __init__(
        self,
        source: ModelSource,
        mode: str = "fast",
        seed: int = DEFAULT_SEED,
        model_id: str | None = None,
        revision: str = "",
    ):
        if mode not in ("fast", "full"):
            raise ValueError(f"mode must be 'fast' or 'full', got {mode!r}")
        self.source = source
        self.mode = mode
        self.seed = seed
        self.model_id = model_id or source.name
        self.revision = revision
        self.index = WeightIndex(source)
        self.cmap: CanonicalMap = canonicalize(self.index.names)
        self.sig: ArchSignature = read_signature(source, self.index, self.cmap)

    # -- per-tensor reads -----------------------------------------------------

    def _tensor_seed(self, role: str, layer: int) -> int:
        # Seeded by canonical role + layer, never by raw tensor name, so
        # models with different naming schemes sample identical positions.
        return fnv1a(self.seed, role, layer)

    def _matrix_sample(self, name: str, role: str, layer: int) -> np.ndarray:
        """Seeded sample of a 2-D tensor (identical positions in both modes)."""
        tseed = self._tensor_seed(role, layer)
        if self.mode == "fast":
            return self.index.read_sample(
                name, seed=tseed, n_blocks=SAMPLE_BLOCKS, block_len=SAMPLE_BLOCK_LEN
            )
        flat = self.index.read_tensor(name).reshape(-1)
        pos = _sample_positions(flat.size, tseed, SAMPLE_BLOCKS, SAMPLE_BLOCK_LEN)
        return flat if pos is None else flat[pos]

    # -- extraction ------------------------------------------------------------

    def run(self) -> Fingerprint:
        fp = Fingerprint(
            model_id=self.model_id,
            arch=self.sig,
            mode=self.mode,
            seed=self.seed,
            revision=self.revision,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        two_d_roles = self.cmap.present_roles(ATTN_ROLES + MLP_ROLES)
        self._prefetch(two_d_roles)
        try:
            self._extract_sigma_and_pcs(fp, two_d_roles)
            self._extract_vectors(fp)
            self._extract_spectra(fp, [r for r in two_d_roles if r in ATTN_ROLES])
        finally:
            self.index.clear_prefetch()
        fp.bytes_read = self.source.bytes_read
        return fp

    def _prefetch(self, two_d_roles: list[str]) -> None:
        """Fetch every byte range this extraction will read in one pass.

        Sample positions are pure functions of (seed, role, layer, tensor
        size), so the complete read plan is known before any data moves.
        Handing it to WeightIndex.prefetch turns thousands of sequential
        round trips into one coalesced, concurrent sweep per shard; the
        extraction below then decodes the very same bytes it would have
        fetched lazily, so fingerprints stay bit-identical.
        """
        if not (self.source.is_remote and self.mode == "fast"):
            # Local reads are already cheap, and full mode would mean
            # holding entire shards in memory.
            return
        plan: list[tuple[str, int, int]] = []
        for role in two_d_roles:
            for layer, name in enumerate(self.cmap.role_by_layer(role)):
                t = self.index.info(name)
                tseed = self._tensor_seed(role, layer)
                for s, e in self.index.plan_sample(
                    name, tseed, SAMPLE_BLOCKS, SAMPLE_BLOCK_LEN
                ):
                    plan.append((t.shard, s, e))
        vector_names: list[str] = []
        for role in self.cmap.present_roles(NORM_ROLES) + self.cmap.bias_roles():
            vector_names.extend(self.cmap.role_by_layer(role))
        if "norm.final" in self.cmap.globals:
            vector_names.append(self.cmap.globals["norm.final"])
        for name in vector_names:
            t = self.index.info(name)
            plan.append((t.shard, t.start, t.end))
        for role in [r for r in two_d_roles if r in ATTN_ROLES]:
            names = self.cmap.role_by_layer(role)
            for layer in range(0, len(names), SPECTRA_STRIDE_FAST):
                t = self.index.info(names[layer])
                if len(t.shape) != 2:
                    continue
                rows, cols = t.shape
                row0, r = _sketch_band(rows, self._tensor_seed(role, layer))
                b0 = t.start + row0 * cols * t.itemsize
                plan.append((t.shard, b0, b0 + r * cols * t.itemsize))
        self.index.prefetch(plan)

    def _extract_sigma_and_pcs(self, fp: Fingerprint, roles: list[str]) -> None:
        n_layers = self.cmap.n_layers
        pcs_per_layer = max(1024, PCS_PER_ROLE // max(n_layers, 1))
        for role in roles:
            sigmas: list[float] = []
            pcs_parts: list[np.ndarray] = []
            for layer, name in enumerate(self.cmap.role_by_layer(role)):
                sample = self._matrix_sample(name, role, layer)
                sigmas.append(float(np.std(sample, dtype=np.float64)))
                if role in PCS_ROLES:
                    pcs_parts.append(sample[:pcs_per_layer])
            fp.sigma_curves[role] = sigmas
            if pcs_parts:
                fp.pcs_samples[role] = _fl(np.concatenate(pcs_parts))

    def _extract_vectors(self, fp: Fingerprint) -> None:
        roles = self.cmap.present_roles(NORM_ROLES) + self.cmap.bias_roles()
        for role in roles:
            norms: list[float] = []
            samples: list[np.ndarray] = []
            for name in self.cmap.role_by_layer(role):
                v = self.index.read_tensor(name).reshape(-1)
                norms.append(float(np.linalg.norm(v)))
                samples.append(_even_subsample(v))
            fp.vector_norms[role] = norms
            fp.vector_samples[role] = _fl(np.concatenate(samples))
        if "norm.final" in self.cmap.globals:
            v = self.index.read_tensor(self.cmap.globals["norm.final"]).reshape(-1)
            fp.vector_norms["norm.final"] = [float(np.linalg.norm(v))]
            fp.vector_samples["norm.final"] = _fl(_even_subsample(v))

    def _extract_spectra(self, fp: Fingerprint, roles: list[str]) -> None:
        stride = SPECTRA_STRIDE_FAST if self.mode == "fast" else 1
        for role in roles:
            names = self.cmap.role_by_layer(role)
            sketch: dict[str, list[float]] = {}
            exact: dict[str, list[float]] = {}
            for layer in range(0, len(names), stride):
                name = names[layer]
                info = self.index.info(name)
                if len(info.shape) != 2:
                    continue
                rows, cols = info.shape
                tseed = self._tensor_seed(role, layer)
                row0, r = _sketch_band(rows, tseed)
                if self.mode == "fast":
                    block = self.index.read_flat_slice(name, row0 * cols, r * cols)
                    block = block.reshape(r, cols)
                else:
                    full = self.index.read_tensor(name)
                    block = full[row0 : row0 + r]
                    exact[str(layer)] = _fl(randomized_svals(full, SVD_TOP_K, seed=tseed))
                sketch[str(layer)] = _fl(randomized_svals(block, SVD_TOP_K, seed=tseed))
            fp.spectra_sketch[role] = sketch
            if exact:
                fp.spectra_exact[role] = exact


def extract_fingerprint(
    source: ModelSource,
    mode: str = "fast",
    seed: int = DEFAULT_SEED,
    model_id: str | None = None,
    revision: str = "",
) -> Fingerprint:
    return FingerprintExtractor(
        source, mode=mode, seed=seed, model_id=model_id, revision=revision
    ).run()
