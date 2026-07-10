"""Tier-0 structural signature: everything knowable without touching weights.

Costs kilobytes (config.json + tokenizer hash + shard headers) and already
does real forensic work: it filters reference candidates down to
shape-compatible ones, and catches renamed-tensor mirrors and exact
re-uploads via the inventory hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from modeldna.arch.canonical import CanonicalMap, canonicalize
from modeldna.io.safetensors import TensorInfo
from modeldna.io.source import ModelSource
from modeldna.io.weights import BaseWeightIndex

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")


@dataclass
class ArchSignature:
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    hidden_size: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    rope_theta: float | None = None
    norm_eps: float | None = None
    torch_dtype: str = ""
    tie_word_embeddings: bool = False
    #: storage the fingerprint was read from ("safetensors", "gguf:<file>")
    weights_format: str = ""

    #: sha256 over the sorted (name, dtype, shape) tensor inventory
    inventory_hash: str = ""
    #: sha256 of the tokenizer file (empty if none found)
    tokenizer_hash: str = ""
    #: total parameter count implied by tensor shapes
    n_params: int = 0
    #: fraction of tensors the canonicalizer understood
    canonical_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchSignature":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def core_shape(self) -> tuple:
        """The tuple that must match for tensors to be directly comparable."""
        return (
            self.hidden_size,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.intermediate_size,
        )

    def shape_compatible(self, other: "ArchSignature") -> bool:
        """Same transformer skeleton (embeddings may differ, e.g. vocab swaps)."""
        return self.core_shape() == other.core_shape()

    def family_compatible(self, other: "ArchSignature") -> bool:
        """Looser: same width/depth, heads may differ (layer surgery flagging)."""
        return (
            self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )

    def summary(self) -> str:
        return (
            f"{self.model_type or '?'} L{self.n_layers} d{self.hidden_size} "
            f"h{self.n_heads}/{self.n_kv_heads} ff{self.intermediate_size} "
            f"v{self.vocab_size} (~{self.n_params / 1e9:.2f}B params)"
        )


def _inventory_hash(tensors: dict[str, TensorInfo]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name]
        h.update(f"{name}|{t.dtype}|{','.join(map(str, t.shape))}\n".encode())
    return h.hexdigest()


def _tokenizer_hash(source: ModelSource) -> str:
    files = source.list_files()
    for fname in TOKENIZER_FILES:
        if fname in files:
            return hashlib.sha256(source.read_file(fname)).hexdigest()
    return ""


def read_config(source: ModelSource) -> dict[str, Any]:
    if "config.json" not in source.list_files():
        return {}
    try:
        return json.loads(source.read_text("config.json"))
    except json.JSONDecodeError:
        return {}


def read_signature(
    source: ModelSource, index: BaseWeightIndex, cmap: CanonicalMap | None = None
) -> ArchSignature:
    cfg = read_config(source)
    if hasattr(index, "hf_config"):
        # GGUF: the embedded metadata describes the actual file (dims, quant
        # dtype) and wins over any config.json the repo happens to ship.
        cfg = {**cfg, **{k: v for k, v in index.hf_config().items() if v}}
    if cmap is None:
        cmap = canonicalize(index.names)

    n_layers = cfg.get("num_hidden_layers") or cmap.n_layers
    n_heads = cfg.get("num_attention_heads", 0)
    return ArchSignature(
        model_type=cfg.get("model_type", ""),
        architectures=cfg.get("architectures", []) or [],
        hidden_size=cfg.get("hidden_size", 0),
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=cfg.get("num_key_value_heads", n_heads),
        intermediate_size=cfg.get("intermediate_size", 0),
        vocab_size=cfg.get("vocab_size", 0),
        rope_theta=cfg.get("rope_theta"),
        norm_eps=cfg.get("rms_norm_eps", cfg.get("layer_norm_eps")),
        torch_dtype=cfg.get("torch_dtype", ""),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        weights_format=getattr(index, "weights_format", ""),
        inventory_hash=_inventory_hash(index.tensors),
        tokenizer_hash=_tokenizer_hash(source) or getattr(index, "tokenizer_hash", ""),
        n_params=sum(t.numel for t in index.tensors.values()),
        canonical_coverage=round(cmap.coverage, 4),
    )
