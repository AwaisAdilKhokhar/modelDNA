"""Synthetic model generation for tests and benchmarks.

Builds tiny llama-architecture checkpoints with the statistical properties
that matter for lineage work: each independently seeded model gets its own
layer-wise sigma profile, norm-gain pattern, and low-rank spectral structure,
while `base_weights` + `noise` simulates fine-tuning / continued training.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_NP_TO_ST = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Serialize tensors in safetensors format (little-endian, C-order)."""
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        st_dtype = _NP_TO_ST[arr.dtype]
        start = len(payload)
        payload += arr.tobytes()
        header[name] = {
            "dtype": st_dtype,
            "shape": list(arr.shape),
            "data_offsets": [start, len(payload)],
        }
    hbytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(payload)


def _rand_matrix(rng: np.random.Generator, rows: int, cols: int, scale: float) -> np.ndarray:
    """Gaussian matrix plus model-specific low-rank structure.

    Pure iid matrices all share the Marchenko-Pastur spectrum, which would
    make every pair of 'independent' test models spectrally identical. The
    low-rank spikes give each synthetic model its own spectral signature,
    like real trained weights have.
    """
    w = rng.normal(0, scale, (rows, cols))
    r = 4
    u = rng.normal(0, 1, (rows, r))
    v = rng.normal(0, 1, (r, cols))
    strengths = scale * (1 + 6 * rng.random(r))
    w += (u * strengths) @ v / np.sqrt(r)
    return w.astype(np.float32)


def make_tiny_llama(
    root: Path,
    *,
    n_layers: int = 4,
    hidden: int = 32,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    intermediate: int = 64,
    vocab: int = 128,
    seed: int = 0,
    sharded: bool = False,
    base_weights: dict[str, np.ndarray] | None = None,
    noise: float = 0.0,
    readme: str | None = None,
) -> dict[str, np.ndarray]:
    """Write a tiny llama-architecture model to `root`.

    If `base_weights` is given, start from those and add gaussian noise of
    scale `noise` — i.e. simulate a fine-tune (small noise) or heavy
    continued training (large noise) of that base.
    """
    rng = np.random.default_rng(seed)
    head_dim = hidden // n_heads
    kv_dim = n_kv_heads * head_dim

    if base_weights is not None:
        tensors = {
            k: (v + rng.normal(0, noise, v.shape)).astype(np.float32)
            for k, v in base_weights.items()
        }
    else:
        # per-model layer-wise scale profile (smooth random walk, positive)
        walk = np.cumsum(rng.normal(0, 1, n_layers))
        walk = (walk - walk.mean()) / (walk.std() + 1e-9)
        profile = 0.02 * (1 + 0.35 * walk)
        profile = np.maximum(profile, 0.005)

        tensors = {}
        tensors["model.embed_tokens.weight"] = rng.normal(0, 0.02, (vocab, hidden)).astype(
            np.float32
        )
        for i in range(n_layers):
            p = f"model.layers.{i}."
            scale = float(profile[i])
            tensors[p + "self_attn.q_proj.weight"] = _rand_matrix(rng, hidden, hidden, scale)
            tensors[p + "self_attn.k_proj.weight"] = _rand_matrix(rng, kv_dim, hidden, scale * 0.9)
            tensors[p + "self_attn.v_proj.weight"] = _rand_matrix(rng, kv_dim, hidden, scale * 1.1)
            tensors[p + "self_attn.o_proj.weight"] = _rand_matrix(rng, hidden, hidden, scale)
            tensors[p + "mlp.gate_proj.weight"] = _rand_matrix(rng, intermediate, hidden, scale)
            tensors[p + "mlp.up_proj.weight"] = _rand_matrix(rng, intermediate, hidden, scale)
            tensors[p + "mlp.down_proj.weight"] = _rand_matrix(rng, hidden, intermediate, scale)
            tensors[p + "input_layernorm.weight"] = (
                1 + rng.normal(0, 0.35, hidden)
            ).astype(np.float32)
            tensors[p + "post_attention_layernorm.weight"] = (
                1 + rng.normal(0, 0.35, hidden)
            ).astype(np.float32)
        tensors["model.norm.weight"] = (1 + rng.normal(0, 0.35, hidden)).astype(np.float32)
        tensors["lm_head.weight"] = rng.normal(0, 0.02, (vocab, hidden)).astype(np.float32)

    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "float32",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {}}, "seed": seed if base_weights is None else "inherited"})
    )
    if readme is not None:
        (root / "README.md").write_text(readme)

    if sharded:
        names = list(tensors)
        half = len(names) // 2
        shards = {
            "model-00001-of-00002.safetensors": {k: tensors[k] for k in names[:half]},
            "model-00002-of-00002.safetensors": {k: tensors[k] for k in names[half:]},
        }
        weight_map = {}
        for shard_name, shard_tensors in shards.items():
            write_safetensors(root / shard_name, shard_tensors)
            for k in shard_tensors:
                weight_map[k] = shard_name
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    else:
        write_safetensors(root / "model.safetensors", tensors)
    return tensors
