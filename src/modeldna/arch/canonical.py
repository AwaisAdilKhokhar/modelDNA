"""Canonical tensor naming across model families.

Different codebases name the same weight differently — Llama's
``model.layers.3.self_attn.q_proj.weight`` is GPT-NeoX's
``gpt_neox.layers.3.attention.query_key_value.weight`` (fused). Everything
downstream (σ-curves, PCS sampling, spectra) works on canonical roles so two
models are always compared attention-to-attention regardless of scheme.

Canonical role vocabulary:

    per layer:  attn.q  attn.k  attn.v  attn.o  attn.qkv (fused)
                mlp.gate  mlp.up  mlp.down  mlp.gate_up (fused)
                mlp.fc1  mlp.fc2 (classic 2-layer MLP)
                norm.in  norm.post  (+ ".bias" suffixed variants of any)
    global:     embed  lm_head  norm.final

A trailing ``.bias`` on the canonical role marks a bias vector, e.g.
``attn.q.bias`` (Qwen2-style attention biases — a key F2 signal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (regex, canonical role) — first match wins. {i} is the layer index group.
_LAYER_PATTERNS: list[tuple[str, str]] = [
    # HF llama family: Llama, Mistral, Qwen2/2.5, Gemma, Yi, InternLM2 (converted), OLMo-2 …
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", "attn.q"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", "attn.k"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", "attn.v"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", "attn.o"),
    (r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)", "attn.qkv"),  # Phi-3
    (r"model\.layers\.(\d+)\.self_attn\.dense\.(weight|bias)", "attn.o"),  # Phi-2
    (r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", "mlp.gate"),
    (r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", "mlp.up"),
    (r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", "mlp.down"),
    (r"model\.layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)", "mlp.gate_up"),  # Phi-3
    (r"model\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)", "mlp.fc1"),  # Phi-2
    (r"model\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)", "mlp.fc2"),
    (r"model\.layers\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", "norm.post"),
    # extra norms some families carry (Gemma2, OLMo-2); tracked, not compared by default
    (r"model\.layers\.(\d+)\.pre_feedforward_layernorm\.(weight|bias)", "norm.pre_ff"),
    (r"model\.layers\.(\d+)\.post_feedforward_layernorm\.(weight|bias)", "norm.post_ff"),
    (r"model\.layers\.(\d+)\.self_attn\.q_norm\.(weight|bias)", "norm.q"),  # Qwen3, OLMo-2
    (r"model\.layers\.(\d+)\.self_attn\.k_norm\.(weight|bias)", "norm.k"),
    # GPT-NeoX / Pythia
    (r"gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.(weight|bias)", "attn.qkv"),
    (r"gpt_neox\.layers\.(\d+)\.attention\.dense\.(weight|bias)", "attn.o"),
    (r"gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)", "mlp.fc1"),
    (r"gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)", "mlp.fc2"),
    (r"gpt_neox\.layers\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", "norm.post"),
    # Falcon
    (r"transformer\.h\.(\d+)\.self_attention\.query_key_value\.(weight|bias)", "attn.qkv"),
    (r"transformer\.h\.(\d+)\.self_attention\.dense\.(weight|bias)", "attn.o"),
    (r"transformer\.h\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)", "mlp.fc1"),
    (r"transformer\.h\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)", "mlp.fc2"),
    (r"transformer\.h\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"transformer\.h\.(\d+)\.ln_attn\.(weight|bias)", "norm.in"),
    (r"transformer\.h\.(\d+)\.ln_mlp\.(weight|bias)", "norm.post"),
    # GPT-2 style (fused Conv1D; kept so Tier-0 can at least inventory them)
    (r"transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)", "attn.qkv"),
    (r"transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)", "attn.o"),
    (r"transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)", "mlp.fc1"),
    (r"transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)", "mlp.fc2"),
    (r"transformer\.h\.(\d+)\.ln_1\.(weight|bias)", "norm.in"),
    (r"transformer\.h\.(\d+)\.ln_2\.(weight|bias)", "norm.post"),
    # GGUF / llama.cpp (served dequantized by io.gguf). llama.cpp uses
    # ffn_up/ffn_down for every family, so fc1/fc2-style models (GPT-2,
    # NeoX, Phi-2) land on mlp.up/mlp.down here — those roles then simply
    # don't overlap with the safetensors fingerprint and are skipped.
    (r"blk\.(\d+)\.attn_q\.(weight|bias)", "attn.q"),
    (r"blk\.(\d+)\.attn_k\.(weight|bias)", "attn.k"),
    (r"blk\.(\d+)\.attn_v\.(weight|bias)", "attn.v"),
    (r"blk\.(\d+)\.attn_output\.(weight|bias)", "attn.o"),
    (r"blk\.(\d+)\.attn_qkv\.(weight|bias)", "attn.qkv"),
    (r"blk\.(\d+)\.ffn_gate\.(weight|bias)", "mlp.gate"),
    (r"blk\.(\d+)\.ffn_up\.(weight|bias)", "mlp.up"),
    (r"blk\.(\d+)\.ffn_down\.(weight|bias)", "mlp.down"),
    (r"blk\.(\d+)\.attn_norm\.(weight|bias)", "norm.in"),
    (r"blk\.(\d+)\.ffn_norm\.(weight|bias)", "norm.post"),
    (r"blk\.(\d+)\.attn_q_norm\.(weight|bias)", "norm.q"),
    (r"blk\.(\d+)\.attn_k_norm\.(weight|bias)", "norm.k"),
]

_GLOBAL_PATTERNS: list[tuple[str, str]] = [
    (r"model\.embed_tokens\.weight", "embed"),
    (r"gpt_neox\.embed_in\.weight", "embed"),
    (r"transformer\.word_embeddings\.weight", "embed"),
    (r"transformer\.wte\.weight", "embed"),
    (r"lm_head\.weight", "lm_head"),
    (r"embed_out\.weight", "lm_head"),
    (r"model\.norm\.(weight|bias)", "norm.final"),
    (r"gpt_neox\.final_layer_norm\.(weight|bias)", "norm.final"),
    (r"transformer\.ln_f\.(weight|bias)", "norm.final"),
    # GGUF / llama.cpp
    (r"token_embd\.weight", "embed"),
    (r"output\.weight", "lm_head"),
    (r"output_norm\.(weight|bias)", "norm.final"),
]

_COMPILED_LAYER = [(re.compile(p + r"$"), role) for p, role in _LAYER_PATTERNS]
_COMPILED_GLOBAL = [(re.compile(p + r"$"), role) for p, role in _GLOBAL_PATTERNS]

#: 2-D attention roles in comparison priority order
ATTN_ROLES = ("attn.q", "attn.k", "attn.v", "attn.o", "attn.qkv")
MLP_ROLES = ("mlp.gate", "mlp.up", "mlp.down", "mlp.gate_up", "mlp.fc1", "mlp.fc2")
NORM_ROLES = ("norm.in", "norm.post", "norm.pre_ff", "norm.post_ff", "norm.q", "norm.k")


@dataclass
class CanonicalMap:
    """Canonical view of a model's tensor inventory."""

    layers: dict[int, dict[str, str]] = field(default_factory=dict)  # layer -> role -> name
    globals: dict[str, str] = field(default_factory=dict)  # role -> name
    unmapped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return max(self.layers) + 1 if self.layers else 0

    def role_by_layer(self, role: str) -> list[str | None]:
        """Tensor names for one role across layers 0..L-1 (None where absent)."""
        return [self.layers.get(i, {}).get(role) for i in range(self.n_layers)]

    def present_roles(self, candidates: tuple[str, ...]) -> list[str]:
        """Roles from `candidates` present in every layer."""
        out = []
        for role in candidates:
            names = self.role_by_layer(role)
            if names and all(n is not None for n in names):
                out.append(role)
        return out

    def bias_roles(self) -> list[str]:
        """Attention/MLP roles that carry biases in every layer."""
        out = []
        for role in ATTN_ROLES + MLP_ROLES:
            names = self.role_by_layer(role + ".bias")
            if names and all(n is not None for n in names):
                out.append(role + ".bias")
        return out

    @property
    def coverage(self) -> float:
        mapped = sum(len(d) for d in self.layers.values()) + len(self.globals)
        total = mapped + len(self.unmapped)
        return mapped / total if total else 0.0


def canonicalize(tensor_names: list[str]) -> CanonicalMap:
    cmap = CanonicalMap()
    for name in tensor_names:
        matched = False
        for rx, role in _COMPILED_LAYER:
            m = rx.match(name)
            if m:
                layer = int(m.group(1))
                kind = m.group(2) if m.lastindex and m.lastindex >= 2 else "weight"
                key = role if kind == "weight" else f"{role}.bias"
                cmap.layers.setdefault(layer, {})[key] = name
                matched = True
                break
        if matched:
            continue
        for rx, role in _COMPILED_GLOBAL:
            if rx.match(name):
                cmap.globals[role] = name
                matched = True
                break
        if not matched:
            cmap.unmapped.append(name)
    return cmap
