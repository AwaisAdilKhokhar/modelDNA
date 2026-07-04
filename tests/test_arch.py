from conftest import make_tiny_llama
from modeldna.arch.canonical import ATTN_ROLES, canonicalize
from modeldna.arch.signature import read_signature
from modeldna.io.source import LocalSource
from modeldna.io.weights import WeightIndex


def llama_names(n_layers=2):
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        names += [
            p + "self_attn.q_proj.weight",
            p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
        ]
    return names


def test_canonicalize_llama():
    cmap = canonicalize(llama_names(4))
    assert cmap.n_layers == 4
    assert cmap.unmapped == []
    assert cmap.coverage == 1.0
    assert cmap.globals["embed"] == "model.embed_tokens.weight"
    assert cmap.layers[2]["attn.q"] == "model.layers.2.self_attn.q_proj.weight"
    assert cmap.present_roles(ATTN_ROLES) == ["attn.q", "attn.k", "attn.v", "attn.o"]


def test_canonicalize_qwen_biases():
    names = llama_names(2) + [
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.1.self_attn.q_proj.bias",
        "model.layers.1.self_attn.k_proj.bias",
        "model.layers.1.self_attn.v_proj.bias",
    ]
    cmap = canonicalize(names)
    assert cmap.layers[1]["attn.q.bias"] == "model.layers.1.self_attn.q_proj.bias"
    assert set(cmap.bias_roles()) == {"attn.q.bias", "attn.k.bias", "attn.v.bias"}


def test_canonicalize_gpt_neox():
    names = [
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.dense.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        "gpt_neox.layers.0.input_layernorm.weight",
        "gpt_neox.layers.0.post_attention_layernorm.weight",
    ]
    cmap = canonicalize(names)
    assert cmap.layers[0]["attn.qkv"] == "gpt_neox.layers.0.attention.query_key_value.weight"
    assert cmap.present_roles(ATTN_ROLES) == ["attn.o", "attn.qkv"]
    assert cmap.unmapped == []


def test_canonicalize_unknown_names():
    cmap = canonicalize(["totally.novel.tensor", "model.layers.0.self_attn.q_proj.weight"])
    assert cmap.unmapped == ["totally.novel.tensor"]
    assert cmap.coverage == 0.5


def test_signature_from_tiny_model(tiny_model):
    root, tensors = tiny_model
    src = LocalSource(root)
    idx = WeightIndex(src)
    sig = read_signature(src, idx)
    assert sig.model_type == "llama"
    assert sig.n_layers == 4
    assert sig.hidden_size == 32
    assert sig.n_kv_heads == 2
    assert sig.inventory_hash and sig.tokenizer_hash
    assert sig.n_params == sum(t.size for t in tensors.values())
    assert sig.canonical_coverage == 1.0


def test_signature_compat(tmp_path):
    r1, r2, r3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    make_tiny_llama(r1, seed=0)
    make_tiny_llama(r2, seed=1)  # same shapes, different weights
    make_tiny_llama(r3, seed=2, n_layers=6)  # different depth
    sigs = []
    for r in (r1, r2, r3):
        src = LocalSource(r)
        sigs.append(read_signature(src, WeightIndex(src)))
    a, b, c = sigs
    assert a.shape_compatible(b)
    assert not a.shape_compatible(c)
    assert a.inventory_hash == b.inventory_hash  # same inventory, weights differ
    assert a.inventory_hash != c.inventory_hash


def test_signature_roundtrip(tiny_model):
    root, _ = tiny_model
    src = LocalSource(root)
    sig = read_signature(src, WeightIndex(src))
    from modeldna.arch.signature import ArchSignature

    sig2 = ArchSignature.from_dict(sig.to_dict())
    assert sig2 == sig
