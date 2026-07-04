"""Base-model family inference and the seed list for building a reference DB.

The family label groups a base with its close variants (base/Instruct/chat)
so the verdict engine can tell "ambiguous within one family" apart from
"evidence split across families" (the merge signal).
"""

from __future__ import annotations

import re

# (pattern on the full repo id, family label) — first match wins
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"meta-llama/Meta-Llama-3\.?1|meta-llama/Llama-3\.1", "llama-3.1"),
    (r"meta-llama/Llama-3\.2", "llama-3.2"),
    (r"meta-llama/Llama-3\.3", "llama-3.3"),
    (r"meta-llama/Meta-Llama-3|meta-llama/Llama-3-", "llama-3"),
    (r"meta-llama/Llama-2", "llama-2"),
    (r"Qwen/Qwen3", "qwen3"),
    (r"Qwen/Qwen2\.5", "qwen2.5"),
    (r"Qwen/Qwen2", "qwen2"),
    (r"Qwen/Qwen1\.5", "qwen1.5"),
    (r"mistralai/Mixtral", "mixtral"),
    (r"mistralai/Mistral", "mistral"),
    (r"google/gemma-3", "gemma-3"),
    (r"google/gemma-2", "gemma-2"),
    (r"google/gemma", "gemma"),
    (r"microsoft/[Pp]hi-4", "phi-4"),
    (r"microsoft/[Pp]hi-3", "phi-3"),
    (r"microsoft/phi-2", "phi-2"),
    (r"deepseek-ai/DeepSeek-V3", "deepseek-v3"),
    (r"deepseek-ai/DeepSeek-V2", "deepseek-v2"),
    (r"deepseek-ai/DeepSeek-R1", "deepseek-r1"),
    (r"deepseek-ai/deepseek-llm", "deepseek-llm"),
    (r"zai-org/GLM|THUDM/glm|THUDM/chatglm", "glm"),
    (r"01-ai/Yi", "yi"),
    (r"internlm/internlm2", "internlm2"),
    (r"tiiuae/falcon", "falcon"),
    (r"allenai/OLMo-2", "olmo-2"),
    (r"allenai/OLMo", "olmo"),
    (r"EleutherAI/pythia", "pythia"),
    (r"openlm-research/open_llama", "openllama"),
    (r"stabilityai/stablelm", "stablelm"),
    (r"HuggingFaceTB/SmolLM2", "smollm2"),
    (r"HuggingFaceTB/SmolLM", "smollm"),
    (r"TinyLlama/", "tinyllama"),
    (r"openbmb/MiniCPM", "minicpm"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), fam) for p, fam in _FAMILY_PATTERNS]


def guess_family(model_id: str) -> str:
    for rx, fam in _COMPILED:
        if rx.search(model_id):
            return fam
    org = model_id.split("/")[0] if "/" in model_id else model_id
    return org.lower()


# Seed reference set: the bases whose derivative populations dominate the Hub,
# plus deliberate hard negatives (OpenLLaMA vs Llama; Pythia seed siblings).
# `modeldna db build` fingerprints these directly from the Hub.
SEED_MODELS: list[str] = [
    # Llama
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Qwen
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen3-8B",
    # Mistral
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # Gemma
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-9b",
    "google/gemma-3-4b-pt",
    # Phi
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-4",
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Others widely forked
    "01-ai/Yi-6B",
    "internlm/internlm2-7b",
    "tiiuae/falcon-7b",
    "allenai/OLMo-2-1124-7B",
    "stabilityai/stablelm-2-1_6b",
    "HuggingFaceTB/SmolLM2-1.7B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openbmb/MiniCPM-2B-sft-bf16",
    # Hard negatives / controls
    "openlm-research/open_llama_7b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-6.9b",
]
