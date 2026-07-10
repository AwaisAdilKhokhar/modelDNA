---
license: apache-2.0
pretty_name: "LineageBench — org-documented lineage ground truth for open-weight LLMs"
tags:
- model-lineage
- provenance
- fingerprinting
- benchmark
- llm
- model-merging
size_categories:
- n<1K
task_categories:
- other
configs:
- config_name: ground_truth
  data_files: ground_truth.jsonl
- config_name: reference_results
  data_files: reference_results.jsonl
---

# LineageBench

A small, **honestly-scoped** benchmark for *weight-based lineage verification*:
given a suspect open-weight model and a set of candidate parents, decide which
foundation base it actually descends from — **from the weights alone**.

The hard part of a lineage benchmark is the ground truth. It cannot come from a
model's own `base_model` tag, because auditing that tag is the entire point of
the exercise. LineageBench instead labels every pair from the **publishing
organization's own documentation** — technical reports, official release notes,
model cards written by the org that trained the parent — and ships a citation
for each label.

- **__N_PAIRS__ suspect models** across __N_FAMILIES__ foundation families
- **__N_PARENTS__ candidate parents** (curated foundation bases)
- Reference implementation: **[modelDNA](https://github.com/AwaisAdilKhokhar/modelDNA)** v__MODELDNA_VERSION__
- Headline: **AUROC __AUROC__ · TPR@FPR1% __TPR__ · __FP_AT_09__ false positives** at the reporting threshold, **__TOP1__ top-1 parent attribution**

Generated __GENERATED_AT__.

## Why this exists

Open-weight model lineage is mostly undocumented or unverifiable: a large
fraction of Hub models carry missing, self-reported, or contradicted parentage.
"Which base is this really derived from?" recurs in license-compliance disputes,
merge archaeology, and safety provenance — and the honest answer has to be
reconstructed from weights, not taken on trust. LineageBench is a named, cited,
reproducible yardstick for that task, so different methods can be compared on
the same ground truth instead of on anecdotes.

## What's here

| file | rows | what |
|---|---|---|
| `ground_truth.jsonl` | __N_PAIRS__ | one row per suspect: `suspect`, `parent` (true base), `kind`, `in_positive_pool`, and a `citation` to org documentation |
| `reference_results.jsonl` | __N_PAIRS__ | modelDNA's output per suspect: calibrated `score_vs_true_parent`, `verdict`, `verdict_best_candidate`, `top1_correct` |
| `parents.json` | __N_PARENTS__ | the candidate foundation parents every suspect is judged against |
| `metrics.json` | — | the headline ship-gate metrics for the reference implementation |

## The suspects

The pairs cover the derivation regimes a scanner meets in the wild, on purpose:

- **fine-tunes** — community SFT/DPO derivatives (Zephyr-7B-β, OpenHermes-2.5, Nous-Hermes, Dolphin-2.9)
- **continued-pretrain** — CodeLlama-7B, Qwen2.5-Coder-7B-Instruct
- **official-instruct** — the orgs' own instruct releases (Llama-3, Qwen2.5, Mistral, Gemma)
- **merge-chain** — AlphaMonarch-7B, a DARE-TIES merge whose every ancestor traces to one Mistral base
- **depth-upscale** — SOLAR-10.7B (a *limitation case*, see below)
- **quantized-copy** — a GPTQ int4 repack (a *limitation case*)

Two cases (`in_positive_pool: false`) are **out-of-scope by design** and scored
separately against their documented expected behavior, which is *abstention*:
depth-changing derivations (SOLAR's 32→48 layer up-scale) and packed-int4
quantization (GPTQ) both fall outside the shape-matched, safetensors-only
comparison the reference tool performs today. A method that scored them as
confident positives would be *wrong*; abstaining is the correct answer.

## The scoring protocol

Each suspect is judged by the **actual verdict engine** against **all
__N_PARENTS__ candidate parents** — the true parent must win, not merely score
well in isolation. The positive pool is the __N_POSITIVES__ in-scope suspects.
The negative pool is **__N_NEGATIVES__ hard negatives**: every positive suspect
scored against each *cross-family* wrong parent, plus all cross-family
parent-versus-parent pairs.

Same-family wrong parents (e.g. Zephyr against Mistral-v0.3 rather than v0.1)
are deliberately **excluded** from the negative pool — they genuinely share
lineage, and scoring them as negatives would punish a method for detecting
something true.

## Reference results (modelDNA v__MODELDNA_VERSION__)

| metric | value |
|---|---|
| AUROC (__N_POSITIVES__ positives vs __N_NEGATIVES__ hard negatives) | __AUROC__ |
| TPR at 1% FPR | __TPR__ |
| false positives at the p ≥ 0.9 reporting threshold | __FP_AT_09__ / __N_NEGATIVES__ |
| weakest positive | __MIN_POS__ |
| strongest negative | __MAX_NEG__ |
| top-1 parent attribution | __TOP1__ |

Clean separation: the entire abstention band (0.50–0.90) is empty on this data.
The two limitation cases behaved exactly as documented — SOLAR produced an
honest `NO_MATCH` on the layer-count mismatch; the GPTQ copy abstained on the
packed tensors while still ranking the true parent first.

## Reproduce it

Every label carries a citation and every fingerprint is cached, so the numbers
regenerate offline in seconds — no Hub round-trip:

```bash
pip install modeldna
git clone https://github.com/AwaisAdilKhokhar/modelDNA
python modelDNA/benchmarks/real_lineagebench.py --no-fetch
# -> ship gates: PASS
```

## Load it

```python
from datasets import load_dataset

gt = load_dataset("AwaisAdilKhokhar/lineagebench", "ground_truth", split="train")
res = load_dataset("AwaisAdilKhokhar/lineagebench", "reference_results", split="train")
```

## Scope, honestly

This is a deliberately **small** slice — __N_PAIRS__ pairs, not the ≥300 the
project's planning document sets as the eventual target. Every label is cited,
but AUROC __AUROC__ should be read as *"no errors at this scale"*, not as a
claim that errors are impossible. Growing the benchmark is mechanical (the
fingerprint pipeline is the expensive part, and it is fast); contributions of
new **org-documented, cited** pairs are welcome via PR to the
[manifest](https://github.com/AwaisAdilKhokhar/modelDNA/blob/main/benchmarks/lineagebench_pairs.json).

Absence of a detected edge is **not** evidence of independence: distillation is
invisible to every weight-space method by construction, and depth/quantization
mismatches abstain rather than resolve. Read a verdict as *statistical
consistency with derivation*, never as an accusation.

## Cite

```bibtex
@misc{lineagebench,
  title  = {LineageBench: org-documented lineage ground truth for open-weight LLMs},
  author = {Awais Bin Adil, Muhammad and Aamir, Saad},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/AwaisAdilKhokhar/lineagebench}},
  note   = {Reference implementation: modelDNA, \url{https://github.com/AwaisAdilKhokhar/modelDNA}}
}
```
