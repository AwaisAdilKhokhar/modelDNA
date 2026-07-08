---
license: apache-2.0
pretty_name: "modelDNA Atlas — weight fingerprints + inferred lineage graph"
tags:
- model-lineage
- provenance
- fingerprinting
- llm
- model-merging
size_categories:
- n<1K
configs:
- config_name: models
  data_files: models.jsonl
- config_name: edges
  data_files: edges.jsonl
- config_name: pairs
  data_files: pairs.jsonl
---

# modelDNA Atlas — fingerprint DB + inferred lineage graph

The data behind **[the modelDNA Atlas](https://awaisadilkhokhar.github.io/modelDNA/)**:
weight-space fingerprints of __N_MODELS__ real Hugging Face models and the
family graph reconstructed from those fingerprints alone — no model card
metadata, no self-reported `base_model` fields, just the weights.

Built with [modelDNA](https://github.com/AwaisAdilKhokhar/modelDNA)
v__MODELDNA_VERSION__ · generated __GENERATED_AT__ · refreshed automatically.
Scan your own model against this DB in the
[live Space](https://huggingface.co/spaces/AwaisAdilKhokhar/modelDNA).

## What's here

| file | rows | what |
|---|---|---|
| `models.jsonl` | __N_MODELS__ | one row per fingerprinted model (architecture, family, fingerprint metadata) |
| `edges.jsonl` | __N_EDGES__ | the inferred graph: __N_DETECTED__ detected edges (calibrated P(derived) ≥ 0.9) plus __N_DOCUMENTED__ externally documented pairs, including two decomposed merges with fitted mixture weights |
| `pairs.jsonl` | __N_PAIRS__ | every depth-compatible pairwise comparison — the full evidence matrix, not just the hits |
| `fingerprints/*.json.gz` | __N_MODELS__ | the raw fingerprints (σ-curves, norm vectors, seeded PCS samples, spectra; ~1.5 MB each) |
| `modeldna-refdb.tar.gz` | __N_REFERENCE__ models | the bases-only reference DB (__REFDB_MB__ MB) that `modeldna db pull` installs |

## Load it

```python
from datasets import load_dataset

edges = load_dataset("AwaisAdilKhokhar/modeldna-atlas", "edges", split="train")
models = load_dataset("AwaisAdilKhokhar/modeldna-atlas", "models", split="train")
```

Use the reference DB directly with the CLI:

```bash
pip install modeldna
modeldna db pull --url https://huggingface.co/datasets/AwaisAdilKhokhar/modeldna-atlas/resolve/main/modeldna-refdb.tar.gz
modeldna scan some-org/suspicious-model
```

## Columns

**`models`** — `model_id`, `family`, `group` (coarse visual grouping, never
itself a lineage claim), architecture fields (`n_layers`, `hidden_size`,
`n_heads`, `n_kv_heads`, `intermediate_size`, `vocab_size`, `n_params`,
`torch_dtype`), `revision`, `fingerprint_mode`/`fingerprint_created_at`/
`fingerprint_file`, `in_reference_db` (true for the curated base-model set),
and — where the model is a LineageBench suspect — `documented_parent` and
`lineagebench_verdict`.

**`pairs` / `edges`** (same schema; `edges` is the filtered graph) —
`model_a`, `model_b`, `p_derived` (calibrated probability that the pair is
weight-derived), the four raw signals (`sigma_r` attention σ-curve
correlation, `vector_cos` norm/bias cosine, `pcs_cos` seeded
parameter-sample cosine, `spectra_r` SVD spectrum correlation), `detected`
(`p_derived ≥ 0.9`), `documented_kind` (fine-tune, official-instruct,
merge-chain, … — from the publishing org's own documentation, with citations
in [`benchmarks/lineagebench_pairs.json`](https://github.com/AwaisAdilKhokhar/modelDNA/blob/main/benchmarks/lineagebench_pairs.json)),
`child` (direction, only where documented or decomposed), and `merge_alpha`
(fitted mixture weight from `modeldna decompose`, validated against
published mergekit configs).

## Read the graph honestly

- **Edges are calibrated probabilities of weight derivation, not
  accusations.** A detected edge means the weights are statistically
  consistent with derivation; the evidence behind it is in the four signal
  columns.
- **Absence of an edge is not evidence of independence.** Pairs with
  different layer counts are never compared (depth-changing derivations such
  as pruning or stacking are out of scope), and quantized/packed weights
  abstain by design. `edges.jsonl` carries those documented-but-abstained
  pairs with `p_derived: null`.
- **Direction is hard from weights alone.** `child` is set only where
  external documentation or a merge decomposition establishes it; detected
  edges without it are undirected.
- **Coverage is curated, not exhaustive.** __N_MODELS__ models is a validated
  seed, not the Hub. The benchmark behind the method (AUROC 1.0,
  0 false positives at p ≥ 0.9 on 13 positives / 107 hard negatives) is
  reproducible from the [repo](https://github.com/AwaisAdilKhokhar/modelDNA).

## Cite

```bibtex
@software{modeldna,
  author = {Awais Bin Adil, Muhammad},
  title = {modelDNA: weight-based lineage verification for open models},
  url = {https://github.com/AwaisAdilKhokhar/modelDNA},
  year = {2026}
}
```
