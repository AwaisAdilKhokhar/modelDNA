# modelDNA

**A 23andMe for open-weight models.** modelDNA fingerprints an LLM from its
weights alone and tells you, with a calibrated probability, which base model
it descends from — regardless of what the README claims.

```
$ modeldna scan some-org/suspicious-model

  SAME_LINEAGE  99.3% likely derived from Qwen/Qwen2.5-14B

  claimed lineage  "from scratch"   [!] INCONSISTENT
  README claims from-scratch training but weights are statistically
  consistent with derivation from Qwen/Qwen2.5-14B (p=0.993)

  signal                                   value   unrelated background
  attention sigma-curve correlation (F1)   0.994              0.30-0.70
  norm/bias vector cosine (F2)             0.995              0.85-0.92
  sampled parameter cosine / PCS (F3)      0.694             -0.01-0.01
  SVD spectrum correlation (F4)            0.897              0.95-0.99
```

Hugging Face's model lineage graph is self-reported: the `base_model`
metadata field is optional and unverified, over 60% of Hub models have no
documented parentage at all, and every laundering controversy so far
(Pangu/Qwen 2025, Reflection-70B 2024, Llama3-V 2024) was investigated ad hoc,
by hand. The detection methods are published and validated — modelDNA
packages them into one command.

**[🧬 Explore the Atlas](https://awaisadilkhokhar.github.io/modelDNA/)** — an
interactive family tree of 48 real Hub models reconstructed from weight
evidence alone: hover any edge for the σ-curve, sample-cosine, and spectral
evidence behind it. Regenerate it yourself with `python scripts/build_atlas.py`.

**[🔬 Scan a model live](https://huggingface.co/spaces/AwaisAdilKhokhar/modelDNA)** —
paste a Hub repo id into the hosted scanner and get a verdict in a minute or
two, no install needed.

## Install

```
pip install modeldna
```

Python 3.10+. CPU only — no GPU or torch required.

## Quick start

```bash
# pull the prebuilt reference DB of known bases (~57 MB, seconds)
modeldna db pull

# …or fingerprint the seed list from the Hub yourself
modeldna db build --limit 10   # quick smoke test, a few minutes
modeldna db build              # full seed list — resumable if interrupted

# scan a model against the DB
modeldna scan mistralai/Mistral-7B-Instruct-v0.3

# test a specific hypothesis pair (the Pangu/Qwen workflow)
modeldna compare suspect-org/model Qwen/Qwen2.5-14B

# machine-readable output, HTML evidence report
modeldna scan org/model --json
modeldna scan org/model --report report.html

# export a fingerprint without judging it
modeldna fingerprint org/model -o fp.json

# raw numbers behind the last verdict
modeldna explain org/model
```

Local model directories work anywhere a repo id does.

## How it works

The safetensors format stores a JSON header (tensor names, shapes, byte
offsets) at the start of each shard, and the Hub serves byte ranges over
HTTP. modelDNA exploits this to fingerprint a 7B model from roughly
100–300 MB of traffic instead of 15 GB:

- **Tier 0 — structure (kilobytes).** `config.json`, tokenizer hash, and the
  exact tensor inventory from shard headers. Filters the reference DB down to
  shape-compatible candidates and catches renamed mirrors outright.
- **Tier 1 — weight statistics (fast mode, ~100–300 MB).** All 1-D tensors
  (norm gains, attention biases) in full, plus seeded deterministic slices of
  every attention/MLP projection.

Four signal families are extracted and compared, following the published
literature:

| signal | method | after | survives |
|---|---|---|---|
| F1 | per-layer σ-curves of Q/K/V/O projections | arXiv:2507.03014 | SFT, continued pretraining, LoRA-merge |
| F2 | norm-gain / bias vector cosines | arXiv:2507.03014 | heavy training |
| F3 | seeded sampled-parameter cosine (PCS) | HuRef, arXiv:2312.04828 | SFT; the cheapest decisive test |
| F4 | top-k singular-value spectra | arXiv:2511.06390 | permutation / rotation re-parameterization |

Sample positions are pure functions of (seed, canonical role, layer, tensor
size), so any two models with matching shapes are sampled at identical
element positions — the reference DB stores only the samples, never full
weights, which keeps it in the single-digit MB per model.

A logistic model over the evidence vector produces the calibrated
probability, and every reported number ships with the unrelated-pair
background distribution measured on the DB itself, because a bare "0.93"
convinces nobody.

### Verdict classes

Every scan resolves to exactly one class:

| class | meaning |
|---|---|
| `EXACT_COPY` | weights byte/near-identical to a DB entry (re-upload or renamed mirror) |
| `QUANTIZED_COPY` | matches a DB entry within quantization noise |
| `FINE_TUNE` | small parameter delta from one parent (SFT/DPO/LoRA-merged) |
| `SAME_LINEAGE` | large delta but lineage signals match one parent (continued-pretrain regime) |
| `LIKELY_MERGE` | evidence splits across ≥2 parents |
| `SAME_FAMILY_UNRESOLVED` | family-level match, exact parent ambiguous |
| `NO_MATCH` | no DB parent supported; consistent with independent training or an unindexed base |
| `INSUFFICIENT` | unsupported format/architecture, or data could not be read |

The tool prefers honest abstention over confident error: thresholds are
conservative, and the `README`-vs-weights consistency flag fires only when
the repo *makes a claim* that the weight evidence contradicts at high
confidence.

### Exit codes (CI-friendly)

- `0` — weight evidence consistent with the repo's claim (or no claim made)
- `3` — weight evidence contradicts the claimed lineage
- `4` — lineage unknown (no DB match, or unsupported format)

## Reference DB

The fastest way to a working DB is to pull a prebuilt one — fingerprints
are small (~55 MB for the full seed list) and derived from public weights:

```bash
modeldna db pull                 # seconds; then a scan only fetches your suspect
```

`db pull` takes the latest published archive from this repo's GitHub
releases (override with `--url` or `MODELDNA_DB_URL`; archives are produced
by `modeldna db export` / the `refdb-release` workflow). To build it
yourself instead:

`modeldna db build` fingerprints a seed list of widely-forked bases (Llama,
Qwen, Mistral, Gemma, Phi, DeepSeek, Yi, Falcon, OLMo, …) plus deliberate
hard negatives — OpenLLaMA (same shapes as Llama, independently trained) and
Pythia seed siblings. Gated repos (Llama, Gemma) need a Hugging Face token
with the license accepted on each gated org (`huggingface-cli login` or
`HF_TOKEN`) — a read-only token is sufficient, no write scope needed.

The seed list has been run against the live Hub: **37 of 41 entries
indexed**, ~55 MB total. The 4 that don't index
(`deepseek-ai/deepseek-llm-7b-base`, `internlm/internlm2-7b`,
`openbmb/MiniCPM-2B-sft-bf16`, `openlm-research/open_llama_7b`) ship only
PyTorch `.bin` checkpoints on the Hub, no safetensors — a hard limitation of
the current safetensors-only I/O layer, not a transient failure. Fast mode
plans every byte range it needs up front and fetches them concurrently
(measured: a 7B model in ~2 min on an ordinary connection), so a full build
is a coffee break, not an overnight job. If your connection drops mid-run,
just rerun — `db build` skips already-indexed entries automatically. Two env
knobs if you need them: `MODELDNA_HTTP_CONCURRENCY` (parallel range reads,
default 32) and `MODELDNA_COALESCE_GAP` (merge nearby ranges into one
request, default 64 KiB — raise it on high-latency, high-bandwidth links).

```bash
modeldna db list                 # what's indexed
modeldna db info Qwen/Qwen2.5-7B
modeldna db add org/new-base --family my-family
modeldna db remove org/old-base
modeldna db export refdb.tar.gz  # package for publishing / air-gapped machines
modeldna db pull --url refdb.tar.gz --overwrite
```

There is also a manual `fingerprint` GitHub Actions workflow: give it Hub
repo ids and it extracts fingerprints on a runner sitting next to the HF
CDN (seconds per model) and uploads them as artifacts — useful for growing
the DB or the benchmark without fetching through your own connection.

The DB lives in `~/.modeldna/db` (override with `MODELDNA_DB`). Entries are
gzipped JSON fingerprints plus a manifest — no weights are ever stored.

## Benchmark

`benchmarks/synthetic_bench.py` runs the evaluation harness on generated
models (independent bases vs fine-tunes, heavy continuations, and fp16
re-casts) and enforces the ship gates in CI: AUROC ≥ 0.99, TPR ≥ 0.95 at
FPR ≤ 1%, and **zero false positives on independent same-architecture
controls at the reporting threshold** — that last gate is the
defamation-risk surface and is non-negotiable. Current results: AUROC 1.0,
0 false positives, max calibration gap 0.045.

**Real-model LineageBench** (`benchmarks/real_lineagebench.py`) runs the
same gates on real Hub weights: 15 suspect models across 6 families whose
parentage is corroborated by the publishing org's own documentation
(fine-tunes, continued-pretrains, official instructs, a merge chain, a
depth-upscale, a GPTQ quantized copy), each judged against all 8 candidate
parents by the actual verdict engine, with 107 wrong-parent and
cross-family comparisons as hard negatives. Results (2026-07-07):

- **AUROC 1.0 · TPR@FPR1% 1.0 · 0 false positives at the 0.9 threshold** —
  ship gates PASS. Weakest positive scored 0.9426 (gemma-2b-it); strongest
  negative 0.6596: clean separation.
- **Top-1 parent attribution: 13/13** on the gate pool — every derivative
  traced to the right base among 8 candidates.
- The two by-design limitation cases behaved exactly as documented: SOLAR
  (depth-upscale) → honest `NO_MATCH` abstention; GPTQ (quantized-copy) →
  abstained on packed int4 tensors while still ranking the true parent
  first.
- One caveat, reported not hidden: the merge-chain case (AlphaMonarch)
  resolved to the correct family as `SAME_FAMILY_UNRESOLVED` rather than
  `LIKELY_MERGE` — consistent with the stated v0.1 limitation that merges
  are flagged, not attributed.

The ground truth manifest (`benchmarks/lineagebench_pairs.json`), the
fingerprint cache, and full per-pair results
(`benchmarks/lineagebench_results.json`) are committed, so
`python benchmarks/real_lineagebench.py --no-fetch` reproduces the numbers
without touching the Hub.

## Limitations — read before citing a verdict

- **Statistical evidence, not proof.** Output is always "weights are
  statistically consistent with derivation from X", never an accusation.
- **Distillation is invisible** to weight forensics by construction; every
  weight-space method fails on it. Behavioral methods are on the roadmap and
  will be labeled experimental.
- **Merges are flagged (`LIKELY_MERGE`), not attributed.** Multi-parent
  decomposition lands in v0.2.
- **Adversarial robustness is bounded.** Deliberate permutation/rotation
  re-parameterization defeats direct-similarity signals (F1–F3); spectral
  invariants (F4) are the counter, and the report surfaces "low direct
  similarity, high invariant similarity" as an anomaly pattern that is
  itself informative. A determined, retraining-scale adversary is out of
  scope and stated so in every report.
- Depth-upscaled / layer-surgery derivatives are detected as shape mismatches
  and reported, not resolved (alignment search planned for v0.3).
- **Safetensors only.** Repos that ship exclusively PyTorch `.bin` /
  `.pt` checkpoints (a handful of older or research releases —
  `deepseek-llm-7b-base`, `internlm2-7b`, `open_llama_7b`, `MiniCPM-2B-sft`
  among the seed list) can't be fingerprinted at all until `.bin` decoding is
  added.

## Library use

```python
from modeldna.scan import scan, compare_pair
from modeldna.db.store import ReferenceDB

res = scan("org/model", db=ReferenceDB())
print(res.verdict.verdict, res.verdict.probability)
print(res.to_dict())

fp_a, fp_b, evidence, verdict = compare_pair("org/suspect", "org/alleged-parent")
print(evidence.sigma_r, evidence.pcs_cos_mean)
```

## Development

```bash
git clone https://github.com/AwaisAdilKhokhar/modelDNA
cd modelDNA
pip install -e .[dev]
pytest -q
ruff check src tests benchmarks
python benchmarks/synthetic_bench.py
```

## References

The methods productized here come from, among others:

- HonestAGI, *Intrinsic Fingerprint of LLMs* (arXiv:2507.03014) — attention σ-curves
- Zeng et al., *HuRef* (NeurIPS 2024, arXiv:2312.04828) — PCS/ICS comparators
- *Ghost in the Transformer* (arXiv:2511.06390) — invariant spectral signatures
- Horwitz et al., *Model Tree Heritage Recovery* (arXiv:2405.18432) and
  *Model Atlas* (arXiv:2503.10633) — population-scale lineage recovery
- *Are Robust LLM Fingerprints Adversarially Robust?* (arXiv:2509.26598) — the threat model

See the PRD (`modeldna_prd.md`) for the full research review and roadmap.

## License

Apache-2.0
