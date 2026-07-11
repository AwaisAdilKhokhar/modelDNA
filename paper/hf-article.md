<!--
Paste this into huggingface.co/new-blog (profile → New article).
Suggested title: Reading a model's DNA: verified lineage for open-weight LLMs, from 300 MB of ranged reads
Suggested cover/tags: model-merging, provenance, safetensors, forensics
The figure is hot-linked from the GitHub repo; you can also upload it
(paper/fig_slerp_tcurves.png) through the article editor instead.
-->

# Reading a model's DNA: verified lineage for open-weight LLMs, from 300 MB of ranged reads

In July 2025, an anonymous analysis claimed Huawei's Pangu Pro MoE was derived from Qwen2.5-14B, based on a 0.927 correlation between their per-layer attention std-dev curves. Huawei denied it, the repo was taken down, and the dispute ran in the international press for days — with no neutral party able to re-run the numbers. Before that it was Reflection-70B (marketed as a breakthrough fine-tune until community weight-diffing said otherwise) and Llama3-V (substantially copied from MiniCPM-Llama3-V 2.5). Each time, the investigation was ad hoc: someone with the right skills happened to spend a weekend on it.

The strange part is that the detection methods aren't the bottleneck. At least ten papers from 2024–2026 demonstrate lineage signals that survive fine-tuning, continued pretraining, merging, and quantization. What's been missing is the unglamorous part: a maintained tool that takes a repo id and returns a verdict, with a reference database, calibrated probabilities, and background distributions printed next to every score.

**[modelDNA](https://github.com/AwaisAdilKhokhar/modelDNA)** is that tool. `pip install modeldna`, then:

```
$ modeldna scan some-org/suspicious-model

  SAME_LINEAGE  99.3% likely derived from Qwen/Qwen2.5-14B

  claimed lineage  "from scratch"   [!] INCONSISTENT

  signal                                   value   unrelated background
  attention sigma-curve correlation (F1)   0.994              0.30-0.70
  norm/bias vector cosine (F2)             0.995              0.85-0.92
  sampled parameter cosine / PCS (F3)      0.694             -0.01-0.01
  SVD spectrum correlation (F4)            0.897              0.95-0.99
```

We've just published the full **[technical report (PDF)](https://github.com/AwaisAdilKhokhar/modelDNA/releases/download/report-v1.0/modeldna-tech-report.pdf)**. Here's the short version.

## Fingerprinting without downloading

The safetensors format puts a JSON header (tensor names, shapes, byte offsets) at the start of each shard, and the Hub serves byte ranges over HTTP. modelDNA reads the headers, plans every byte range the extraction needs, and fetches the plan concurrently. A 7B model fingerprints in about two minutes from roughly 100–300 MB of traffic instead of a 15 GB download — CPU only, no GPU, no cooperation from the suspect repo. GGUF quants work too: the sampled positions are dequantized with llama.cpp's reference kernels, so a Q4_K_M compares straight against the fp16 original.

Four signal families from the literature are extracted and compared as an ensemble: per-layer σ-curves of attention projections (the Pangu-dispute signal), norm/bias vector cosines, seeded sampled-parameter cosines (HuRef's PCS), and top-k singular-value spectra (invariant to permutation/rotation re-parameterization). Every reported number ships with the unrelated-pair background distribution measured on the database itself — because a bare "0.93" convinces nobody, which is the actual lesson of the Pangu dispute.

The design is shaped by one asymmetry: the worst failure mode of a lineage tool is not a missed detection but a false accusation. So the verdict engine is calibrated and abstention-first — eight verdict classes, conservative thresholds, and a hard ship gate of **zero false positives on independent same-architecture controls**. On LineageBench — 15 real Hub models whose parentage is corroborated by the publishing org's own documentation, judged against 8 candidate bases (13 positives, 107 hard negatives) — it scores AUROC 1.0, zero false positives at the reporting threshold, and 13/13 correct top-1 parent attribution.

## The fun part: reading merge recipes back out of fingerprints

One sampling-design decision turned out to matter more than expected: fingerprint sample positions are pure functions of *(seed, canonical role, layer, tensor size)* — tensor identity, never tensor content. So fingerprints of different models are element-aligned. And since every mainstream mergekit method (linear, slerp, task arithmetic, TIES/DARE) is (near-)linear per tensor, a merged model's fingerprint is the *same* linear combination of its parents' fingerprints.

That means mixture weights can be recovered from fingerprints alone, by sum-to-one constrained least squares — algebraically a regression in task-vector space, where it's well-conditioned even though the parents themselves are 0.99+ cosine-similar. Against merges with published mergekit configs as ground truth:

- **[mlabonne/NeuralPipe-7B-slerp](https://huggingface.co/mlabonne/NeuralPipe-7B-slerp)** — the per-layer fit recovers the model card's opposite attention/MLP interpolation curves at **r = 0.999** (attention) and 0.97 (MLP):

![Recovered slerp t-curves vs the published mergekit config](https://raw.githubusercontent.com/AwaisAdilKhokhar/modelDNA/main/paper/fig_slerp_tcurves.png)

- **[mlabonne/Monarch-7B](https://huggingface.co/mlabonne/Monarch-7B)** — a dare_ties merge; fitted weights **0.371 / 0.347 / 0.291** vs the published 0.36 / 0.34 / 0.30 (max error 0.011).

No weights beyond the fingerprints were downloaded for any of this. To our knowledge it's the first demonstration that merge recipes can be read back out of *sampled* fingerprints.

## Try it

- 🔬 **[Scan a model live](https://huggingface.co/spaces/AwaisAdilKhokhar/modelDNA)** — paste a repo id, get a verdict in a minute or two.
- 🧬 **[Explore the Atlas](https://awaisadilkhokhar.github.io/modelDNA/)** — an interactive family tree of 55 real Hub models reconstructed from weight evidence alone, merges drawn with their fitted mixture weights.
- 📊 **[Build on the data](https://huggingface.co/datasets/AwaisAdilKhokhar/modeldna-atlas)** — fingerprints and the inferred lineage graph, refreshed weekly; plus **[LineageBench](https://huggingface.co/datasets/AwaisAdilKhokhar/lineagebench)**, the org-documented ground-truth benchmark.
- 📄 **[Read the tech report](https://github.com/AwaisAdilKhokhar/modelDNA/releases/download/report-v1.0/modeldna-tech-report.pdf)** — fingerprint design, verdict engine, merge-decomposition math, and every benchmark number, reproducible offline from committed caches.

Known limits, stated up front: distillation is invisible to weight forensics by construction; merge *attribution* needs a candidate list; a determined retraining-scale adversary is out of scope. The output is always "weights are statistically consistent with derivation from X" — never an accusation.

*Muhammad Awais Bin Adil & Saad Aamir — Independent. Code Apache-2.0, report CC BY 4.0.*
