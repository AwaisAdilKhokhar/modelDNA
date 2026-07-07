---
title: modelDNA — lineage scanner
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Which base model does it really descend from?
---

# modelDNA — live lineage scanner

Paste a Hub repo id and modelDNA fingerprints the model **from its weights
alone** — a few hundred MB of sampled slices, never the full checkpoint —
and reports, with a calibrated probability, which indexed base model it
descends from. When the evidence doesn't single out a parent, it abstains
instead of guessing: the worst failure mode of a tool like this is a false
accusation.

The reference DB of base-model fingerprints is pulled at startup from the
[modelDNA GitHub releases](https://github.com/AwaisAdilKhokhar/modelDNA/releases);
each scan then only reads the suspect model.

Source, method documentation, CLI, and the interactive family-tree Atlas:
**[github.com/AwaisAdilKhokhar/modelDNA](https://github.com/AwaisAdilKhokhar/modelDNA)**.

This Space is deployed from the `space/` directory of that repository —
edit there, not here.
