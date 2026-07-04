# modeldna

**A 23andMe for open-weight models.** `modeldna` fingerprints an open-weight LLM
from its weights alone and tells you, with a calibrated probability, which base
model it descends from — regardless of what the README claims.

```
$ modeldna scan some-org/suspicious-model

  Verdict: 94.2% likely derived from Qwen/Qwen2.5-14B
  Evidence: attention σ-curve correlation 0.93 (unrelated baseline: 0.3–0.7)
            norm-vector cosine 0.991 · SVD spectrum match 0.96
  Claimed lineage in README: "trained from scratch"   ⚠ INCONSISTENT
```

Work in progress — see `modeldna_prd.md` for the full product spec.

## License

Apache-2.0
