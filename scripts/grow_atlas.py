"""Grow the Atlas: fingerprint trending Hub models into a committed cache.

This is the step that turns the weekly refresh from "rebuild the same graph"
into "the graph gets bigger". It asks the Hub for the currently trending
text-generation models, skips the ones already fingerprinted anywhere
(reference DB, LineageBench/mergebench caches, or a previous grow run), and
fast-fingerprints the new ones into ``benchmarks/atlas_growth_cache/`` — a
committed ReferenceDB that ``build_atlas.py`` and ``export_dataset.py`` load
alongside the other sources.

Design choices that keep the weekly cron safe and the repo bounded:

* **Fast fingerprints only.** Sampled slices, a few MB read per model; the
  growth cache stores gzipped fingerprints, never weights.
* **Every fetch is best-effort.** Gated repos, ``.bin``-only repos, depth
  weirdness, network blips — any single failure is logged and skipped, never
  fatal, so a bad trending entry can't break the refresh.
* **Bounded.** ``--limit`` caps additions per run (default 8); ``--max-total``
  caps the cache so it can't grow without limit (default 80). No pruning, so
  the graph never flaps week to week.
* **Idempotent + resumable.** Anything already cached is skipped; interrupted
  runs pick up where they left off.

Usage:
    python scripts/grow_atlas.py                 # fetch + fingerprint
    python scripts/grow_atlas.py --dry-run       # list candidates, fetch nothing
    python scripts/grow_atlas.py --limit 4 --sort downloads
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from modeldna.db.families import SEED_MODELS, guess_family
from modeldna.db.store import ReferenceDB, default_db_path
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import open_source

REPO = Path(__file__).resolve().parents[1]
BENCH_CACHE = REPO / "benchmarks" / "lineagebench_cache"
MERGE_CACHE = REPO / "benchmarks" / "mergebench_cache"
GROWTH_CACHE = REPO / "benchmarks" / "atlas_growth_cache"


def known_ids() -> set[str]:
    """Every model id already fingerprinted somewhere we load from."""
    ids = set(SEED_MODELS)  # curated bases: never re-fingerprint as "trending"
    for root in (default_db_path(), BENCH_CACHE, MERGE_CACHE, GROWTH_CACHE):
        if Path(root).exists():
            ids.update(e.model_id for e in ReferenceDB(root).entries())
    return ids


def trending_candidates(sort: str, fetch: int) -> list[str]:
    """Trending text-generation repo ids, most-trending first.

    ``sort`` returns descending (most-trending / most-downloaded first) and
    ``filter="safetensors"`` restricts to repos tagged with safetensors, which
    is what the fingerprinter can actually read — .bin-only repos are dropped
    before we waste a fetch on them.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(
        sort=sort, limit=fetch, pipeline_tag="text-generation", filter="safetensors"
    )
    return [m.id for m in models]


def grow(
    sort: str = "trendingScore",
    fetch: int = 60,
    limit: int = 8,
    max_total: int = 80,
    budget_s: float = 2400.0,
    dry_run: bool = False,
) -> dict:
    cache = ReferenceDB(GROWTH_CACHE)
    have = known_ids()
    current_total = len(cache)

    candidates = [c for c in trending_candidates(sort, fetch) if c not in have]
    print(f"{len(candidates)} trending candidates not yet fingerprinted "
          f"(cache holds {current_total}/{max_total})")

    if dry_run:
        for c in candidates[:limit]:
            print(f"  would fingerprint {c}")
        return {"candidates": candidates[:limit], "added": 0, "dry_run": True}

    added, failed, skipped_cap = [], [], 0
    started = time.monotonic()
    for repo in candidates:
        if len(added) >= limit:
            break
        if current_total + len(added) >= max_total:
            skipped_cap = len(candidates) - candidates.index(repo)
            print(f"cache at cap ({max_total}); stopping, {skipped_cap} candidate(s) left")
            break
        if time.monotonic() - started > budget_s:
            print("time budget exhausted; stopping")
            break
        print(f"fingerprinting {repo} ...")
        try:
            src = open_source(repo)
            fp = extract_fingerprint(src, mode="fast", model_id=repo)
            cache.add(fp, family=guess_family(repo),
                      meta={"source": "hub-trending", "sort": sort})
            added.append(repo)
            print(f"  ok ({src.bytes_read / 1e6:.0f} MB read) · {fp.arch.summary()}")
        except Exception as e:  # one bad trending repo must never break the refresh
            failed.append(repo)
            print(f"  skipped ({type(e).__name__}): {e}")

    print(f"\ngrew Atlas by {len(added)} model(s); {len(failed)} skipped; "
          f"cache now {len(cache)}/{max_total} · db v{cache.version}")
    return {"added": added, "failed": failed, "skipped_at_cap": skipped_cap,
            "cache_total": len(cache)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sort", default="trendingScore",
                    help="Hub sort key (trendingScore, downloads, likes).")
    ap.add_argument("--fetch", type=int, default=60,
                    help="How many trending repos to consider before filtering.")
    ap.add_argument("--limit", type=int, default=8,
                    help="Max models to fingerprint this run.")
    ap.add_argument("--max-total", type=int, default=80,
                    help="Stop adding once the growth cache reaches this size.")
    ap.add_argument("--budget-s", type=float, default=2400.0,
                    help="Wall-clock budget for fingerprinting, seconds.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be fingerprinted; fetch nothing.")
    args = ap.parse_args()

    grow(sort=args.sort, fetch=args.fetch, limit=args.limit,
         max_total=args.max_total, budget_s=args.budget_s, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
