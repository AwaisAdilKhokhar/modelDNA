"""Schema checks for benchmarks/lineagebench_pairs.json. No network access --
just catches manifest typos before a real_lineagebench.py run burns Hub
traffic on a broken entry."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = (
    Path(__file__).parent.parent / "benchmarks" / "lineagebench_pairs.json"
)
VALID_KINDS = {
    "fine-tune",
    "continued-pretrain",
    "official-instruct",
    "merge-chain",
    "depth-upscale",
    "quantized-copy",
}
REQUIRED_PAIR_FIELDS = {"suspect", "parent", "kind", "in_positive_pool", "citation"}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_loads():
    m = _manifest()
    assert m["parents"]
    assert m["pairs"]


def test_parents_are_unique_repo_ids():
    m = _manifest()
    parents = m["parents"]
    assert len(parents) == len(set(parents))
    for p in parents:
        assert "/" in p, f"{p!r} doesn't look like an org/name Hub repo id"


def test_pairs_well_formed():
    m = _manifest()
    parents = set(m["parents"])
    seen_suspects = set()
    for pair in m["pairs"]:
        missing = REQUIRED_PAIR_FIELDS - pair.keys()
        assert not missing, f"pair {pair.get('suspect')!r} missing fields: {missing}"
        assert pair["kind"] in VALID_KINDS, f"unknown kind {pair['kind']!r}"
        assert pair["parent"] in parents, (
            f"{pair['suspect']!r} claims parent {pair['parent']!r}, "
            "which isn't in the parents list"
        )
        assert isinstance(pair["in_positive_pool"], bool)
        assert pair["citation"].strip(), f"{pair['suspect']!r} has an empty citation"
        assert pair["suspect"] not in seen_suspects, f"duplicate suspect {pair['suspect']!r}"
        seen_suspects.add(pair["suspect"])


def test_limitation_kinds_excluded_from_positive_pool():
    """depth-upscale and quantized-copy are documented structural limitations
    (layer-count mismatch / packed quantization) -- they must never silently
    count toward the AUROC ship gate."""
    m = _manifest()
    for pair in m["pairs"]:
        if pair["kind"] in ("depth-upscale", "quantized-copy"):
            assert pair["in_positive_pool"] is False
