"""Model-card claim extraction.

Reads what a repo *says* about its lineage — the `base_model` field in the
README front matter and explicit "trained from scratch" language — so a scan
can flag weight evidence that contradicts the stated story. The consistency
check only ever fires when the repo actually makes a claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from modeldna.io.source import ModelSource

_FROM_SCRATCH_PATTERNS = [
    r"train(?:ed)?\s+(?:entirely\s+)?from\s+scratch",
    r"pre-?train(?:ed)?\s+from\s+scratch",
    r"from-scratch\s+(?:pre-?)?train",
    r"not\s+(?:a\s+)?fine-?tun(?:e|ed)\s+of\s+any",
]
_FROM_SCRATCH_RX = re.compile("|".join(_FROM_SCRATCH_PATTERNS), re.IGNORECASE)


@dataclass
class Claims:
    has_readme: bool = False
    base_models: list[str] = field(default_factory=list)
    license: str = ""
    from_scratch: bool = False

    @property
    def makes_lineage_claim(self) -> bool:
        return bool(self.base_models) or self.from_scratch

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_readme": self.has_readme,
            "base_models": self.base_models,
            "license": self.license,
            "claims_from_scratch": self.from_scratch,
        }


def _parse_front_matter(text: str) -> dict[str, Any]:
    """Tiny YAML-subset parser for the fields we need (no yaml dependency).

    Handles `key: value`, `key: [a, b]`, and block lists under a key.
    """
    m = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}
    out: dict[str, Any] = {}
    current_key: str | None = None
    for line in m.group(1).splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        item = re.match(r"\s+-\s*(.+)", line)
        if item and current_key:
            out.setdefault(current_key, [])
            if isinstance(out[current_key], list):
                out[current_key].append(item.group(1).strip().strip("\"'"))
            continue
        kv = re.match(r"([A-Za-z0-9_-]+):\s*(.*)", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            current_key = key
            if not val:
                out[key] = []
            elif val.startswith("[") and val.endswith("]"):
                out[key] = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
            else:
                out[key] = val.strip("\"'")
    return out


def read_claims(source: ModelSource) -> Claims:
    claims = Claims()
    files = source.list_files()
    readme = next((f for f in ("README.md", "readme.md", "Readme.md") if f in files), None)
    if readme is None:
        return claims
    claims.has_readme = True
    text = source.read_text(readme)

    fm = _parse_front_matter(text)
    base = fm.get("base_model")
    if isinstance(base, str) and base:
        claims.base_models = [base]
    elif isinstance(base, list):
        claims.base_models = [b for b in base if b]
    lic = fm.get("license")
    if isinstance(lic, str):
        claims.license = lic

    body = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    if _FROM_SCRATCH_RX.search(body):
        claims.from_scratch = True
    return claims


def check_consistency(claims: Claims, verdict_dict: dict[str, Any]) -> dict[str, Any]:
    """Compare claimed lineage against the weight verdict.

    Returns {status: CONSISTENT|INCONSISTENT|NO_CLAIM|UNVERIFIED, detail}.
    Fires INCONSISTENT only on a high-confidence contradiction.
    """
    positive_classes = {"EXACT_COPY", "QUANTIZED_COPY", "FINE_TUNE", "SAME_LINEAGE",
                        "LIKELY_MERGE"}
    verdict = verdict_dict.get("verdict", "")
    best = verdict_dict.get("best_candidate") or ""
    prob = verdict_dict.get("probability")
    # SAME_FAMILY_UNRESOLVED at high confidence is still a lineage detection —
    # the ambiguity is only about *which* family member is the parent
    lineage_detected = (prob or 0) >= 0.9 and (
        verdict in positive_classes or verdict == "SAME_FAMILY_UNRESOLVED"
    )
    # candidates the evidence actually supports (for claim matching)
    supported = [
        c["candidate_id"]
        for c in verdict_dict.get("candidates", [])
        if (c.get("probability") or 0) >= 0.9
    ] or ([best] if best else [])

    if not claims.makes_lineage_claim:
        return {"status": "NO_CLAIM",
                "detail": "repo does not state a base_model or a from-scratch claim"}

    if claims.from_scratch and lineage_detected:
        return {
            "status": "INCONSISTENT",
            "detail": (
                "README claims from-scratch training but weights are "
                f"statistically consistent with derivation from {best} "
                f"(p={prob:.3f})"
            ),
        }

    if claims.base_models:
        claimed_norm = {c.lower() for c in claims.base_models}

        def matches(candidate: str) -> bool:
            cand = candidate.lower()
            name_only = cand.split("/")[-1]
            return any(
                cand == c or c.endswith("/" + name_only) or name_only == c.split("/")[-1]
                for c in claimed_norm
            )

        if lineage_detected:
            if any(matches(c) for c in supported):
                return {"status": "CONSISTENT",
                        "detail": "weight evidence supports the declared base "
                                  f"({', '.join(c for c in supported if matches(c))})"}
            return {
                "status": "INCONSISTENT",
                "detail": (
                    f"README declares base_model {claims.base_models} but weight "
                    f"evidence points to {best} (p={prob:.3f})"
                ),
            }
        return {
            "status": "UNVERIFIED",
            "detail": "declared base could not be confirmed from weights "
                      "(it may not be in the reference DB)",
        }

    return {"status": "UNVERIFIED", "detail": "no verdict overlap with the stated claim"}
