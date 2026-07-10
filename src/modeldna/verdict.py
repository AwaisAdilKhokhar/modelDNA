"""Verdict engine: evidence -> calibrated class + probability.

Every scan resolves to exactly one verdict class (PRD 5.2). The engine
prefers honest abstention (`SAME_FAMILY_UNRESOLVED`, `NO_MATCH`) over
confident error: thresholds are deliberately conservative because the
worst failure mode of this tool is a false accusation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modeldna.calibration import DEFAULT_CALIBRATOR, LogisticCalibrator
from modeldna.compare import Evidence


class VerdictClass(str, Enum):
    EXACT_COPY = "EXACT_COPY"
    QUANTIZED_COPY = "QUANTIZED_COPY"
    FINE_TUNE = "FINE_TUNE"
    SAME_LINEAGE = "SAME_LINEAGE"
    LIKELY_MERGE = "LIKELY_MERGE"
    SAME_FAMILY_UNRESOLVED = "SAME_FAMILY_UNRESOLVED"
    NO_MATCH = "NO_MATCH"
    INSUFFICIENT = "INSUFFICIENT"


DESCRIPTIONS: dict[VerdictClass, str] = {
    VerdictClass.EXACT_COPY: "weights byte/near-identical to a reference entry "
    "(possible re-upload or renamed mirror)",
    VerdictClass.QUANTIZED_COPY: "weights match a reference entry within quantization noise",
    VerdictClass.FINE_TUNE: "small parameter delta from one parent "
    "(SFT/DPO/LoRA-merged regime)",
    VerdictClass.SAME_LINEAGE: "large delta but lineage signals match one parent "
    "(continued-pretraining / heavy RL regime)",
    VerdictClass.LIKELY_MERGE: "evidence splits across two or more parents",
    VerdictClass.SAME_FAMILY_UNRESOLVED: "family-level match, exact parent ambiguous",
    VerdictClass.NO_MATCH: "no reference parent supported; consistent with independent "
    "training or an unindexed base",
    VerdictClass.INSUFFICIENT: "format or architecture not supported, or required "
    "data could not be read",
}


@dataclass
class Thresholds:
    """Decision thresholds. Conservative by default; see PRD section 9."""

    positive: float = 0.90  # P(derived) to call lineage at all
    abstain: float = 0.50  # gray zone lower bound -> SAME_FAMILY_UNRESOLVED
    fine_tune_pcs: float = 0.98  # PCS above this + positive -> FINE_TUNE
    exact_pcs: float = 0.99995
    exact_sigma: float = 0.9999
    quant_pcs: float = 0.999
    family_tie: float = 0.03  # probability gap treated as a tie within a family


@dataclass
class CandidateResult:
    candidate_id: str
    family: str
    probability: float | None  # None when not comparable
    evidence: Evidence
    ruled_out: str | None = None  # reason, when probability is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "probability": self.probability,
            "ruled_out": self.ruled_out,
            "evidence": self.evidence.to_dict(),
        }


@dataclass
class Verdict:
    verdict: VerdictClass
    probability: float | None  # probability attached to the headline verdict
    best: CandidateResult | None
    candidates: list[CandidateResult] = field(default_factory=list)  # ranked
    notes: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return DESCRIPTIONS[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "description": self.description,
            "probability": self.probability,
            "best_candidate": self.best.candidate_id if self.best else None,
            "notes": self.notes,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _score_candidate(
    ev: Evidence,
    family: str,
    calibrator: LogisticCalibrator,
) -> CandidateResult:
    if not ev.layers_match:
        return CandidateResult(
            ev.candidate_id, family, None, ev, ruled_out="layer count mismatch"
        )
    if not ev.shape_compatible:
        return CandidateResult(
            ev.candidate_id, family, None, ev, ruled_out="incompatible tensor shapes"
        )
    if not any(v is not None for v in ev.features().values()):
        return CandidateResult(
            ev.candidate_id, family, None, ev, ruled_out="no comparable signals"
        )
    p = calibrator.predict(ev.features())
    return CandidateResult(ev.candidate_id, family, p, ev)


def _classify_positive(ev: Evidence, suspect_dtype: str, candidate_dtype: str,
                       th: Thresholds) -> VerdictClass:
    pcs = ev.pcs_cos_mean
    sigma = ev.sigma_r_mean or 0.0
    if pcs is not None and pcs >= th.exact_pcs and sigma >= th.exact_sigma:
        if ev.inventory_match:
            return VerdictClass.EXACT_COPY
        return VerdictClass.QUANTIZED_COPY  # same values, different packaging
    if pcs is not None and pcs >= th.quant_pcs and suspect_dtype and candidate_dtype \
            and suspect_dtype != candidate_dtype:
        return VerdictClass.QUANTIZED_COPY
    if pcs is not None and pcs >= th.fine_tune_pcs:
        return VerdictClass.FINE_TUNE
    return VerdictClass.SAME_LINEAGE


def judge(
    evidences: list[tuple[Evidence, str]],  # (evidence, candidate family)
    suspect_dtype: str = "",
    candidate_dtypes: dict[str, str] | None = None,
    calibrator: LogisticCalibrator | None = None,
    thresholds: Thresholds | None = None,
) -> Verdict:
    """Aggregate per-candidate evidence into one scan verdict."""
    cal = calibrator or DEFAULT_CALIBRATOR
    th = thresholds or Thresholds()
    dtypes = candidate_dtypes or {}

    results = [_score_candidate(ev, fam, cal) for ev, fam in evidences]
    results.sort(key=lambda r: (r.probability is not None, r.probability or 0.0), reverse=True)

    comparable = [r for r in results if r.probability is not None]
    if not results:
        return Verdict(VerdictClass.INSUFFICIENT, None, None,
                       notes=["no reference candidates to compare against"])
    if not comparable:
        return Verdict(
            VerdictClass.NO_MATCH, None, None, results,
            notes=["no candidate had compatible structure; "
                   "the base may simply not be indexed"],
        )

    best = comparable[0]
    positives = [r for r in comparable if r.probability >= th.positive]

    if not positives:
        if best.probability >= th.abstain:
            return Verdict(
                VerdictClass.SAME_FAMILY_UNRESOLVED, best.probability, best, results,
                notes=[f"strongest candidate {best.candidate_id} sits in the "
                       "abstention band; evidence is suggestive, not conclusive"],
            )
        return Verdict(VerdictClass.NO_MATCH, best.probability, best, results)

    families = {r.family for r in positives}
    if len(positives) >= 2 and len(families) >= 2:
        note = (
            "evidence exceeds the positive threshold for parents in different "
            f"families ({', '.join(sorted(families))}); multi-parent merge is "
            "the most consistent explanation (estimate mixture weights with "
            "`modeldna decompose <target> <parent> <parent> …`)"
        )
        return Verdict(VerdictClass.LIKELY_MERGE, best.probability, best, results, [note])

    if len(positives) >= 2:
        p0, p1 = positives[0].probability, positives[1].probability
        if p0 - p1 <= th.family_tie:
            note = (
                f"{positives[0].candidate_id} and {positives[1].candidate_id} "
                f"are statistically tied (Δp={p0 - p1:.3f}); the parent is one "
                "of them but the data cannot say which"
            )
            return Verdict(
                VerdictClass.SAME_FAMILY_UNRESOLVED, p0, positives[0], results, [note]
            )

    cls = _classify_positive(
        best.evidence, suspect_dtype, dtypes.get(best.candidate_id, ""), th
    )
    verdict = Verdict(cls, best.probability, best, results)
    if cls == VerdictClass.FINE_TUNE and re.match(r"i?q\d", suspect_dtype or ""):
        # deep quants (Q4 and below) put more noise on the samples than a
        # light SFT does — the two regimes overlap and we refuse to guess
        verdict.notes.append(
            f"suspect weights are {suspect_dtype}-quantized; at this depth "
            "quantization noise and a light fine-tune are not separable from "
            "sampled weights alone, so this may equally be a quantized copy "
            f"of {best.candidate_id}"
        )
    return verdict
