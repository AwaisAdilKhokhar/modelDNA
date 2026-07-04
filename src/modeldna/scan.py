"""Scan orchestration: target -> fingerprint -> DB comparison -> verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modeldna import __version__
from modeldna.calibration import Background
from modeldna.card import Claims, check_consistency, read_claims
from modeldna.compare import Evidence, compare_fingerprints
from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import Fingerprint, extract_fingerprint
from modeldna.io.source import ModelSource, open_source
from modeldna.verdict import Verdict, VerdictClass, judge

#: CI-friendly exit codes (PRD 5.1)
EXIT_CONSISTENT = 0
EXIT_INCONSISTENT = 3
EXIT_UNKNOWN = 4


@dataclass
class ScanResult:
    target: str
    mode: str
    fingerprint: Fingerprint
    verdict: Verdict
    claims: Claims
    consistency: dict[str, Any]
    background: Background
    db_version: int
    bytes_read: int = 0
    tool_version: str = __version__
    #: fingerprint of the winning candidate (for plots), when available
    best_fingerprint: Fingerprint | None = None

    @property
    def exit_code(self) -> int:
        if self.consistency.get("status") == "INCONSISTENT":
            return EXIT_INCONSISTENT
        if self.verdict.verdict in (VerdictClass.NO_MATCH, VerdictClass.INSUFFICIENT):
            return EXIT_UNKNOWN
        return EXIT_CONSISTENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_version": self.tool_version,
            "target": self.target,
            "mode": self.mode,
            "db_version": self.db_version,
            "bytes_read": self.bytes_read,
            "arch": self.fingerprint.arch.to_dict(),
            "verdict": self.verdict.to_dict(),
            "claims": self.claims.to_dict(),
            "consistency": self.consistency,
            "background_ranges": {
                k: self.background.range_str(k)
                for k in ("sigma_r", "vector_cos", "pcs_cos", "spectra_r")
            },
            "curves": {
                "suspect_sigma": self.fingerprint.sigma_curves,
                "best_sigma": (
                    self.best_fingerprint.sigma_curves if self.best_fingerprint else {}
                ),
                "best_id": (
                    self.best_fingerprint.model_id if self.best_fingerprint else ""
                ),
            },
        }


def scan(
    target: str,
    db: ReferenceDB | None = None,
    mode: str = "fast",
    revision: str = "main",
    source: ModelSource | None = None,
) -> ScanResult:
    """Scan a model against the reference DB and return a full result."""
    db = db or ReferenceDB()
    src = source or open_source(target, revision=revision)
    fp = extract_fingerprint(src, mode=mode, model_id=target, revision=revision)
    claims = read_claims(src)

    evidences: list[tuple[Evidence, str]] = []
    dtypes: dict[str, str] = {}
    ref_fps: dict[str, Fingerprint] = {}
    for entry in db.candidates_for(fp.arch):
        # never judge a model against its own reference entry
        if entry.model_id == target:
            continue
        ref_fp = db.load_fingerprint(entry.model_id)
        ref_fps[entry.model_id] = ref_fp
        evidences.append((compare_fingerprints(fp, ref_fp), entry.family))
        dtypes[entry.model_id] = entry.torch_dtype

    verdict = judge(
        evidences,
        suspect_dtype=fp.arch.torch_dtype,
        candidate_dtypes=dtypes,
        calibrator=db.calibrator(),
    )
    if len(db) == 0:
        verdict.notes.append(
            "reference DB is empty — run `modeldna db build` or add entries "
            "with `modeldna db add`"
        )

    consistency = check_consistency(claims, verdict.to_dict())
    return ScanResult(
        target=target,
        mode=mode,
        fingerprint=fp,
        verdict=verdict,
        claims=claims,
        consistency=consistency,
        background=db.background(),
        db_version=db.version,
        bytes_read=src.bytes_read,
        best_fingerprint=(
            ref_fps.get(verdict.best.candidate_id) if verdict.best else None
        ),
    )


def compare_pair(
    target_a: str,
    target_b: str,
    mode: str = "fast",
    revision: str = "main",
) -> tuple[Fingerprint, Fingerprint, Evidence, Verdict]:
    """The hypothesis-pair workflow: is A derived from B?"""
    src_a = open_source(target_a, revision=revision)
    src_b = open_source(target_b, revision=revision)
    fp_a = extract_fingerprint(src_a, mode=mode, model_id=target_a)
    fp_b = extract_fingerprint(src_b, mode=mode, model_id=target_b)
    ev = compare_fingerprints(fp_a, fp_b)
    verdict = judge(
        [(ev, target_b)],
        suspect_dtype=fp_a.arch.torch_dtype,
        candidate_dtypes={target_b: fp_b.arch.torch_dtype},
    )
    return fp_a, fp_b, ev, verdict
