"""Pairwise evidence between two fingerprints.

Produces one `Evidence` record per (suspect, candidate) pair: every signal
that could be computed, every signal that couldn't (and why), and the scalar
feature vector the verdict engine consumes. No thresholds live here — this
module reports numbers, the verdict engine interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modeldna.fingerprint.extract import Fingerprint
from modeldna.fingerprint.methods import cosine, log_spectrum_distance, pearson, zscore


@dataclass
class Evidence:
    suspect_id: str
    candidate_id: str

    shape_compatible: bool = False
    layers_match: bool = False
    inventory_match: bool = False
    tokenizer_match: bool = False
    seed_match: bool = True

    # per-role detail (kept for reports and `explain`)
    sigma_r: dict[str, float] = field(default_factory=dict)
    vector_cos: dict[str, float] = field(default_factory=dict)
    vector_norm_r: dict[str, float] = field(default_factory=dict)
    pcs_cos: dict[str, float] = field(default_factory=dict)
    spectra_r: dict[str, float] = field(default_factory=dict)
    spectra_logdist: dict[str, float] = field(default_factory=dict)
    spectra_kind: str = ""  # "exact" or "sketch"

    notes: list[str] = field(default_factory=list)

    # -- aggregates -----------------------------------------------------------

    @staticmethod
    def _mean(d: dict[str, float]) -> float | None:
        return float(np.mean(list(d.values()))) if d else None

    @property
    def sigma_r_mean(self) -> float | None:
        return self._mean(self.sigma_r)

    @property
    def vector_cos_mean(self) -> float | None:
        return self._mean(self.vector_cos)

    @property
    def pcs_cos_mean(self) -> float | None:
        return self._mean(self.pcs_cos)

    @property
    def spectra_r_mean(self) -> float | None:
        return self._mean(self.spectra_r)

    def features(self) -> dict[str, float | None]:
        """Scalar feature vector for the verdict engine."""
        return {
            "sigma_r": self.sigma_r_mean,
            "vector_cos": self.vector_cos_mean,
            "pcs_cos": self.pcs_cos_mean,
            "spectra_r": self.spectra_r_mean,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspect_id": self.suspect_id,
            "candidate_id": self.candidate_id,
            "shape_compatible": self.shape_compatible,
            "layers_match": self.layers_match,
            "inventory_match": self.inventory_match,
            "tokenizer_match": self.tokenizer_match,
            "sigma_r": self.sigma_r,
            "sigma_r_mean": self.sigma_r_mean,
            "vector_cos": self.vector_cos,
            "vector_norm_r": self.vector_norm_r,
            "vector_cos_mean": self.vector_cos_mean,
            "pcs_cos": self.pcs_cos,
            "pcs_cos_mean": self.pcs_cos_mean,
            "spectra_r": self.spectra_r,
            "spectra_logdist": self.spectra_logdist,
            "spectra_r_mean": self.spectra_r_mean,
            "spectra_kind": self.spectra_kind,
            "notes": self.notes,
        }


def _shared_roles(a: dict, b: dict) -> list[str]:
    return sorted(set(a) & set(b))


def compare_fingerprints(suspect: Fingerprint, candidate: Fingerprint) -> Evidence:
    ev = Evidence(suspect_id=suspect.model_id, candidate_id=candidate.model_id)

    sa, sc = suspect.arch, candidate.arch
    ev.shape_compatible = sa.shape_compatible(sc)
    ev.layers_match = sa.n_layers == sc.n_layers
    ev.inventory_match = bool(sa.inventory_hash) and sa.inventory_hash == sc.inventory_hash
    ev.tokenizer_match = bool(sa.tokenizer_hash) and sa.tokenizer_hash == sc.tokenizer_hash
    ev.seed_match = suspect.seed == candidate.seed

    if not ev.layers_match:
        ev.notes.append(
            f"layer count differs ({sa.n_layers} vs {sc.n_layers}); "
            "per-layer signals skipped (depth surgery needs alignment search, v0.3)"
        )
        return ev
    if not ev.shape_compatible:
        ev.notes.append(
            "tensor shapes differ; only shape-independent signals compared"
        )

    _compare_sigma(ev, suspect, candidate)
    _compare_vectors(ev, suspect, candidate)
    if ev.shape_compatible:
        if ev.seed_match:
            _compare_pcs(ev, suspect, candidate)
        else:
            ev.notes.append(
                f"sampling seeds differ ({suspect.seed} vs {candidate.seed}); "
                "PCS not comparable"
            )
        _compare_spectra(ev, suspect, candidate)
    return ev


def _compare_sigma(ev: Evidence, a: Fingerprint, b: Fingerprint) -> None:
    for role in _shared_roles(a.sigma_curves, b.sigma_curves):
        ca, cb = np.array(a.sigma_curves[role]), np.array(b.sigma_curves[role])
        if len(ca) != len(cb) or len(ca) < 3:
            continue
        ev.sigma_r[role] = pearson(zscore(ca), zscore(cb))


def _compare_vectors(ev: Evidence, a: Fingerprint, b: Fingerprint) -> None:
    for role in _shared_roles(a.vector_samples, b.vector_samples):
        va, vb = np.array(a.vector_samples[role]), np.array(b.vector_samples[role])
        if va.size != vb.size:
            continue
        ev.vector_cos[role] = cosine(va, vb)
        na, nb = np.array(a.vector_norms[role]), np.array(b.vector_norms[role])
        if na.size == nb.size and na.size >= 3:
            ev.vector_norm_r[role] = pearson(zscore(na), zscore(nb))


def _compare_pcs(ev: Evidence, a: Fingerprint, b: Fingerprint) -> None:
    for role in _shared_roles(a.pcs_samples, b.pcs_samples):
        va, vb = np.array(a.pcs_samples[role]), np.array(b.pcs_samples[role])
        if va.size != vb.size:
            ev.notes.append(f"PCS sample size mismatch for {role}; skipped")
            continue
        ev.pcs_cos[role] = cosine(va, vb)


def _compare_spectra(ev: Evidence, a: Fingerprint, b: Fingerprint) -> None:
    # exact spectra when both sides have them, sketches otherwise
    if a.spectra_exact and b.spectra_exact:
        sa, sb, kind = a.spectra_exact, b.spectra_exact, "exact"
    else:
        sa, sb, kind = a.spectra_sketch, b.spectra_sketch, "sketch"
    ev.spectra_kind = kind
    for role in _shared_roles(sa, sb):
        rs, ds = [], []
        for layer in sorted(set(sa[role]) & set(sb[role]), key=int):
            s1, s2 = np.array(sa[role][layer]), np.array(sb[role][layer])
            n = min(s1.size, s2.size)
            if n < 4:
                continue
            rs.append(pearson(np.log(np.maximum(s1[:n], 1e-12)),
                              np.log(np.maximum(s2[:n], 1e-12))))
            ds.append(log_spectrum_distance(s1, s2))
        if rs:
            ev.spectra_r[role] = float(np.mean(rs))
            ev.spectra_logdist[role] = float(np.mean(ds))
