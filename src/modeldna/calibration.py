"""Score -> probability calibration for the verdict engine.

A small logistic model over the evidence features, fittable from labeled
pairs (``modeldna``'s LineageBench workflow) and shipped with bootstrap
coefficients. Missing features (e.g. PCS skipped because shapes differ) are
imputed with their unrelated-pair background mean, so absence of a signal
never counts as evidence either way.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FEATURES = ("sigma_r", "vector_cos", "pcs_cos", "spectra_r")

#: unrelated-pair means used to impute missing features; conservative values
#: (sigma_r from the published unrelated-pair background, the rest measured
#: on the synthetic benchmark), overridden by DB background when available
DEFAULT_NEUTRAL = {
    "sigma_r": 0.3,
    "vector_cos": 0.85,
    "pcs_cos": 0.0,
    "spectra_r": 0.95,
}


@dataclass
class LogisticCalibrator:
    coef: dict[str, float]
    intercept: float
    neutral: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NEUTRAL))
    fitted_on: str = "bootstrap defaults"

    def score(self, features: dict[str, float | None]) -> float:
        z = self.intercept
        for name, w in self.coef.items():
            v = features.get(name)
            if v is None:
                v = self.neutral.get(name, 0.0)
            z += w * v
        return z

    def predict(self, features: dict[str, float | None]) -> float:
        return 1.0 / (1.0 + math.exp(-self.score(features)))

    # -- fitting ---------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        X: Sequence[dict[str, float | None]],
        y: Sequence[int],
        neutral: dict[str, float] | None = None,
        l2: float = 1e-3,
        lr: float = 0.5,
        epochs: int = 4000,
        nonnegative: bool = True,
        fitted_on: str = "custom fit",
    ) -> "LogisticCalibrator":
        neutral = neutral or dict(DEFAULT_NEUTRAL)
        mat = np.array(
            [
                [x.get(f) if x.get(f) is not None else neutral[f] for f in FEATURES]
                for x in X
            ],
            dtype=np.float64,
        )
        target = np.asarray(y, dtype=np.float64)
        w = np.zeros(len(FEATURES))
        b = 0.0
        n = len(target)
        for _ in range(epochs):
            z = mat @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = mat.T @ (p - target) / n + l2 * w
            grad_b = float(np.mean(p - target))
            w -= lr * grad_w
            b -= lr * grad_b
            if nonnegative:
                # every feature is oriented "higher = more similar", so a
                # negative weight can only ever be an overfit artifact — and
                # one an adversary could exploit
                w = np.maximum(w, 0.0)
        return cls(
            coef=dict(zip(FEATURES, map(float, w))),
            intercept=float(b),
            neutral=neutral,
            fitted_on=fitted_on,
        )

    # -- persistence -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": self.coef,
            "intercept": self.intercept,
            "neutral": self.neutral,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogisticCalibrator":
        return cls(
            coef=d["coef"],
            intercept=d["intercept"],
            neutral=d.get("neutral", dict(DEFAULT_NEUTRAL)),
            fitted_on=d.get("fitted_on", "loaded"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LogisticCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Bootstrap coefficients: non-negative fit on the synthetic pair benchmark
# (bases x {fine-tunes, heavy-noise continuations, independent same-arch
# controls}) puts nearly all weight on sigma_r and pcs_cos; vector_cos and
# spectra_r keep modest positive priors reflecting their published
# discriminative power on real models (HuRef, Ghost-in-the-Transformer).
# Refit against LineageBench pairs for production calibration.
DEFAULT_CALIBRATOR = LogisticCalibrator(
    coef={"sigma_r": 5.5, "vector_cos": 1.5, "pcs_cos": 4.5, "spectra_r": 1.0},
    intercept=-6.0,
    fitted_on="synthetic bootstrap v0.1",
)


@dataclass
class Background:
    """Unrelated-pair feature distributions, for report context.

    A bare '0.93' convinces nobody; '0.93 against an unrelated background of
    0.3-0.7' is the actual argument.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)

    def add(self, features: dict[str, float | None]) -> None:
        for k, v in features.items():
            if v is not None:
                self.samples.setdefault(k, []).append(float(v))

    def range_str(self, feature: str, lo_q: float = 10, hi_q: float = 90) -> str:
        vals = self.samples.get(feature)
        if not vals or len(vals) < 5:
            return "n/a"
        lo, hi = np.percentile(vals, [lo_q, hi_q])
        return f"{lo:.2f}–{hi:.2f}"

    def neutral_means(self) -> dict[str, float]:
        return {
            k: float(np.mean(v)) for k, v in self.samples.items() if len(v) >= 5
        }

    def to_dict(self) -> dict[str, Any]:
        return {"samples": self.samples}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Background":
        return cls(samples=d.get("samples", {}))
