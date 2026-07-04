"""Numerical primitives shared by the fingerprint extractors and comparators."""

from __future__ import annotations

import numpy as np


def fnv1a(*parts: str | int) -> int:
    """Stable 64-bit hash for deriving per-tensor sampling seeds.

    Must never change: sample positions in every published fingerprint
    depend on it.
    """
    h = 0xCBF29CE484222325
    for part in parts:
        for b in str(part).encode() + b"\x1f":  # separator so (1,"x") != ("1x",)
            h ^= b
            h = (h * 0x100000001B3) & ((1 << 64) - 1)
    return h


def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either side is degenerate."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def randomized_svals(
    mat: np.ndarray, k: int, seed: int = 0, oversample: int = 10, n_iter: int = 3
) -> np.ndarray:
    """Top-k singular values via randomized subspace iteration (Halko et al.).

    A handful of matmuls and a small exact SVD instead of a full
    decomposition — what makes full-mode spectra tractable on CPU for
    4096x4096 matrices. Power iterations keep the estimate tight even for
    slowly decaying spectra.
    """
    mat = np.asarray(mat, dtype=np.float32)
    m, n = mat.shape
    if min(m, n) <= 2 * (k + oversample):
        return np.linalg.svd(mat, compute_uv=False)[:k].astype(np.float64)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, k + oversample)).astype(np.float32)
    q, _ = np.linalg.qr(mat @ omega)
    for _ in range(n_iter):
        q, _ = np.linalg.qr(mat.T @ q)
        q, _ = np.linalg.qr(mat @ q)
    b = q.T @ mat
    s = np.linalg.svd(b, compute_uv=False)
    return s[:k].astype(np.float64)


def log_spectrum_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """L2 distance between log-spectra, length-normalized."""
    n = min(len(s1), len(s2))
    if n == 0:
        return float("inf")
    a = np.log(np.maximum(np.asarray(s1[:n], dtype=np.float64), 1e-12))
    b = np.log(np.maximum(np.asarray(s2[:n], dtype=np.float64), 1e-12))
    return float(np.linalg.norm(a - b) / np.sqrt(n))
