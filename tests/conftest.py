"""Shared test fixtures. The synthetic model factory lives in modeldna.testing
so the benchmark scripts can reuse it; re-exported here for the tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from modeldna.testing import make_tiny_llama, write_safetensors  # noqa: F401

__all__ = ["make_tiny_llama", "write_safetensors", "tiny_model"]


@pytest.fixture
def tiny_model(tmp_path: Path):
    root = tmp_path / "base"
    tensors = make_tiny_llama(root)
    return root, tensors
