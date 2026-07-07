"""The prefetch fast path must be invisible: same bytes, same fingerprints.

These tests pin the Phase-A guarantees: (1) a fingerprint extracted through
the planned/prefetched path is bit-identical to the lazy path, so every
stored fingerprint stays valid; (2) after prefetch the extraction issues no
further reads against the source; (3) an incomplete plan degrades to direct
reads, never to wrong data; (4) HubSource resolves the CDN redirect once,
re-resolves on signature expiry, and never sends the Hub token to the CDN.
"""

import numpy as np
import pytest

from conftest import make_tiny_llama, write_safetensors
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import HubSource, LocalSource, SourceError
from modeldna.io.weights import WeightIndex, _coalesce


class CountingSource(LocalSource):
    """Local files pretending to be remote, with read accounting."""

    is_remote = True

    def __init__(self, path):
        super().__init__(path)
        self.direct_reads = 0
        self.batch_calls = 0
        self._in_batch = False

    def read_range(self, filename, start, end):
        if not self._in_batch:
            self.direct_reads += 1
        return super().read_range(filename, start, end)

    def read_ranges(self, filename, ranges):
        self.batch_calls += 1
        self._in_batch = True
        try:
            return super().read_ranges(filename, ranges)
        finally:
            self._in_batch = False


@pytest.mark.parametrize("sharded", [False, True])
def test_prefetched_fingerprint_bit_identical(tmp_path, sharded):
    root = tmp_path / "m"
    make_tiny_llama(root, sharded=sharded)
    lazy = extract_fingerprint(LocalSource(root), mode="fast", model_id="m").to_dict()
    fast = extract_fingerprint(CountingSource(root), mode="fast", model_id="m").to_dict()
    for volatile in ("created_at", "bytes_read"):
        lazy.pop(volatile), fast.pop(volatile)
    assert fast == lazy


def test_extraction_reads_only_through_prefetch(tmp_path):
    from modeldna.fingerprint.extract import FingerprintExtractor

    root = tmp_path / "m"
    make_tiny_llama(root, sharded=True)
    src = CountingSource(root)
    ex = FingerprintExtractor(src, mode="fast")  # headers + config read here
    src.direct_reads = 0
    ex.run()
    assert src.batch_calls >= 1  # one concurrent sweep per shard
    assert src.direct_reads == 0  # the plan covered every byte the run needed


def test_plan_sample_matches_read_sample(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(600, 700)).astype(np.float32)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})

    expected = WeightIndex(LocalSource(tmp_path)).read_sample(
        "w", seed=42, n_blocks=16, block_len=1024
    )

    src = CountingSource(tmp_path)
    idx = WeightIndex(src)
    t = idx.info("w")
    plan = idx.plan_sample("w", seed=42, n_blocks=16, block_len=1024)
    assert len(plan) > 1  # exercises the block-sampled path, not whole-tensor
    idx.prefetch((t.shard, s, e) for s, e in plan)
    src.direct_reads = 0
    got = idx.read_sample("w", seed=42, n_blocks=16, block_len=1024)
    np.testing.assert_array_equal(got, expected)
    assert src.direct_reads == 0


def test_reads_outside_prefetch_fall_back(tmp_path):
    arr = np.arange(10_000, dtype=np.float32)
    write_safetensors(tmp_path / "model.safetensors", {"w": arr})
    src = CountingSource(tmp_path)
    idx = WeightIndex(src)
    t = idx.info("w")
    idx.prefetch([(t.shard, t.start, t.start + 100)])  # covers almost nothing
    np.testing.assert_array_equal(idx.read_tensor("w"), arr)
    assert src.direct_reads > 0  # fell back to the source, correctly


def test_coalesce_merges_and_sorts():
    assert _coalesce([(100, 200), (0, 50), (210, 300)], gap=16) == [(0, 50), (100, 300)]
    assert _coalesce([(0, 10), (40, 50)], gap=16) == [(0, 10), (40, 50)]


# -- HubSource HTTP behavior --------------------------------------------------


class FakeResponse:
    def __init__(self, status, headers=None, content=b""):
        self.status_code = status
        self.headers = headers or {}
        self.content = content


class FakeHub:
    """Stands in for requests.Session: resolve endpoint + expiring CDN URLs."""

    def __init__(self, blob):
        self.blob = blob
        self.calls = []
        self.signature = 1
        self.expire_remaining = None  # int -> that many GETs before a 403

    def request(self, method, url, headers=None, allow_redirects=True, timeout=None):
        self.calls.append((method.upper(), url, dict(headers or {})))
        if "/resolve/" in url:
            assert not allow_redirects
            return FakeResponse(
                302, {"location": f"https://cdn.example/f?sig={self.signature}"}
            )
        if url.startswith("https://cdn.example/"):
            if f"sig={self.signature}" not in url:
                return FakeResponse(403)
            if self.expire_remaining is not None:
                if self.expire_remaining == 0:
                    self.signature += 1  # old URL now stale
                    self.expire_remaining = None
                    return FakeResponse(403)
                self.expire_remaining -= 1
            a, b = headers["Range"].removeprefix("bytes=").split("-")
            return FakeResponse(206, {}, self.blob[int(a) : int(b) + 1])
        return FakeResponse(404)


def _hub_source(fake):
    src = HubSource("org/name", token="hf_secret")
    src._session = fake
    return src


def test_hubsource_resolves_redirect_once():
    fake = FakeHub(bytes(range(256)))
    src = _hub_source(fake)
    assert src.read_range("model.safetensors", 5, 10) == bytes([5, 6, 7, 8, 9])
    assert src.read_range("model.safetensors", 0, 3) == bytes([0, 1, 2])
    resolves = [c for c in fake.calls if "/resolve/" in c[1]]
    assert len(resolves) == 1
    assert src.bytes_read == 8


def test_hubsource_token_never_sent_to_cdn():
    fake = FakeHub(bytes(256))
    src = _hub_source(fake)
    src.read_range("model.safetensors", 0, 16)
    for method, url, headers in fake.calls:
        auth = {k.lower(): v for k, v in headers.items()}.get("authorization")
        if url.startswith("https://cdn.example/"):
            assert auth is None
        else:
            assert auth == "Bearer hf_secret"


def test_hubsource_reresolves_expired_url():
    fake = FakeHub(bytes(range(256)))
    src = _hub_source(fake)
    assert src.read_range("model.safetensors", 0, 4) == bytes([0, 1, 2, 3])
    fake.expire_remaining = 0  # next GET against the old signature 403s
    assert src.read_range("model.safetensors", 4, 8) == bytes([4, 5, 6, 7])
    assert len([c for c in fake.calls if "/resolve/" in c[1]]) == 2


def test_hubsource_retries_mid_body_break(monkeypatch):
    # a connection that dies while reading the body (IncompleteRead ->
    # ChunkedEncodingError) must be retried like any transport failure
    import requests

    import modeldna.io.source as source_mod

    monkeypatch.setattr(source_mod.time, "sleep", lambda _: None)

    class FlakyHub(FakeHub):
        def __init__(self, blob, fail_first_n):
            super().__init__(blob)
            self.fail_remaining = fail_first_n

        def request(self, method, url, headers=None, allow_redirects=True, timeout=None):
            if url.startswith("https://cdn.example/") and self.fail_remaining > 0:
                self.fail_remaining -= 1
                raise requests.exceptions.ChunkedEncodingError("Connection broken")
            return super().request(method, url, headers=headers,
                                   allow_redirects=allow_redirects, timeout=timeout)

    src = _hub_source(FlakyHub(bytes(range(256)), fail_first_n=2))
    assert src.read_range("model.safetensors", 5, 10) == bytes([5, 6, 7, 8, 9])


def test_hubsource_retries_then_fails(monkeypatch):
    import modeldna.io.source as source_mod

    sleeps = []
    monkeypatch.setattr(source_mod.time, "sleep", sleeps.append)

    class AlwaysBusy(FakeHub):
        def request(self, method, url, headers=None, allow_redirects=True, timeout=None):
            return FakeResponse(503)

    src = _hub_source(AlwaysBusy(b""))
    with pytest.raises(SourceError, match="503"):
        src.read_range("model.safetensors", 0, 4)
    assert len(sleeps) == HubSource._MAX_ATTEMPTS - 1


def test_hubsource_read_ranges_order_preserved():
    fake = FakeHub(bytes(range(256)))
    src = _hub_source(fake)
    src.concurrency = 4
    ranges = [(200, 210), (0, 5), (50, 60), (5, 10)]
    got = src.read_ranges("model.safetensors", ranges)
    assert got == [bytes(range(s, e)) for s, e in ranges]
