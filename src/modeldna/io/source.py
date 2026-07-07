"""Uniform byte-level access to a model, whether it lives on disk or the Hub.

The whole tool is built on top of two primitives: list the files in a repo,
and read an arbitrary byte range from one of them. Locally that's a seek;
on the Hub it's an HTTP Range request against the resolve endpoint, which
the CDN serves without us ever downloading the shard.
"""

from __future__ import annotations

import os
import random
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urljoin


class SourceError(RuntimeError):
    """Raised when a model source can't be listed or read."""


class ModelSource(ABC):
    """Byte-range access to the files of one model repo."""

    #: identifier used in reports (repo id or local path)
    name: str

    #: running total of payload bytes fetched, for the traffic report
    bytes_read: int = 0

    #: True when reads cross the network — callers use this to decide
    #: whether planning/prefetching reads up front is worth it
    is_remote: bool = False

    @abstractmethod
    def list_files(self) -> list[str]: ...

    @abstractmethod
    def read_range(self, filename: str, start: int, end: int) -> bytes:
        """Read bytes [start, end) of a file."""

    @abstractmethod
    def size(self, filename: str) -> int: ...

    def read_ranges(self, filename: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        """Read many byte ranges of one file; results match input order.

        The base implementation is a sequential loop; remote sources
        override it to issue the requests concurrently.
        """
        return [self.read_range(filename, s, e) for s, e in ranges]

    def read_file(self, filename: str) -> bytes:
        return self.read_range(filename, 0, self.size(filename))

    def exists(self, filename: str) -> bool:
        return filename in self.list_files()

    def read_text(self, filename: str, encoding: str = "utf-8") -> str:
        return self.read_file(filename).decode(encoding, errors="replace")


class LocalSource(ModelSource):
    def __init__(self, path: str | os.PathLike):
        self.root = Path(path)
        if not self.root.is_dir():
            raise SourceError(f"not a directory: {self.root}")
        self.name = str(self.root)

    def list_files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in self.root.rglob("*")
            if p.is_file()
        )

    def size(self, filename: str) -> int:
        return (self.root / filename).stat().st_size

    def read_range(self, filename: str, start: int, end: int) -> bytes:
        with open(self.root / filename, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        self.bytes_read += len(data)
        return data


class HubSource(ModelSource):
    """Hugging Face Hub repo accessed via HTTP range requests.

    File listing goes through the Hub API. Byte reads resolve the
    ``/resolve/`` redirect once per file, cache the signed CDN URL, and then
    hit the CDN directly — one HTTP exchange per read instead of two. Batch
    reads (``read_ranges``) run over a thread pool, which is what makes
    sparse fingerprint extraction latency-bound work finish in minutes
    instead of hours.
    """

    is_remote = True

    _REDIRECT_STATUS = (301, 302, 303, 307, 308)
    _RETRY_STATUS = (429, 500, 502, 503, 504)
    _MAX_ATTEMPTS = 4
    _TIMEOUT = (10, 60)  # connect, read (seconds)

    def __init__(
        self,
        repo_id: str,
        revision: str = "main",
        token: str | None = None,
        concurrency: int | None = None,
    ):
        import requests
        from requests.adapters import HTTPAdapter

        self.repo_id = repo_id
        self.revision = revision
        self.name = repo_id
        if concurrency is None:
            concurrency = int(os.environ.get("MODELDNA_HTTP_CONCURRENCY", "32"))
        self.concurrency = max(1, concurrency)
        self._session = requests.Session()
        # default pool is 10 connections; size it to the worker count
        self._session.mount(
            "https://", HTTPAdapter(pool_connections=4, pool_maxsize=self.concurrency)
        )
        self._files: list[str] | None = None
        self._sizes: dict[str, int] = {}
        self._direct_urls: dict[str, str] = {}
        self._lock = threading.Lock()

        if token is None:
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:
                token = os.environ.get("HF_TOKEN")
        self._token = token

    def _url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"

    def _auth_headers(self, url: str) -> dict[str, str]:
        # The token authenticates against the Hub only. CDN URLs carry their
        # own signature, and forwarding the bearer token to another host
        # would leak it (requests strips it on cross-host redirects for the
        # same reason).
        if self._token and url.startswith("https://huggingface.co/"):
            return {"authorization": f"Bearer {self._token}"}
        return {}

    def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        import requests

        h = dict(headers or {})
        h.update(self._auth_headers(url))
        last: Exception | str = "no attempt made"
        for attempt in range(self._MAX_ATTEMPTS):
            if attempt:
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.25)
            try:
                r = self._session.request(
                    method, url, headers=h,
                    allow_redirects=allow_redirects, timeout=self._TIMEOUT,
                )
            except requests.RequestException as e:
                # covers timeouts, refused/reset connections, and mid-body
                # breaks (ChunkedEncodingError/IncompleteRead) — all
                # transient at the transport level, all worth retrying
                last = e
                continue
            if r.status_code not in self._RETRY_STATUS:
                return r
            last = f"HTTP {r.status_code}"
        raise SourceError(
            f"{method.upper()} {url} failed after {self._MAX_ATTEMPTS} attempts: {last}"
        )

    def _direct_url(self, filename: str) -> str:
        """CDN URL for a file, resolving the /resolve/ redirect once."""
        with self._lock:
            cached = self._direct_urls.get(filename)
        if cached:
            return cached
        resolve = self._url(filename)
        r = self._request("head", resolve, allow_redirects=False)
        if r.status_code in self._REDIRECT_STATUS:
            url = urljoin(resolve, r.headers["location"])
        elif r.status_code == 200:
            url = resolve  # small non-LFS file, served by the Hub directly
        else:
            raise SourceError(f"HEAD {filename} on {self.repo_id}: HTTP {r.status_code}")
        with self._lock:
            self._direct_urls[filename] = url
        return url

    def list_files(self) -> list[str]:
        if self._files is None:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

            api = HfApi(token=self._token)
            try:
                info = api.model_info(self.repo_id, revision=self.revision, files_metadata=True)
            except RepositoryNotFoundError as e:
                raise SourceError(f"repo not found on the Hub: {self.repo_id}") from e
            except HfHubHTTPError as e:
                raise SourceError(f"Hub API error for {self.repo_id}: {e}") from e
            self._files = []
            for sib in info.siblings or []:
                self._files.append(sib.rfilename)
                if sib.size is not None:
                    self._sizes[sib.rfilename] = sib.size
        return self._files

    def size(self, filename: str) -> int:
        if filename not in self._sizes:
            r = self._request("head", self._url(filename))
            if r.status_code != 200:
                raise SourceError(f"HEAD {filename} on {self.repo_id}: HTTP {r.status_code}")
            self._sizes[filename] = int(r.headers["content-length"])
        return self._sizes[filename]

    def read_range(self, filename: str, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        headers = {"Range": f"bytes={start}-{end - 1}"}
        r = self._request("get", self._direct_url(filename), headers=headers)
        if r.status_code in (401, 403, 404):
            # signed CDN URLs expire; drop the cached one and re-resolve once
            with self._lock:
                self._direct_urls.pop(filename, None)
            r = self._request("get", self._direct_url(filename), headers=headers)
        if r.status_code not in (200, 206):
            raise SourceError(f"GET {filename} on {self.repo_id}: HTTP {r.status_code}")
        data = r.content
        if r.status_code == 200 and len(data) > end - start:
            # server ignored the Range header; take what we asked for
            data = data[start:end]
        with self._lock:
            self.bytes_read += len(data)
        return data

    def read_ranges(self, filename: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        if len(ranges) <= 1 or self.concurrency <= 1:
            return [self.read_range(filename, s, e) for s, e in ranges]
        self._direct_url(filename)  # resolve once before the pool races on it
        from concurrent.futures import ThreadPoolExecutor

        workers = min(self.concurrency, len(ranges))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda se: self.read_range(filename, se[0], se[1]), ranges))


def open_source(target: str, revision: str = "main") -> ModelSource:
    """Resolve a CLI target (local path or Hub repo id) to a ModelSource."""
    p = Path(target)
    if p.exists():
        if p.is_file():  # allow pointing at a file inside the model dir
            p = p.parent
        return LocalSource(p)
    if "/" in target and not target.startswith((".", "/", "~")):
        return HubSource(target, revision=revision)
    raise SourceError(
        f"{target!r} is neither a local directory nor a Hub repo id (org/name)"
    )
