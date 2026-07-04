"""Uniform byte-level access to a model, whether it lives on disk or the Hub.

The whole tool is built on top of two primitives: list the files in a repo,
and read an arbitrary byte range from one of them. Locally that's a seek;
on the Hub it's an HTTP Range request against the resolve endpoint, which
the CDN serves without us ever downloading the shard.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class SourceError(RuntimeError):
    """Raised when a model source can't be listed or read."""


class ModelSource(ABC):
    """Byte-range access to the files of one model repo."""

    #: identifier used in reports (repo id or local path)
    name: str

    #: running total of payload bytes fetched, for the traffic report
    bytes_read: int = 0

    @abstractmethod
    def list_files(self) -> list[str]: ...

    @abstractmethod
    def read_range(self, filename: str, start: int, end: int) -> bytes:
        """Read bytes [start, end) of a file."""

    @abstractmethod
    def size(self, filename: str) -> int: ...

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

    File listing goes through the Hub API; byte reads hit the
    ``/resolve/`` endpoint with a Range header, which redirects to the CDN
    and costs only the bytes requested.
    """

    def __init__(self, repo_id: str, revision: str = "main", token: str | None = None):
        import requests

        self.repo_id = repo_id
        self.revision = revision
        self.name = repo_id
        self._session = requests.Session()
        self._files: list[str] | None = None
        self._sizes: dict[str, int] = {}

        if token is None:
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:
                token = os.environ.get("HF_TOKEN")
        self._token = token
        if token:
            self._session.headers["authorization"] = f"Bearer {token}"

    def _url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"

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
            r = self._session.head(self._url(filename), allow_redirects=True, timeout=30)
            if r.status_code != 200:
                raise SourceError(f"HEAD {filename} on {self.repo_id}: HTTP {r.status_code}")
            self._sizes[filename] = int(r.headers["content-length"])
        return self._sizes[filename]

    def read_range(self, filename: str, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        headers = {"Range": f"bytes={start}-{end - 1}"}
        r = self._session.get(self._url(filename), headers=headers, timeout=120)
        if r.status_code not in (200, 206):
            raise SourceError(f"GET {filename} on {self.repo_id}: HTTP {r.status_code}")
        data = r.content
        if r.status_code == 200 and len(data) > end - start:
            # server ignored the Range header; take what we asked for
            data = data[start:end]
        self.bytes_read += len(data)
        return data


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
