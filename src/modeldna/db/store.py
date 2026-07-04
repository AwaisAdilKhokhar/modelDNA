"""Local reference database of base-model fingerprints.

Layout (default ``~/.modeldna/db``, override with ``MODELDNA_DB``):

    manifest.json                 db version + entry index (cheap to load)
    fingerprints/<id>.json.gz     one Fingerprint per reference model
    background.json               cached unrelated-pair feature distributions
    calibrator.json               optional custom-fitted calibrator

The manifest carries just enough per entry (shape tuple, hashes, dtype) to
filter candidates without loading fingerprints.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from modeldna.arch.signature import ArchSignature
from modeldna.calibration import DEFAULT_CALIBRATOR, Background, LogisticCalibrator
from modeldna.compare import compare_fingerprints
from modeldna.db.families import guess_family
from modeldna.fingerprint.extract import Fingerprint

MANIFEST = "manifest.json"


def default_db_path() -> Path:
    env = os.environ.get("MODELDNA_DB")
    if env:
        return Path(env)
    return Path.home() / ".modeldna" / "db"


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__").replace(":", "_")


@dataclass
class DBEntry:
    model_id: str
    family: str
    file: str
    revision: str = ""
    core_shape: list[int] = field(default_factory=list)
    n_layers: int = 0
    inventory_hash: str = ""
    tokenizer_hash: str = ""
    torch_dtype: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    added_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "file": self.file,
            "revision": self.revision,
            "core_shape": self.core_shape,
            "n_layers": self.n_layers,
            "inventory_hash": self.inventory_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "torch_dtype": self.torch_dtype,
            "meta": self.meta,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DBEntry":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class ReferenceDB:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_db_path()
        self.fp_dir = self.root / "fingerprints"
        self._manifest: dict[str, Any] | None = None

    # -- manifest ---------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            mf = self.root / MANIFEST
            if mf.exists():
                self._manifest = json.loads(mf.read_text())
            else:
                self._manifest = {"db_version": 0, "updated_at": "", "entries": []}
        return self._manifest

    def _save_manifest(self) -> None:
        m = self._load_manifest()
        m["db_version"] = m.get("db_version", 0) + 1
        m["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / MANIFEST).write_text(json.dumps(m, indent=2))
        # any change to contents invalidates the cached background
        (self.root / "background.json").unlink(missing_ok=True)

    @property
    def version(self) -> int:
        return self._load_manifest().get("db_version", 0)

    def entries(self) -> list[DBEntry]:
        return [DBEntry.from_dict(d) for d in self._load_manifest()["entries"]]

    def get(self, model_id: str) -> DBEntry | None:
        for e in self.entries():
            if e.model_id == model_id:
                return e
        return None

    def __len__(self) -> int:
        return len(self._load_manifest()["entries"])

    # -- content ------------------------------------------------------------------

    def add(
        self,
        fp: Fingerprint,
        family: str | None = None,
        meta: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> DBEntry:
        m = self._load_manifest()
        existing = self.get(fp.model_id)
        if existing and not overwrite:
            raise ValueError(f"{fp.model_id} already in DB (pass overwrite=True to replace)")
        if existing:
            m["entries"] = [e for e in m["entries"] if e["model_id"] != fp.model_id]

        fname = _safe_name(fp.model_id) + ".json.gz"
        self.fp_dir.mkdir(parents=True, exist_ok=True)
        fp.save(self.fp_dir / fname)

        entry = DBEntry(
            model_id=fp.model_id,
            family=family or guess_family(fp.model_id),
            file=fname,
            revision=fp.revision,
            core_shape=list(fp.arch.core_shape()),
            n_layers=fp.arch.n_layers,
            inventory_hash=fp.arch.inventory_hash,
            tokenizer_hash=fp.arch.tokenizer_hash,
            torch_dtype=fp.arch.torch_dtype,
            meta=meta or {},
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        m["entries"].append(entry.to_dict())
        self._save_manifest()
        return entry

    def remove(self, model_id: str) -> bool:
        m = self._load_manifest()
        entry = self.get(model_id)
        if entry is None:
            return False
        (self.fp_dir / entry.file).unlink(missing_ok=True)
        m["entries"] = [e for e in m["entries"] if e["model_id"] != model_id]
        self._save_manifest()
        return True

    def load_fingerprint(self, model_id: str) -> Fingerprint:
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(f"{model_id} not in reference DB")
        return Fingerprint.load(self.fp_dir / entry.file)

    # -- candidate filtering ---------------------------------------------------------

    def candidates_for(self, sig: ArchSignature) -> list[DBEntry]:
        """Entries worth comparing: same depth, or same inventory hash.

        Depth (layer count) is the hard gate — everything per-layer needs it.
        Inventory-hash matches are always included so renamed mirrors of a
        base with unusual metadata still surface.
        """
        out = []
        for e in self.entries():
            if e.n_layers == sig.n_layers or (
                sig.inventory_hash and e.inventory_hash == sig.inventory_hash
            ):
                out.append(e)
        return out

    # -- background & calibration -------------------------------------------------------

    def background(self, recompute: bool = False) -> Background:
        """Unrelated-pair (cross-family) feature distributions over the DB."""
        cache = self.root / "background.json"
        if cache.exists() and not recompute:
            return Background.from_dict(json.loads(cache.read_text()))
        bg = Background()
        entries = self.entries()
        fps: dict[str, Fingerprint] = {}
        for a, b in combinations(entries, 2):
            if a.family == b.family:
                continue
            if a.n_layers != b.n_layers:
                continue
            for e in (a, b):
                if e.model_id not in fps:
                    fps[e.model_id] = self.load_fingerprint(e.model_id)
            ev = compare_fingerprints(fps[a.model_id], fps[b.model_id])
            bg.add(ev.features())
        if bg.samples:
            self.root.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(bg.to_dict()))
        return bg

    def calibrator(self) -> LogisticCalibrator:
        custom = self.root / "calibrator.json"
        if custom.exists():
            return LogisticCalibrator.load(custom)
        cal = LogisticCalibrator(
            coef=dict(DEFAULT_CALIBRATOR.coef),
            intercept=DEFAULT_CALIBRATOR.intercept,
            neutral=dict(DEFAULT_CALIBRATOR.neutral),
            fitted_on=DEFAULT_CALIBRATOR.fitted_on,
        )
        try:
            means = self.background().neutral_means()
            cal.neutral.update(means)
        except Exception:
            pass
        return cal

    def info(self) -> dict[str, Any]:
        m = self._load_manifest()
        fams: dict[str, int] = {}
        for e in self.entries():
            fams[e.family] = fams.get(e.family, 0) + 1
        return {
            "path": str(self.root),
            "db_version": m.get("db_version", 0),
            "updated_at": m.get("updated_at", ""),
            "n_entries": len(m["entries"]),
            "families": dict(sorted(fams.items())),
        }
