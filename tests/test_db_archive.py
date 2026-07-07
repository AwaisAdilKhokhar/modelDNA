"""db export / pull: ship the reference DB as one archive.

Pins the Phase-B contract: an exported archive re-imports losslessly
(fingerprints, families, meta survive), merges skip-by-default /
replace-on-overwrite, and `modeldna db pull` works from a local archive
path with no network.
"""

import pytest
from typer.testing import CliRunner

from conftest import make_tiny_llama
from modeldna.cli import app
from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import LocalSource

runner = CliRunner()


@pytest.fixture(scope="module")
def source_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("archive_zoo")
    db = ReferenceDB(tmp / "refdb")
    for seed, (name, model_id, family) in enumerate([
        ("base_a", "org-a/base-a", "family-a"),
        ("base_b", "org-b/base-b", "family-b"),
    ]):
        make_tiny_llama(tmp / name, seed=seed)
        fp = extract_fingerprint(LocalSource(tmp / name), model_id=model_id)
        db.add(fp, family=family, meta={"license": "apache-2.0"})
    return db


def test_export_import_roundtrip(source_db, tmp_path):
    archive = source_db.export_archive(tmp_path / "refdb.tar.gz")
    assert archive.exists()

    dest = ReferenceDB(tmp_path / "pulled")
    added, skipped = dest.import_archive(archive)
    assert (added, skipped) == (2, 0)

    for model_id in ("org-a/base-a", "org-b/base-b"):
        src_e, dst_e = source_db.get(model_id), dest.get(model_id)
        assert dst_e.family == src_e.family
        assert dst_e.meta == {"license": "apache-2.0"}
        assert dst_e.inventory_hash == src_e.inventory_hash
        a = source_db.load_fingerprint(model_id).to_dict()
        b = dest.load_fingerprint(model_id).to_dict()
        assert a == b  # fingerprints survive the archive bit-exactly


def test_import_skips_existing_unless_overwrite(source_db, tmp_path):
    archive = source_db.export_archive(tmp_path / "refdb.tar.gz")
    dest = ReferenceDB(tmp_path / "pulled")
    dest.add(source_db.load_fingerprint("org-a/base-a"), family="local-family")

    added, skipped = dest.import_archive(archive)
    assert (added, skipped) == (1, 1)
    assert dest.get("org-a/base-a").family == "local-family"  # kept

    added, skipped = dest.import_archive(archive, overwrite=True)
    assert (added, skipped) == (2, 0)
    assert dest.get("org-a/base-a").family == "family-a"  # replaced


def test_import_rejects_non_db_archive(tmp_path):
    import tarfile

    bogus = tmp_path / "bogus.tar.gz"
    (tmp_path / "junk.txt").write_text("junk")
    with tarfile.open(bogus, "w:gz") as tar:
        tar.add(tmp_path / "junk.txt", arcname="junk.txt")
    with pytest.raises(ValueError, match="not a modeldna DB archive"):
        ReferenceDB(tmp_path / "db").import_archive(bogus)


def test_cli_pull_from_local_archive(source_db, tmp_path):
    archive = source_db.export_archive(tmp_path / "refdb.tar.gz")
    dest = tmp_path / "cli_db"
    res = runner.invoke(
        app, ["db", "pull", "--url", str(archive), "--db", str(dest)]
    )
    assert res.exit_code == 0, res.output
    assert "pulled 2 entries" in res.output
    assert len(ReferenceDB(dest)) == 2


def test_cli_export_empty_db_fails(tmp_path):
    res = runner.invoke(
        app, ["db", "export", str(tmp_path / "out.tar.gz"), "--db", str(tmp_path / "empty")]
    )
    assert res.exit_code == 1
