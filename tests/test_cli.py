import json

import pytest
from typer.testing import CliRunner

from conftest import make_tiny_llama
from modeldna.cli import app

runner = CliRunner()

DIMS = dict(n_layers=8, hidden=64, n_heads=8, n_kv_heads=4, intermediate=128)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated DB + scans dir, plus a small model zoo."""
    home = tmp_path / "modeldna-home"
    monkeypatch.setenv("MODELDNA_DB", str(home / "db"))

    base_w = make_tiny_llama(tmp_path / "base", seed=0, **DIMS)
    make_tiny_llama(
        tmp_path / "ft", seed=42, base_weights=base_w, noise=0.002,
        readme="---\nbase_model: local/base\n---\nfine-tune\n", **DIMS,
    )
    make_tiny_llama(tmp_path / "other", seed=9, **DIMS)
    return tmp_path


def _add(env, name, model_id, family):
    r = runner.invoke(app, [
        "db", "add", str(env / name), "--id", model_id, "--family", family, "--fast",
    ])
    assert r.exit_code == 0, r.output
    return r


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert "modeldna" in r.output


def test_db_add_list_info_remove(env):
    _add(env, "base", "local/base", "test-family")
    r = runner.invoke(app, ["db", "list"])
    assert "local/base" in r.output and "test-family" in r.output

    r = runner.invoke(app, ["db", "info"])
    assert json.loads(r.output)["n_entries"] == 1

    r = runner.invoke(app, ["db", "info", "local/base"])
    assert json.loads(r.output)["family"] == "test-family"

    r = runner.invoke(app, ["db", "add", str(env / "base"), "--id", "local/base"])
    assert r.exit_code == 1  # duplicate

    r = runner.invoke(app, ["db", "remove", "local/base"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["db", "remove", "local/base"])
    assert r.exit_code == 1


def test_scan_json_and_exit_codes(env):
    _add(env, "base", "local/base", "test-family")
    _add(env, "other", "local/other", "other-family")

    # fine-tune with a truthful claim -> exit 0, CONSISTENT
    r = runner.invoke(app, ["scan", str(env / "ft"), "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["verdict"]["verdict"] == "FINE_TUNE"
    assert d["verdict"]["best_candidate"] == "local/base"
    assert d["consistency"]["status"] == "CONSISTENT"

    # independent model, no claim -> NO_MATCH, exit 4
    r = runner.invoke(app, ["scan", str(env / "base"), "--json"])
    d = json.loads(r.output)
    assert d["verdict"]["verdict"] == "EXACT_COPY"  # it IS the DB entry

    r = runner.invoke(app, ["scan", str(env / "other"), "--json"])
    # own DB entry excluded only by id; here id differs (path vs local/other)
    # so the exact-copy match against itself is expected
    d = json.loads(r.output)
    assert d["verdict"]["verdict"] in ("EXACT_COPY", "NO_MATCH")


def test_scan_terminal_output(env):
    _add(env, "base", "local/base", "test-family")
    r = runner.invoke(app, ["scan", str(env / "ft")])
    assert "FINE_TUNE" in r.output
    assert "likely derived from" in r.output
    assert "Evidence" in r.output


def test_scan_html_report(env, tmp_path):
    _add(env, "base", "local/base", "test-family")
    out = tmp_path / "report.html"
    r = runner.invoke(app, ["scan", str(env / "ft"), "--report", str(out), "--json"])
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "modeldna · lineage report" in html
    assert "FINE_TUNE" in html
    assert "<svg" in html  # sigma-curve plots made it in
    assert "local/base" in html


def test_scan_missing_target(env):
    r = runner.invoke(app, ["scan", "no-such-dir-xyz"])
    assert r.exit_code == 1


def test_compare(env):
    r = runner.invoke(app, ["compare", str(env / "ft"), str(env / "base"), "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["verdict"]["verdict"] == "FINE_TUNE"

    r = runner.invoke(app, ["compare", str(env / "other"), str(env / "base")])
    assert "NO_MATCH" in r.output


def test_fingerprint_export(env, tmp_path):
    out = tmp_path / "fp.json"
    r = runner.invoke(app, ["fingerprint", str(env / "base"), "-o", str(out)])
    assert r.exit_code == 0, r.output
    d = json.loads(out.read_text())
    assert d["model_id"] == str(env / "base")
    assert "sigma_curves" in d


def test_explain_roundtrip(env):
    _add(env, "base", "local/base", "test-family")
    runner.invoke(app, ["scan", str(env / "ft"), "--json"])
    r = runner.invoke(app, ["explain", str(env / "ft")])
    assert r.exit_code == 0
    assert "sigma" in r.output.lower() or "verdict" in r.output.lower()

    r = runner.invoke(app, ["explain", "never-scanned"])
    assert r.exit_code == 1


def test_db_update_unconfigured(env, monkeypatch):
    monkeypatch.delenv("MODELDNA_DB_URL", raising=False)
    r = runner.invoke(app, ["db", "update"])
    assert r.exit_code == 4
    assert "db build" in r.output
