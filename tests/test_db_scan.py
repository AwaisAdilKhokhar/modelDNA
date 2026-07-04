import numpy as np
import pytest

from conftest import make_tiny_llama
from modeldna.card import Claims, check_consistency, read_claims
from modeldna.db.families import guess_family
from modeldna.db.store import ReferenceDB
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.io.source import LocalSource
from modeldna.scan import compare_pair, scan
from modeldna.verdict import VerdictClass

DIMS = dict(n_layers=8, hidden=64, n_heads=8, n_kv_heads=4, intermediate=128)


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zoo")
    base_w = make_tiny_llama(tmp / "base_a", seed=0, **DIMS)
    make_tiny_llama(tmp / "base_a_instruct", seed=60, base_weights=base_w,
                    noise=0.0008, **DIMS)
    make_tiny_llama(tmp / "base_b", seed=1, **DIMS)
    make_tiny_llama(
        tmp / "suspect_ft", seed=70, base_weights=base_w, noise=0.003,
        readme="---\nbase_model: org-a/base-a\nlicense: apache-2.0\n---\nA fine-tune.\n",
        **DIMS,
    )
    make_tiny_llama(
        tmp / "suspect_laundered", seed=71, base_weights=base_w, noise=0.12,
        readme="---\nlicense: mit\n---\nOur new model, trained entirely from scratch "
               "on 2T tokens.\n",
        **DIMS,
    )
    make_tiny_llama(
        tmp / "honest_original", seed=5,
        readme="---\nlicense: apache-2.0\n---\nTrained from scratch on open data.\n",
        **DIMS,
    )
    return tmp


@pytest.fixture()
def db(zoo, tmp_path):
    db = ReferenceDB(tmp_path / "refdb")
    for name, model_id, family in [
        ("base_a", "org-a/base-a", "family-a"),
        ("base_a_instruct", "org-a/base-a-instruct", "family-a"),
        ("base_b", "org-b/base-b", "family-b"),
    ]:
        fp = extract_fingerprint(LocalSource(zoo / name), model_id=model_id)
        db.add(fp, family=family, meta={"license": "apache-2.0"})
    return db


# -- store -----------------------------------------------------------------


def test_db_roundtrip(db):
    assert len(db) == 3
    assert db.version == 3  # one bump per add
    e = db.get("org-a/base-a")
    assert e.family == "family-a"
    fp = db.load_fingerprint("org-a/base-a")
    assert fp.model_id == "org-a/base-a"
    info = db.info()
    assert info["n_entries"] == 3
    assert info["families"] == {"family-a": 2, "family-b": 1}


def test_db_duplicate_add(db):
    fp = db.load_fingerprint("org-b/base-b")
    with pytest.raises(ValueError):
        db.add(fp, family="family-b")
    db.add(fp, family="family-b", overwrite=True)  # fine


def test_db_remove(db):
    assert db.remove("org-b/base-b")
    assert not db.remove("org-b/base-b")
    assert len(db) == 2


def test_db_candidates_filter(db, zoo, tmp_path):
    deep = tmp_path / "deep"
    make_tiny_llama(deep, seed=3, n_layers=12, hidden=64, n_heads=8,
                    n_kv_heads=4, intermediate=128)
    fp = extract_fingerprint(LocalSource(deep))
    assert db.candidates_for(fp.arch) == []  # nothing at 12 layers
    fp2 = db.load_fingerprint("org-a/base-a")
    assert len(db.candidates_for(fp2.arch)) == 3


def test_db_background(db):
    bg = db.background()
    # only cross-family pairs contribute: (a, b) and (a-instruct, b)
    assert len(bg.samples["pcs_cos"]) == 2
    assert db.background().samples == bg.samples  # cached


def test_guess_family():
    assert guess_family("meta-llama/Llama-3.1-8B") == "llama-3.1"
    assert guess_family("Qwen/Qwen2.5-14B") == "qwen2.5"
    assert guess_family("EleutherAI/pythia-1.4b") == "pythia"
    assert guess_family("some-org/mystery-model") == "some-org"


# -- card parsing ------------------------------------------------------------


def test_read_claims(zoo):
    claims = read_claims(LocalSource(zoo / "suspect_ft"))
    assert claims.base_models == ["org-a/base-a"]
    assert claims.license == "apache-2.0"
    assert not claims.from_scratch

    claims2 = read_claims(LocalSource(zoo / "suspect_laundered"))
    assert claims2.from_scratch
    assert claims2.base_models == []

    claims3 = read_claims(LocalSource(zoo / "base_a"))
    assert not claims3.has_readme


def test_front_matter_list():
    from modeldna.card import _parse_front_matter

    fm = _parse_front_matter(
        "---\nbase_model:\n  - org/one\n  - org/two\nlicense: mit\n---\nbody\n"
    )
    assert fm["base_model"] == ["org/one", "org/two"]
    fm2 = _parse_front_matter("---\nbase_model: [org/one, org/two]\n---\n")
    assert fm2["base_model"] == ["org/one", "org/two"]


# -- scan --------------------------------------------------------------------


def test_scan_finetune_consistent(zoo, db):
    res = scan(str(zoo / "suspect_ft"), db=db)
    assert res.verdict.verdict in (VerdictClass.FINE_TUNE,
                                   VerdictClass.SAME_FAMILY_UNRESOLVED)
    assert res.verdict.best.candidate_id.startswith("org-a/")
    assert res.consistency["status"] == "CONSISTENT"
    assert res.exit_code == 0
    d = res.to_dict()
    assert d["verdict"]["probability"] > 0.9
    assert d["background_ranges"]["pcs_cos"] != ""


def test_scan_laundered_inconsistent(zoo, db):
    """The Pangu pattern: heavy delta, from-scratch claim, lineage match."""
    res = scan(str(zoo / "suspect_laundered"), db=db)
    # base-a and its near-identical instruct variant tie -> family-level match
    assert res.verdict.verdict in (VerdictClass.SAME_LINEAGE, VerdictClass.FINE_TUNE,
                                   VerdictClass.SAME_FAMILY_UNRESOLVED)
    assert res.verdict.probability > 0.9
    assert res.consistency["status"] == "INCONSISTENT"
    assert res.exit_code == 3


def test_scan_honest_original(zoo, db):
    res = scan(str(zoo / "honest_original"), db=db)
    assert res.verdict.verdict == VerdictClass.NO_MATCH
    # from-scratch claim + NO_MATCH is not a contradiction
    assert res.consistency["status"] != "INCONSISTENT"
    assert res.exit_code == 4  # unknown lineage


def test_scan_empty_db(zoo, tmp_path):
    res = scan(str(zoo / "suspect_ft"), db=ReferenceDB(tmp_path / "empty"))
    assert res.verdict.verdict == VerdictClass.INSUFFICIENT
    assert any("empty" in n for n in res.verdict.notes)


def test_compare_pair(zoo):
    _, _, ev, verdict = compare_pair(str(zoo / "suspect_ft"), str(zoo / "base_a"))
    assert ev.pcs_cos_mean > 0.95
    assert verdict.verdict == VerdictClass.FINE_TUNE

    _, _, ev2, verdict2 = compare_pair(str(zoo / "base_b"), str(zoo / "base_a"))
    assert verdict2.verdict == VerdictClass.NO_MATCH


def test_consistency_no_claim():
    out = check_consistency(Claims(), {"verdict": "FINE_TUNE", "probability": 0.99,
                                       "best_candidate": "x/y"})
    assert out["status"] == "NO_CLAIM"
