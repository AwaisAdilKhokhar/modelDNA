"""modeldna command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from modeldna import __version__
from modeldna.db.store import ReferenceDB, _safe_name, default_db_path
from modeldna.io.source import SourceError, open_source
from modeldna.io.weights import WeightIndexError
from modeldna.io.safetensors import SafetensorsError

app = typer.Typer(
    name="modeldna",
    help="Forensic lineage detection for open-weight LLMs: find out what a "
    "model really descends from, from its weights alone.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
db_app = typer.Typer(help="Manage the reference database of base-model fingerprints.")
app.add_typer(db_app, name="db")

console = Console()
err_console = Console(stderr=True)

READ_ERRORS = (SourceError, WeightIndexError, SafetensorsError)


def _fail(msg: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/bold red] {msg}")
    raise typer.Exit(code)


def _scans_dir() -> Path:
    return default_db_path().parent / "scans"


def _save_scan(target: str, d: dict) -> Path:
    out = _scans_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / (_safe_name(target) + ".json")
    path.write_text(json.dumps(d, indent=1))
    return path


def _mode(full: bool) -> str:
    return "full" if full else "fast"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"modelDNA {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", help="Print version and exit.",
        callback=_version_callback, is_eager=True,
    ),
) -> None:
    pass


@app.command()
def scan(
    target: str = typer.Argument(..., help="Hub repo id (org/name) or local model dir."),
    full: bool = typer.Option(False, "--full", help="Download complete attention "
                              "tensors instead of sampled slices."),
    top_k: int = typer.Option(3, "--top-k", help="Candidates to display."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
    report: Optional[Path] = typer.Option(None, "--report", help="Write an HTML "
                                          "evidence report to this path."),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Reference DB directory."),
    revision: str = typer.Option("main", "--revision", help="Hub revision to scan."),
):
    """Scan a model against the reference DB and print a lineage verdict.

    Exit codes: 0 = consistent with claimed lineage (or no claim),
    3 = weight evidence contradicts the repo's claim, 4 = unknown lineage.
    """
    from modeldna.scan import scan as run_scan

    try:
        res = run_scan(target, db=ReferenceDB(db_path), mode=_mode(full),
                       revision=revision)
    except READ_ERRORS as e:
        _fail(str(e))
    d = res.to_dict()
    _save_scan(target, d)
    if report:
        from modeldna.report.html import render_html

        report.write_text(render_html(d), encoding="utf-8")
    if json_out:
        console.print_json(json.dumps(d))
    else:
        from modeldna.report.terminal import print_scan

        print_scan(res, top_k=top_k, console=console)
        if report:
            console.print(f"[dim]report written to {report}[/dim]")
    raise typer.Exit(res.exit_code)


@app.command()
def compare(
    suspect: str = typer.Argument(..., help="Model to test (repo id or local dir)."),
    candidate: str = typer.Argument(..., help="Alleged parent (repo id or local dir)."),
    full: bool = typer.Option(False, "--full", help="Use complete tensors."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
    report: Optional[Path] = typer.Option(None, "--report", help="Write an HTML "
                                          "evidence report to this path."),
    revision: str = typer.Option("main", "--revision"),
):
    """Test a specific hypothesis pair: is SUSPECT derived from CANDIDATE?"""
    from modeldna.scan import compare_pair

    try:
        fp_a, fp_b, ev, verdict = compare_pair(
            suspect, candidate, mode=_mode(full), revision=revision
        )
    except READ_ERRORS as e:
        _fail(str(e))

    d = {
        "tool_version": __version__,
        "target": suspect,
        "mode": _mode(full),
        "db_version": 0,
        "bytes_read": 0,
        "arch": fp_a.arch.to_dict(),
        "verdict": verdict.to_dict(),
        "claims": {"has_readme": False, "base_models": [], "license": "",
                   "claims_from_scratch": False},
        "consistency": {"status": "NO_CLAIM",
                        "detail": "compare mode tests a pair, not a repo claim"},
        "background_ranges": {},
        "curves": {
            "suspect_sigma": fp_a.sigma_curves,
            "best_sigma": fp_b.sigma_curves,
            "best_id": candidate,
        },
    }
    _save_scan(f"compare {suspect} {candidate}", d)
    if report:
        from modeldna.report.html import render_html

        report.write_text(render_html(d), encoding="utf-8")
    if json_out:
        console.print_json(json.dumps(d))
    else:
        from modeldna.report.terminal import print_compare

        print_compare(ev, verdict, console=console)
        if report:
            console.print(f"[dim]report written to {report}[/dim]")


@app.command()
def fingerprint(
    target: str = typer.Argument(..., help="Hub repo id or local model dir."),
    output: Path = typer.Option(Path("fingerprint.json"), "-o", "--output",
                                help="Output path (.json or .json.gz)."),
    full: bool = typer.Option(False, "--full"),
    revision: str = typer.Option("main", "--revision"),
):
    """Extract and save a fingerprint without judging it."""
    from modeldna.fingerprint.extract import extract_fingerprint

    try:
        src = open_source(target, revision=revision)
        fp = extract_fingerprint(src, mode=_mode(full), model_id=target,
                                 revision=revision)
    except READ_ERRORS as e:
        _fail(str(e))
    fp.save(output)
    console.print(
        f"fingerprint of [bold]{target}[/bold] "
        f"({fp.arch.summary()}) -> {output} "
        f"[dim]({src.bytes_read / 1e6:.1f} MB read)[/dim]"
    )


@app.command()
def explain(
    target: str = typer.Argument(..., help="Target of a previous scan or compare."),
    report: Optional[Path] = typer.Option(None, "--report", help="Also write the "
                                          "HTML report here."),
):
    """Show the raw numbers behind the most recent verdict for TARGET."""
    path = _scans_dir() / (_safe_name(target) + ".json")
    if not path.exists():
        _fail(f"no saved scan for {target!r} — run `modeldna scan {target}` first")
    d = json.loads(path.read_text())
    console.print_json(json.dumps(d))
    if report:
        from modeldna.report.html import render_html

        report.write_text(render_html(d), encoding="utf-8")
        console.print(f"[dim]report written to {report}[/dim]")


# -- db commands ---------------------------------------------------------------


@db_app.command("list")
def db_list(db_path: Optional[Path] = typer.Option(None, "--db")):
    """List reference DB entries."""
    from rich.table import Table

    db = ReferenceDB(db_path)
    t = Table(show_edge=False)
    t.add_column("model")
    t.add_column("family")
    t.add_column("layers", justify="right")
    t.add_column("dtype")
    t.add_column("added")
    for e in db.entries():
        t.add_row(e.model_id, e.family, str(e.n_layers), e.torch_dtype,
                  e.added_at[:10])
    console.print(t)
    console.print(f"[dim]{len(db)} entries · db v{db.version} · {db.root}[/dim]")


@db_app.command("info")
def db_info(
    model_id: Optional[str] = typer.Argument(None),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Show DB summary, or details for one entry."""
    db = ReferenceDB(db_path)
    if model_id is None:
        console.print_json(json.dumps(db.info()))
        return
    entry = db.get(model_id)
    if entry is None:
        _fail(f"{model_id} not in reference DB")
    console.print_json(json.dumps(entry.to_dict()))


@db_app.command("add")
def db_add(
    target: str = typer.Argument(..., help="Hub repo id or local model dir."),
    model_id: Optional[str] = typer.Option(None, "--id", help="Store under this id "
                                           "(defaults to the target)."),
    family: Optional[str] = typer.Option(None, "--family"),
    full: bool = typer.Option(True, "--full/--fast", help="Reference entries "
                              "default to full-mode fingerprints."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    db_path: Optional[Path] = typer.Option(None, "--db"),
    revision: str = typer.Option("main", "--revision"),
):
    """Fingerprint a model and add it to the reference DB."""
    from modeldna.fingerprint.extract import extract_fingerprint

    db = ReferenceDB(db_path)
    try:
        src = open_source(target, revision=revision)
        fp = extract_fingerprint(src, mode=_mode(full), model_id=model_id or target,
                                 revision=revision)
        entry = db.add(fp, family=family, overwrite=overwrite)
    except READ_ERRORS as e:
        _fail(str(e))
    except ValueError as e:
        _fail(str(e))
    console.print(f"added [bold]{entry.model_id}[/bold] (family: {entry.family}) "
                  f"· db v{db.version}")


@db_app.command("remove")
def db_remove(
    model_id: str = typer.Argument(...),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Remove an entry from the reference DB."""
    db = ReferenceDB(db_path)
    if not db.remove(model_id):
        _fail(f"{model_id} not in reference DB")
    console.print(f"removed {model_id} · db v{db.version}")


@db_app.command("build")
def db_build(
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N "
                                        "seed models."),
    fast: bool = typer.Option(True, "--fast/--full", help="Fingerprint mode for "
                              "seed entries."),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Build the reference DB by fingerprinting the seed list from the Hub.

    Needs network access (and an HF token for gated repos like Llama).
    Interrupted builds resume where they left off.
    """
    from modeldna.db.families import SEED_MODELS
    from modeldna.fingerprint.extract import extract_fingerprint

    db = ReferenceDB(db_path)
    todo = SEED_MODELS[:limit] if limit else SEED_MODELS
    done = {e.model_id for e in db.entries()}
    failures = 0
    for i, repo in enumerate(todo, 1):
        if repo in done:
            console.print(f"[dim][{i}/{len(todo)}] {repo} already indexed[/dim]")
            continue
        console.print(f"[{i}/{len(todo)}] fingerprinting [bold]{repo}[/bold] …")
        try:
            src = open_source(repo)
            fp = extract_fingerprint(src, mode="fast" if fast else "full",
                                     model_id=repo)
            db.add(fp)
            console.print(f"  [green]ok[/green] ({src.bytes_read / 1e6:.0f} MB read)")
        except Exception as e:  # keep building; report at the end
            failures += 1
            err_console.print(f"  [red]failed:[/red] {e}")
    console.print(f"\ndone: {len(db)} entries · db v{db.version} · "
                  f"{failures} failure(s)")
    if failures:
        raise typer.Exit(1)


#: release asset published by .github/workflows/refdb-release.yml
DEFAULT_DB_URL = (
    "https://github.com/AwaisAdilKhokhar/modelDNA/releases/latest/download/modeldna-refdb.tar.gz"
)


@db_app.command("export")
def db_export(
    output: Path = typer.Argument(Path("modeldna-refdb.tar.gz"),
                                  help="Archive path (.tar.gz)."),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Package the reference DB into an archive that `db pull` can consume."""
    db = ReferenceDB(db_path)
    if len(db) == 0:
        _fail("reference DB is empty; nothing to export")
    path = db.export_archive(output)
    console.print(f"exported {len(db)} entries -> {path} "
                  f"[dim]({path.stat().st_size / 1e6:.1f} MB)[/dim]")


@db_app.command("pull")
def db_pull(
    url: Optional[str] = typer.Option(None, "--url", help="Archive URL or local "
                                      "path; defaults to $MODELDNA_DB_URL, then "
                                      "the latest GitHub release."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace entries "
                                   "that already exist locally."),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Fetch a published reference DB instead of fingerprinting the seed list.

    Pulling ~55 MB of prebuilt fingerprints takes seconds; after that a scan
    only ever fetches one model from the Hub — the suspect.
    """
    import os
    import tempfile

    import requests

    url = url or os.environ.get("MODELDNA_DB_URL") or DEFAULT_DB_URL
    db = ReferenceDB(db_path)
    try:
        if Path(url).exists():
            added, skipped = db.import_archive(url, overwrite=overwrite)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "refdb.tar.gz"
                with requests.get(url, stream=True, timeout=120) as r:
                    if r.status_code != 200:
                        _fail(f"GET {url}: HTTP {r.status_code}"
                              + (" (no release published yet?)"
                                 if r.status_code == 404 else ""))
                    with open(archive, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                added, skipped = db.import_archive(archive, overwrite=overwrite)
    except (ValueError, OSError, requests.RequestException) as e:
        _fail(str(e))
    console.print(f"pulled {added} entries ({skipped} already present) · "
                  f"{len(db)} total · db v{db.version}")


@db_app.command("update")
def db_update(
    url: Optional[str] = typer.Option(None, "--url"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    db_path: Optional[Path] = typer.Option(None, "--db"),
):
    """Alias for `db pull`."""
    db_pull(url=url, overwrite=overwrite, db_path=db_path)


if __name__ == "__main__":
    app()
