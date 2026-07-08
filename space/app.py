"""modelDNA live scanner — the Hugging Face Space frontend.

Paste a Hub repo id, get a calibrated lineage verdict. This is a thin
Gradio wrapper around `modeldna.scan`: on startup it pulls the prebuilt
reference DB (the same archive `modeldna db pull` uses), then each scan
reads only the sampled weight slices of the suspect model (~250 MB for a
7B), so a verdict lands in a couple of minutes on Space networking.

Visual language follows the Atlas (site/index.html): paper/ink palette,
mono eyebrows, --g0 blue accent, light/dark via CSS variables.

Deployed from the `space/` directory of
https://github.com/AwaisAdilKhokhar/modelDNA — edit there, not on the Hub.
"""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import time
from pathlib import Path

import gradio as gr
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from modeldna import __version__
from modeldna.cli import DEFAULT_DB_URL
from modeldna.db.store import ReferenceDB
from modeldna.io.safetensors import SafetensorsError
from modeldna.io.source import SourceError
from modeldna.io.weights import WeightIndexError
from modeldna.merge import decompose_targets
from modeldna.report.html import _CONSISTENCY_COLOR, _VERDICT_COLOR, render_html
from modeldna.report.terminal import mergekit_yaml
from modeldna.scan import scan

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
#: refuse live scans of repos whose safetensors exceed this (≈70B bf16);
#: a fast scan of something bigger stops being "live" on shared hardware
MAX_WEIGHT_BYTES = 180e9
READ_ERRORS = (SourceError, WeightIndexError, SafetensorsError)

REPO_URL = "https://github.com/AwaisAdilKhokhar/modelDNA"
ATLAS_URL = "https://awaisadilkhokhar.github.io/modelDNA/"

_SIGNALS = [
    ("attention σ-curve correlation (F1)", "sigma_r_mean", "sigma_r"),
    ("norm/bias vector cosine (F2)", "vector_cos_mean", "vector_cos"),
    ("sampled parameter cosine / PCS (F3)", "pcs_cos_mean", "pcs_cos"),
    ("SVD spectrum correlation (F4)", "spectra_r_mean", "spectra_r"),
]

#: (repo@revision, db_version) -> (result dict, elapsed seconds)
_CACHE: dict[str, tuple[dict, float]] = {}

#: decompose: cap candidate models per request (each costs a ~300 MB read)
MAX_PARENTS = 4
_DEC_CACHE: dict[str, tuple[dict, float]] = {}

_SUMMARY_COLOR = {
    "MERGE": "#8e44ad",
    "SINGLE_PARENT": "#b7950b",
    "AMBIGUOUS": "#2471a3",
    "UNEXPLAINED": "#1e8449",
}


def ensure_db() -> ReferenceDB:
    """Pull the published reference DB on first boot (the `db pull` flow)."""
    db = ReferenceDB()
    if len(db) == 0:
        url = os.environ.get("MODELDNA_DB_URL") or DEFAULT_DB_URL
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "refdb.tar.gz"
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(archive, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            db.import_archive(archive)
    if len(db) == 0:
        raise RuntimeError("reference DB is empty after pull — cannot serve scans")
    return db


DB = ensure_db()


def _preflight(repo_id: str, revision: str) -> int:
    """Reject missing/gated/oversized repos with a readable message."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError:
        raise gr.Error(
            f"{repo_id} is gated. This public Space can only scan repos its "
            "token has access to — run `modeldna scan` locally with your own "
            "HF login instead."
        )
    except RepositoryNotFoundError:
        raise gr.Error(f"{repo_id} not found on the Hub (or it is private).")
    except RevisionNotFoundError:
        raise gr.Error(f"revision {revision!r} does not exist in {repo_id}.")
    except HfHubHTTPError as e:
        raise gr.Error(f"Hub error while checking {repo_id}: {e}")
    weights = [s for s in (info.siblings or []) if s.rfilename.endswith(".safetensors")]
    if not weights:
        raise gr.Error(
            f"{repo_id} ships no .safetensors weights — modelDNA reads "
            "safetensors only (PyTorch .bin and GGUF are not supported)."
        )
    total = sum(s.size or 0 for s in weights)
    if total > MAX_WEIGHT_BYTES:
        raise gr.Error(
            f"{repo_id} weighs {total / 1e9:.0f} GB; this Space caps live scans "
            f"at {MAX_WEIGHT_BYTES / 1e9:.0f} GB of weights. Run "
            f"`modeldna scan {repo_id}` locally for big models."
        )
    return total


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _table(headers: list[tuple[str, bool]], rows: list[list[str]]) -> str:
    """headers: (label, right-aligned); cell strings are pre-escaped."""
    head = "".join(
        f"<th class='{'num' if right else ''}'>{label}</th>" for label, right in headers
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'num' if right else ''}'>{cell}</td>"
            for cell, (_, right) in zip(row, headers)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table class='mdna-table'><tr>{head}</tr>{body}</table>"


def _result_html(d: dict, elapsed: float, cached: bool) -> str:
    v = d["verdict"]
    vclass = v["verdict"]
    prob = v.get("probability")
    best = v.get("best_candidate")
    vcolor = _VERDICT_COLOR.get(vclass, "#566573")

    headline = f"<b>{html.escape(vclass)}</b>"
    if prob is not None and best and vclass not in ("NO_MATCH", "INSUFFICIENT"):
        headline += (
            f" · {prob * 100:.1f}% likely derived from {html.escape(best)}"
        )

    cons = d.get("consistency", {})
    cstatus = cons.get("status", "NO_CLAIM")
    ccolor = _CONSISTENCY_COLOR.get(cstatus, "#566573")
    claims = d.get("claims", {})
    claimed = ", ".join(html.escape(b) for b in claims.get("base_models", [])) or (
        "&quot;trained from scratch&quot;"
        if claims.get("claims_from_scratch")
        else "none stated"
    )

    parts = [
        "<div class='mdna-card'>",
        f"<div class='mdna-eyebrow'>VERDICT — {html.escape(d.get('target', ''))}</div>",
        f"<div class='mdna-verdict' style='background:{vcolor}'>{headline}</div>",
        f"<p class='mdna-desc'>{html.escape(v.get('description', ''))}</p>",
        "<p class='mdna-claims'>claimed lineage: "
        f"<b>{claimed}</b>&ensp;"
        f"<span class='mdna-pill' style='background:{ccolor}'>{html.escape(cstatus)}</span>"
        f"<br><span class='mdna-dim'>{html.escape(cons.get('detail', ''))}</span></p>",
    ]

    best_cand = next(
        (c for c in v.get("candidates", []) if c["candidate_id"] == best), None
    )
    if best_cand:
        bg = d.get("background_ranges", {})
        parts.append(_table(
            [("signal", False), ("value", True), ("unrelated background", True)],
            [
                [
                    html.escape(label),
                    f"<b>{_fmt(best_cand['evidence'].get(ev_key))}</b>",
                    f"<span class='mdna-dim'>{html.escape(bg.get(bg_key, '—'))}</span>",
                ]
                for label, ev_key, bg_key in _SIGNALS
            ],
        ))

    cands = [c for c in v.get("candidates", []) if c.get("probability") is not None]
    if len(cands) > 1:
        parts.append(_table(
            [("#", False), ("candidate", False), ("family", False),
             ("P(derived)", True)],
            [
                [str(i), html.escape(c["candidate_id"]),
                 html.escape(c["family"]), f"{c['probability']:.3f}"]
                for i, c in enumerate(cands[:5], 1)
            ],
        ))

    for n in v.get("notes", []):
        parts.append(f"<p class='mdna-dim mdna-note'>note: {html.escape(n)}</p>")

    report_doc = html.escape(render_html(d))
    raw = html.escape(json.dumps(d, indent=1))
    parts.append(
        "<details class='mdna-details'><summary>Full evidence report</summary>"
        f"<iframe sandbox loading='lazy' srcdoc=\"{report_doc}\"></iframe></details>"
        "<details class='mdna-details'><summary>Raw JSON</summary>"
        f"<pre>{raw}</pre></details>"
    )

    stamp = (
        f"fast mode · {d.get('bytes_read', 0) / 1e6:.0f} MB read · "
        f"{elapsed:.0f} s · reference DB v{d.get('db_version')} "
        f"({len(DB)} bases) · modeldna {__version__}"
    )
    if cached:
        stamp = "cached result · " + stamp
    parts.append(f"<div class='mdna-stamp'>{stamp}</div></div>")
    return "".join(parts)


def do_scan(repo_id: str, revision: str, progress=gr.Progress()):
    repo_id = (repo_id or "").strip().strip("/")
    revision = (revision or "main").strip() or "main"
    if not REPO_RE.match(repo_id):
        raise gr.Error("enter a Hub repo id like `org/model-name`")
    if not REVISION_RE.match(revision):
        raise gr.Error("revision contains unexpected characters")

    key = f"{repo_id}@{revision}#db{DB.version}"
    if key in _CACHE:
        d, elapsed = _CACHE[key]
        return _result_html(d, elapsed, cached=True)

    progress(0.05, desc=f"checking {repo_id} on the Hub")
    total = _preflight(repo_id, revision)
    progress(
        0.15,
        desc=f"{total / 1e9:.1f} GB of weights — reading sampled slices "
        "(a 7B takes ~1–2 min)",
    )

    t0 = time.time()
    try:
        res = scan(repo_id, db=DB, mode="fast", revision=revision)
    except READ_ERRORS as e:
        raise gr.Error(f"could not read {repo_id}: {e}")
    elapsed = time.time() - t0

    progress(0.95, desc="rendering report")
    d = res.to_dict()
    _CACHE[key] = (d, elapsed)
    return _result_html(d, elapsed, cached=False)


def _decompose_html(dec: dict, elapsed: float, cached: bool) -> str:
    color = _SUMMARY_COLOR.get(dec["summary"], "#566573")
    parts = [
        "<div class='mdna-card'>",
        f"<div class='mdna-eyebrow'>MIXTURE — {html.escape(dec['target_id'])}</div>",
        f"<div class='mdna-verdict' style='background:{color}'>"
        f"<b>{html.escape(dec['summary'])}</b></div>",
        f"<p class='mdna-desc'>{html.escape(dec.get('description', ''))}</p>",
    ]

    ordered = sorted(dec["parent_ids"], key=lambda p: -dec["alphas"][p])
    if dec.get("base_id"):
        ordered.append(dec["base_id"])
    rows = []
    for cid in ordered:
        label = html.escape(cid) + (
            " <span class='mdna-dim'>(base)</span>" if cid == dec.get("base_id") else ""
        )
        f2 = dec.get("f2_alphas", {}).get(cid)
        rows.append([
            label,
            f"<b>{dec['alphas'][cid]:+.3f}</b>",
            _fmt(dec.get("alpha_spread", {}).get(cid)),
            _fmt(f2),
        ])
    parts.append(_table(
        [("candidate", False), ("weight", True), ("± roles", True),
         ("F2 check", True)],
        rows,
    ))

    parts.append(
        "<p class='mdna-claims'>reconstruction cosine "
        f"<b>{dec['recon_cos']:.5f}</b> · best single candidate "
        f"{html.escape(dec['best_single'])} at {dec['best_single_cos']:.5f}<br>"
        f"<span class='mdna-dim'>the mixture removes "
        f"{max(dec['gain_vs_single'], 0) * 100:.1f}% of the residual the best "
        f"single candidate leaves ({dec['n_samples']:,} sampled elements)</span></p>"
    )

    for n in dec.get("notes", []):
        parts.append(f"<p class='mdna-dim mdna-note'>note: {html.escape(n)}</p>")

    if dec.get("mergekit_yaml"):
        parts.append(
            "<details class='mdna-details'><summary>Nearest linear mergekit config"
            f"</summary><pre>{html.escape(dec['mergekit_yaml'])}</pre></details>"
        )
    parts.append(
        "<details class='mdna-details'><summary>Raw JSON</summary>"
        f"<pre>{html.escape(json.dumps(dec, indent=1))}</pre></details>"
    )

    stamp = (
        f"fast mode · {elapsed:.0f} s · reference DB v{DB.version} · "
        f"modeldna {__version__}"
    )
    if cached:
        stamp = "cached result · " + stamp
    parts.append(f"<div class='mdna-stamp'>{stamp}</div></div>")
    return "".join(parts)


def do_decompose(target: str, parents_raw: str, base: str, progress=gr.Progress()):
    target = (target or "").strip().strip("/")
    base = (base or "").strip().strip("/") or None
    parents = [p.strip().strip("/") for p in re.split(r"[,\s]+", parents_raw or "") if p.strip()]

    if not REPO_RE.match(target):
        raise gr.Error("enter the merge's Hub repo id like `org/model-name`")
    for p in parents + ([base] if base else []):
        if not REPO_RE.match(p):
            raise gr.Error(f"{p!r} doesn't look like a Hub repo id (org/model-name)")
    if len(parents) < 2:
        raise gr.Error("give at least two candidate parents (comma or space separated)")
    if len(parents) > MAX_PARENTS:
        raise gr.Error(f"at most {MAX_PARENTS} candidate parents per request on this Space")
    if len(set(parents + [target] + ([base] if base else []))) != len(parents) + 1 + bool(base):
        raise gr.Error("target, parents, and base must all be different repos")

    key = "|".join([target] + sorted(parents) + [base or ""]) + f"#db{DB.version}"
    if key in _DEC_CACHE:
        d, elapsed = _DEC_CACHE[key]
        return _decompose_html(d, elapsed, cached=True)

    todo = [target] + parents + ([base] if base else [])
    fetches = [r for r in todo if DB.get(r) is None]
    for i, repo in enumerate(todo):
        progress(0.02 + 0.06 * i / len(todo), desc=f"checking {repo} on the Hub")
        _preflight(repo, "main")
    progress(
        0.1,
        desc=f"fingerprinting {len(fetches)} model(s) "
        f"({len(todo) - len(fetches)} already in the reference DB) — "
        "roughly a minute per 7B",
    )

    t0 = time.time()
    try:
        dec = decompose_targets(target, parents, base=base, db=DB, mode="fast")
    except READ_ERRORS as e:
        raise gr.Error(f"could not read weights: {e}")
    except ValueError as e:
        raise gr.Error(str(e))
    elapsed = time.time() - t0

    progress(0.95, desc="rendering")
    d = dec.to_dict()
    d["mergekit_yaml"] = mergekit_yaml(dec)
    _DEC_CACHE[key] = (d, elapsed)
    return _decompose_html(d, elapsed, cached=False)


CSS = """
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; }
footer { display: none !important; }

.gradio-container {
  --mdna-ink: #0b0b0b; --mdna-ink2: #52514e; --mdna-muted: #898781;
  --mdna-grid: #e1e0d9; --mdna-axis: #c3c2b7; --mdna-accent: #2a78d6;
  --mdna-card: #ffffff; --mdna-surface: #fcfcfb;
  --mdna-shadow: 0 10px 30px rgba(11,11,11,.08);
}
.dark .gradio-container, .gradio-container.dark {
  --mdna-ink: #ffffff; --mdna-ink2: #c3c2b7; --mdna-muted: #898781;
  --mdna-grid: #2c2c2a; --mdna-axis: #383835; --mdna-accent: #3987e5;
  --mdna-card: #222221; --mdna-surface: #1a1a19;
  --mdna-shadow: 0 10px 30px rgba(0,0,0,.5);
}

.mdna-mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }

/* the theme font ships only 400 and 600 faces — <b> must use the real 600,
   never a synthetic 700 (faux bold thins diagonals: the A in DNA) */
.mdna-brand b, .mdna-thesis b, .mdna-card b, .mdna-foot b, .mdna-verdict b {
  font-weight: 600; }
.mdna-brand h1 { font-weight: 400; }

/* header */
.mdna-brand { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  margin: 6px 0 2px; }
.mdna-brand h1 { font-size: 28px; margin: 0; letter-spacing: -.01em;
  color: var(--mdna-ink); }
.mdna-brand h1 .dna { color: var(--mdna-accent); }
.mdna-tag { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
  letter-spacing: .24em; color: var(--mdna-ink2);
  border: 1px solid var(--mdna-axis); border-radius: 999px;
  padding: 4px 12px 3px; background: var(--mdna-surface); }
.mdna-links { display: flex; gap: 8px; }
.mdna-links a { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
  letter-spacing: .12em; text-decoration: none; color: var(--mdna-ink2);
  border: 1px solid var(--mdna-axis); border-radius: 999px;
  padding: 5px 13px 4px; background: var(--mdna-surface); }
.mdna-links a:hover { border-color: var(--mdna-ink2); color: var(--mdna-ink); }
.mdna-thesis { max-width: 68ch; color: var(--mdna-ink2); margin: 10px 0 4px;
  font-size: 15px; line-height: 1.55; text-wrap: balance; }
.mdna-thesis b { color: var(--mdna-ink); }

/* scan / decompose buttons */
#mdna-scan, #mdna-decompose { background: var(--mdna-accent) !important;
  color: #fff !important; border: none !important; box-shadow: none !important; }
#mdna-scan:hover, #mdna-decompose:hover { filter: brightness(1.08); }
#mdna-decompose { align-self: end; }

/* result card */
.mdna-card { border: 1px solid var(--mdna-grid); border-radius: 14px;
  background: var(--mdna-card); box-shadow: var(--mdna-shadow);
  padding: 20px 22px 14px; margin: 6px 0 2px; color: var(--mdna-ink);
  line-height: 1.5; }
.mdna-eyebrow { font-family: ui-monospace, Consolas, monospace;
  font-size: 11.5px; letter-spacing: .22em; color: var(--mdna-muted);
  margin-bottom: 10px; }
.mdna-verdict { display: inline-block; padding: .38rem .9rem;
  border-radius: 8px; color: #fff; font-size: 1.06rem; }
.mdna-desc { color: var(--mdna-ink2); margin: .6rem 0 .2rem; }
.mdna-claims { margin: .6rem 0; }
.mdna-pill { display: inline-block; padding: .1rem .55rem; border-radius: 999px;
  color: #fff; font-size: .8rem; font-weight: 600; vertical-align: 1px; }
.mdna-dim { color: var(--mdna-muted); font-size: .9rem; }
.mdna-note { margin: .25rem 0; }
.mdna-table { border-collapse: collapse; margin: .75rem 0; width: 100%; }
.mdna-table th, .mdna-table td { text-align: left; padding: .32rem .6rem;
  border-bottom: 1px solid var(--mdna-grid); font-size: .92rem; }
.mdna-table th { font-family: ui-monospace, Consolas, monospace;
  font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--mdna-muted); border-bottom: 1px solid var(--mdna-axis); }
.mdna-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.mdna-details { margin: .55rem 0; border: 1px solid var(--mdna-grid);
  border-radius: 10px; background: var(--mdna-surface); }
.mdna-details summary { cursor: pointer; padding: .55rem .9rem;
  font-family: ui-monospace, Consolas, monospace; font-size: 12px;
  letter-spacing: .14em; color: var(--mdna-ink2); }
.mdna-details[open] summary { border-bottom: 1px solid var(--mdna-grid); }
.mdna-details iframe { width: 100%; height: 1250px; border: none;
  border-radius: 0 0 10px 10px; background: #fff; display: block; }
.mdna-details pre { margin: 0; padding: .8rem 1rem; max-height: 480px;
  overflow: auto; font-size: 12px; }
.mdna-stamp { color: var(--mdna-muted); font-size: .82rem; margin-top: .8rem;
  font-family: ui-monospace, Consolas, monospace; }

/* footer */
.mdna-foot { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
  margin-top: 10px; padding-top: 16px; border-top: 1px solid var(--mdna-grid); }
@media (max-width: 760px) { .mdna-foot { grid-template-columns: 1fr; } }
.mdna-foot h4 { font-family: ui-monospace, Consolas, monospace;
  font-size: 11.5px; letter-spacing: .22em; color: var(--mdna-muted);
  margin: 0 0 6px; font-weight: 600; }
.mdna-foot p { color: var(--mdna-ink2); font-size: .86rem; line-height: 1.55;
  margin: 0; }
.mdna-foot code { font-size: .8rem; }
.mdna-foot a { color: var(--mdna-accent); text-decoration: none; }
.mdna-foot a:hover { text-decoration: underline; }
"""

HEADER = f"""
<div class="mdna-brand">
  <h1><span class="dna">🧬</span> model<b>DNA</b></h1>
  <span class="mdna-tag">LIVE SCANNER</span>
  <div class="mdna-links">
    <a href="{ATLAS_URL}" target="_blank">ATLAS ↗</a>
    <a href="{REPO_URL}" target="_blank">GITHUB ↗</a>
  </div>
</div>
<p class="mdna-thesis">Paste an open-weight model's Hub repo id. modelDNA
reads a few hundred MB of <b>sampled weight slices</b> — never the full
checkpoint — fingerprints them, and reports with a calibrated probability
which of <b>{len(DB)} indexed base models</b> it descends from. When the
evidence doesn't single out a parent, it says so instead of guessing.</p>
"""

FOOTER = f"""
<div class="mdna-foot">
  <div>
    <h4>READING THE VERDICT</h4>
    <p><code>NO_MATCH</code> means <i>no indexed parent matched</i> — the
    model may be trained from scratch <b>or</b> descend from a base that
    isn't in the reference DB yet. Thresholds are deliberately conservative:
    ambiguous evidence resolves to abstention, not to a confident claim.</p>
  </div>
  <div>
    <h4>LIMITS</h4>
    <p>Safetensors repos only (no <code>.bin</code>, no GGUF). Scans here
    are capped at {MAX_WEIGHT_BYTES / 1e9:.0f}&nbsp;GB of weights. Gated
    repos (Llama, Gemma…) need your own HF login — scan those locally.</p>
  </div>
  <div>
    <h4>RUN IT YOURSELF</h4>
    <p><code>pip install "modeldna @ git+{REPO_URL}"</code><br>
    then <code>modeldna db pull</code> · <code>modeldna scan org/model</code>.
    Source &amp; method docs on
    <a href="{REPO_URL}" target="_blank">GitHub</a>, the reconstructed
    family tree in the <a href="{ATLAS_URL}" target="_blank">Atlas</a>.</p>
  </div>
</div>
"""

EXAMPLES = [
    ["HuggingFaceTB/SmolLM2-1.7B-Instruct"],
    ["HuggingFaceH4/zephyr-7b-beta"],
    ["teknium/OpenHermes-2.5-Mistral-7B"],
    ["cognitivecomputations/dolphin-2.9-llama3-8b"],
]

#: real merges validated against their published mergekit configs
#: (benchmarks/merge_decompose_bench.py in the repo)
DECOMPOSE_EXAMPLES = [
    ["mlabonne/NeuralPipe-7B-slerp",
     "OpenPipe/mistral-ft-optimized-1218, mlabonne/NeuralHermes-2.5-Mistral-7B",
     ""],
    ["mlabonne/Monarch-7B",
     "mlabonne/OmniTruthyBeagle-7B-v0, mlabonne/NeuBeagle-7B, "
     "mlabonne/NeuralOmniBeagle-7B",
     "mistralai/Mistral-7B-v0.1"],
]

with gr.Blocks(title="modelDNA — lineage scanner") as demo:
    gr.HTML(HEADER)
    with gr.Tab("Scan"):
        with gr.Row():
            repo_in = gr.Textbox(
                label="Hub repo id",
                placeholder="org/model — e.g. HuggingFaceH4/zephyr-7b-beta",
                scale=5,
            )
            rev_in = gr.Textbox(label="revision", value="main", scale=1)
            scan_btn = gr.Button("Scan", variant="primary", scale=1,
                                 elem_id="mdna-scan")
        result_out = gr.HTML()
        gr.Examples(examples=EXAMPLES, inputs=[repo_in])
    with gr.Tab("Decompose a merge"):
        gr.HTML(
            "<p class='mdna-thesis'>Name a suspected <b>merge</b> and its "
            "candidate parents; modelDNA fits the merged weights as a "
            "sum-to-one mixture of the parents and reports each parent's "
            "share — the estimated merge recipe, recovered from weights "
            "alone. Both examples below are real merges whose fitted weights "
            "match the mergekit config published on their model card.</p>"
        )
        dec_target_in = gr.Textbox(
            label="merged model",
            placeholder="org/suspected-merge",
        )
        with gr.Row():
            dec_parents_in = gr.Textbox(
                label=f"candidate parents (2–{MAX_PARENTS}, comma separated)",
                placeholder="org/parent-a, org/parent-b",
                scale=4,
            )
            dec_base_in = gr.Textbox(
                label="shared base (optional)",
                placeholder="org/base-model",
                scale=2,
            )
            dec_btn = gr.Button("Decompose", variant="primary", scale=1,
                                elem_id="mdna-decompose")
        dec_out = gr.HTML()
        gr.Examples(examples=DECOMPOSE_EXAMPLES,
                    inputs=[dec_target_in, dec_parents_in, dec_base_in])
    gr.HTML(FOOTER)

    scan_btn.click(do_scan, inputs=[repo_in, rev_in], outputs=[result_out],
                   concurrency_limit=1)
    repo_in.submit(do_scan, inputs=[repo_in, rev_in], outputs=[result_out],
                   concurrency_limit=1)
    dec_btn.click(do_decompose, inputs=[dec_target_in, dec_parents_in, dec_base_in],
                  outputs=[dec_out], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(css=CSS)
