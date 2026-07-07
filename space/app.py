"""modelDNA live scanner — the Hugging Face Space frontend.

Paste a Hub repo id, get a calibrated lineage verdict. This is a thin
Gradio wrapper around `modeldna.scan`: on startup it pulls the prebuilt
reference DB (the same archive `modeldna db pull` uses), then each scan
reads only the sampled weight slices of the suspect model (~250 MB for a
7B), so a verdict lands in a couple of minutes on Space networking.

Deployed from the `space/` directory of
https://github.com/AwaisAdilKhokhar/modelDNA — edit there, not on the Hub.
"""

from __future__ import annotations

import html
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
from modeldna.report.html import _CONSISTENCY_COLOR, _VERDICT_COLOR, render_html
from modeldna.scan import scan

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
#: refuse live scans of repos whose safetensors exceed this (≈70B bf16);
#: a fast scan of something bigger stops being "live" on shared hardware
MAX_WEIGHT_BYTES = 180e9
READ_ERRORS = (SourceError, WeightIndexError, SafetensorsError)

REPO_URL = "https://github.com/AwaisAdilKhokhar/modelDNA"

_SIGNALS = [
    ("attention σ-curve correlation (F1)", "sigma_r_mean", "sigma_r"),
    ("norm/bias vector cosine (F2)", "vector_cos_mean", "vector_cos"),
    ("sampled parameter cosine / PCS (F3)", "pcs_cos_mean", "pcs_cos"),
    ("SVD spectrum correlation (F4)", "spectra_r_mean", "spectra_r"),
]

#: (repo@revision, db_version) -> (result dict, elapsed seconds)
_CACHE: dict[str, tuple[dict, float]] = {}


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


def _summary_html(d: dict, elapsed: float, cached: bool) -> str:
    v = d["verdict"]
    vclass = v["verdict"]
    prob = v.get("probability")
    best = v.get("best_candidate")
    vcolor = _VERDICT_COLOR.get(vclass, "#566573")

    headline = html.escape(vclass)
    if prob is not None and best and vclass not in ("NO_MATCH", "INSUFFICIENT"):
        headline += (
            f" <span style='font-weight:400'>· {prob * 100:.1f}% likely derived "
            f"from <b>{html.escape(best)}</b></span>"
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

    badge = (
        "display:inline-block;padding:.3rem .8rem;border-radius:6px;color:#fff;"
        "font-weight:600;font-size:1.05rem"
    )
    parts = [
        "<div style='font-family:ui-sans-serif,system-ui,sans-serif;line-height:1.5'>",
        f"<p><span style='{badge};background:{vcolor}'>{headline}</span></p>",
        f"<p style='color:#6c757d'>{html.escape(v.get('description', ''))}</p>",
        "<p>claimed lineage: <b>" + claimed + "</b> &nbsp; "
        f"<span style='{badge};font-size:.85rem;padding:.15rem .5rem;"
        f"background:{ccolor}'>{html.escape(cstatus)}</span><br>"
        f"<span style='color:#6c757d;font-size:.9rem'>"
        f"{html.escape(cons.get('detail', ''))}</span></p>",
    ]

    th = "text-align:left;padding:.3rem .6rem;border-bottom:1px solid #adb5bd"
    td = "padding:.3rem .6rem;border-bottom:1px solid #dee2e6"
    num = ";text-align:right"

    def _table(headers: list[tuple[str, bool]], rows: list[list[str]]) -> str:
        # headers: (label, right-aligned); cell strings are pre-escaped
        head = "".join(
            f"<th style='{th}{num if right else ''}'>{label}</th>"
            for label, right in headers
        )
        body = "".join(
            "<tr>" + "".join(
                f"<td style='{td}{num if right else ''}'>{cell}</td>"
                for cell, (_, right) in zip(row, headers)
            ) + "</tr>"
            for row in rows
        )
        return ("<table style='border-collapse:collapse;margin:.5rem 0'>"
                f"<tr>{head}</tr>{body}</table>")

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
                    f"<span style='color:#6c757d'>{html.escape(bg.get(bg_key, '—'))}</span>",
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
        parts.append(f"<p style='color:#6c757d;font-size:.9rem'>note: {html.escape(n)}</p>")

    stamp = (
        f"fast mode · {d.get('bytes_read', 0) / 1e6:.0f} MB read · "
        f"{elapsed:.0f} s · reference DB v{d.get('db_version')} "
        f"({len(DB)} bases) · modeldna {__version__}"
    )
    if cached:
        stamp = "cached result · " + stamp
    parts.append(f"<p style='color:#adb5bd;font-size:.85rem'>{stamp}</p></div>")
    return "".join(parts)


def _report_iframe(d: dict) -> str:
    doc = render_html(d)
    return (
        f"<iframe sandbox srcdoc=\"{html.escape(doc)}\" "
        "style='width:100%;height:1300px;border:1px solid #dee2e6;"
        "border-radius:8px;background:#fff'></iframe>"
    )


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
        return _summary_html(d, elapsed, cached=True), _report_iframe(d), d

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
    return _summary_html(d, elapsed, cached=False), _report_iframe(d), d


HEADER = f"""
# 🧬 modelDNA — live lineage scanner

Paste an open-weight model's Hub repo id. modelDNA reads a few hundred MB of
**sampled weight slices** (never the full checkpoint), fingerprints them, and
reports — with a calibrated probability — which of **{len(DB)} indexed base
models** the model descends from. When the evidence doesn't single out a
parent, it says so instead of guessing.
"""

FOOTER = f"""
----
**Reading the verdict.** `NO_MATCH` means *no indexed parent matched* — the
model may be trained from scratch **or** descend from a base that isn't in the
reference DB yet. Thresholds are deliberately conservative: the worst failure
mode of a tool like this is a false accusation, so ambiguous evidence resolves
to `SAME_FAMILY_UNRESOLVED` or `NO_MATCH`, not to a confident claim.

**Limits.** Safetensors repos only (no `.bin`, no GGUF); scans here are capped
at {MAX_WEIGHT_BYTES / 1e9:.0f} GB of weights; gated repos (Llama, Gemma…)
need your own HF login, so scan those locally.

**Run it yourself:** `pip install "modeldna @ git+{REPO_URL}"`,
then `modeldna db pull` (seconds) and `modeldna scan org/model`.
Source, method docs and the interactive family-tree Atlas:
[github.com/AwaisAdilKhokhar/modelDNA]({REPO_URL}).
"""

EXAMPLES = [
    ["HuggingFaceTB/SmolLM2-1.7B-Instruct"],
    ["HuggingFaceH4/zephyr-7b-beta"],
    ["teknium/OpenHermes-2.5-Mistral-7B"],
    ["cognitivecomputations/dolphin-2.9-llama3-8b"],
]

with gr.Blocks(title="modelDNA — lineage scanner") as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        repo_in = gr.Textbox(
            label="Hub repo id",
            placeholder="org/model — e.g. HuggingFaceH4/zephyr-7b-beta",
            scale=4,
        )
        rev_in = gr.Textbox(label="revision", value="main", scale=1)
        scan_btn = gr.Button("Scan", variant="primary", scale=1)
    summary_out = gr.HTML()
    with gr.Accordion("Full evidence report", open=False):
        report_out = gr.HTML()
    with gr.Accordion("Raw JSON", open=False):
        json_out = gr.JSON()
    gr.Examples(examples=EXAMPLES, inputs=[repo_in])
    gr.Markdown(FOOTER)

    scan_btn.click(
        do_scan,
        inputs=[repo_in, rev_in],
        outputs=[summary_out, report_out, json_out],
        concurrency_limit=1,
    )
    repo_in.submit(
        do_scan,
        inputs=[repo_in, rev_in],
        outputs=[summary_out, report_out, json_out],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
