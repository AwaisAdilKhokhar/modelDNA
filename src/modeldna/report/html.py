"""Self-contained HTML evidence report (no external assets, inline SVG plots).

Renders from a ScanResult dict so `modeldna explain` can rebuild a report
from a saved scan without re-downloading anything.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np

_VERDICT_COLOR = {
    "EXACT_COPY": "#c0392b",
    "QUANTIZED_COPY": "#c0392b",
    "FINE_TUNE": "#d68910",
    "SAME_LINEAGE": "#d68910",
    "LIKELY_MERGE": "#8e44ad",
    "SAME_FAMILY_UNRESOLVED": "#2471a3",
    "NO_MATCH": "#1e8449",
    "INSUFFICIENT": "#566573",
}

_CONSISTENCY_COLOR = {
    "CONSISTENT": "#1e8449",
    "INCONSISTENT": "#c0392b",
    "UNVERIFIED": "#d68910",
    "NO_CLAIM": "#566573",
}

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 900px; padding: 0 1rem; color: #212529; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
.badge { display: inline-block; padding: .25rem .7rem; border-radius: 4px;
         color: #fff; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #dee2e6;
         font-size: .92rem; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: #6c757d; font-size: .88rem; }
.plot { margin: 1rem 0; }
.footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #dee2e6;
          font-size: .85rem; color: #6c757d; }
code { background: #f1f3f5; padding: .1rem .3rem; border-radius: 3px; }
"""


def _svg_curves(
    series: list[tuple[str, list[float], str]],
    title: str,
    width: int = 400,
    height: int = 180,
) -> str:
    """Small line chart: z-normalized per-layer curves overlaid."""
    pad_l, pad_r, pad_t, pad_b = 36, 10, 26, 22
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    all_vals = [v for _, vals, _ in series for v in vals]
    if not all_vals:
        return ""
    lo, hi = min(all_vals), max(all_vals)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    n = max(len(vals) for _, vals, _ in series)

    def x(i: int) -> float:
        return pad_l + (plot_w * i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{pad_l}" y="14" font-size="12" font-weight="600">'
        f"{html.escape(title)}</text>",
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#ced4da"/>',
    ]
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + plot_h * frac
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            'stroke="#e9ecef"/>'
        )
    parts.append(
        f'<text x="{pad_l - 4}" y="{pad_t + 10}" font-size="9" text-anchor="end" '
        f'fill="#6c757d">{hi:.2f}</text>'
        f'<text x="{pad_l - 4}" y="{pad_t + plot_h}" font-size="9" text-anchor="end" '
        f'fill="#6c757d">{lo:.2f}</text>'
        f'<text x="{pad_l + plot_w / 2}" y="{height - 6}" font-size="9" '
        f'text-anchor="middle" fill="#6c757d">layer</text>'
    )
    legend_x = pad_l + 6
    for label, vals, color in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            'stroke-width="1.8"/>'
        )
        parts.append(
            f'<rect x="{legend_x}" y="{pad_t + 5}" width="10" height="3" fill="{color}"/>'
            f'<text x="{legend_x + 14}" y="{pad_t + 10}" font-size="9" fill="#495057">'
            f"{html.escape(label)}</text>"
        )
        legend_x += 16 + 7 * len(label)
    parts.append("</svg>")
    return "".join(parts)


def _zs(vals: list[float]) -> list[float]:
    a = np.asarray(vals, dtype=np.float64)
    sd = a.std()
    return list((a - a.mean()) / sd) if sd > 0 else [0.0] * len(vals)


def _sigma_plots(d: dict[str, Any]) -> str:
    curves = d.get("curves", {})
    suspect = curves.get("suspect_sigma", {})
    best = curves.get("best_sigma", {})
    best_id = curves.get("best_id", "candidate")
    out = []
    for role in ("attn.q", "attn.k", "attn.v", "attn.o"):
        if role in suspect and role in best and len(suspect[role]) == len(best[role]):
            out.append(
                _svg_curves(
                    [
                        ("suspect", _zs(suspect[role]), "#c0392b"),
                        (best_id.split("/")[-1][:24], _zs(best[role]), "#2471a3"),
                    ],
                    f"σ-curve · {role} (z-normalized)",
                )
            )
    if not out:
        return ""
    return "<h2>Attention σ-curves</h2><div>" + "".join(out) + "</div>"


def _evidence_rows(d: dict[str, Any]) -> str:
    verdict = d["verdict"]
    cands = verdict.get("candidates", [])
    best = next(
        (c for c in cands if c["candidate_id"] == verdict.get("best_candidate")), None
    )
    if best is None:
        return ""
    ev = best["evidence"]
    bg = d.get("background_ranges", {})
    rows = [
        ("attention σ-curve correlation (F1)", ev.get("sigma_r_mean"), bg.get("sigma_r")),
        ("norm/bias vector cosine (F2)", ev.get("vector_cos_mean"), bg.get("vector_cos")),
        ("sampled parameter cosine / PCS (F3)", ev.get("pcs_cos_mean"), bg.get("pcs_cos")),
        (f"SVD spectrum correlation (F4, {ev.get('spectra_kind') or 'n/a'})",
         ev.get("spectra_r_mean"), bg.get("spectra_r")),
    ]
    body = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td class='num'>{'—' if v is None else f'{v:.4f}'}</td>"
        f"<td class='num dim'>{html.escape(rng or 'n/a')}</td></tr>"
        for name, v, rng in rows
    )
    return (
        "<h2>Evidence vs best candidate</h2>"
        "<table><tr><th>signal</th><th class='num'>value</th>"
        "<th class='num'>unrelated background</th></tr>" + body + "</table>"
    )


def _candidates_table(d: dict[str, Any], top_k: int = 5) -> str:
    cands = [
        c for c in d["verdict"].get("candidates", []) if c.get("probability") is not None
    ][:top_k]
    if not cands:
        return ""
    body = "".join(
        f"<tr><td>{i}</td><td>{html.escape(c['candidate_id'])}</td>"
        f"<td>{html.escape(c['family'])}</td>"
        f"<td class='num'>{c['probability']:.3f}</td></tr>"
        for i, c in enumerate(cands, 1)
    )
    return (
        "<h2>Ranked candidates</h2>"
        "<table><tr><th>#</th><th>candidate</th><th>family</th>"
        "<th class='num'>P(derived)</th></tr>" + body + "</table>"
    )


def render_html(d: dict[str, Any]) -> str:
    """Render a ScanResult dict (ScanResult.to_dict()) to standalone HTML."""
    verdict = d["verdict"]
    vclass = verdict["verdict"]
    vcolor = _VERDICT_COLOR.get(vclass, "#566573")
    prob = verdict.get("probability")
    best = verdict.get("best_candidate")

    headline = vclass
    if prob is not None and best and vclass not in ("NO_MATCH", "INSUFFICIENT"):
        headline = f"{vclass} · {prob * 100:.1f}% likely derived from {best}"

    cons = d.get("consistency", {})
    cstatus = cons.get("status", "NO_CLAIM")
    ccolor = _CONSISTENCY_COLOR.get(cstatus, "#566573")

    claims = d.get("claims", {})
    claimed = ", ".join(claims.get("base_models", [])) or (
        '"trained from scratch"' if claims.get("claims_from_scratch") else "none stated"
    )

    arch = d.get("arch", {})
    notes = verdict.get("notes", [])
    notes_html = "".join(f"<p class='dim'>note: {html.escape(n)}</p>" for n in notes)

    repro = f"modeldna scan {d['target']} --{d.get('mode', 'fast')} --report report.html"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>modelDNA report · {html.escape(d['target'])}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>modelDNA · lineage report</h1>
<p><code>{html.escape(d['target'])}</code></p>
<p><span class="badge" style="background:{vcolor}">{html.escape(headline)}</span></p>
<p class="dim">{html.escape(verdict.get('description', ''))}</p>

<h2>Claimed vs detected lineage</h2>
<table>
<tr><th>claimed in README</th><td>{html.escape(claimed)}</td></tr>
<tr><th>detected from weights</th><td>{html.escape(best or '—')}</td></tr>
<tr><th>consistency</th>
    <td><span class="badge" style="background:{ccolor}">{html.escape(cstatus)}</span>
        <span class="dim"> {html.escape(cons.get('detail', ''))}</span></td></tr>
</table>

<p class="dim">architecture: {html.escape(str(arch.get('model_type', '?')))} ·
{arch.get('n_layers', '?')} layers · hidden {arch.get('hidden_size', '?')} ·
vocab {arch.get('vocab_size', '?')} · ~{(arch.get('n_params') or 0) / 1e9:.2f}B params</p>

{_evidence_rows(d)}
{_sigma_plots(d)}
{_candidates_table(d)}
{notes_html}

<div class="footer">
<p>Statistical evidence, not proof. Merges are flagged but not attributed;
distillation is undetectable from weights by construction; adversarial
re-parameterization can defeat direct-similarity signals (spectral signals
are the countermeasure and are reported separately). Findings are statements
of statistical consistency with derivation, never accusations.</p>
<p>reference DB v{d.get('db_version', '?')} · scan mode: {html.escape(d.get('mode', ''))} ·
{(d.get('bytes_read') or 0) / 1e6:.1f} MB downloaded ·
modelDNA {html.escape(d.get('tool_version', ''))}</p>
<p>reproduce: <code>{html.escape(repro)}</code></p>
</div>
<script type="application/json" id="modeldna-data">
{json.dumps(d, indent=1)}
</script>
</body>
</html>
"""
