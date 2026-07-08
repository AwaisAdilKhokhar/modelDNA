"""Rich terminal rendering of scan and compare results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modeldna.compare import Evidence
from modeldna.merge import MergeDecomposition
from modeldna.scan import ScanResult
from modeldna.verdict import DESCRIPTIONS, Verdict, VerdictClass

_VERDICT_STYLE = {
    VerdictClass.EXACT_COPY: "bold red",
    VerdictClass.QUANTIZED_COPY: "bold red",
    VerdictClass.FINE_TUNE: "bold yellow",
    VerdictClass.SAME_LINEAGE: "bold yellow",
    VerdictClass.LIKELY_MERGE: "bold magenta",
    VerdictClass.SAME_FAMILY_UNRESOLVED: "bold cyan",
    VerdictClass.NO_MATCH: "bold green",
    VerdictClass.INSUFFICIENT: "bold white",
}

# ASCII-only labels: legacy Windows consoles (cp1252) choke on ✓ / ⚠
_CONSISTENCY_STYLE = {
    "CONSISTENT": ("[OK] CONSISTENT", "green"),
    "INCONSISTENT": ("[!] INCONSISTENT", "bold red"),
    "UNVERIFIED": ("[?] UNVERIFIED", "yellow"),
    "NO_CLAIM": ("[-] NO CLAIM", "dim"),
}

LIMITATIONS = (
    "Limitations: statistical evidence, not proof. Merges are flagged here; "
    "`modeldna decompose` estimates mixture weights when candidate parents are "
    "known; distillation is undetectable from weights by construction; "
    "determined adversarial re-parameterization can defeat direct-similarity "
    "signals (spectral signals are the countermeasure and are reported "
    "separately). Findings are statements of statistical consistency, never "
    "accusations."
)


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _headline(verdict: Verdict) -> Text:
    style = _VERDICT_STYLE[verdict.verdict]
    t = Text()
    t.append(verdict.verdict.value, style=style)
    if verdict.probability is not None and verdict.best is not None:
        if verdict.verdict not in (VerdictClass.NO_MATCH, VerdictClass.INSUFFICIENT):
            t.append(
                f"  {verdict.probability * 100:.1f}% likely derived from ",
                style="bold",
            )
            t.append(verdict.best.candidate_id, style="bold white")
        else:
            t.append(
                f"  best candidate {verdict.best.candidate_id} at "
                f"p={verdict.probability:.3f}",
                style="dim",
            )
    return t


def _evidence_table(ev: Evidence, background_ranges: dict[str, str] | None) -> Table:
    bg = background_ranges or {}
    table = Table(title="Evidence", title_justify="left", show_edge=False)
    table.add_column("signal")
    table.add_column("value", justify="right")
    table.add_column("unrelated background", justify="right", style="dim")
    rows = [
        ("attention sigma-curve correlation (F1)", ev.sigma_r_mean, bg.get("sigma_r", "")),
        ("norm/bias vector cosine (F2)", ev.vector_cos_mean, bg.get("vector_cos", "")),
        ("sampled parameter cosine / PCS (F3)", ev.pcs_cos_mean, bg.get("pcs_cos", "")),
        (f"SVD spectrum correlation (F4, {ev.spectra_kind or 'n/a'})",
         ev.spectra_r_mean, bg.get("spectra_r", "")),
    ]
    for name, value, rng in rows:
        table.add_row(name, _fmt(value), rng or "n/a")
    return table


def print_scan(result: ScanResult, top_k: int = 3, console: Console | None = None) -> None:
    console = console or Console()
    v = result.verdict

    console.print(Panel(_headline(v), title=f"modelDNA scan · {result.target}"))
    console.print(f"[dim]{v.verdict.value}: {DESCRIPTIONS[v.verdict]}[/dim]\n")

    arch = result.fingerprint.arch
    console.print(f"architecture  {arch.summary()}")

    label, style = _CONSISTENCY_STYLE[result.consistency["status"]]
    claimed = ", ".join(result.claims.base_models) or (
        '"from scratch"' if result.claims.from_scratch else "none"
    )
    console.print(f"claimed lineage  {claimed}   [{style}]{label}[/{style}]")
    console.print(f"[dim]{result.consistency['detail']}[/dim]\n")

    d = result.to_dict()
    if v.best is not None:
        console.print(_evidence_table(v.best.evidence, d["background_ranges"]))
        console.print()

    ranked = [c for c in v.candidates if c.probability is not None][:top_k]
    if ranked:
        t = Table(title=f"Top {len(ranked)} candidates", title_justify="left",
                  show_edge=False)
        t.add_column("#", justify="right")
        t.add_column("candidate")
        t.add_column("family")
        t.add_column("P(derived)", justify="right")
        for i, c in enumerate(ranked, 1):
            t.add_row(str(i), c.candidate_id, c.family, f"{c.probability:.3f}")
        console.print(t)
        console.print()

    for note in v.notes:
        console.print(f"[yellow]note[/yellow] {note}")
    for c in v.candidates:
        for note in c.evidence.notes:
            console.print(f"[dim]note ({c.candidate_id}): {note}[/dim]")

    mb = result.bytes_read / 1e6
    console.print(
        f"\n[dim]db v{result.db_version} | {result.mode} mode | "
        f"{mb:.1f} MB downloaded | modelDNA {result.tool_version}[/dim]"
    )
    console.print(f"[dim]{LIMITATIONS}[/dim]")


_SUMMARY_STYLE = {
    "MERGE": "bold magenta",
    "SINGLE_PARENT": "bold yellow",
    "AMBIGUOUS": "bold cyan",
    "UNEXPLAINED": "bold green",
}


def print_decompose(
    dec: MergeDecomposition,
    show_layers: bool = False,
    console: Console | None = None,
) -> None:
    console = console or Console()
    head = Text()
    head.append(dec.summary, style=_SUMMARY_STYLE[dec.summary])
    head.append(f"  {dec.description}", style="bold")
    console.print(Panel(head, title=f"modelDNA decompose · {dec.target_id}"))

    t = Table(title="Mixture fit (sum-to-one least squares on F3 samples)",
              title_justify="left", show_edge=False)
    t.add_column("candidate")
    t.add_column("weight", justify="right")
    t.add_column("± roles", justify="right", style="dim")
    t.add_column("F2 check", justify="right", style="dim")
    ordered = sorted(dec.parent_ids, key=lambda p: -dec.alphas[p])
    if dec.base_id:
        ordered.append(dec.base_id)
    for cid in ordered:
        label = f"{cid} [dim](base)[/dim]" if cid == dec.base_id else cid
        t.add_row(
            label,
            f"{dec.alphas[cid]:+.3f}",
            _fmt(dec.alpha_spread.get(cid)),
            _fmt(dec.f2_alphas.get(cid)),
        )
    console.print(t)

    console.print(
        f"\nreconstruction cosine [bold]{dec.recon_cos:.5f}[/bold]   "
        f"best single candidate {dec.best_single} at {dec.best_single_cos:.5f}"
    )
    console.print(
        f"mixture removes [bold]{max(dec.gain_vs_single, 0.0) * 100:.1f}%[/bold] of the "
        f"residual the best single candidate leaves "
        f"[dim]({dec.n_samples:,} sampled elements)[/dim]"
    )

    if show_layers and dec.per_layer:
        lt = Table(title="Per-layer mixture", title_justify="left", show_edge=False)
        lt.add_column("layer", justify="right")
        for cid in ordered:
            lt.add_column(cid.split("/")[-1], justify="right")
        n_layers = len(next(iter(dec.per_layer.values())))
        for layer in range(n_layers):
            lt.add_row(str(layer),
                       *(f"{dec.per_layer[cid][layer]:+.2f}" for cid in ordered))
        console.print()
        console.print(lt)

    for note in dec.notes:
        console.print(f"[yellow]note[/yellow] {note}")
    console.print(f"\n[dim]{LIMITATIONS}[/dim]")


def mergekit_yaml(dec: MergeDecomposition) -> str:
    """An equivalent `merge_method: linear` mergekit config for the fitted mix."""
    lines = ["# nearest linear mergekit config fitted by modeldna decompose",
             "merge_method: linear", "models:"]
    for cid in dec.parent_ids:
        lines += [f"  - model: {cid}", "    parameters:",
                  f"      weight: {dec.alphas[cid]:.4f}"]
    if dec.base_id:
        lines += [f"  - model: {dec.base_id}  # shared base", "    parameters:",
                  f"      weight: {dec.alphas[dec.base_id]:.4f}"]
    lines.append("dtype: bfloat16")
    return "\n".join(lines) + "\n"


def print_compare(
    ev: Evidence,
    verdict: Verdict,
    console: Console | None = None,
) -> None:
    console = console or Console()
    t = Text()
    t.append(f"{ev.suspect_id}  vs  {ev.candidate_id}\n", style="bold")
    t.append_text(_headline(verdict))
    console.print(Panel(t, title="modelDNA compare"))
    console.print(f"[dim]{verdict.verdict.value}: {DESCRIPTIONS[verdict.verdict]}[/dim]\n")
    console.print(_evidence_table(ev, None))
    if ev.sigma_r:
        t = Table(title="sigma-curve correlation by projection", title_justify="left",
                  show_edge=False)
        t.add_column("projection")
        t.add_column("pearson r", justify="right")
        for role in sorted(ev.sigma_r):
            t.add_row(role, f"{ev.sigma_r[role]:.4f}")
        console.print()
        console.print(t)
    for note in verdict.notes + ev.notes:
        console.print(f"[yellow]note[/yellow] {note}")
    console.print(f"\n[dim]{LIMITATIONS}[/dim]")
