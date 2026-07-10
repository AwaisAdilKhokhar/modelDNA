"""Figure for the tech report: fitted per-layer slerp t-curves vs the published
mergekit config (NeuralPipe-7B-slerp), from benchmarks/merge_decompose_results.json.

Usage: python paper/make_figures.py
Writes paper/fig_slerp_tcurves.png (+ .pdf for LaTeX).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE.parent / "benchmarks" / "merge_decompose_results.json"

INK = "#0b0b0b"
INK_2 = "#52514e"  # published-config reference line (neutral ink, not a series hue)
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # fitted estimates (categorical slot 1)
SURFACE = "#ffffff"


def published_t_curves(n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    # mergekit interpolates the 5 t anchors evenly across the layer range
    # (same construction as benchmarks/merge_decompose_bench.py).
    anchors = np.linspace(0, n_layers - 1, 5)
    t_attn = np.interp(np.arange(n_layers), anchors, [0, 0.5, 0.3, 0.7, 1])
    t_mlp = np.interp(np.arange(n_layers), anchors, [1, 0.5, 0.7, 0.3, 0])
    return t_attn, t_mlp


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def main() -> None:
    res = json.loads(RESULTS.read_text())["neuralpipe_slerp"]
    attn_fit = np.asarray(res["attn_t_fitted"])
    mlp_fit = np.asarray(res["mlp_t_fitted"])
    n_layers = len(attn_fit)
    t_attn, t_mlp = published_t_curves(n_layers)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 3.3), dpi=220, sharey=True, facecolor=SURFACE
    )

    panels = [
        (axes[0], t_attn, attn_fit, "Attention (mean of Q/K/V/O)",
         res["attn_t_corr_vs_config"]),
        (axes[1], t_mlp, mlp_fit, "MLP (down projection)",
         res["mlp_t_corr_vs_config"]),
    ]
    for ax, published, fitted, title, r in panels:
        style_axis(ax)
        ax.plot(layers, published, color=INK_2, linewidth=1.8, zorder=2,
                solid_capstyle="round")
        ax.plot(layers, fitted, linestyle="none", marker="o", markersize=4.6,
                markerfacecolor=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=0.7, zorder=3)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8, loc="left")
        ax.text(0.99, 1.02, f"r = {r:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK_2)
        ax.set_xlabel("layer", fontsize=8.5, color=MUTED)
        ax.set_xticks([0, 8, 16, 24, 31])

    axes[0].set_ylabel("interpolation weight $t$\n(share of NeuralHermes)",
                       fontsize=8.5, color=MUTED)

    # direct labels (first panel), doubling as the legend for both
    axes[0].annotate("published config", xy=(20.5, t_attn[20] - 0.02),
                     xytext=(21.5, 0.30), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.7))
    axes[0].annotate("fitted from fingerprints", xy=(7, attn_fit[7] + 0.02),
                     xytext=(1.0, 0.72), fontsize=8, color=BLUE,
                     arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.7))

    # the one visible deviation: MLP layer 0, where the parents' task-vector
    # norm collapses and t is unidentifiable by construction
    axes[1].annotate("layer 0: parents near-identical,\n$t$ unidentifiable",
                     xy=(0, mlp_fit[0]), xytext=(3.2, 0.30), fontsize=7.5,
                     color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.7))

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_slerp_tcurves.{ext}", facecolor=SURFACE,
                    bbox_inches="tight")
    print("wrote", HERE / "fig_slerp_tcurves.png", "and .pdf")


if __name__ == "__main__":
    main()
