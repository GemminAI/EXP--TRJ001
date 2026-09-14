#!/usr/bin/env python3
"""Generate EXP--TRJ001 paper figures (academic PDF/PNG).

Figure 1  Experimental pipeline (Raw vs PCA branches)
Figure 2  Layer 8 / 16 / 24 AUROC with 95% CI
Figure 3  Hero comparison: PCA 64D, PC4--67, BL-LEN, Raw 3584D
Figure 4  Conceptual: global variance vs task-relevant geometry

Frozen metrics are taken from docs/Paper/phase3_metrics.json (rounded as
in the manuscript). The script does not re-estimate statistics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

try:
    import seaborn as sns  # type: ignore

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    sns = None
    HAS_SEABORN = False


OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe--Ito colorblind-safe palette
C = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
    "lightgray": "#D0D0D0",
    "conf": "#0072B2",
    "expl": "#D55E00",
    "paper": "#F7F4EF",
    "ink": "#1B1B1B",
}

# Frozen point estimates and CIs (manuscript rounding)
LAYER_AUROC = {
    8: {"auroc": 0.7090, "lo": 0.6554, "hi": 0.7599},
    16: {"auroc": 0.7806, "lo": 0.7298, "hi": 0.8232},
    24: {"auroc": 0.7167, "lo": 0.6570, "hi": 0.7693},
}

HERO = [
    {"name": "PCA 64D", "auroc": 0.7806, "kind": "exploratory", "delta": -0.0271, "p": 0.8685},
    {"name": "PC4--67", "auroc": 0.7969, "kind": "ablation", "delta": -0.0108, "p": 0.6935},
    {"name": "BL-LEN", "auroc": 0.8077, "kind": "baseline", "delta": 0.0, "p": None},
    {"name": "Raw 3584D", "auroc": 0.8472, "kind": "exploratory", "delta": 0.0395, "p": 0.004},
    {"name": "BL-NORM", "auroc": 0.8588, "kind": "magnitude", "delta": 0.0511, "p": None},
]


def _apply_style() -> None:
    if HAS_SEABORN:
        sns.set_theme(
            style="whitegrid",
            font="DejaVu Serif",
            rc={
                "axes.edgecolor": C["ink"],
                "axes.labelcolor": C["ink"],
                "text.color": C["ink"],
                "xtick.color": C["ink"],
                "ytick.color": C["ink"],
                "grid.color": "#E6E6E6",
                "grid.linewidth": 0.6,
            },
        )
    else:
        if "seaborn-v0_8-whitegrid" in plt.style.available:
            plt.style.use("seaborn-v0_8-whitegrid")
        else:
            plt.style.use("seaborn-whitegrid") if "seaborn-whitegrid" in plt.style.available else None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"wrote {pdf.name}  {png.name}")
    plt.close(fig)


def _round_box(ax, xy, w, h, text, facecolor, edgecolor, fontsize=8.2, textcolor="white", lw=1.1, radius=0.18):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
        mutation_aspect=0.6,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2.0,
        xy[1] + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        zorder=4,
        linespacing=1.25,
        fontweight="medium",
    )
    return box


def _arrow(ax, start, end, color, lw=1.4, style="-|>"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color=color,
        zorder=2,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)


def figure1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 6.35))
    ax.set_xlim(-0.15, 12.15)
    ax.set_ylim(-0.25, 11.15)
    ax.axis("off")
    ax.set_title(
        "Experimental pipeline: 64D PCA vs. uncompressed raw trajectories (both exploratory)",
        fontsize=11.2,
        pad=8,
        color=C["ink"],
    )

    left_panel = FancyBboxPatch(
        (0.20, 3.55),
        5.25,
        3.15,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor="#FDE8DC",
        edgecolor=C["expl"],
        linewidth=1.15,
        linestyle=(0, (3.2, 2.0)),
        zorder=0,
    )
    right_panel = FancyBboxPatch(
        (6.55, 3.55),
        5.25,
        3.15,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor="#E4F0F8",
        edgecolor=C["conf"],
        linewidth=1.15,
        zorder=0,
    )
    ax.add_patch(left_panel)
    ax.add_patch(right_panel)
    ax.text(2.82, 6.45, "Raw 3584D  (exploratory)", ha="center", va="center", fontsize=8.0, color=C["expl"], fontweight="bold", zorder=5)
    ax.text(9.17, 6.45, "64D PCA  (exploratory)", ha="center", va="center", fontsize=8.0, color=C["conf"], fontweight="bold", zorder=5)

    _round_box(ax, (4.05, 10.05), 3.90, 0.78, "Prompt", C["ink"], C["ink"], fontsize=9.2)
    _round_box(ax, (3.25, 8.90), 5.50, 0.78, "LLM generation   (Qwen2.5-7B-Instruct)", C["ink"], C["ink"], fontsize=8.3)
    _round_box(
        ax,
        (2.35, 7.75),
        7.30,
        0.78,
        r"Token-level hidden-state trajectory   $\mathbf{h}_t \in \mathbb{R}^{3584}$",
        "#333333",
        "#333333",
        fontsize=8.1,
    )
    _arrow(ax, (6.0, 10.05), (6.0, 9.68), C["ink"])
    _arrow(ax, (6.0, 8.90), (6.0, 8.53), C["ink"])

    _arrow(ax, (4.15, 7.75), (2.82, 6.85), C["expl"], lw=1.55)
    _arrow(ax, (7.85, 7.75), (9.17, 6.85), C["conf"], lw=1.55)

    _round_box(
        ax,
        (0.48, 4.85),
        4.70,
        1.15,
        "Raw 3584D hidden states\n(no dimensionality reduction)",
        C["expl"],
        C["expl"],
        fontsize=8.0,
    )
    _round_box(
        ax,
        (6.82, 5.45),
        4.70,
        0.72,
        "Variance-maximizing PCA",
        C["conf"],
        C["conf"],
        fontsize=8.2,
    )
    _round_box(
        ax,
        (6.82, 3.80),
        4.70,
        1.20,
        "64D (PC1--64)    /    PC4--67\nexplained var.  PC1--64  =  0.99992",
        C["sky"],
        C["conf"],
        fontsize=7.6,
        textcolor=C["ink"],
    )
    _arrow(ax, (9.17, 5.45), (9.17, 5.00), C["conf"])

    _arrow(ax, (2.82, 4.85), (2.82, 3.15), C["expl"], lw=1.55)
    _arrow(ax, (9.17, 3.80), (9.17, 3.15), C["conf"], lw=1.55)
    ax.plot([2.82, 9.17], [3.00, 3.00], color=C["gray"], lw=1.15, zorder=1)
    _arrow(ax, (6.0, 3.00), (6.0, 2.85), C["ink"])

    _round_box(
        ax,
        (2.85, 1.85),
        6.30,
        0.95,
        r"$\kappa$-derived trajectory features" + "\nmean  /  max  /  $p_{95}$  /  std  /  AUC-density",
        "#333333",
        "#333333",
        fontsize=8.0,
    )
    _arrow(ax, (6.0, 1.85), (6.0, 1.58), C["ink"])
    _round_box(ax, (3.55, 0.95), 4.90, 0.58, r"Classifier  $\rightarrow$  test AUROC", C["ink"], C["ink"], fontsize=8.3)
    _arrow(ax, (6.0, 0.95), (6.0, 0.70), C["ink"])
    _round_box(
        ax,
        (2.35, 0.02),
        7.30,
        0.58,
        "Comparison to response-length baseline   (BL-LEN)",
        "#444444",
        "#444444",
        fontsize=7.8,
    )

    conf_p = mpatches.Patch(facecolor="#E4F0F8", edgecolor=C["conf"], label="64D PCA (exploratory)")
    expl_p = mpatches.Patch(facecolor="#FDE8DC", edgecolor=C["expl"], label="Raw 3584D (exploratory)")
    ax.legend(handles=[conf_p, expl_p], loc="upper left", bbox_to_anchor=(0.0, 1.01), frameon=False, fontsize=8)

    _save(fig, "fig1_pipeline")


def figure2_layer_auroc() -> None:
    layers = [8, 16, 24]
    y = np.array([LAYER_AUROC[L]["auroc"] for L in layers])
    lo = np.array([LAYER_AUROC[L]["lo"] for L in layers])
    hi = np.array([LAYER_AUROC[L]["hi"] for L in layers])
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(6.15, 3.85))
    x = np.arange(len(layers))
    colors = [C["gray"], C["blue"], C["gray"]]
    edge = [C["gray"], C["blue"], C["gray"]]
    widths = [0.55, 0.62, 0.55]

    if HAS_SEABORN:
        sns.despine(ax=ax, offset=2)

    ax.bar(
        x,
        y,
        width=widths,
        color=colors,
        edgecolor=edge,
        linewidth=1.05,
        zorder=2,
        alpha=0.92,
    )
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=C["ink"],
        ecolor=C["ink"],
        elinewidth=1.2,
        capsize=5.0,
        capthick=1.2,
        markersize=4.2,
        zorder=3,
    )

    for i, (yi, loi, hii) in enumerate(zip(y, lo, hi)):
        weight = "bold" if i == 1 else "regular"
        ax.text(x[i], hii + 0.014, f"{yi:.4f}", ha="center", va="bottom", fontsize=8.6, fontweight=weight)
        ax.text(x[i], 0.492, f"[{loi:.4f}, {hii:.4f}]", ha="center", va="bottom", fontsize=6.5, color=C["gray"])

    ax.axhline(0.5, color=C["lightgray"], ls="--", lw=1.0, zorder=1)
    ax.axhline(0.8077, color=C["vermillion"], ls=":", lw=1.15, zorder=1)
    ax.text(2.48, 0.816, "BL-LEN  0.8077", ha="right", va="bottom", fontsize=7.2, color=C["vermillion"])
    ax.text(-0.48, 0.508, "chance", ha="left", va="bottom", fontsize=7.0, color=C["gray"])

    ax.annotate(
        "peak among evaluated layers\nHolm $p<0.001$ vs. L8 and L24",
        xy=(1, 0.8232),
        xytext=(1.42, 0.60),
        fontsize=7.3,
        color=C["blue"],
        arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=0.9),
        ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {L}" for L in layers])
    ax.set_ylabel(r"PROP-$\kappa$  test AUROC")
    ax.set_ylim(0.47, 0.94)
    ax.set_xlim(-0.55, 2.55)
    ax.set_title("Layer localization of trajectory discrimination", fontsize=11)
    ax.yaxis.grid(True, ls=":", lw=0.55, color="#D8D8D8")
    ax.xaxis.grid(False)

    _save(fig, "fig2_layer_auroc")


def figure3_hero() -> None:
    vals = np.array([h["auroc"] for h in HERO])
    colors = [C["gray"], "#5B7C99", C["blue"], C["green"], C["vermillion"]]
    fig, ax = plt.subplots(figsize=(7.05, 4.45))

    if HAS_SEABORN:
        sns.despine(ax=ax, offset=2)

    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.68, color=colors, edgecolor=colors, linewidth=0.8, zorder=2)
    bars[3].set_linewidth(1.5)
    bars[3].set_edgecolor("#005C44")
    bars[4].set_linewidth(1.8)
    bars[4].set_edgecolor("#8C3A00")

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.005,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold" if i >= 3 else "regular",
        )

    ax.axhline(0.8077, color=C["blue"], ls="--", lw=0.95, alpha=0.70, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "PCA 64D\n" + r"$\Delta=-0.0271$" + "\n" + r"$p=0.8685$",
            "PC4--67\n" + r"$\Delta=-0.0108$" + "\n" + r"$p=0.6935$",
            "BL-LEN\nreference",
            "Raw 3584D\n" + r"$\Delta=+0.0395$" + "\n" + r"$p=0.004$",
            "BL-NORM\n" + r"$\Delta=+0.0511$" + "\nmagnitude",
        ]
    )
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.74, 0.92)
    ax.set_title("Paired Layer-16 comparison on the 98-prompt test set", fontsize=11)
    ax.yaxis.grid(True, ls=":", lw=0.55, color="#D8D8D8")
    ax.xaxis.grid(False)

    handles = [
        mpatches.Patch(color=C["gray"], label="PCA 64D  (exploratory)"),
        mpatches.Patch(color="#5B7C99", label="PC4--67  (ablation)"),
        mpatches.Patch(color=C["blue"], label="BL-LEN  (output baseline)"),
        mpatches.Patch(color=C["green"], label="Raw 3584D  (exploratory)"),
        mpatches.Patch(color=C["vermillion"], label="BL-NORM  (magnitude)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False, ncol=1)

    _save(fig, "fig3_hero_comparison")


def figure4_geometry() -> None:
    rng = np.random.default_rng(7)

    # Elongated cloud: global variance along x; class means separated along y
    # (a possible task-relevant direction, shown as a hypothesis only).
    n = 180
    cov_shared = np.array([[2.40, 0.04], [0.04, 0.10]])
    mean_a = np.array([0.00, 0.72])
    mean_b = np.array([0.12, -0.72])
    a = rng.multivariate_normal(mean_a, cov_shared, n)
    b = rng.multivariate_normal(mean_b, cov_shared, n)
    X = np.vstack([a, b])
    Xc = X - X.mean(axis=0)
    a = a - X.mean(axis=0)
    b = b - X.mean(axis=0)

    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]
    if pc1[0] < 0:
        pc1 = -pc1
    task = mean_a - mean_b
    task = task / np.linalg.norm(task)

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.75), sharey=True)

    if HAS_SEABORN:
        for ax in axes:
            sns.despine(ax=ax, offset=1)

    def _scatter(ax, title):
        ax.scatter(a[:, 0], a[:, 1], s=16, c=C["sky"], alpha=0.62, linewidths=0, zorder=2)
        ax.scatter(b[:, 0], b[:, 1], s=16, c=C["vermillion"], alpha=0.62, linewidths=0, zorder=2)
        ax.set_xlim(-5.1, 5.1)
        ax.set_ylim(-2.05, 2.15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("synthetic dim. 1")
        ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, ls=":", lw=0.45, color="#E0E0E0")
        ax.xaxis.grid(True, ls=":", lw=0.45, color="#E0E0E0")

    _scatter(axes[0], "Variance-maximizing geometry")
    _scatter(axes[1], "Possible task-relevant geometry")
    axes[0].set_ylabel("synthetic dim. 2")

    origin = np.zeros(2)
    axes[0].annotate(
        "",
        xy=origin + 4.3 * pc1,
        xytext=origin - 4.3 * pc1,
        arrowprops=dict(arrowstyle="<|-|>", color=C["blue"], lw=1.85, mutation_scale=11),
    )
    axes[0].text(3.55, 1.55, "PC1\n(max. variance)", color=C["blue"], fontsize=7.5, ha="center")

    axes[1].annotate(
        "",
        xy=origin + 4.3 * pc1,
        xytext=origin - 4.3 * pc1,
        arrowprops=dict(arrowstyle="<|-|>", color=C["lightgray"], lw=1.25, mutation_scale=10),
    )
    axes[1].annotate(
        "",
        xy=origin + 1.55 * task,
        xytext=origin - 1.55 * task,
        arrowprops=dict(arrowstyle="<|-|>", color=C["green"], lw=2.05, mutation_scale=12),
    )
    axes[1].text(3.55, 1.55, "PC1 (retained)", color=C["gray"], fontsize=7.4, ha="center")
    axes[1].text(
        -3.55,
        1.72,
        "possible location of\ntask-relevant structure",
        color=C["green"],
        fontsize=7.3,
        ha="center",
        fontweight="bold",
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["sky"], markersize=6, label="reliable (schematic)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["vermillion"], markersize=6, label="unreliable (schematic)"),
        Line2D([0], [0], color=C["blue"], lw=1.8, label="global-variance axis"),
        Line2D([0], [0], color=C["green"], lw=2.0, label="possible task-relevant axis"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Global variance is not necessarily the geometry of the task", fontsize=11.1, y=1.03)
    fig.text(
        0.5,
        -0.10,
        "Schematic only. The experiment does not establish that the signal occupies a specific low-variance subspace.",
        ha="center",
        fontsize=7.0,
        color=C["gray"],
        style="italic",
    )

    _save(fig, "fig4_geometry")


def main() -> int:
    _apply_style()
    figure1_pipeline()
    figure2_layer_auroc()
    figure3_hero()
    figure4_geometry()
    print(f"figures written to {OUT_DIR}")
    print(f"seaborn={'yes' if HAS_SEABORN else 'no (matplotlib seaborn-v0_8 style)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
