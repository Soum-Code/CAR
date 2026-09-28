"""Conceptual diagrams for the thesis. [CPU]

`make_figures.py` produces twelve plots, and every one of them is a
measurement. What the draft has none of is a diagram: a picture of the *object*
being measured. Three ideas carry the argument and are currently carried by
prose alone --

  D1  the decomposition a step fails under (C1), which is the whole thesis in
      one 2x2 -- a verifier reads one axis and the answer depends on both;
  D2  the control loop, so a reader can see which chapter measures which
      component before meeting any of them;
  D3  the absorption process, which prose describes as "near-absorbing" and a
      transition diagram states exactly -- including that the persistence rate
      is a property of the generator, measured on three of them.

They are drawn in matplotlib rather than TikZ so they share the plots' visual
language, and so they render without a TeX toolchain -- which this machine does
not have.

    python scripts/make_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (  # noqa: E402
    CRITICAL,
    GOOD,
    GRID,
    INK,
    INK_2,
    INK_3,
    OUT,
    S1,
    S2,
    S3,
    SURFACE,
    note,
    save,
)


def box(ax, x, y, w, h, label, *, face=SURFACE, edge=INK_3, fs=9,
        weight="normal", tcol=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tcol, weight=weight, zorder=3, linespacing=1.4)


def arrow(ax, p, q, *, col=INK_3, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12,
                                 color=col, linewidth=lw, zorder=2,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ---------------------------------------------------------------- D1
def d1_decomposition():
    """The 2x2 the whole thesis turns on."""
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    blank(ax)

    x0, y0, w, h = 0.30, 0.30, 0.30, 0.28
    cells = [
        (x0,     y0 + h, "GLOBALLY WRONG\n\ninherited corruption\n\n" r"$\bf{49.6\%}$ of all steps",
         "#fdeaea", CRITICAL),
        (x0 + w, y0 + h, "GLOBALLY WRONG\n\nlocally wrong too",
         "#fdeaea", CRITICAL),
        (x0,     y0,     "CORRECT",
         "#eaf7ee", GOOD),
        (x0 + w, y0,     "GLOBALLY WRONG\n\nlocally wrong",
         "#fdeaea", CRITICAL),
    ]
    for cx, cy, lab, face, edge in cells:
        box(ax, cx, cy, w, h, lab, face=face, edge=edge, fs=7.6, tcol=INK_2)

    ax.text(x0 + w, y0 + 2 * h + 0.105, "local arithmetic", ha="center",
            fontsize=9, color=INK, weight="bold")
    ax.text(x0 + w / 2, y0 + 2 * h + 0.055, "holds", ha="center", fontsize=8.2, color=INK_2)
    ax.text(x0 + 1.5 * w, y0 + 2 * h + 0.055, "fails", ha="center", fontsize=8.2, color=INK_2)

    ax.text(x0 - 0.055, y0 + h, "premises", ha="center", va="center",
            fontsize=9, color=INK, weight="bold", rotation=90)
    ax.text(x0 - 0.018, y0 + 1.5 * h, "corrupt", ha="right", va="center",
            fontsize=8.2, color=INK_2)
    ax.text(x0 - 0.018, y0 + 0.5 * h, "clean", ha="right", va="center",
            fontsize=8.2, color=INK_2)

    # what the verifier can see: the right-hand column only
    ax.add_patch(mpatches.Rectangle((x0 + w - 0.004, y0 - 0.008), w + 0.008,
                                    2 * h + 0.016, fill=False, edgecolor=S1,
                                    linewidth=2.0, linestyle=(0, (4, 3)), zorder=4))
    ax.annotate("a calculator sees\nthis column only",
                xy=(x0 + 2 * w + 0.01, y0 + h), xytext=(x0 + 2 * w + 0.10, y0 + 1.45 * h),
                fontsize=8.4, color=S1, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S1, lw=1.2))

    ax.annotate("", xy=(x0 - 0.012, y0 + 1.5 * h), xytext=(x0 - 0.012, y0 + 0.5 * h),
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0))

    ax.text(0.5, 0.155,
            r"$\bf{78.5\%}$ of globally-wrong steps are arithmetically perfect"
            "\nThey sit in the top-left cell, and nothing on the right-hand axis reaches them.",
            ha="center", fontsize=8.8, color=INK, linespacing=1.6)

    ax.set_title("A step fails two separable ways", loc="left", x=0.02, y=0.99,
                 fontsize=10.5, color=INK, weight="bold")
    note(fig, "Rates are Mistral-7B-SFT within wrong-answer solutions (ch. 4). "
              "Verification certifies the horizontal axis; the answer depends on both.",
         y=-0.03)
    save(fig, "diag1-decomposition")


# ---------------------------------------------------------------- D2
def d2_loop():
    """The control loop, annotated with which chapter measures what."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    blank(ax)

    y, h, w = 0.38, 0.22, 0.155
    gap = 0.035
    stages = [
        ("generator\nwrites step $t$", SURFACE, INK_3, ""),
        ("uncertainty\nscore", "#eaf1fb", S1, "ch. 7: AUROC 0.574"),
        ("conformal\ncalibrator", "#eaf1fb", S1, "ch. 7: holds coverage,\nnot risk"),
        ("gate\nverify or not", "#fff4e6", S2, "ch. 6: where to spend"),
        ("verifier", "#eaf7ee", S3, "ch. 5: scope 0.00-0.90"),
    ]
    xs = []
    for i, (lab, face, edge, _) in enumerate(stages):
        x = 0.045 + i * (w + gap)
        xs.append(x)
        box(ax, x, y, w, h, lab, face=face, edge=edge, fs=8.4, tcol=INK)
        if i:
            arrow(ax, (x - gap + 0.004, y + h / 2), (x - 0.004, y + h / 2))

    for (x, (_, _, edge, sub)) in zip(xs, stages):
        if sub:
            ax.text(x + w / 2, y - 0.055, sub, ha="center", va="top",
                    fontsize=7.4, color=edge, linespacing=1.4)

    # repair loop back to the generator
    x_last = xs[-1] + w
    arrow(ax, (x_last - w / 2, y + h + 0.02), (xs[0] + w / 2, y + h + 0.02),
          col=INK_3, rad=0.22, ls=(0, (4, 3)))
    ax.text((xs[0] + x_last) / 2, y + h + 0.30, "repair, and continue",
            ha="center", fontsize=8, color=INK_2)

    ax.text(0.045, y - 0.22,
            "Chapter 4 measures what the verifier is being asked to catch; "
            "chapter 8 asks whether the target is attainable at all.",
            fontsize=8.4, color=INK_2)

    ax.set_title("The gate, and which chapter measures which part", loc="left",
                 x=0.02, y=1.02, fontsize=10.5, color=INK, weight="bold")
    note(fig, "Each component was measured separately before chapter 7 ran the "
              "assembled loop. The assembled result is negative, and locating "
              "which component causes it is what the separate measurements buy.",
         y=-0.06)
    save(fig, "diag2-gate-loop")


# ---------------------------------------------------------------- D3
def d3_absorption():
    """Corruption as a transition system, with the measured rates on it."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    blank(ax)

    r, yc = 0.085, 0.56
    states = [(0.16, "CLEAN", "#eaf7ee", GOOD), (0.52, "CORRUPT", "#fdeaea", CRITICAL)]
    for x, lab, face, edge in states:
        ax.add_patch(plt.Circle((x, yc), r, facecolor=face, edgecolor=edge,
                                linewidth=1.6, zorder=2))
        ax.text(x, yc, lab, ha="center", va="center", fontsize=9,
                color=INK, weight="bold", zorder=3)

    arrow(ax, (0.16 + r, yc + 0.03), (0.52 - r, yc + 0.03), col=CRITICAL, lw=1.6)
    ax.text(0.34, yc + 0.11, "a step goes wrong", ha="center", fontsize=8.4,
            color=CRITICAL)

    arrow(ax, (0.52 - r, yc - 0.045), (0.16 + r, yc - 0.045), col=INK_3,
          lw=1.2, ls=(0, (3, 3)))
    ax.text(0.34, yc - 0.135, "recovery\n" r"Mistral $\bf{0}$ of 14,573",
            ha="center", va="top", fontsize=8.4, color=INK_2, linespacing=1.5)

    # self-loop on CORRUPT
    # Endpoints on the circle's right edge and the arc bowing OUTWARD: with
    # rad positive it bulges left of travel, i.e. back through the node label.
    ax.add_patch(FancyArrowPatch((0.52 + r * 0.72, yc + r * 0.70),
                                 (0.52 + r * 0.72, yc - r * 0.70),
                                 connectionstyle="arc3,rad=-1.5",
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=CRITICAL, linewidth=1.8, zorder=1))
    ax.text(0.82, yc, "stays corrupt\n" r"$\bf{95.9\%}$ of later steps",
            ha="center", va="center", fontsize=8.6, color=CRITICAL, linespacing=1.5)

    ax.text(0.5, 0.055,
            "Llama 3.1 8B 81.6% (12 of 141 recover)   |   Qwen2.5-7B 66.4% (6 of 83)\n"
            "Absorption decays $\\it{monotonically}$ as the generator improves.",
            ha="center", va="bottom", fontsize=8.4, color=INK_2, linespacing=1.7)

    ax.set_title("Corruption is close to absorbing", loc="left", x=0.02, y=0.94,
                 fontsize=10.5, color=INK, weight="bold")
    note(fig, "Measured, not modelled (ch. 4). Entering corruption does not depend "
              "on the verifier; escaping it does -- which is why reach, not budget, "
              "is the controlling variable.", y=-0.06)
    save(fig, "diag3-absorption")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing diagrams to", OUT)
    d1_decomposition()
    d2_loop()
    d3_absorption()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
