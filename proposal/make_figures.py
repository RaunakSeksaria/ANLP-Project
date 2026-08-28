"""Schematics for the proposal, sized for a single ACL column.

    uv run python proposal/make_figures.py

Column width with this document's geometry (A4, 2cm margins, 0.65cm columnsep)
is (21 - 4.0 - 0.65)/2 = 8.18 cm = 3.22 in. Both figures are drawn at 3.15 in
so they sit inside the column with a hair of slack, and use 6-8 pt type, which
is what ACL figures normally run at.

Writes PDF (for \\includegraphics) and PNG (for quick viewing) into
proposal/figures/.

Triplet form follows Baly et al. (2020) SS4.2.2, after Schroff et al. (2015):
    L = max(D(a,p) - D(a,n) + eps, 0)
Baly holds *ideology* fixed and varies the medium. We hold the *event* fixed and
vary, in turn, the publication (a) and the ideology (b).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
COL = 3.15   # single ACL column, inches
WIDE = 6.60  # full text width, for figure*

INK = "#1A1A1A"
MUTED = "#6B7280"
PULL = "#1B6B47"
PUSH = "#A8441C"
ACCENT = "#14606B"
LEFT = "#1B4F9C"      # ideology: left
RIGHT = "#B02418"     # ideology: right
# Outlet colours are deliberately NOT blue/red: in panel (a) the axis is
# publication, and a red/blue pair would be read as ideology instead.
SRC_A = "#6B3FA0"     # outlet 1
SRC_B = "#C2610A"     # outlet 2
GRAD = "#8A2846"
FILL = "#F3F5F7"
FILL_A = "#E2EEF0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "text.color": INK,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def box(ax, x, y, w, h, label, sub=None, ec=INK, fc=FILL, lw=0.9, fs=6.6, bold=False):
    """x, y is the CENTRE."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y + (0.19 * h if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=ec, fontweight="bold" if bold else "normal", zorder=3)
    if sub:
        ax.text(x, y - 0.26 * h, sub, ha="center", va="center",
                fontsize=fs - 1.1, color=ec, zorder=3)


def arrow(ax, a, b, color=INK, ls="-", lw=0.9, rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4,
        shrinkA=1.5, shrinkB=1.5))


# --------------------------------------------------------------------------- #
# Figure 1 - architecture, vertical flow to fit a narrow column
# --------------------------------------------------------------------------- #

def figure_architecture():
    """Full text width (figure*): a standard LED seq2seq spine, with the
    triplet branch hanging off the encoder output."""
    fig, ax = plt.subplots(figsize=(WIDE, 2.55))
    ax.set_xlim(0, 21); ax.set_ylim(0, 9.2); ax.axis("off")

    BLK_W, BLK_H = 1.95, 3.30     # encoder and decoder: identical footprint

    # --- inputs -----------------------------------------------------------
    ys = [5.85, 4.60, 3.35]
    for i, y in enumerate(ys, 1):
        box(ax, 1.05, y, 1.5, 0.82, f"$I_{i}$", fs=7.2)
        arrow(ax, (1.80, y), (2.60, y), ms=6)
    ax.text(1.05, 2.42, "source\narticles", ha="center", va="top",
            fontsize=6.0, color=MUTED, style="italic")

    # --- encoder ----------------------------------------------------------
    ex = 3.58
    ax.add_patch(FancyBboxPatch((ex - BLK_W / 2, 4.60 - BLK_H / 2), BLK_W, BLK_H,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.6, edgecolor=INK, facecolor=FILL, zorder=2))
    ax.text(ex, 4.60, "ENCODER", rotation=90, ha="center", va="center",
            fontsize=7.6, fontweight="bold", zorder=3)
    ax.text(ex, 6.62, "LED", ha="center", va="center", fontsize=6.4, color=MUTED)

    # --- hidden states + branch point -------------------------------------
    for i, y in enumerate(ys, 1):
        arrow(ax, (ex + BLK_W / 2, y), (5.55, y), ms=6)
        ax.text(5.80, y, f"$h_{i}$", ha="center", va="center", fontsize=7.2)

    bx = 6.45
    for y in ys:
        ax.plot([6.05, bx], [y, y], color=INK, linewidth=0.85, zorder=1)
    ax.plot([bx, bx], [ys[-1], 7.60], color=INK, linewidth=0.85, zorder=1)
    ax.plot([bx, bx], [ys[-1], 4.60], color=INK, linewidth=0.85, zorder=1)

    # --- upper branch: pooling -> z -> L2 -> triplet loss ------------------
    arrow(ax, (bx, 7.60), (7.55, 7.60), ms=6)
    box(ax, 8.78, 7.60, 2.55, 1.15, "pooling", ec=ACCENT, fc=FILL_A, fs=6.6)
    arrow(ax, (10.06, 7.60), (10.60, 7.60), color=ACCENT, ms=6)

    ax.add_patch(FancyBboxPatch((10.62, 6.55), 1.60, 2.10,
                                boxstyle="round,pad=0,rounding_size=0.40",
                                linewidth=1.1, edgecolor=ACCENT, facecolor=FILL_A,
                                zorder=2))
    for k, yy in enumerate((8.12, 7.60, 7.08), 1):
        ax.text(11.42, yy, f"$z_{k}$", ha="center", va="center", fontsize=6.8,
                color=ACCENT, zorder=3)
    arrow(ax, (12.22, 7.60), (12.90, 7.60), color=ACCENT, ms=6)

    box(ax, 14.10, 7.60, 2.35, 1.15, r"$\ell_2$ distance", ec=ACCENT, fs=6.6)
    arrow(ax, (15.28, 7.60), (16.02, 7.60), color=ACCENT, ms=6)
    box(ax, 17.28, 7.60, 2.48, 1.15, r"$\mathcal{L}_{\mathrm{trip}}$",
        sub="triplet loss", ec=ACCENT, fs=8.0, bold=True)

    # --- lower branch: decoder -> text -> generation loss ------------------
    arrow(ax, (bx, 4.60), (7.55, 4.60), ms=6)
    dx = 8.55
    ax.add_patch(FancyBboxPatch((dx - BLK_W / 2, 4.60 - BLK_H / 2), BLK_W, BLK_H,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=1.6, edgecolor=INK, facecolor=FILL, zorder=2))
    ax.text(dx, 4.60, "DECODER", rotation=90, ha="center", va="center",
            fontsize=7.6, fontweight="bold", zorder=3)

    arrow(ax, (dx + BLK_W / 2, 4.60), (10.38, 4.60), ms=6)
    box(ax, 11.88, 4.60, 2.95, 1.15, "neutral article", fs=6.6)
    arrow(ax, (13.38, 4.60), (14.12, 4.60), ms=6)
    box(ax, 15.52, 4.60, 2.72, 1.15, r"$\mathcal{L}_{\mathrm{gen}}$",
        sub="cross-entropy", fs=8.0, bold=True)

    # --- combined objective -----------------------------------------------
    box(ax, 19.35, 6.10, 3.05, 1.15,
        r"$\mathcal{L}_{\mathrm{gen}}\!+\!\lambda\mathcal{L}_{\mathrm{trip}}$",
        fs=7.8, lw=1.4, fc="white", bold=True)
    arrow(ax, (18.52, 7.28), (19.35, 6.70), color=ACCENT, ms=6)
    arrow(ax, (16.88, 4.72), (17.90, 5.58), ms=6)

    # --- gradient path back to the encoder --------------------------------
    dash = dict(color=GRAD, linewidth=1.0, linestyle=(0, (4, 2.6)), zorder=1)
    ax.plot([19.35, 19.35], [5.52, 1.30], **dash)
    ax.plot([19.35, ex], [1.30, 1.30], **dash)
    # the combined objective backpropagates into BOTH blocks
    for tap in (dx, ex):
        arrow(ax, (tap, 1.30), (tap, 2.90), color=GRAD, ls=(0, (4, 2.6)),
              lw=1.0, ms=7)
    ax.text(14.30, 1.02, "backpropagation", ha="center", va="center",
            fontsize=6.4, color=GRAD, style="italic")
    ax.text(6.05, 0.42, r"$\mathcal{L}_{\mathrm{trip}}$ has no gradient w.r.t. the "
                        r"decoder — it reaches the encoder only",
            ha="center", va="center", fontsize=5.9, color=GRAD, style="italic")

    fig.savefig(OUT / "architecture.pdf")
    fig.savefig(OUT / "architecture.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 - the two triplet constructions, stacked
# --------------------------------------------------------------------------- #

def node(ax, x, y, name, attrs, color, lx, ly, ha="left", r=115):
    """A dot in embedding space plus a Baly-style label block beside it."""
    ax.scatter([x], [y], s=r, c=color, edgecolors=INK, linewidths=0.9, zorder=5)
    ax.text(lx, ly, name, ha=ha, va="top", fontsize=6.6, fontweight="bold",
            color=INK, zorder=6)
    for i, a in enumerate(attrs):
        text, col = a if isinstance(a, tuple) else (a, INK)
        ax.text(lx, ly - 0.52 - 0.46 * i, text, ha=ha, va="top", fontsize=6.0,
                color=col, zorder=6)


def triplet_panel(ax, title, attr_name, anchor_v, positive_v, negative_v, rule,
                  value_colors=None):
    """Baly et al. (2020) Fig. 3 layout: anchor at left, arrows to a nearby
    positive and a distant negative, each labelled with its two attributes."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(5.0, 6.15, title, ha="center", va="center", fontsize=7.0,
            fontweight="bold")

    ax_, ay = 1.55, 3.65          # anchor
    px, py = 3.95, 1.95           # positive - near
    nx, ny = 7.55, 5.15           # negative - far

    # shrink past the marker radius (s=115 -> ~6 pt) so the head stays visible
    for tx, ty in ((nx, ny), (px, py)):
        ax.add_patch(FancyArrowPatch(
            (ax_, ay), (tx, ty), arrowstyle="-|>", mutation_scale=9,
            linewidth=0.85, color=INK, zorder=4, shrinkA=7, shrinkB=9))

    vc = value_colors or {}
    line = lambda v: (f"{attr_name}: {v}", vc.get(v, INK))

    node(ax, ax_, ay, "Anchor", ["Event: $A$", line(anchor_v)],
         "#1A1A1A", 0.15, ay - 0.42)
    node(ax, nx, ny, "Negative", ["Event: $B$", line(negative_v)],
         "#4C9A5C", 5.55, ny - 0.42)
    node(ax, px, py, "Positive", ["Event: $A$", line(positive_v)],
         "#4A6FB5", 4.35, py + 0.30)

    ax.text(5.0, 0.18, rule, ha="center", va="center", fontsize=5.9,
            color=MUTED, style="italic")


def figure_triplets():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.3))
    triplet_panel(
        axes[0], "(a) publication-invariant", "Source",
        "CNN", "Fox News", "CNN",
        "positive shares the event; negative shares the publication",
        value_colors={"CNN": SRC_A, "Fox News": SRC_B})
    triplet_panel(
        axes[1], "(b) ideology-invariant", "Ideology",
        "Left", "Right", "Left",
        "positive shares the event; negative shares the ideology",
        value_colors={"Left": LEFT, "Right": RIGHT})
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(OUT / "triplets.pdf")
    fig.savefig(OUT / "triplets.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_architecture()
    figure_triplets()
    print(f"wrote {OUT.name}/architecture.[pdf|png] and {OUT.name}/triplets.[pdf|png]")
