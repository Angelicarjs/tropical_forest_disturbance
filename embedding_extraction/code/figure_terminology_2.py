"""Terminology diagram: tile, patch, token and mosaic.

Revision of figure_terminology.py, keeping its single row and its arrow chain:
  * the patch annotation carries a leader to the marked patch, so it labels the
    patch and not the whole tile;
  * panel (d) adds the mosaic, the token grids of every tile of one polygon
    stitched over its footprint.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

try:  # same style as the other manuscript figures
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except ImportError:  # fall back to an equivalent inline style
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 0.8})

OUTDIR = os.environ.get("TERMDIR", ".")
TEXT_W = 160 / 25.4  # text width of the manuscript, in inches
BLUE, GREY = "#2a78d6", "0.55"
FS_NOTE, FS_LABEL = 8, 9

plt.rcParams.update({"savefig.bbox": "tight"})

POLY_XY = [(0.45, 0.60), (1.35, 0.40), (1.75, 1.05),
           (1.45, 1.65), (0.80, 1.60), (0.35, 1.10)]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    1, 4, figsize=(TEXT_W, TEXT_W * 0.34),
    gridspec_kw={"width_ratios": [1.0, 1.0, 0.85, 1.0], "wspace": 0.28},
)

# ── (a) a polygon spanning four tiles ──
for i in range(2):
    for j in range(2):
        ax1.add_patch(Rectangle((i, j), 1, 1, fill=False, ec=GREY, lw=0.8))
ax1.add_patch(MplPolygon(POLY_XY, closed=True, fc=BLUE, ec=BLUE, alpha=0.25, lw=1.4))
ax1.add_patch(Rectangle((1, 1), 1, 1, fill=False, ec=BLUE, lw=1.8))

ax1.set_xlim(-0.08, 2.08)
ax1.set_ylim(-0.08, 2.12)
ax1.set_aspect("equal")
ax1.axis("off")

# ── (b) that tile split into 15 x 15 patches, one of them marked ──
ax2.add_patch(Rectangle((0, 0), 15, 15, fill=False, ec=BLUE, lw=1.8))
for k in range(1, 15):
    ax2.plot([k, k], [0, 15], color=GREY, lw=0.35)
    ax2.plot([0, 15], [k, k], color=GREY, lw=0.35)
PX, PY = 11, 2                           # the marked patch
ax2.add_patch(Rectangle((PX, PY), 1, 1, fc=BLUE, ec=BLUE, lw=1.0))
ax2.set_xlim(-0.5, 15.5)
ax2.set_ylim(-0.5, 15.5)
ax2.set_aspect("equal")
ax2.axis("off")

# ── (c) the token the encoder returns for that patch ──
ax3.text(0.5, 0.58, r"$[\,x_1\ \ x_2\ \ \cdots\ \ x_{768}\,]$",
         ha="center", va="center", fontsize=9.5, color=BLUE)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")

# ── (d) the mosaic: the token grids of the four tiles, stitched ──
N = 15
for i in range(2):
    for j in range(2):
        for k in range(1, N):
            s = k / N
            ax4.plot([i + s, i + s], [j, j + 1], color=GREY, lw=0.22)
            ax4.plot([i, i + 1], [j + s, j + s], color=GREY, lw=0.22)
for i in range(2):
    for j in range(2):
        ax4.add_patch(Rectangle((i, j), 1, 1, fill=False, ec=BLUE, lw=1.5))
ax4.add_patch(MplPolygon(POLY_XY, closed=True, fill=False, ec=BLUE, lw=1.5))
ax4.set_xlim(-0.08, 2.08)
ax4.set_ylim(-0.08, 2.12)
ax4.set_aspect("equal")
ax4.axis("off")

# ── arrows: tile to its patches, patch to its token, token to the mosaic ──
for xy_a, ax_a, xy_b, ax_b, shrink_b in [
        ((2.08, 1.5), ax1, (-0.5, 7.5), ax2, 4),
        ((PX + 1, PY + 0.5), ax2, (0.02, 0.58), ax3, 12)]:
    fig.add_artist(ConnectionPatch(xyA=xy_a, coordsA=ax_a.transData,
                                   xyB=xy_b, coordsB=ax_b.transData,
                                   arrowstyle="->", lw=1.0, color=GREY,
                                   shrinkA=2, shrinkB=shrink_b))

# ── panel labels, all four on the same line ──
LABELS = ["(a) polygon and tiles", "(b) patches in a tile",
          "(c) token", "(d) mosaic"]
NOTES = [("tile\n120 x 120 px = 1200 m", GREY),
         ("patch\n8 x 8 px = 80 m", BLUE),
         ("token\n768 dimensions", GREY),
         ("mosaic\n15 x 15 tokens per tile", GREY)]
fig.canvas.draw()  # resolve the aspect ratios before reading the boxes
boxes = [ax.get_window_extent().transformed(fig.transFigure.inverted())
         for ax in (ax1, ax2, ax3, ax4)]
label_y = max(box.y1 for box in boxes) + 0.02
note_y = min(box.y0 for box in boxes) - 0.02
for box, label, (note, colour) in zip(boxes, LABELS, NOTES):
    xc = 0.5 * (box.x0 + box.x1)
    fig.text(xc, label_y, label, ha="center", va="bottom", fontsize=FS_LABEL)
    fig.text(xc, note_y, note, ha="center", va="top",
             fontsize=FS_NOTE, color=colour)

# leader from the (b) note up to the filled patch
xc_b = 0.5 * (boxes[1].x0 + boxes[1].x1)
ax2.annotate("", xy=(PX + 0.5, PY), xycoords=ax2.transData,
             xytext=(xc_b, note_y + 0.012), textcoords=fig.transFigure,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE,
                             shrinkA=1, shrinkB=2))

out = os.path.join(OUTDIR, "terminology_tile_patch_token_2.pdf")
fig.savefig(out)
print("written", out)
