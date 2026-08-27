"""Terminology diagram: polygon to tile to patch to token.

Same drawing as the terminology cell of croma_fid_embeddings_charlotte.ipynb, with the
three panel labels placed at one common height. In the notebook version the labels were
axes titles, and since panels (a) and (b) use set_aspect('equal') their axes box is
shrunk to satisfy the aspect ratio, which moved their titles away from the title of (c).
Here the labels are drawn in figure coordinates after the layout is resolved, so the
three sit on the same line whatever the aspect ratios do.
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

FIGDIR = os.path.expanduser(
    "~/Desktop/Master/thesis/manuscript/manuscript_tropical_forest_disturbance/figures"
)
TEXT_W = 160 / 25.4  # text width of the manuscript, in inches
BLUE, GREY = "#2a78d6", "0.55"

plt.rcParams.update({"savefig.bbox": "tight"})

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(TEXT_W, TEXT_W * 0.34),
    gridspec_kw={"width_ratios": [1.0, 1.0, 0.95]},
)

# ── (a) polygon spanning four tiles ──
poly_xy = [(0.45, 0.60), (1.35, 0.40), (1.75, 1.05),
           (1.45, 1.65), (0.80, 1.60), (0.35, 1.10)]
for i in range(2):
    for j in range(2):
        ax1.add_patch(Rectangle((i, j), 1, 1, fill=False, ec=GREY, lw=0.8))
ax1.add_patch(MplPolygon(poly_xy, closed=True, fc=BLUE, ec=BLUE, alpha=0.25, lw=1.4))
ax1.add_patch(Rectangle((1, 1), 1, 1, fill=False, ec=BLUE, lw=1.8))
ax1.set_xlim(-0.08, 2.08)
ax1.set_ylim(-0.08, 2.08)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.text(1.0, -0.32, "tile: 120 x 120 px = 1200 m", ha="center", fontsize=9, color=GREY)

# ── (b) one tile split into 15 x 15 patches ──
ax2.add_patch(Rectangle((0, 0), 15, 15, fill=False, ec=BLUE, lw=1.8))
for k in range(1, 15):
    ax2.plot([k, k], [0, 15], color=GREY, lw=0.4)
    ax2.plot([0, 15], [k, k], color=GREY, lw=0.4)
ax2.add_patch(Rectangle((9, 5), 1, 1, fc=BLUE, ec=BLUE, lw=1.0))
ax2.set_xlim(-0.5, 15.5)
ax2.set_ylim(-0.5, 15.5)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.text(7.5, -2.0, "patch: 8 x 8 px = 80 m", ha="center", fontsize=9, color=GREY)

# ── (c) the token as a vector ──
ax3.text(0.5, 0.56, r"$[\,x_1\ \ x_2\ \ x_3\ \ \cdots\ \ x_{768}\,]$",
         ha="center", va="center", fontsize=11, color=BLUE)
ax3.text(0.5, 0.40, "768 dimensions", ha="center", va="top", fontsize=9, color=GREY)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")

# ── arrows between panels ──
# shrinkB keeps the arrow head clear of the text it points at
for xy_a, ax_a, xy_b, ax_b, shrink_b in [((2.08, 1.5), ax1, (-0.5, 7.5), ax2, 4),
                                         ((10, 5.5), ax2, (0.02, 0.56), ax3, 14)]:
    fig.add_artist(ConnectionPatch(xyA=xy_a, coordsA=ax_a.transData,
                                   xyB=xy_b, coordsB=ax_b.transData,
                                   arrowstyle="->", lw=1.0, color=GREY,
                                   shrinkA=2, shrinkB=shrink_b))

# ── panel labels, all three on the same line ──
LABELS = ["(a) polygon and tiles", "(b) patches in a tile", "(c) token"]
fig.canvas.draw()  # resolve the aspect ratios before reading the boxes
boxes = [ax.get_window_extent().transformed(fig.transFigure.inverted())
         for ax in (ax1, ax2, ax3)]
label_y = max(box.y1 for box in boxes) + 0.02
for box, label in zip(boxes, LABELS):
    fig.text(0.5 * (box.x0 + box.x1), label_y, label,
             ha="center", va="bottom", fontsize=10)

out = os.path.join(FIGDIR, "terminology_tile_patch_token.pdf")
fig.savefig(out)
print("written", out)
