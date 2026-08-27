"""Schematic of the three temporal windows around a disturbance event.

before: 1 January of the event year to 8 days before the event.
during: the reference date plus or minus 7 days.
after:  8 to 28 days after the event.

The before period is drawn broken, because its length depends on when the event
occurred and is therefore different for every polygon.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:  # same style as the other manuscript figures
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except ImportError:  # fall back to an equivalent inline style
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 0.8})

FIGDIR = os.path.expanduser(
    "~/Desktop/Master/thesis/manuscript/manuscript_tropical_forest_disturbance/figures"
)
TEXT_W = 160 / 25.4
# Okabe-Ito, colorblind safe and dark enough for white labels
BEFORE, DURING, AFTER = "#0072B2", "#D55E00", "#009E73"
GREY = "0.45"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "savefig.bbox": "tight",
})

# left edge of the drawing, standing for 1 January rather than for a fixed offset
LEFT, Y, H = -40, 0.0, 0.30

fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 0.20))

bars = [
    (LEFT, -8 - LEFT, BEFORE, "before", "1 Jan to $-8$ d"),
    (-7, 14, DURING, "during", "$-7$ to $+7$ d"),
    (8, 20, AFTER, "after", "$+8$ to $+28$ d"),
]
for x0, width, color, name, detail in bars:
    ax.add_patch(Rectangle((x0, Y - H / 2), width, H, facecolor=color, edgecolor="none"))
    ax.text(x0 + width / 2, Y, name, ha="center", va="center", color="white", fontsize=11)
    ax.text(x0 + width / 2, Y - H / 2 - 0.10, detail, ha="center", va="top",
            fontsize=9, color=GREY)

# the break that marks the variable length of the before period
for dx in (0.0, 2.2):
    ax.plot([LEFT + 3 + dx, LEFT + 6 + dx], [Y - H / 2 - 0.03, Y + H / 2 + 0.03],
            color="white", lw=1.6, solid_capstyle="butt", zorder=3)

# the reference date
# a marker above the bars, rather than a line across them, so no label is cut
ax.plot([0], [Y + H / 2 + 0.05], marker="v", color="black", markersize=5, zorder=4,
        clip_on=False)
ax.text(0, Y + H / 2 + 0.13, "reference date", ha="center", va="bottom", fontsize=9.5)

ax.set_xlim(LEFT - 2, 33)
ax.set_ylim(-0.60, 0.48)
ax.set_yticks([])
ax.set_xticks([-8, 0, 8, 28])
ax.set_xticklabels(["$-8$", "0", "$+8$", "$+28$"])
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)

out = os.path.join(FIGDIR, "temporal_windows.pdf")
fig.savefig(out)
print("written", out)
