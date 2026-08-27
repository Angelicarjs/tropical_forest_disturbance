"""Near real time rule on one polygon, with the date the rule reports marked.

Same score series as the top panel of nrt_fid.py, plus the three things that turn a
series into a detection: the threshold, the first acquisition that crosses it, and the
second one that confirms it within `confirm_days`. The reference date of DETER is drawn
alongside, so the lead the rule gains is readable off the axis.

Usage:
    python near_real_time/code/figure_nrt_example.py --fid 389 --tau 0.851
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

try:  # same style as the other manuscript figures
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except ImportError:
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 0.8})

FIGDIR = os.path.expanduser(
    "~/Desktop/Master/thesis/manuscript/manuscript_tropical_forest_disturbance/figures"
)
SCORES = os.path.expanduser(
    "~/Desktop/Master/thesis/tropical_forest_disturbance/near_real_time/results/nrt_scores"
)
TILES = os.path.expanduser("~/Desktop/Master/thesis/tiles_120px/s2_l2a")
SHP = os.path.expanduser(
    "~/Desktop/Master/thesis/tropical_forest_disturbance/data/data_shp/label_polygons.shp"
)
TILE_M, TILE_PX = 1200, 120
TEXT_W = 160 / 25.4
WIN_COLOR = {"bef": "#0072B2", "evt": "#D55E00", "aft": "#009E73"}
WIN_NAME = {"bef": "before", "evt": "event", "aft": "after"}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
})


def rgb_frames(fid, rows):
    """One true color mosaic per date, plus the polygon rasterized over that mosaic.

    Same construction as nrt_fid.py: the tiles of a polygon are placed on the grid by
    their bounds, each date is stretched over the whole mosaic so the tiles do not look
    mismatched, and the outline comes from the label geometry rather than from the tokens.
    """
    bounds, paths = {}, {}
    for _, r in rows.iterrows():
        d = Path(TILES) / f"fid_{fid}" / r.win / r.s2_id
        for tif in sorted(d.glob("tile_*.tif")):
            with rasterio.open(tif) as src:
                bounds[tif.stem] = src.bounds
            paths.setdefault(r.date, {})[tif.stem] = tif
    minx = min(b.left for b in bounds.values())
    maxy = max(b.top for b in bounds.values())
    pos = {t: (int(round((maxy - b.top) / TILE_M)), int(round((b.left - minx) / TILE_M)))
           for t, b in bounds.items()}
    nrow = max(r for r, _ in pos.values()) + 1
    ncol = max(c for _, c in pos.values()) + 1

    frames = {}
    for date, tiles in paths.items():
        img = np.full((nrow * TILE_PX, ncol * TILE_PX, 3), np.nan, "float32")
        for tile, tif in tiles.items():
            r, c = pos[tile]
            with rasterio.open(tif) as src:
                img[r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX] = \
                    np.stack([src.read(b).astype("float32") for b in (4, 3, 2)], -1)
        lo, hi = np.nanpercentile(img, [2, 98])
        frames[date] = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    with rasterio.open(next(iter(next(iter(paths.values())).values()))) as src:  # noqa
        crs = src.crs
    polys = gpd.read_file(SHP).to_crs(crs)
    polys = polys[polys["fid"].astype(float).astype(int) == fid]
    outline = rasterize(polys.geometry, (nrow * TILE_PX, ncol * TILE_PX),
                        transform=from_origin(minx, maxy, 10, 10))

    # crop every date to a square window around the polygon, so the disturbance fills
    # the thumbnail instead of sitting in a corner of the tile
    ys, xs = np.where(outline > 0)
    half = max(np.ptp(ys), np.ptp(xs)) * 0.85 + 12
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    H, W = outline.shape
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    frames = {d: f[y0:y1, x0:x1] for d, f in frames.items()}
    return frames, outline[y0:y1, x0:x1]


def outline_on(ax, outline, h, w):
    """Draw the polygon boundary on an axis showing an h x w image of the same extent."""
    if not (outline > 0).any():
        return
    H, W = outline.shape
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    ax.contour(xs, ys, (outline > 0).astype(float), levels=[0.5], colors="red",
               linewidths=1.0)


def first_alert(x, tau, confirm_days):
    """First acquisition at or above tau that a later one confirms within the window."""
    above = x[x.p_dist >= tau]
    for _, row in above.iterrows():
        later = above[above.date > row.date]
        if not later.empty and (later.date - row.date).dt.days.min() <= confirm_days:
            confirming = later[(later.date - row.date).dt.days <= confirm_days].iloc[0]
            return row.date, confirming.date
    return None, None


ap = argparse.ArgumentParser()
ap.add_argument("--fid", type=int, default=389)
ap.add_argument("--tau", type=float, default=0.851)
ap.add_argument("--mode", default="joint")
ap.add_argument("--model", default="log_reg")
ap.add_argument("--version", type=int, default=3)
ap.add_argument("--confirm-days", type=int, default=90)
ap.add_argument("--thumbs", type=int, default=6,
                help="how many true color thumbnails to draw, 0 for all")
args = ap.parse_args()

csv = Path(SCORES) / f"{args.mode}_{args.model}_v{args.version}_test" / f"fid_{args.fid}.csv"
x = pd.read_csv(csv, parse_dates=["date", "view_date"]).sort_values("date")
view = x.view_date.iloc[0]
alert, confirm = first_alert(x, args.tau, args.confirm_days)
delay = (alert - view).days if alert is not None else None
print(f"fid {args.fid} | view {view.date()} | alert {alert.date() if alert is not None else None} "
      f"| confirmed by {confirm.date() if confirm is not None else None} | delay {delay} d")

frames, outline = rgb_frames(args.fid, x)
T = len(x)
xi = {d: i for i, d in enumerate(x.date)}          # index positions, one per date
def X(d):                                          # a date on the index axis
    return float(np.interp(d.value, [t.value for t in x.date], range(T)))

fig = plt.figure(figsize=(TEXT_W, TEXT_W * 0.62))
ax = fig.add_axes([0.10, 0.34, 0.88, 0.62])

ax.plot(range(T), x.p_dist, color="0.75", lw=1.0, zorder=1)
for win, g in x.groupby("win"):
    ax.scatter([xi[d] for d in g.date], g.p_dist, s=34, color=WIN_COLOR[win],
               edgecolor="white", linewidth=0.6, zorder=3, label=WIN_NAME[win])

ax.axhline(args.tau, color="black", ls=":", lw=1.1, zorder=2)
ax.annotate(f"threshold $\\tau = {args.tau:.3f}$", xy=(0.015, args.tau),
            xycoords=("axes fraction", "data"), xytext=(0, 4),
            textcoords="offset points", fontsize=9)

ax.axvline(X(view), color="black", ls="--", lw=1.1, zorder=2)
ax.annotate("\\textsc{deter} date".replace("\\textsc{deter}", "DETER date").replace(" date date", " date"),
            xy=(X(view), 1.0), xycoords=("data", "axes fraction"), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=9)

if alert is not None:
    ax.axvline(X(alert), color="#B03A2E", lw=1.4, zorder=2)
    ax.annotate(f"alert, {alert.strftime('%d %b')}", xy=(X(alert), 1.0),
                xycoords=("data", "axes fraction"), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color="#B03A2E")
    ax.scatter([X(alert)], [x.loc[x.date == alert, "p_dist"].iloc[0]], s=130, facecolor="none",
               edgecolor="#B03A2E", linewidth=1.6, zorder=4)
    if confirm is not None:
        p_conf = x.loc[x.date == confirm, "p_dist"].iloc[0]
        ax.scatter([X(confirm)], [p_conf], s=130, facecolor="none", edgecolor="#B03A2E",
                   linewidth=1.0, linestyle=":", zorder=4)
        ax.annotate(f"confirmed {(confirm - alert).days} d later", xy=(X(confirm), p_conf),
                    xytext=(10, 9), textcoords="offset points", fontsize=9, color="#B03A2E")
    # the lead the rule gains over the operational date
    ax.annotate("", xy=(X(view), 0.42), xytext=(X(alert), 0.42),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.35"))
    ax.annotate(f"{abs(delay)} days earlier", xy=((X(alert) + X(view)) / 2, 0.42),
                xycoords=("data", "axes fraction"), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom", fontsize=9,
                color="0.35")

ax.set_ylim(0, 1.16)
ax.set_xlim(-0.6, T - 0.4)
ax.set_xticks(range(T))
ax.set_xticklabels([])
ax.set_ylabel("$p$(non forest)")

# a thumbnail is placed under the point it belongs to, so fewer of them can be larger
if args.thumbs and args.thumbs < T:
    must = {x.date.iloc[0], x.date.iloc[-1], x.loc[x.p_dist.idxmin(), "date"]}
    if alert is not None:
        must |= {alert}
        if confirm is not None:
            must |= {confirm}
    must |= {min(x.date, key=lambda d: abs((d - view).days))}
    rest = [d for d in x.date if d not in must]
    step = max(1, len(rest) // max(1, args.thumbs - len(must)))
    sel = sorted(must | set(rest[::step][: max(0, args.thumbs - len(must))]))
else:
    sel = list(x.date)

fig.canvas.draw()
W, H = fig.get_size_inches()
n = len(sel)
w = min(0.145, 0.88 / n * 0.94)
left, right = 0.10, 0.98
y_top = 0.30                                   # top edge of the thumbnail row
for k, d in enumerate(sel):
    xf = left + (right - left) * (k + 0.5) / n     # evenly spread, not on the point
    axr = fig.add_axes([xf - w / 2, y_top - w * W / H, w, w * W / H])
    axr.imshow(np.nan_to_num(frames[d], nan=1.0))
    outline_on(axr, outline, frames[d].shape[0], frames[d].shape[1])
    win = x.loc[x.date == d, "win"].iloc[0]
    axr.text(0.5, -0.05, d.strftime("%d %b"), transform=axr.transAxes, ha="center",
             va="top", fontsize=7.5, color=WIN_COLOR[win])
    axr.axis("off")
    # a thin leader from the thumbnail to the acquisition it shows
    fig.add_artist(ConnectionPatch(
        xyA=(0.5, 1.0), coordsA=axr.transAxes,
        xyB=(xi[d], x.loc[x.date == d, "p_dist"].iloc[0]), coordsB=ax.transData,
        lw=0.6, color="0.55", ls=":"))

out = os.path.join(FIGDIR, f"nrt_example_fid{args.fid}_{args.mode}.pdf")
fig.savefig(out)
print("written", out)
