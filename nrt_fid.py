"""
Near-real-time detection for one FID, across the four embedding / label-space
combinations. For each mode a trained classifier is applied to every acquisition
(bef + evt + aft) and the predictions are plotted over time.

Modes:
    joint           CROMA joint embeddings,   7-class model
    joint_binary    CROMA joint embeddings,   forest vs. non-forest model
    optical         CROMA optical (S2),       7-class model
    optical_binary  CROMA optical (S2),       forest vs. non-forest model

Figures, per FID:
    nrt_fid<FID>_<mode>_<model>.pdf   one panel per mode: curve + class mosaic + RGB
    nrt_fid<FID>_compare_<model>.pdf  multiclass modes on top, binary modes below

The binary modes keep their own panel and their own row of the comparison figure.
Their label space is forest vs. clear-cut bare soil only, so their curve answers a
different question than the multiclass one and the two never share an axes.

Usage:
    python nrt_fid.py --fid 28
    python nrt_fid.py --fid 28 63 --model rf --save results_nrt
    python nrt_fid.py --fid 28 --mode joint optical
"""
import argparse
import glob
import os
import re

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from rand_forest import majority_downsample
from seg_dataset import CLASS_TO_ID

TILES = "/share/castor/home/e2406749/thesis_tiles_120px"
SHP = "data_shp/label_polygons.shp"

# One entry per (embedding modality, label space). `res` is the --results-root
# the matching run_eval*.sh wrote into, so the model always matches the tokens.
MODES = {
    "joint":          {"emb": "embeddings/joint",  "res": "results",                "binary": False},
    "joint_binary":   {"emb": "embeddings/joint",  "res": "results_binary",         "binary": True},
    "optical":        {"emb": "embeddings/s2_l2a", "res": "results_optical",        "binary": False},
    "optical_binary": {"emb": "embeddings/s2_l2a", "res": "results_binary_optical", "binary": True},
}
# joint and its binary twin share a color, so the comparison figure reads as
# "modality = color, label space = row".
MODE_COLORS = {"joint": "tab:blue", "joint_binary": "tab:blue",
               "optical": "tab:orange", "optical_binary": "tab:orange"}

WIN_COLORS = {"bef": "tab:blue", "evt": "tab:red", "aft": "tab:green"}
BIN_COLORS = ["#1b7837", "#a6611a"]        # forest, non-forest (same hues as ids 0 and 1)
BIN_NAMES = ["forest", "non-forest"]
TILE_M = 1200.0                            # one tile = 120 px x 10 m
GRID = 15                                  # tokens per tile side


def _parts(npy, emb_root):
    """(window, image_dir, tile) from an embedding path, independent of emb_root depth."""
    rel = os.path.relpath(npy, emb_root).split(os.sep)   # fid_X / win / img_dir / tile.npy
    return rel[1], rel[2], rel[3].replace(".npy", "")


def _palette(binary):
    """Colormap, norm, class names and legend colors for one label space."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    if binary:
        cmap = ListedColormap(BIN_COLORS)
        return cmap, BoundaryNorm([-0.5, 0.5, 1.5], cmap.N), BIN_NAMES, BIN_COLORS
    from eval_plots import CMAP, NORM, NAMES, CLASS_COLORS
    return CMAP, NORM, NAMES, [CLASS_COLORS[c] for c in range(7)]


def fid_curve(fid, clf, polys, emb_root, cls_id=None):
    """One row per acquisition, pooled over all polygon tokens of the FID.

        p_dist   1 - p(forest)   -- did it stop being forest?
        p_class  p(cls_id)       -- did it become this FID's actual class?
                                    multiclass only; the binary label space holds
                                    just forest and clear-cut, so a FID of any
                                    other class has no column to read.
    plus n_nonforest, the count of tokens whose argmax is not forest.
    """
    rows = []
    masks = {}
    for npy in glob.glob(f"{emb_root}/fid_{fid}/*/*/tile_*.npy"):
        win, img_dir, tile = _parts(npy, emb_root)

        if tile not in masks:
            tifs = glob.glob(f"{TILES}/s2_l2a/fid_{fid}/*/*/{tile}.tif")
            with rasterio.open(tifs[0]) as src:
                fine = rasterize(polys, (src.height, src.width), transform=src.transform)
            masks[tile] = majority_downsample(fine, GRID, GRID, 2).reshape(-1) > 0
        inside = masks[tile]
        if not inside.any():
            continue                               # tile misses the polygon

        emb = np.load(npy)
        proba = clf.predict_proba(emb.reshape(-1, emb.shape[-1])[inside])
        # Sums, not means: tiles hold unequal token counts and are pooled below.
        row = {"date": re.search(r"(\d{8})T", img_dir).group(1),
               "win": win,
               "tile": tile,
               "n_tok": int(inside.sum()),
               "p_sum": float((1 - proba[:, 0]).sum()),
               "n_nonforest": int((proba.argmax(1) != 0).sum())}
        if cls_id is not None:
            row["pc_sum"] = float(proba[:, cls_id].sum())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    agg = {"n_tok": ("n_tok", "sum"), "p_sum": ("p_sum", "sum"),
           "n_nonforest": ("n_nonforest", "sum"), "n_tiles": ("tile", "nunique")}
    if cls_id is not None:
        agg["pc_sum"] = ("pc_sum", "sum")
    out = pd.DataFrame(rows).groupby(["date", "win"], as_index=False).agg(**agg)

    out["p_dist"] = out.p_sum / out.n_tok          # token-weighted over the polygon
    if cls_id is not None:
        out["p_class"] = out.pc_sum / out.n_tok
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    return out.sort_values("date").reset_index(drop=True)


def build_mosaics(fid, clf, polys, emb_root):
    """Class-map and RGB mosaic per date, plus the polygon mask over the mosaic.

    The tiling grid is regular and non-overlapping (tile_pipeline.create_tile_grid),
    so tiles can be placed back by their geographic corner. RGB always comes from
    s2_l2a, whatever modality the embeddings are.
    """
    bounds = {}
    for tif in glob.glob(f"{TILES}/s2_l2a/fid_{fid}/*/*/tile_*.tif"):
        t = tif.split("/")[-1].replace(".tif", "")
        if t not in bounds:
            with rasterio.open(tif) as src:
                bounds[t] = src.bounds
    minx = min(b.left for b in bounds.values())
    maxy = max(b.top for b in bounds.values())
    pos = {t: (int(round((maxy - b.top) / TILE_M)), int(round((b.left - minx) / TILE_M)))
           for t, b in bounds.items()}
    nrow = max(r for r, _ in pos.values()) + 1
    ncol = max(c for _, c in pos.values()) + 1

    frames = {}
    for npy in glob.glob(f"{emb_root}/fid_{fid}/*/*/tile_*.npy"):
        win, img_dir, tile = _parts(npy, emb_root)
        s2_id = img_dir.split("__")[0]             # joint dirs are "<s2id>__<s1id>"
        date = re.search(r"(\d{8})T", img_dir).group(1)

        f = frames.setdefault(date, {
            "win": win,
            "pred": np.full((nrow * GRID, ncol * GRID), np.nan),   # NaN = no image
            "rgb": np.full((nrow * 120, ncol * 120, 3), np.nan),
        })
        r, c = pos[tile]

        # Full token grid, not only polygon tokens, so the map also shows what
        # the model says about the surrounding forest.
        emb = np.load(npy)
        f["pred"][r * GRID:(r + 1) * GRID, c * GRID:(c + 1) * GRID] = \
            clf.predict(emb.reshape(-1, emb.shape[-1])).reshape(GRID, GRID)

        tif = f"{TILES}/s2_l2a/fid_{fid}/{win}/{s2_id}/{tile}.tif"
        if os.path.exists(tif):
            with rasterio.open(tif) as src:
                f["rgb"][r * 120:(r + 1) * 120, c * 120:(c + 1) * 120] = \
                    np.stack([src.read(b).astype("float32") for b in (4, 3, 2)], -1)

    # Stretch each date over the whole mosaic, so tiles don't look mismatched.
    for f in frames.values():
        lo, hi = np.nanpercentile(f["rgb"], [2, 98])
        f["rgb"] = np.clip((f["rgb"] - lo) / (hi - lo + 1e-6), 0, 1)

    fine = rasterize(polys, (nrow * 120, ncol * 120),
                     transform=from_origin(minx, maxy, 10, 10))
    return frames, fine, nrow, ncol


def plot_fid(fid, curve, frames, fine, nrow, ncol, sub, mode, model_name):
    """Curve on top, class mosaic and RGB mosaic aligned underneath."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from eval_plots import _outline_polygon

    binary = MODES[mode]["binary"]
    cmap, norm, names, colors = _palette(binary)
    cmap = cmap.copy()
    cmap.set_bad("white")                          # tiles missing on that date

    keys = curve.date.dt.strftime("%Y%m%d")
    T = len(curve)

    fig = plt.figure(figsize=(max(11, 1.45 * T), 9))
    gs = gridspec.GridSpec(3, T, height_ratios=[3, 1.3, 1.3], hspace=0.12, wspace=0.06)

    # Index positions on the x axis, not real dates, so each point sits above
    # its thumbnails. VIEW_DATE is interpolated to the same scale.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(range(T), curve.p_dist, "-", color="lightgray", zorder=1)
    ax.scatter(range(T), curve.p_dist, c=[WIN_COLORS[w] for w in curve.win],
               s=70, edgecolors="black", lw=0.5, zorder=3,
               label="p(non-forest)" if binary else "1 - p(forest)")
    if not binary:
        # Second criterion: probability of this FID's actual disturbance class.
        ax.plot(range(T), curve.p_class, "--", color="tab:purple", lw=1.5, zorder=2,
                label=f"p({sub['CLASSNAME'].iloc[0]})")

    view = pd.to_datetime(sub["VIEW_DATE"].iloc[0])
    p = float(np.interp(view.value, [t.value for t in curve.date], range(T)))
    ax.axvline(p, color="black", ls="--", lw=1.5, zorder=0, label="VIEW_DATE")

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xticks(range(T))
    ax.set_xticklabels([])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("probability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"fid {fid} - {sub['CLASSNAME'].iloc[0]} - {mode} / {model_name}\n"
                 f"{curve.n_tok.iloc[0]} polygon tokens over {nrow}x{ncol} tiles",
                 fontsize=10)

    for i, (key, win) in enumerate(zip(keys, curve.win)):
        f = frames.get(key)
        if f is None:
            continue

        ax_map = fig.add_subplot(gs[1, i])
        ax_map.imshow(np.ma.masked_invalid(f["pred"]), cmap=cmap, norm=norm,
                      interpolation="nearest")
        _outline_polygon(ax_map, fine, nrow * GRID, ncol * GRID)
        ax_map.axis("off")

        ax_rgb = fig.add_subplot(gs[2, i])
        ax_rgb.imshow(np.nan_to_num(f["rgb"], nan=1.0))           # gaps -> white
        _outline_polygon(ax_rgb, fine, nrow * 120, ncol * 120)
        ax_rgb.set_title(f"{key[:4]}-{key[4:6]}-{key[6:]}\n({win})",
                         fontsize=6, color=WIN_COLORS[win], y=-0.40)
        ax_rgb.axis("off")

    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
               names, loc="lower center", ncol=min(4, len(names)), fontsize=8)
    fig.subplots_adjust(bottom=0.14)
    return fig


def plot_compare(fid, curves, sub, model_name):
    """Multiclass modes on the top axes, binary modes on the bottom one.

    Split by label space on purpose: 1 - p(forest) from a 7-class model and
    p(non-forest) from a forest-vs-clear-cut model are not the same quantity.
    Real dates on the x axis, because joint only exists where an S1 pair was
    found and therefore has fewer acquisitions than optical.
    """
    import matplotlib.pyplot as plt

    groups = [(False, "1 - p(forest)", "multiclass (7 classes)"),
              (True, "p(non-forest)", "binary (forest vs. clear-cut)")]
    groups = [(b, yl, t) for b, yl, t in groups
              if any(MODES[m]["binary"] == b for m in curves)]

    view = pd.to_datetime(sub["VIEW_DATE"].iloc[0])
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 3.4 * len(groups)),
                             sharex=True, squeeze=False)

    for ax, (binary, ylab, title) in zip(axes[:, 0], groups):
        for mode, c in curves.items():
            if MODES[mode]["binary"] != binary:
                continue
            col = MODE_COLORS[mode]
            ax.plot(c.date, c.p_dist, "-", color=col, lw=1.5, zorder=2, label=mode)
            # marker fill = window, marker edge = modality
            ax.scatter(c.date, c.p_dist, s=55, zorder=3, lw=1.2, edgecolors=col,
                       c=[WIN_COLORS[w] for w in c.win])
        ax.axvline(view, color="black", ls="--", lw=1.5, zorder=1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=WIN_COLORS[w],
                          markeredgecolor="black", label=w) for w in ("bef", "evt", "aft")]
    handles.append(plt.Line2D([], [], color="black", ls="--", label="VIEW_DATE"))
    axes[0, 0].legend(handles=handles + axes[0, 0].get_legend_handles_labels()[0],
                      fontsize=8, loc="upper left", ncol=2)

    fig.suptitle(f"fid {fid} - {sub['CLASSNAME'].iloc[0]} - {model_name}", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", nargs="+", required=True)
    ap.add_argument("--mode", nargs="+", default=["all"],
                    help=f"any of {list(MODES)}, or 'all'")
    ap.add_argument("--model", default="log_reg", help="log_reg | rf")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="save PDFs into DIR; omitted -> show on screen")
    ap.add_argument("--no-panels", action="store_true",
                    help="only the comparison figure, skip the per-mode panels")
    args = ap.parse_args()

    modes = list(MODES) if "all" in args.mode else args.mode
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown mode(s) {bad}; pick from {list(MODES)} or 'all'")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
        os.makedirs(args.save, exist_ok=True)
    import matplotlib.pyplot as plt

    # Load each mode's classifier once. A mode whose model or embeddings are
    # missing is skipped with a warning instead of killing the whole run.
    clfs = {}
    for mode in modes:
        path = os.path.join(MODES[mode]["res"], args.model, "model.joblib")
        if not os.path.exists(path):
            print(f"[skip] {mode}: no model at {path}")
            continue
        if not os.path.isdir(MODES[mode]["emb"]):
            print(f"[skip] {mode}: no embeddings at {MODES[mode]['emb']}")
            continue
        clfs[mode] = joblib.load(path)
    if not clfs:
        ap.error("no usable mode: check --model and the results*/ directories")

    gdf = gpd.read_file(SHP).to_crs("EPSG:3857")
    gdf["fid"] = gdf["fid"].astype(int).astype(str)

    def out(fig, name):
        if args.save:
            path = os.path.join(args.save, name)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")
        else:
            plt.show()

    for fid in args.fid:
        sub = gdf[gdf["fid"] == str(fid)]
        polys = [(g, 1) for g in sub.geometry]
        cls_id = CLASS_TO_ID[sub["CLASSNAME"].iloc[0]]
        print(f"\n=== fid {fid} | {sub['CLASSNAME'].iloc[0]} | "
              f"VIEW_DATE {pd.to_datetime(sub['VIEW_DATE'].iloc[0]).date()}")

        curves = {}
        for mode, clf in clfs.items():
            binary = MODES[mode]["binary"]
            curve = fid_curve(fid, clf, polys, MODES[mode]["emb"],
                              cls_id=None if binary else cls_id)
            if curve.empty:
                print(f"[skip] {mode}: no tile of fid {fid} touches the polygon")
                continue
            curves[mode] = curve

            cols = ["date", "win", "n_tok", "n_tiles", "p_dist", "n_nonforest"]
            if not binary:
                cols.insert(5, "p_class")
            print(f"\n--- {mode} / {args.model}")
            print(curve[cols].to_string(index=False))

            if not args.no_panels:
                frames, fine, nrow, ncol = build_mosaics(fid, clf, polys, MODES[mode]["emb"])
                fig = plot_fid(fid, curve, frames, fine, nrow, ncol, sub, mode, args.model)
                out(fig, f"nrt_fid{fid}_{mode}_{args.model}.pdf")

        if len(curves) > 1:
            out(plot_compare(fid, curves, sub, args.model),
                f"nrt_fid{fid}_compare_{args.model}.pdf")


if __name__ == "__main__":
    main()
