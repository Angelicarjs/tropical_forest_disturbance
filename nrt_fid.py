"""
Near-real-time detection for one FID: apply a trained classifier to every
acquisition (bef + evt + aft) and plot how the predictions evolve.

    top     1 - p(forest), pooled over every polygon token of the FID
    middle  predicted class mosaic (all tiles stitched together)
    bottom  S2 true-color mosaic, same layout

Usage:
    python nrt_fid.py --fid 28
    python nrt_fid.py --fid 28 63 --model rf --save results_nrt
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

TILES = "/share/castor/home/e2406749/thesis_tiles_120px"
EMB_ROOT = "embeddings/joint"
SHP = "data_shp/label_polygons.shp"
MODELS = {"log_reg": "results/log_reg/model.joblib",
          "rf": "results/rf/model.joblib"}

WIN_COLORS = {"bef": "tab:blue", "evt": "tab:red", "aft": "tab:green"}
TILE_M = 1200.0                       # one tile = 120 px x 10 m
GRID = 15                             # tokens per tile side


def fid_curve(fid, clf, polys):
    """One row per acquisition: 1 - p(forest) over all polygon tokens of the FID."""
    rows = []
    masks = {}
    for npy in glob.glob(f"{EMB_ROOT}/fid_{fid}/*/*/tile_*.npy"):
        tile = npy.split("/")[-1].replace(".npy", "")

        if tile not in masks:
            tifs = glob.glob(f"{TILES}/s2_l2a/fid_{fid}/*/*/{tile}.tif")
            with rasterio.open(tifs[0]) as src:
                fine = rasterize(polys, (src.height, src.width), transform=src.transform)
            masks[tile] = majority_downsample(fine, GRID, GRID, 2).reshape(-1) > 0
        inside = masks[tile]
        if not inside.any():
            continue                               # tile misses the polygon

        proba = clf.predict_proba(np.load(npy).reshape(-1, 768)[inside])
        # Sums, not means: tiles hold unequal token counts and are pooled below.
        rows.append({"date": re.search(r"(\d{8})T", npy).group(1),
                     "win": npy.split("/")[3],
                     "tile": tile,
                     "n_tok": int(inside.sum()),
                     "p_sum": float((1 - proba[:, 0]).sum())})

    out = (pd.DataFrame(rows).groupby(["date", "win"], as_index=False)
             .agg(n_tok=("n_tok", "sum"), p_sum=("p_sum", "sum"),
                  n_tiles=("tile", "nunique")))
    out["p_dist"] = out.p_sum / out.n_tok          # token-weighted over the polygon
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    return out.sort_values("date").reset_index(drop=True)


def build_mosaics(fid, clf, polys):
    """Class-map and RGB mosaic per date, plus the polygon mask over the mosaic.

    The tiling grid is regular and non-overlapping (tile_pipeline.create_tile_grid),
    so tiles can be placed back by their geographic corner.
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
    for npy in glob.glob(f"{EMB_ROOT}/fid_{fid}/*/*/tile_*.npy"):
        tile = npy.split("/")[-1].replace(".npy", "")
        win, img_dir = npy.split("/")[3], npy.split("/")[4]
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
        f["pred"][r * GRID:(r + 1) * GRID, c * GRID:(c + 1) * GRID] = \
            clf.predict(np.load(npy).reshape(-1, 768)).reshape(GRID, GRID)

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


def plot_fid(fid, curve, frames, fine, nrow, ncol, sub, model_name):
    """Curve on top, class mosaic and RGB mosaic aligned underneath."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from eval_plots import CMAP, NORM, NAMES, CLASS_COLORS, _outline_polygon

    keys = curve.date.dt.strftime("%Y%m%d")
    T = len(curve)
    cmap = CMAP.copy()
    cmap.set_bad("white")                          # tiles missing on that date

    fig = plt.figure(figsize=(max(11, 1.45 * T), 9))
    gs = gridspec.GridSpec(3, T, height_ratios=[3, 1.3, 1.3], hspace=0.12, wspace=0.06)

    # Index positions on the x axis, not real dates, so each point sits above
    # its thumbnails. VIEW_DATE is interpolated to the same scale.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(range(T), curve.p_dist, "-", color="lightgray", zorder=1)
    ax.scatter(range(T), curve.p_dist, c=[WIN_COLORS[w] for w in curve.win],
               s=70, edgecolors="black", lw=0.5, zorder=3)

    view = pd.to_datetime(sub["VIEW_DATE"].iloc[0])
    p = float(np.interp(view.value, [t.value for t in curve.date], range(T)))
    ax.axvline(p, color="black", ls="--", lw=1.5, zorder=0, label="VIEW_DATE")

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xticks(range(T))
    ax.set_xticklabels([])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("1 - p(forest)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"fid {fid} - {sub['CLASSNAME'].iloc[0]} - {model_name}\n"
                 f"{curve.n_tok.iloc[0]} polygon tokens over {nrow}x{ncol} tiles",
                 fontsize=10)

    for i, (key, win) in enumerate(zip(keys, curve.win)):
        f = frames.get(key)
        if f is None:
            continue

        ax_map = fig.add_subplot(gs[1, i])
        ax_map.imshow(np.ma.masked_invalid(f["pred"]), cmap=cmap, norm=NORM,
                      interpolation="nearest")
        _outline_polygon(ax_map, fine, nrow * GRID, ncol * GRID)
        ax_map.axis("off")

        ax_rgb = fig.add_subplot(gs[2, i])
        ax_rgb.imshow(np.nan_to_num(f["rgb"], nan=1.0))           # gaps -> white
        _outline_polygon(ax_rgb, fine, nrow * 120, ncol * 120)
        ax_rgb.set_title(f"{key[:4]}-{key[4:6]}-{key[6:]}\n({win})",
                         fontsize=6, color=WIN_COLORS[win], y=-0.40)
        ax_rgb.axis("off")

    fig.legend([plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in range(7)],
               NAMES, loc="lower center", ncol=4, fontsize=8)
    fig.subplots_adjust(bottom=0.14)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", nargs="+", required=True)
    ap.add_argument("--model", default="log_reg", help="log_reg | rf | path to .joblib")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="save one PDF per FID into DIR (nrt_fid<FID>_<model>.pdf); "
                         "omitted -> show on screen")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
        os.makedirs(args.save, exist_ok=True)
    import matplotlib.pyplot as plt

    clf = joblib.load(MODELS.get(args.model, args.model))
    gdf = gpd.read_file(SHP).to_crs("EPSG:3857")
    gdf["fid"] = gdf["fid"].astype(int).astype(str)

    for fid in args.fid:
        sub = gdf[gdf["fid"] == str(fid)]
        polys = [(g, 1) for g in sub.geometry]

        curve = fid_curve(fid, clf, polys)
        print(f"\n=== fid {fid} | {sub['CLASSNAME'].iloc[0]} | "
              f"VIEW_DATE {pd.to_datetime(sub['VIEW_DATE'].iloc[0]).date()}")
        print(curve[["date", "win", "n_tok", "n_tiles", "p_dist"]].to_string(index=False))

        frames, fine, nrow, ncol = build_mosaics(fid, clf, polys)
        fig = plot_fid(fid, curve, frames, fine, nrow, ncol, sub, args.model)

        if args.save:
            path = os.path.join(args.save, f"nrt_fid{fid}_{args.model}.pdf")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
