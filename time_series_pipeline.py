"""Per-(fid, tile, token) visual pipeline matching time_series_tile.ipynb.

Cloud-filter version (v1/v2/v3) is applied end-to-end so that the optical, SAR
and joint analyses all use the same image set as the joint pairing.

Usage:
    from time_series_pipeline import run
    run(fid=12, tile=2)                       # random token with seed=42, v3
    run(12, 2, version="v1")                  # compare against v1 cloud filter
    run(12, 2, seed=9)                        # different random token
    run(12, 2, tok_r=5, tok_c=0)              # explicit token override
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from rasterio.features import geometry_mask
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

from obs_date import obs_date
from make_embeddings import (
    embed_image,
    embed_image_joint,
    iter_joint_pairs_from_csv,
    load_model,
    load_or_compute_stats,
)

# The tiles live outside the repository. $HOME/thesis_tiles_120px on the cluster,
# the same location run_eval.py and run_embeddings.sh assume; override with THESIS_TILES.
TILES_ROOT = os.environ.get("THESIS_TILES", os.path.expanduser("~/thesis_tiles_120px"))
EMB_ROOT = "embeddings"
SHP_PATH = "data/data_shp/label_polygons.shp"
CSV_TEMPLATE = "data/data_csv/{version}_images_s2_s1.csv"
_VERSION = "v3"  # current cloud-filter version; set by run(), shown in plot titles
_VERSIONS = ("v1", "v2", "v3")  # all cloud-filter versions (for the common PCA base)

SUBDIRS = {"optical": "s2_l2a", "sar": "s1_grd", "joint": "joint"}
WIN_COLORS = {"bef": "tab:blue", "evt": "tab:red", "aft": "tab:green"}

_MODELS: dict[str, object] = {}
_STATS: dict[str, object] = {}


def _get_model(modality: str):
    if modality not in _MODELS:
        _MODELS[modality] = load_model(modality)
    return _MODELS[modality]


def _get_stats(product: str):
    if product not in _STATS:
        _STATS[product] = load_or_compute_stats(TILES_ROOT, product)
    return _STATS[product]


def _by_s2_date(p: str) -> str:
    return re.search(r"/(\d{8})T", p).group(1)


def _by_s1_date(p: str) -> str:
    return re.search(r"_(\d{8})T", p).group(1)


def _by_date_any(p: str) -> str:
    """Date of an observation, joint pairs included. See obs_date.py: a pair is
    dated by the later of its two acquisitions, and the same helper is used by
    the near real time analysis so both date an observation identically."""
    return obs_date(p)


def _by_win(p: str) -> str:
    return re.search(r"/(bef|evt|aft)/", p).group(1)


def _split_ids(cell) -> list[str]:
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]


def _allowed_ids_for_fid(csv_path: str, fid: int) -> dict[str, set[str]]:
    """Union of bef/evt/aft S2 and S1 image_ids for this fid in the given CSV."""
    df = pd.read_csv(csv_path)
    df = df[df["fid"].astype(int) == int(fid)]
    s2_ids: set[str] = set()
    s1_ids: set[str] = set()
    for win in ("bef", "evt", "aft"):
        for _, row in df.iterrows():
            s2_ids.update(_split_ids(row.get(f"{win}IdsS2")))
            s1_ids.update(_split_ids(row.get(f"{win}IdsS1")))
    return {"s2": s2_ids, "s1": s1_ids}


def _on_disk_s2_ids(fid: int, tile: int) -> set[str]:
    """S2 image_ids whose tile_{tile}.tif survived tile verification on disk.

    Tile verification drops empty/border tiles per (image, tile_idx), so an
    image listed in the CSVs may have no tile_{tile} on disk for this fid.
    """
    return {_image_id_of(p) for p in
            glob.glob(f"{TILES_ROOT}/s2_l2a/fid_{fid}/*/*/tile_{tile}.tif")}


def _common_s2_ids_across_versions(fid: int, tile: int) -> set[str]:
    """S2 image_ids common to ALL cloud-filter versions (v1∩v2∩v3) and on disk."""
    per_version = [_allowed_ids_for_fid(CSV_TEMPLATE.format(version=v), fid)["s2"]
                   for v in _VERSIONS]
    return set.intersection(*per_version) & _on_disk_s2_ids(fid, tile)

# to print 
def _first_common_s2_date(fid: int, tile: int) -> str | None:
    """Earliest acquisition date (YYYYMMDD) of the S2 images common to every version."""
    dates = [m.group(1) for i in _common_s2_ids_across_versions(fid, tile)
             if (m := re.search(r"(\d{8})T", i))]
    print(min(dates) if dates else None)
    return min(dates) if dates else None


def _image_id_of(path: str) -> str:
    """The image_id is the parent dir of a tile_N.{tif,npy} path."""
    return Path(path).parent.name


def _filter_s2(paths: list[str], allowed_s2: set[str]) -> list[str]:
    return [p for p in paths if _image_id_of(p) in allowed_s2]


def _filter_s1(paths: list[str], allowed_s1: set[str]) -> list[str]:
    return [p for p in paths if _image_id_of(p) in allowed_s1]


def _filter_joint(paths: list[str], allowed_s2: set[str], allowed_s1: set[str]) -> list[str]:
    out = []
    for p in paths:
        name = _image_id_of(p)
        if "__" not in name:
            continue
        s2_id, s1_id = name.split("__", 1)
        if s2_id in allowed_s2 and s1_id in allowed_s1:
            out.append(p)
    return out


def _create_embeddings(fid: int, csv_path: str, max_gap_days: int) -> None:
    """Embed any missing tiles for this fid. embed_image skips existing .npy."""
    s2_dirs = sorted(glob.glob(f"{TILES_ROOT}/s2_l2a/fid_{fid}/*/*"))
    s1_dirs = sorted(glob.glob(f"{TILES_ROOT}/s1_grd/fid_{fid}/*/*"))
    s2_stats = _get_stats("s2_l2a")
    s1_stats = _get_stats("s1_grd")
    for d in s2_dirs:
        embed_image(d, model=_get_model("optical"), stats=s2_stats)
    for d in s1_dirs:
        embed_image(d, model=_get_model("sar"), stats=s1_stats)
    for f, _win, s2_dir, s1_dir, _gap in iter_joint_pairs_from_csv(
        csv_path, TILES_ROOT, max_gap_days=max_gap_days
    ):
        if f != fid:
            continue
        embed_image_joint(
            s2_dir,
            s1_dir,
            model=_get_model("joint"),
            s2_stats=s2_stats,
            s1_stats=s1_stats,
        )


def _load_polygon(fid: int):
    gdf = gpd.read_file(SHP_PATH)
    gdf["fid"] = gdf["fid"].astype(int).astype(str)
    return gdf, gdf[gdf["fid"] == str(fid)].geometry.iloc[0]


def _event_date(fid: int):
    """Official disturbance detection date (VIEW_DATE) for this fid, or None."""
    gdf = gpd.read_file(SHP_PATH)
    gdf["fid"] = gdf["fid"].astype(int)
    sub = gdf[gdf["fid"] == int(fid)]
    if sub.empty or pd.isna(sub["VIEW_DATE"].iloc[0]):
        return None
    return pd.to_datetime(sub["VIEW_DATE"].iloc[0])


def _add_event_line(ax, event_date):
    """Dashed vertical line at the disturbance event date (VIEW_DATE)."""
    if event_date is not None:
        ax.axvline(event_date, color="black", ls="--", lw=1.5, zorder=0)


def _first_s2_tile(fid: int, tile: int) -> str:
    matches = sorted(glob.glob(f"{TILES_ROOT}/s2_l2a/fid_{fid}/*/*/tile_{tile}.tif"))
    if not matches:
        raise FileNotFoundError(f"No S2 tile_{tile}.tif found for fid {fid}")
    return matches[0]


def _pick_random_token(fid: int, tile: int, seed: int):
    """Pick a random pixel inside the polygon mask. Returns the marker plot inputs."""
    gdf, poly = _load_polygon(fid)
    with rasterio.open(_first_s2_tile(fid, tile)) as src:
        poly_t = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(src.crs).iloc[0]
        mask = geometry_mask(
            [poly_t.__geo_interface__],
            (src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        rgb_tile = src.read([4, 3, 2]).transpose(1, 2, 0)
        poly_pix = [~src.transform * (x, y) for x, y in zip(*poly_t.exterior.xy)]

    np.random.seed(seed)
    ys, xs = np.where(mask)
    i = np.random.randint(len(ys))
    pix_row, pix_col = int(ys[i]), int(xs[i])
    tok_r, tok_c = pix_row // 8, pix_col // 8
    return tok_r, tok_c, pix_row, pix_col, rgb_tile, poly_pix


def _show_tile_with_pixel(fid: int, tile: int, pix_row: int, pix_col: int, rgb_tile, poly_pix):
    plt.imshow(rgb_tile / 3000, vmin=0, vmax=1)
    plt.gca().add_patch(MplPolygon(poly_pix, fill=False, edgecolor="red", lw=2))
    plt.scatter(pix_col, pix_row, c="yellow", s=120, edgecolors="black", zorder=5)
    plt.title(f"FID {fid}, tile_{tile} — pixel ({pix_row}, {pix_col})")
    plt.show()

def _show_explicit_token(fid: int, tile: int, tok_r: int, tok_c: int):
      """Draw the first S2 tile with the (tok_r, tok_c) token center marked."""
      gdf, poly = _load_polygon(fid)
      with rasterio.open(_first_s2_tile(fid, tile)) as src:
          poly_t = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(src.crs).iloc[0]
          rgb_tile = src.read([4, 3, 2]).transpose(1, 2, 0)
          poly_pix = [~src.transform * (x, y) for x, y in zip(*poly_t.exterior.xy)]
      pix_row, pix_col = tok_r * 8 + 4, tok_c * 8 + 4    # token center
      _show_tile_with_pixel(fid, tile, pix_row, pix_col, rgb_tile, poly_pix)

def _paths_for(fid: int, tile: int, modality: str, allowed_s2, allowed_s1) -> list[str]:
    subdir = SUBDIRS[modality]
    paths = sorted(glob.glob(f"{EMB_ROOT}/{subdir}/fid_{fid}/*/*/tile_{tile}.npy"))
    if modality == "optical":
        paths = _filter_s2(paths, allowed_s2)
    elif modality == "sar":
        paths = _filter_s1(paths, allowed_s1)
    else:
        paths = _filter_joint(paths, allowed_s2, allowed_s1)
    paths.sort(key=_by_date_any)
    return paths


def _cosine_sim_plot(fid: int, tile: int, tok_r: int, tok_c: int, modality: str,
                     allowed_s2, allowed_s1):
    paths = _paths_for(fid, tile, modality, allowed_s2, allowed_s1)
    if not paths:
        print(f"[{modality}] no embeddings after filter — skipping cosine sim")
        return
    cube = np.stack([np.load(p) for p in paths])
    ts = cube[:, tok_r, tok_c, :]

    dates = pd.to_datetime([_by_date_any(p) for p in paths])
    wins = [_by_win(p) for p in paths]
    labels = [f"{d.strftime('%Y-%m-%d')} ({w})" for d, w in zip(dates, wins)]

    X = normalize(ts, axis=1)
    S = X @ X.T
    vmin, vmax = np.percentile(S, [1, 99])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if i != j:
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(f"Cosine sim {modality.upper()} TxT — token ({tok_r},{tok_c}) — {_VERSION}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.show()


def _ndvi_profile(fid: int, tile: int, tok_r: int, tok_c: int, allowed_s2,
                  red_idx: int = 4, nir_idx: int = 8):
    paths = sorted(
        glob.glob(f"{TILES_ROOT}/s2_l2a/fid_{fid}/*/*/tile_{tile}.tif"),
        key=_by_s2_date,
    )
    paths = _filter_s2(paths, allowed_s2)
    if not paths:
        print("[ndvi] no S2 tiles after filter — skipping")
        return

    r0, c0 = tok_r * 8, tok_c * 8
    dates, ndvi, wins = [], [], []
    for p in paths:
        with rasterio.open(p) as src:
            red = src.read(red_idx)[r0:r0 + 8, c0:c0 + 8].astype("float32")
            nir = src.read(nir_idx)[r0:r0 + 8, c0:c0 + 8].astype("float32")
        ndvi.append(float(((nir - red) / (nir + red + 1e-8)).mean()))
        dates.append(_by_s2_date(p))
        wins.append(_by_win(p))

    dates = pd.to_datetime(dates)
    colors = [WIN_COLORS[w] for w in wins]

    plt.figure(figsize=(8, 3))
    plt.plot(dates, ndvi, "-", color="lightgray", zorder=1)
    plt.scatter(dates, ndvi, c=colors, s=40, zorder=2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WIN_COLORS[w], label=w, markersize=8)
        for w in ("bef", "evt", "aft") if w in wins
    ]
    _add_event_line(plt.gca(), _event_date(fid))
    plt.legend(handles=handles, loc="best")
    plt.title(f"NDVI — fid {fid}, tile {tile}, token ({tok_r},{tok_c}) — {_VERSION}")
    plt.ylabel("NDVI"); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _fit_common_base_pca(fid: int, tile: int, modality: str, num_components: int = 3):
    """Fit PCA on the base image anchored at the earliest S2 date common to every
    cloud-filter version (v1-v2-v3), so PC axes are comparable across versions and
    modalities.Returns (pca, base_id, base_date) or None if there is no common S2 date / no image after it.
    """
    subdir = SUBDIRS[modality]
    paths_all = sorted(glob.glob(f"{EMB_ROOT}/{subdir}/fid_{fid}/*/*/tile_{tile}.npy"),
                       key=_by_date_any)
    if not paths_all:
        return None
    base_date = _first_common_s2_date(fid, tile)
    if base_date is None:
        return None
    base_paths = [p for p in paths_all if _by_date_any(p) >= base_date]
    if not base_paths:
        return None
    base_img = np.load(base_paths[0])                       # (R, C, D)
    R, C, D = base_img.shape
    pca = PCA(n_components=num_components).fit(base_img.reshape(R * C, D))
    return pca, _image_id_of(base_paths[0]), base_date


def _pca_first_image(fid: int, tile: int, tok_r: int, tok_c: int, modality: str,
                     allowed_s2, allowed_s1, num_components: int = 3):
    # create the cube of embeddings for this fid/tile/modality
    paths = _paths_for(fid, tile, modality, allowed_s2, allowed_s1)
    if not paths:
        print(f"[{modality}] no embeddings after filter — skipping PCA")
        return

    #cube depending on the version of cloud filter
    cube = np.stack([np.load(p) for p in paths])
    #time, rows, cols, dims 
    T,R,C,D = cube.shape
    print(f"[{modality}] cube shape: {cube.shape}")

    # COMMON BASE: PCA fitted on the image anchored at the earliest S2 date that
    # survives EVERY cloud filter (v1-v2-v3), so PC1 is comparable across versions.
    base = _fit_common_base_pca(fid, tile, modality, num_components)
    if base is None:
        print(f"[{modality}] no common-S2 base image — skipping PCA")
        return
    pca, base_id, base_date = base
    print(f"[{modality}] PCA base = {base_id} (common S2 date {base_date})")

    #single-token time series: pick the (tok_r, tok_c) token at every time step -> (T, 768)
    token = cube[:, tok_r, tok_c, :]              # (T, 768)

    #project the token series onto the PCA components
    scores = pca.transform(token)                    # (T, 3)
    evr = pca.explained_variance_ratio_
    print(f"[{modality}] PCA on first image ({R*C} tokens) — EVR={evr[:num_components].round(3).tolist()}") 
    
    #fusion PC1, PC2, PC3 into a single value for visualization
    mn, mx = scores.min(axis=0), scores.max(axis=0)
    rgb = (scores - mn) / (mx - mn) #normalize to [0,1]

    dates = pd.to_datetime([_by_date_any(p) for p in paths])
    wins = [_by_win(p) for p in paths]
    labels = [f"{d.strftime('%Y-%m-%d')} ({w})" for d, w in zip(dates, wins)]
    colors = [WIN_COLORS[w] for w in wins]
    evt = _event_date(fid)                       # disturbance detection date (VIEW_DATE)

    # plot 1: PC1 only (identical style to most-variable-dim)
    plt.figure(figsize=(10, 4))
    plt.plot(dates, scores[:, 0], "-", color="gray", alpha=0.5)
    plt.scatter(dates, scores[:, 0], c=colors, s=70,
                edgecolors="black", linewidths=0.5, zorder=3)
    if evt is not None:
        plt.axvline(evt, color="black", ls="--", lw=1.5, label="event")
    plt.title(f"PCA PC1 (var: {evr[0]:.1%}) — fid {fid}, tile {tile}, {modality} — {_VERSION}")
    plt.ylabel("PC1 score")
    plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor=c, label=w) for w, c in WIN_COLORS.items() if w in wins],
        title="window",
    )
    plt.tight_layout(); plt.show()

    # plot 2: PC1, PC2, PC3 as three lines on one plot
    pc_colors = ["tab:purple", "tab:orange", "tab:cyan"]   # one per component

    plt.figure(figsize=(10, 4))
    for k in range(scores.shape[1]):
        plt.plot(dates, scores[:, k], "-", color=pc_colors[k], alpha=0.7, zorder=1)
        plt.scatter(dates, scores[:, k], c=colors, s=70,
                    edgecolors="black", linewidths=0.5, zorder=3)
    if evt is not None:
        plt.axvline(evt, color="black", ls="--", lw=1.5)

    # two legends: window (point color) + component (line color)
    win_handles = [Patch(facecolor=c, label=w) for w, c in WIN_COLORS.items() if w in wins]
    pc_handles = [plt.Line2D([0], [0], color=pc_colors[k], lw=2, label=f"PC{k+1} ({evr[k]:.1%})")
                for k in range(scores.shape[1])]
    leg1 = plt.legend(handles=win_handles, title="window", loc="upper left")
    plt.gca().add_artist(leg1)
    plt.legend(handles=pc_handles, title="component", loc="upper right")

    plt.title(f"PCA PC1–PC3 — fid {fid}, tile {tile}, {modality} — {_VERSION}")
    plt.ylabel("PC score")
    plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def _most_variable_dim(fid: int, tile: int, modality: str, allowed_s2, allowed_s1,
                       top_k: int = 5, tok_r: int = None, tok_c: int = None):
    """Follow the most variable embedding dimension of one token through time.

    The unit is the token, the same one the cosine and the PCA readings follow, so
    the three readings describe the same location. Collapsing the whole tile instead
    would reintroduce the tile level quantity that the mosaic analysis showed to be
    spatially inconsistent, and on a small polygon the tokens outside it dominate
    the mean. The tile mean is kept as the fallback for callers that follow no
    particular token.
    """
    paths = _paths_for(fid, tile, modality, allowed_s2, allowed_s1)
    if not paths:
        print(f"[{modality}] no embeddings after filter — skipping most-variable-dim")
        return

    cube = np.stack([np.load(p) for p in paths])      # (T, R, C, 768)
    if tok_r is not None and tok_c is not None:
        series = cube[:, tok_r, tok_c, :]             # (T, 768)
        unit = f"token ({tok_r},{tok_c})"
    else:
        series = cube.mean(axis=(1, 2))               # (T, 768)
        unit = "tile mean"

    dim_scores = series.std(axis=0)               # (768,)
    sorted_dims = np.argsort(dim_scores)[::-1]
    top_dim = sorted_dims[0]
    print(f"[{modality}] cube shape: {cube.shape} | unit: {unit}")
    print(f"[{modality}] top dim: {top_dim} (std={dim_scores[top_dim]:.4f})")
    print(f"[{modality}] top {top_k} dims: {sorted_dims[:top_k].tolist()}")
    print(f"[{modality}] top {top_k} scores: {dim_scores[sorted_dims[:top_k]].round(4).tolist()}")

    dates = pd.to_datetime([_by_date_any(p) for p in paths])
    wins = [_by_win(p) for p in paths]
    colors = [WIN_COLORS[w] for w in wins]

    plt.figure(figsize=(10, 4))
    plt.plot(dates, series[:, top_dim], "-", color="gray", alpha=0.5)
    plt.scatter(dates, series[:, top_dim], c=colors, s=70,
                edgecolors="black", linewidths=0.5, zorder=3)
    _add_event_line(plt.gca(), _event_date(fid))
    plt.title(f"Dim {top_dim} (std={dim_scores[top_dim]:.3f}) — fid {fid}, tile {tile}, "
              f"{modality}, {unit} — {_VERSION}")
    plt.ylabel(f"embedding value (dim {top_dim})")
    plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor=c, label=w) for w, c in WIN_COLORS.items() if w in wins],
        title="window",
    )
    plt.tight_layout(); plt.show()


def _optical_rgb_grid(fid: int, tile: int, allowed_s2, ncols: int = 5, vmax: int = 3000):
    paths = sorted(
        glob.glob(f"{TILES_ROOT}/s2_l2a/fid_{fid}/*/*/tile_{tile}.tif"),
        key=_by_s2_date,
    )
    paths = _filter_s2(paths, allowed_s2)
    if not paths:
        print("[rgb-grid] no S2 tiles after filter — skipping")
        return

    gdf, poly = _load_polygon(fid)
    with rasterio.open(paths[0]) as src:
        poly_t = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(src.crs).iloc[0]
        poly_pix = [~src.transform * (x, y) for x, y in zip(*poly_t.exterior.xy)]

    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, paths):
        with rasterio.open(p) as src:
            rgb = src.read([4, 3, 2]).transpose(1, 2, 0)
        date = _by_s2_date(p)
        win = _by_win(p)
        ax.imshow(rgb / vmax, vmin=0, vmax=1)
        ax.add_patch(MplPolygon(poly_pix, fill=False, edgecolor="red", lw=1.2))
        ax.set_title(f"{date[:4]}-{date[4:6]}-{date[6:]}\n({win})",
                     fontsize=9, color=WIN_COLORS[win])
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"S2 L2A — fid {fid}, tile {tile}  ({n} dates) — {_VERSION}", y=1.0)
    fig.tight_layout(); plt.show()


def _vh_time_series(fid: int, tile: int, tok_r: int, tok_c: int, allowed_s1):
    paths = sorted(
        glob.glob(f"{TILES_ROOT}/s1_grd/fid_{fid}/*/*/tile_{tile}.tif"),
        key=_by_s1_date,
    )
    paths = _filter_s1(paths, allowed_s1)
    if not paths:
        print("[vh] no S1 tiles after filter — skipping")
        return

    vh_values, dates, wins = [], [], []
    for p in paths:
        with rasterio.open(p) as src:
            vh = src.read(1)[tok_r * 8:(tok_r + 1) * 8, tok_c * 8:(tok_c + 1) * 8].astype("float32")
        vh_values.append(float(vh.mean()))
        dates.append(_by_s1_date(p))
        wins.append(_by_win(p))

    dates = pd.to_datetime(dates)
    colors = [WIN_COLORS[w] for w in wins]

    plt.figure(figsize=(8, 3))
    plt.plot(dates, vh_values, "-", color="lightgray", zorder=1)
    plt.scatter(dates, vh_values, c=colors, s=40, zorder=2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WIN_COLORS[w], label=w, markersize=8)
        for w in ("bef", "evt", "aft") if w in wins
    ]
    _add_event_line(plt.gca(), _event_date(fid))
    plt.legend(handles=handles, loc="best")
    plt.title(f"VH — fid {fid}, tile {tile}, token ({tok_r},{tok_c}) — {_VERSION}")
    plt.ylabel("VH"); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _plot_series(ax, dates, values, colors, title, line_color="gray", label=None):
    """Time series on a given ax: background line + points colored by window."""
    ax.plot(dates, values, "-", color=line_color, alpha=0.6, zorder=1, label=label)
    ax.scatter(dates, values, c=colors, s=55, edgecolors="black", lw=0.5, zorder=3)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)


def compare_versions_mvd(fid: int, tok_r: int, tok_c: int, tile: int = 0,
                         modality: str = "optical", versions=_VERSIONS,
                         max_gap_days: int = 7, num_components: int = 3):
    """One figure: MVD + PCA (PC1, PC1-PC3) per cloud-filter version for a single
    (fid, tile, token, modality). PCA is fit ONCE on the common-S2-date base image
    (same convention as _pca_first_image), so PC axes are comparable across versions.
    """
    # make sure embeddings exist (resumable)
    _create_embeddings(fid, CSV_TEMPLATE.format(version="v3"), max_gap_days)

    subdir = SUBDIRS[modality]
    paths_all = sorted(glob.glob(f"{EMB_ROOT}/{subdir}/fid_{fid}/*/*/tile_{tile}.npy"),
                       key=_by_date_any)
    if not paths_all:
        print(f"[{modality}] no embeddings on disk — nothing to plot")
        return

    # COMMON BASE (same anchor as _pca_first_image): PCA fit once on the common-S2 base.
    base = _fit_common_base_pca(fid, tile, modality, num_components)
    if base is None:
        print(f"[{modality}] no common-S2 base image — skipping")
        return
    pca, _, _ = base
    evr = pca.explained_variance_ratio_

    # most-variable dim for this token, fixed across versions (from the full on-disk set)
    cube_all = np.stack([np.load(p) for p in paths_all])
    top = int(cube_all[:, tok_r, tok_c, :].std(axis=0).argmax())

    nver = len(versions)
    fig, axes = plt.subplots(3, nver, figsize=(6 * nver, 12), squeeze=False)
    pc_colors = ["tab:purple", "tab:orange", "tab:cyan"]
    evt = _event_date(fid)

    for j, version in enumerate(versions):
        ax_mvd, ax_pc1, ax_pc13 = axes[0, j], axes[1, j], axes[2, j]
        allowed = _allowed_ids_for_fid(CSV_TEMPLATE.format(version=version), fid)
        paths = _paths_for(fid, tile, modality, allowed["s2"], allowed["s1"])
        if not paths:
            for ax in (ax_mvd, ax_pc1, ax_pc13):
                ax.set_title(f"{version}: no data"); ax.axis("off")
            continue

        cube = np.stack([np.load(p) for p in paths])   # filtered cube
        dates = pd.to_datetime([_by_date_any(p) for p in paths])
        colors = [WIN_COLORS[_by_win(p)] for p in paths]
        token = cube[:, tok_r, tok_c, :]               # (T, D)
        scores = pca.transform(token)                  # project onto common base

        _plot_series(ax_mvd, dates, token[:, top], colors,
                     f"{version} — dim {top} (n={len(paths)})")
        _plot_series(ax_pc1, dates, scores[:, 0], colors,
                     f"{version} — PC1 (var {evr[0]:.1%})")
        for k in range(num_components):
            _plot_series(ax_pc13, dates, scores[:, k], colors,
                         f"{version} — PC1-PC3", line_color=pc_colors[k],
                         label=f"PC{k+1} ({evr[k]:.1%})")
        for ax in (ax_mvd, ax_pc1, ax_pc13):
            _add_event_line(ax, evt)
        ax_pc13.legend(title="component", fontsize=8, loc="upper right")

    fig.legend(handles=[Patch(facecolor=c, label=w) for w, c in WIN_COLORS.items()],
               title="window", loc="upper left")
    fig.suptitle(f"MVD + PCA (token {tok_r},{tok_c}, common S2 base) — "
                 f"fid {fid}, tile {tile}, {modality}", y=1.0)
    plt.tight_layout(); plt.show()


def compare_mvd(fid: int, tok_r: int, tok_c: int, tile: int = 0,
                modalities=("optical", "sar", "joint")):
    """Run compare_versions_mvd for several modalities (one figure each)."""
    for m in modalities:
        compare_versions_mvd(fid, tok_r=tok_r, tok_c=tok_c, tile=tile, modality=m)

def complete_analysis(fid: int, tile: int, seed: int = 42, version: str = "v3",
        tok_r: int | None = None, tok_c: int | None = None,
        max_gap_days: int = 7) -> dict:
    """Run all notebook steps for (fid, tile) at a token, filtered by `version`."""
    csv_path = CSV_TEMPLATE.format(version=version)
    global _VERSION
    _VERSION = version
    print(f"[run] fid={fid} tile={tile} seed={seed} version={version} csv={csv_path}")

    _create_embeddings(fid, csv_path, max_gap_days)

    allowed = _allowed_ids_for_fid(csv_path, fid)
    allowed_s2, allowed_s1 = allowed["s2"], allowed["s1"]
    print(f"[run] allowed S2 ids: {len(allowed_s2)}  S1 ids: {len(allowed_s1)}")

    if tok_r is None or tok_c is None:
        tok_r, tok_c, pix_row, pix_col, rgb_tile, poly_pix = _pick_random_token(fid, tile, seed)
        print(f"[run] random pixel ({pix_row},{pix_col}) -> token ({tok_r},{tok_c})")
        _show_tile_with_pixel(fid, tile, pix_row, pix_col, rgb_tile, poly_pix)
    else:
        print(f"[run] using explicit token ({tok_r},{tok_c})")
        _show_explicit_token(fid, tile, tok_r, tok_c)

    for modality in ("optical", "sar", "joint"):
        _cosine_sim_plot(fid, tile, tok_r, tok_c, modality, allowed_s2, allowed_s1)

    _ndvi_profile(fid, tile, tok_r, tok_c, allowed_s2)

    for modality in ("optical", "sar", "joint"):
        _most_variable_dim(fid, tile, modality, allowed_s2, allowed_s1,
                           tok_r=tok_r, tok_c=tok_c)
        _pca_first_image(fid, tile, tok_r, tok_c, modality, allowed_s2, allowed_s1)

    _optical_rgb_grid(fid, tile, allowed_s2)
    _vh_time_series(fid, tile, tok_r, tok_c, allowed_s1)

    return {"fid": fid, "tile": tile, "tok_r": tok_r, "tok_c": tok_c,
            "version": version, "csv_path": csv_path,
            "n_allowed_s2": len(allowed_s2), "n_allowed_s1": len(allowed_s1)}

def _s2_tile_for_joint(p, fid, tile):
      return f"{TILES_ROOT}/s2_l2a/fid_{fid}/{_by_win(p)}/{_image_id_of(p).split('__')[0]}/tile_{tile}.tif"

#nc = number of PCA components to plot
#vmax = max value for RGB normalization of S2 tile
def joint_pca_with_thumbs(fid, tile, tok_r, tok_c, version="v3", vmax=3000, nc=3):
      """Token location + JOINT PCA PC1-PC3 with the S2 RGB tile under each point."""
      import matplotlib.gridspec as gridspec
      global _VERSION
      _VERSION = version

      # 0) token location on the tile (separate figure)
      _show_explicit_token(fid, tile, tok_r, tok_c)

      allowed = _allowed_ids_for_fid(CSV_TEMPLATE.format(version=version), fid)
      paths = _paths_for(fid, tile, "joint", allowed["s2"], allowed["s1"])
      base = _fit_common_base_pca(fid, tile, "joint", nc)
      if not paths or base is None:
          print("[joint] nothing to plot"); return
      pca, evr = base[0], base[0].explained_variance_ratio_

      cube = np.stack([np.load(p) for p in paths])
      scores = pca.transform(cube[:, tok_r, tok_c, :])         # (T, nc)
      T = len(paths)
      dates = [_by_date_any(p) for p in paths]
      wins = [_by_win(p) for p in paths]
      colors = [WIN_COLORS[w] for w in wins]

      fig = plt.figure(figsize=(max(10, 1.4 * T), 7.5))
      gs = gridspec.GridSpec(2, T, height_ratios=[3, 1], hspace=0.35, wspace=0.1)

      # top: PC1-PC3 lines (dates go on the thumbnails, not here)
      ax = fig.add_subplot(gs[0, :])
      for k in range(nc):
          ax.plot(range(T), scores[:, k], "-", alpha=0.7, label=f"PC{k+1} ({evr[k]:.1%})")
          ax.scatter(range(T), scores[:, k], c=colors, s=70, edgecolors="black", lw=0.5, zorder=3)

      # event date (VIEW_DATE) as a dashed line, interpolated to index position
      evt = _event_date(fid)
      if evt is not None:
          dts = pd.to_datetime([d[:8] for d in dates])
          pos = float(np.interp(evt.value, [t.value for t in dts], range(T)))
          ax.axvline(pos, color="black", ls="--", lw=1.5, zorder=0, label="event")

      ax.set_xlim(-0.5, T - 0.5); ax.set_xticks(range(T)); ax.set_xticklabels([])
      ax.set_ylabel("PC score"); ax.grid(alpha=0.3); ax.legend(title="component", fontsize=8)
      ax.set_title(f"JOINT PCA PC1-PC3 — fid {fid}, tile {tile}, token ({tok_r},{tok_c}) — {version}")

      # bottom: one S2 RGB thumbnail per point, date as title colored by window
      for i, p in enumerate(paths):
          axi = fig.add_subplot(gs[1, i])
          with rasterio.open(_s2_tile_for_joint(p, fid, tile)) as src:
              axi.imshow(src.read([4, 3, 2]).transpose(1, 2, 0) / vmax, vmin=0, vmax=1)
          d = dates[i]
          axi.set_title(f"{d[:4]}-{d[4:6]}-{d[6:8]}\n({wins[i]})", fontsize=7, color=colors[i])
          axi.axis("off")
      plt.show()


