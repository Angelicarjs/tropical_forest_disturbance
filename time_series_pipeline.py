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

from make_embeddings import (
    embed_image,
    embed_image_joint,
    iter_joint_pairs_from_csv,
    load_model,
    load_or_compute_stats,
)

TILES_ROOT = "/share/castor/home/e2406749/thesis_tiles_120px"
EMB_ROOT = "embeddings"
SHP_PATH = "data_shp/label_polygons.shp"
CSV_TEMPLATE = "data_csv/{version}_images_s2_s1.csv"
_VERSION = "v3"  # current cloud-filter version; set by run(), shown in plot titles

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
    return re.search(r"(\d{8})T", p).group(1)


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
    plt.legend(handles=handles, loc="best")
    plt.title(f"NDVI — fid {fid}, tile {tile}, token ({tok_r},{tok_c}) — {_VERSION}")
    plt.ylabel("NDVI"); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def _pca_first_image(fid: int, tile: int, modality: str, allowed_s2, allowed_s1,
                     num_components: int = 3):
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

    # COMMON BASE: fit PCA on the first image of the FULL set on disk
    # (version-independent, so PC1 is comparable across v1/v2/v3)
    subdir = SUBDIRS[modality]
    paths_all = sorted(glob.glob(f"{EMB_ROOT}/{subdir}/fid_{fid}/*/*/tile_{tile}.npy"),
                        key=_by_date_any)
    cube_all = np.stack([np.load(p) for p in paths_all])
    

    # make PCA on the first image (T=0) and project all images onto the first 3 components
    pca = PCA(n_components=num_components)
    pca.fit(cube_all[0].reshape(R * C, D)) #flatten rows and columns in one dimension (N_patches, 768) and use the cube of all the images

    #temporal reduction by doing the mean of each time step (T) over the spatial dimensions (R, C) to get a (T, 768) matrix
    tile_means = cube.mean(axis=(1, 2))           # (T, 768)
    
    #project the tile means onto the PCA components
    scores = pca.transform(tile_means)               # (T, 3)
    evr = pca.explained_variance_ratio_
    print(f"[{modality}] PCA on first image ({R*C} tokens) — EVR={evr[:num_components].round(3).tolist()}") 
    
    #fusion PC1, PC2, PC3 into a single value for visualization
    mn, mx = scores.min(axis=0), scores.max(axis=0)
    rgb = (scores - mn) / (mx - mn) #normalize to [0,1]

    dates = pd.to_datetime([_by_date_any(p) for p in paths])
    wins = [_by_win(p) for p in paths]
    labels = [f"{d.strftime('%Y-%m-%d')} ({w})" for d, w in zip(dates, wins)]
    colors = [WIN_COLORS[w] for w in wins]

    #plot 1: RGB image of the PCA scores over time
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(rgb[None,:,:], aspect="auto") #auto is to stretch the image to fill the axes
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"PCA RGB (PC1=R, PC2=G, PC3=B; var: {pca.explained_variance_ratio_.sum():.1%}) — "
                f"fid {fid}, tile {tile}, {modality} — {_VERSION}")
    
    # plot 2: PC1 only (identical style to most-variable-dim)
    plt.figure(figsize=(10, 4))
    plt.plot(dates, scores[:, 0], "-", color="gray", alpha=0.5)
    plt.scatter(dates, scores[:, 0], c=colors, s=70,
                edgecolors="black", linewidths=0.5, zorder=3)
    plt.title(f"PCA PC1 (var: {evr[0]:.1%}) — fid {fid}, tile {tile}, {modality} — {_VERSION}")
    plt.ylabel("PC1 score")
    plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor=c, label=w) for w, c in WIN_COLORS.items() if w in wins],
        title="window",
    )
    plt.tight_layout(); plt.show()

    # plot 3: PC1, PC2, PC3 as three lines on one plot
    pc_colors = ["tab:purple", "tab:orange", "tab:cyan"]   # one per component

    plt.figure(figsize=(10, 4))
    for k in range(scores.shape[1]):
        plt.plot(dates, scores[:, k], "-", color=pc_colors[k], alpha=0.7, zorder=1)
        plt.scatter(dates, scores[:, k], c=colors, s=70,
                    edgecolors="black", linewidths=0.5, zorder=3)

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
                       top_k: int = 5):
    paths = _paths_for(fid, tile, modality, allowed_s2, allowed_s1)
    if not paths:
        print(f"[{modality}] no embeddings after filter — skipping most-variable-dim")
        return

    cube = np.stack([np.load(p) for p in paths])
    tile_means = cube.mean(axis=(1, 2))           # (T, 768)
    dim_scores = tile_means.std(axis=0)           # (768,)
    sorted_dims = np.argsort(dim_scores)[::-1]
    top_dim = sorted_dims[0]
    print(f"[{modality}] cube shape: {cube.shape}")
    print(f"[{modality}] top dim: {top_dim} (std={dim_scores[top_dim]:.4f})")
    print(f"[{modality}] top {top_k} dims: {sorted_dims[:top_k].tolist()}")
    print(f"[{modality}] top {top_k} scores: {dim_scores[sorted_dims[:top_k]].round(4).tolist()}")

    dates = pd.to_datetime([_by_date_any(p) for p in paths])
    wins = [_by_win(p) for p in paths]
    colors = [WIN_COLORS[w] for w in wins]

    plt.figure(figsize=(10, 4))
    plt.plot(dates, tile_means[:, top_dim], "-", color="gray", alpha=0.5)
    plt.scatter(dates, tile_means[:, top_dim], c=colors, s=70,
                edgecolors="black", linewidths=0.5, zorder=3)
    plt.title(f"Dim {top_dim} (std={dim_scores[top_dim]:.3f}) — fid {fid}, tile {tile}, {modality} — {_VERSION}")
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
    plt.legend(handles=handles, loc="best")
    plt.title(f"VH — fid {fid}, tile {tile}, token ({tok_r},{tok_c}) — {_VERSION}")
    plt.ylabel("VH"); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def run(fid: int, tile: int, seed: int = 42, version: str = "v3",
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
        _most_variable_dim(fid, tile, modality, allowed_s2, allowed_s1)
        _pca_first_image(fid, tile, modality, allowed_s2, allowed_s1)

    _optical_rgb_grid(fid, tile, allowed_s2)
    _vh_time_series(fid, tile, tok_r, tok_c, allowed_s1)

    return {"fid": fid, "tile": tile, "tok_r": tok_r, "tok_c": tok_c,
            "version": version, "csv_path": csv_path,
            "n_allowed_s2": len(allowed_s2), "n_allowed_s1": len(allowed_s1)}
