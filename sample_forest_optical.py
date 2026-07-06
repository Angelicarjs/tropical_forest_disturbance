"""Generate optical (S2) stable-forest embeddings, mirroring the joint forest samples.

Forest is defined purely spatially: a cell is forest when PRODES == 100 (forest)
AND it lies outside the disturbance polygon. The optical embedding values are only
read at the already-selected cells (the selection never looks at the embeddings).

For every optical tile (evt/aft windows) it writes one CSV per fid/window/date/tile
with 768 columns (lyr.1..lyr.768) -- the same format as the joint forest CSVs -- so
that build_pixel_dataset_forest() can read them unchanged. Point FOREST_ROOT at the
output folder and DisturbanceSegDataset at "s2_l2a" to run the optical baseline.

Paths are read from environment variables so the same script works locally and on
the cluster:
    OPTICAL_ROOT  optical .npy tiles     (fid_*/<window>/<date>/tile_*.npy)
    PRODES_ROOT   PRODES masks           (fid_*/prodes_mask_tile_*.tif)
    FOREST_SHP    label polygons shapefile
    OUT_ROOT      output forest CSVs
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from seg_dataset import CLASS_TO_ID

OPTICAL_ROOT = os.environ.get("OPTICAL_ROOT", "embeddings/s2_l2a")
PRODES_ROOT = os.environ.get("PRODES_ROOT", os.path.expanduser("~/Downloads/samples/joint"))
FOREST_SHP = os.environ.get("FOREST_SHP", "data_shp/label_polygons.shp")
OUT_ROOT = os.environ.get("OUT_ROOT", "embeddings/s2_forest")
WINDOWS = ("evt", "aft")
PRODES_FOREST = 100                       # PRODES value for stable forest
SAMPLE_N = int(os.environ.get("FOREST_SAMPLE_N", "10"))   # forest px per tile/date (match joint; 0 = all)
COLS = [f"lyr.{i}" for i in range(1, 769)]


def load_polygons(prodes_crs):
    """fid -> list of disturbance-polygon geometries, reprojected to the PRODES CRS."""
    gdf = gpd.read_file(FOREST_SHP)
    gdf["fid"] = gdf["fid"].astype(int).astype(str)
    gdf = gdf.to_crs(prodes_crs)
    fid_polys = {}
    for _, r in gdf.iterrows():
        if r["CLASSNAME"] in CLASS_TO_ID:
            fid_polys.setdefault(r["fid"], []).append(r.geometry)
    return fid_polys


def main():
    # CRS is read from one PRODES mask; all masks share the same tiling CRS
    first = next(iter(glob.glob(str(Path(PRODES_ROOT) / "fid_*" / "prodes_mask_tile_*.tif"))))
    with rasterio.open(first) as s:
        prodes_crs = s.crs
    fid_polys = load_polygons(prodes_crs)

    rng = np.random.default_rng(0)        # reproducible forest subsampling
    n_csv = n_px = n_skip = 0
    for npy in glob.glob(str(Path(OPTICAL_ROOT) / "fid_*" / "*" / "*" / "tile_*.npy")):
        p = Path(npy)
        window = p.parent.parent.name
        if window not in WINDOWS:
            continue
        fid = p.parent.parent.parent.name.replace("fid_", "")
        tile = p.stem                                       # "tile_N"
        prodes_tif = Path(PRODES_ROOT) / f"fid_{fid}" / f"prodes_mask_{tile}.tif"
        if fid not in fid_polys or not prodes_tif.exists():
            n_skip += 1
            continue

        with rasterio.open(prodes_tif) as s:
            prodes = s.read(1)                              # (15, 15)
            transform = s.transform
        poly = rasterize(fid_polys[fid], out_shape=prodes.shape, transform=transform,
                         fill=0, dtype="uint8", all_touched=False)   # >0 = inside polygon
        forest = (prodes == PRODES_FOREST) & (poly == 0)             # forest AND outside polygon
        idx = np.flatnonzero(forest.reshape(-1))
        if idx.size == 0:
            continue
        if SAMPLE_N and idx.size > SAMPLE_N:                          # match joint: ~10 px per tile/date
            idx = rng.choice(idx, size=SAMPLE_N, replace=False)

        rows = np.load(npy).reshape(-1, 768)[idx]                    # (n_forest, 768)
        out = Path(OUT_ROOT) / f"fid_{fid}" / window / p.parent.name
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=COLS).to_csv(out / f"{tile}.csv", index=False)
        n_csv += 1
        n_px += len(rows)

    print(f"wrote {n_csv} CSVs, {n_px} forest pixels to {OUT_ROOT}  (skipped {n_skip} tiles w/o PRODES/polygon)")


if __name__ == "__main__":
    main()
