#!/usr/bin/env python3
"""
Download Sentinel-1 RTC tiles from Microsoft Planetary Computer (MPC).

Same tiling logic as tile_pipeline.py:
- Polygon projected to EPSG:3857
- Tile grid: tile_size_m = size_px * 10, snapped to multiples of tile_size_m
  from the projection origin; keep tiles that intersect the polygon
- Output written in EPSG:3857 at 10 m, so RTC tiles align pixel-for-pixel
  with the existing GRD / S2 tiles produced by tile_pipeline.py

Each (fid, window, scene) may produce 1+ tiles per size depending on
polygon footprint vs tile size.

The three CSVs are merged and (fid, window, scene_id) tuples are
deduplicated, matching the existing pipeline's convention.

GRD scene id → RTC item id mapping on MPC:
    S1A_..._047405_05B0D3_E8BA  →  S1A_..._047405_05B0D3_rtc
(replace last underscore-delimited segment with "rtc")

Fallback when the derived id is not found: STAC search by
sat:absolute_orbit + acquisition date (parsed from the GRD id itself).

Output structure (matches tile_pipeline.py layout, sibling to existing s1_rtc/):
    {output}_{size}px/
    └── s1_rtc_mpc/
        └── fid_{fid}/
            └── {window}/
                └── {scene_id}/
                    ├── tile_0.tif    (2 bands: VV, VH, EPSG:3857 @ 10 m)
                    └── tile_1.tif    (only if polygon spans more than one grid cell)

Required packages (install in your env):
    pip install planetary-computer pystac-client rasterio shapely pyproj

Default FID filter: data_csv/sample_10pct_stratified.txt (the 10% stratified
sample). Pass --all-fids to disable filtering and process the full dataset.

Usage:
    python s1_rtc_download_mpc.py --dry-run                 # preview counts
    python s1_rtc_download_mpc.py --workers 4               # 10% sample (default)
    python s1_rtc_download_mpc.py --all-fids --workers 4    # full dataset
"""

import argparse
import csv
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
BANDS = ["vv", "vh"]
RESOLUTION_M = 10
CRS_OUT = "EPSG:3857"

WINDOW_COLUMNS = {
    "bef": "befIdsS1",
    "evt": "evtIdsS1",
    "aft": "aftIdsS1",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.path.join(_SCRIPT_DIR, "data_csv")
DEFAULT_FID_LIST = os.path.join(DEFAULT_CSV_DIR, "sample_10pct_stratified.txt")

# Per-size base becomes f"{DEFAULT_OUTPUT_BASE}_{size}px".
DEFAULT_OUTPUT_BASE = "/share/castor/home/e2406749/thesis_tiles"

PRODUCT_NAME = "s1_rtc_mpc"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_SCRIPT_DIR, "s1_rtc_download_mpc.log")),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GRD id parsing
# ──────────────────────────────────────────────────────────────────────────────

def grd_to_rtc_id(grd_id: str) -> str:
    """S1A_..._047405_05B0D3_E8BA → S1A_..._047405_05B0D3_rtc"""
    return grd_id.rsplit("_", 1)[0] + "_rtc"


def parse_grd_id(grd_id: str):
    """Return (start_datetime, absolute_orbit) for STAC fallback search."""
    parts = grd_id.split("_")
    start = datetime.strptime(parts[4], "%Y%m%dT%H%M%S")
    orbit = int(parts[6])
    return start, orbit


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate / grid helpers (same logic as tile_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

_transformer_4326_to_3857 = Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
)


def project_to_3857(geom_4326):
    return shapely_transform(_transformer_4326_to_3857.transform, geom_4326)


def create_tile_grid(polygon_3857, tile_size_m):
    """Grid of tile_size_m × tile_size_m boxes (EPSG:3857) intersecting the polygon."""
    minx, miny, maxx, maxy = polygon_3857.bounds
    grid_minx = math.floor(minx / tile_size_m) * tile_size_m
    grid_miny = math.floor(miny / tile_size_m) * tile_size_m
    grid_maxx = math.ceil(maxx / tile_size_m) * tile_size_m
    grid_maxy = math.ceil(maxy / tile_size_m) * tile_size_m

    tiles = []
    x = grid_minx
    while x < grid_maxx:
        y = grid_miny
        while y < grid_maxy:
            tile = box(x, y, x + tile_size_m, y + tile_size_m)
            if tile.intersects(polygon_3857):
                tiles.append(tile)
            y += tile_size_m
        x += tile_size_m
    return tiles


# ──────────────────────────────────────────────────────────────────────────────
# CSV iteration
# ──────────────────────────────────────────────────────────────────────────────

def load_fid_list(path):
    """Load a set of FIDs (one per line) for filtering. Returns None if path is None."""
    if not path:
        return None
    if not os.path.exists(path):
        logger.warning(f"FID list not found: {path} — processing all FIDs")
        return None
    with open(path) as f:
        fids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(fids)} FIDs from {path}")
    return fids


def iter_rows(csv_dir, csv_files, fid_filter=None):
    """Yield (fid, geometry_geojson, window, scene_id) per UNIQUE (fid, window, scene_id).

    Iterates over all CSVs in order; first occurrence wins. This matches the
    existing pipeline's convention of merging v1/v2/v3 to avoid duplicate downloads.
    """
    seen = set()
    for csv_name in csv_files:
        path = os.path.join(csv_dir, csv_name)
        if not os.path.exists(path):
            logger.warning(f"CSV not found: {path}")
            continue
        version = csv_name.split("_")[0]
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = (row.get("fid") or "").strip()
                if not fid:
                    continue
                if fid_filter is not None and fid not in fid_filter:
                    continue
                geo_raw = row.get(".geo") or ""
                if not geo_raw:
                    continue
                try:
                    geometry = json.loads(geo_raw)
                except Exception as e:
                    logger.warning(f"{version} fid={fid}: bad .geo JSON: {e}")
                    continue
                for window, col in WINDOW_COLUMNS.items():
                    raw = (row.get(col) or "").strip().strip('"')
                    if not raw:
                        continue
                    for scene_id in (s.strip() for s in raw.split(",") if s.strip()):
                        key = (fid, window, scene_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield fid, geometry, window, scene_id


# ──────────────────────────────────────────────────────────────────────────────
# MPC item lookup
# ──────────────────────────────────────────────────────────────────────────────

def open_catalog():
    return pystac_client.Client.open(
        MPC_STAC_URL, modifier=planetary_computer.sign_inplace
    )


def get_rtc_item(catalog, grd_id):
    """Return RTC pystac.Item (signed) for the given GRD scene id, or None."""
    rtc_id = grd_to_rtc_id(grd_id)
    try:
        items = list(catalog.search(collections=[COLLECTION], ids=[rtc_id]).items())
        if items:
            return items[0]
    except Exception as e:
        logger.debug(f"Direct id lookup failed for {rtc_id}: {e}")

    try:
        start, orbit = parse_grd_id(grd_id)
        date = start.date()
        next_date = date + timedelta(days=1)
        items = list(
            catalog.search(
                collections=[COLLECTION],
                datetime=f"{date.isoformat()}/{next_date.isoformat()}",
                query={"sat:absolute_orbit": {"eq": orbit}},
            ).items()
        )
        if items:
            logger.info(f"Fallback search hit for {grd_id}: {items[0].id}")
            return items[0]
    except Exception as e:
        logger.debug(f"Fallback search failed for {grd_id}: {e}")

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tile download
# ──────────────────────────────────────────────────────────────────────────────

def out_path(output_base, size, fid, window, scene_id, tile_idx):
    """{base}_{size}px/s1_rtc_mpc/fid_{fid}/{window}/{scene_id}/tile_{idx}.tif"""
    return os.path.join(
        f"{output_base.rstrip('/')}_{size}px",
        PRODUCT_NAME,
        f"fid_{fid}",
        window,
        scene_id.replace("/", "_"),
        f"tile_{tile_idx}.tif",
    )


def write_one_tile(src_handles, tile_bounds_3857, size_px, path, nodata, dtype):
    """Write a single size_px × size_px tile from open band sources."""
    minx, _miny, _maxx, maxy = tile_bounds_3857
    transform = from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M)

    arrays = []
    for src in src_handles:
        with WarpedVRT(
            src,
            crs=CRS_OUT,
            transform=transform,
            width=size_px,
            height=size_px,
            resampling=Resampling.bilinear,
        ) as vrt:
            arrays.append(vrt.read(1))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=size_px, width=size_px,
        count=len(BANDS),
        dtype=dtype,
        crs=CRS_OUT,
        transform=transform,
        compress="deflate",
        nodata=nodata,
    ) as dst:
        for i, arr in enumerate(arrays):
            dst.write(arr, i + 1)
        dst.descriptions = tuple(b.upper() for b in BANDS)


def process_task(catalog, output_base, sizes, fid, geometry, window, scene_id):
    """For one (fid × window × scene), write all grid tiles for all sizes."""
    polygon_3857 = project_to_3857(shape(geometry))

    # Build list of pending writes: one entry per (size, tile_idx) not on disk
    todo = []  # (size_px, tile_idx, tile_bounds, out_path)
    for size in sizes:
        grid = create_tile_grid(polygon_3857, size * RESOLUTION_M)
        for tile_idx, tile in enumerate(grid):
            p = out_path(output_base, size, fid, window, scene_id, tile_idx)
            if not os.path.exists(p):
                todo.append((size, tile_idx, tile.bounds, p))

    if not todo:
        return 0

    item = get_rtc_item(catalog, scene_id)
    if item is None:
        logger.warning(f"No RTC for {scene_id} (fid={fid} {window})")
        return 0

    # Open both band COGs once for the whole scene
    src_handles = []
    written = 0
    try:
        for band in BANDS:
            src_handles.append(rasterio.open(item.assets[band].href))
        nodata = src_handles[0].nodata
        dtype = src_handles[0].dtypes[0]

        for size_px, tile_idx, bounds, path in todo:
            try:
                write_one_tile(src_handles, bounds, size_px, path, nodata, dtype)
                written += 1
            except Exception as e:
                logger.error(f"Tile write failed {path}: {e}")
    finally:
        for src in src_handles:
            try:
                src.close()
            except Exception:
                pass
    return written


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Download Sentinel-1 RTC tiles from Microsoft Planetary Computer."
    )
    p.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT_BASE,
                   help=f"Output base directory; '_<size>px' is appended per size "
                        f"(default: {DEFAULT_OUTPUT_BASE})")
    p.add_argument("--sizes", type=int, nargs="+", default=[224, 256],
                   help="Tile sizes in pixels (default: 224 256)")
    p.add_argument("--csv-versions", nargs="+", default=["v1", "v2", "v3"],
                   choices=["v1", "v2", "v3"])
    p.add_argument("--fid-list", default=DEFAULT_FID_LIST,
                   help=f"File with FIDs to process, one per line "
                        f"(default: {DEFAULT_FID_LIST}). "
                        f"Pass --fid-list '' or --all-fids to process every FID.")
    p.add_argument("--all-fids", action="store_true",
                   help="Disable FID filtering (overrides --fid-list).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tasks (testing)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download threads (default: 4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Count tasks/tiles without downloading")
    args = p.parse_args()

    csv_files = [f"{v}_images_s2_s1.csv" for v in args.csv_versions]
    fid_filter = None if args.all_fids else load_fid_list(args.fid_list)

    logger.info(f"CSV dir: {args.csv_dir}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Sizes: {args.sizes}")
    logger.info(f"CSVs: {csv_files}")
    logger.info(f"FID filter: {len(fid_filter) if fid_filter else 'disabled (all FIDs)'}")

    tasks = list(iter_rows(args.csv_dir, csv_files, fid_filter))
    logger.info(f"Total scenes (fid × window × scene): {len(tasks)}")

    if args.limit:
        tasks = tasks[: args.limit]
        logger.info(f"Limited to {len(tasks)} scenes")

    if args.dry_run:
        existing = pending = 0
        per_size = {s: 0 for s in args.sizes}
        for fid, geometry, window, scene_id in tasks:
            polygon_3857 = project_to_3857(shape(geometry))
            for size in args.sizes:
                grid = create_tile_grid(polygon_3857, size * RESOLUTION_M)
                per_size[size] += len(grid)
                for tile_idx in range(len(grid)):
                    p = out_path(args.output, size, fid, window, scene_id, tile_idx)
                    if os.path.exists(p):
                        existing += 1
                    else:
                        pending += 1
        for s, c in per_size.items():
            logger.info(f"  {s}px: {c} tiles total ({c / max(1, len(tasks)):.2f} per scene)")
        logger.info(f"DRY RUN: {pending} tiles to download, {existing} already exist")
        return

    catalog = open_catalog()
    completed = failed = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(process_task, catalog, args.output, args.sizes,
                          fid, geometry, window, scene_id)
                for fid, geometry, window, scene_id in tasks
            ]
            for i, fut in enumerate(as_completed(futures)):
                try:
                    completed += fut.result()
                except Exception as e:
                    logger.error(f"Task error: {e}")
                    failed += 1
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(futures)} scenes")
    else:
        for i, (fid, geometry, window, scene_id) in enumerate(tasks):
            try:
                completed += process_task(
                    catalog, args.output, args.sizes,
                    fid, geometry, window, scene_id,
                )
            except Exception as e:
                logger.error(f"Task error: {e}")
                failed += 1
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} scenes")

    logger.info(f"Done. {completed} tiles written, {failed} scene errors.")


if __name__ == "__main__":
    main()
