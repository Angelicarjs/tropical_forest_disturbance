#!/usr/bin/env python3
"""
Download Sentinel-1 RTC tiles from Microsoft Planetary Computer (MPC).

For each row in v1/v2/v3 CSVs and each S1 GRD scene id listed in
befIdsS1 / evtIdsS1 / aftIdsS1, fetches the matching RTC item from MPC
and writes a tile centered on the polygon centroid at one or more
output sizes (default: 224 and 256 px @ 10 m).

The three CSVs are merged and (fid, window, scene_id) tuples are
deduplicated, matching the existing pipeline's convention.

GRD scene id → RTC item id mapping on MPC:
    S1A_..._047405_05B0D3_E8BA  →  S1A_..._047405_05B0D3_rtc
(replace last underscore-delimited segment with "rtc")

Fallback when the derived id is not found: STAC search by
sat:absolute_orbit + acquisition date (parsed from the GRD id itself).

Output structure (sibling to existing s1_rtc/ to avoid touching manual tiles):
    {output}_{size}px/
    └── s1_rtc_mpc/
        └── fid_{fid}/
            └── {window}/
                └── {scene_id}.tif    (2 bands: VV, VH, UTM 10 m)

Required packages (install in your env):
    pip install planetary-computer pystac-client rasterio shapely pyproj numpy

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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import shape

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
BANDS = ["vv", "vh"]
RESOLUTION_M = 10

WINDOW_COLUMNS = {
    "bef": "befIdsS1",
    "evt": "evtIdsS1",
    "aft": "aftIdsS1",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.path.join(_SCRIPT_DIR, "data_csv")
DEFAULT_FID_LIST = os.path.join(DEFAULT_CSV_DIR, "sample_10pct_stratified.txt")

# Cluster vs local auto-detect (matches tile_pipeline.py convention).
# Per-size base becomes f"{DEFAULT_OUTPUT_BASE}_{size}px".
if os.path.exists(os.path.expanduser("~/thesis_scripts")):
    DEFAULT_OUTPUT_BASE = os.path.expanduser("~/thesis_tiles")
else:
    DEFAULT_OUTPUT_BASE = "/Users/angelicamariamorenorojas/Desktop/Master/thesis/tiles"

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
# UTM helper
# ──────────────────────────────────────────────────────────────────────────────

def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


# ──────────────────────────────────────────────────────────────────────────────
# CSV iteration (no dedup, row-by-row)
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

    If fid_filter is a set, only rows whose fid is in the set are yielded.
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

def write_tiles(item, polygon_4326, sizes, out_paths):
    """Read bands once at max size, then center-crop to each requested size."""
    centroid = polygon_4326.centroid
    utm = utm_crs_for_lonlat(centroid.x, centroid.y)
    transformer = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    cx, cy = transformer.transform(centroid.x, centroid.y)

    max_size = max(sizes)
    half = max_size * RESOLUTION_M / 2
    minx, miny = cx - half, cy - half
    maxx, maxy = cx + half, cy + half
    transform_max = from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M)

    band_arrays = []
    nodata_val = None
    dtype = None
    for band in BANDS:
        href = item.assets[band].href
        with rasterio.open(href) as src:
            if nodata_val is None:
                nodata_val = src.nodata
                dtype = src.dtypes[0]
            with WarpedVRT(
                src,
                crs=utm,
                transform=transform_max,
                width=max_size,
                height=max_size,
                resampling=Resampling.bilinear,
            ) as vrt:
                band_arrays.append(vrt.read(1))

    stack = np.stack(band_arrays, axis=0)

    written = {}
    for size in sizes:
        out_path = out_paths[size]
        if size == max_size:
            arr = stack
            t = transform_max
        else:
            offset = (max_size - size) // 2
            arr = stack[:, offset:offset + size, offset:offset + size]
            t = from_origin(
                minx + offset * RESOLUTION_M,
                maxy - offset * RESOLUTION_M,
                RESOLUTION_M,
                RESOLUTION_M,
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=size, width=size,
            count=len(BANDS),
            dtype=dtype,
            crs=utm,
            transform=t,
            compress="deflate",
            nodata=nodata_val,
        ) as dst:
            dst.write(arr)
            dst.descriptions = tuple(b.upper() for b in BANDS)
        written[size] = True

    return written


def out_path(output_base, size, fid, window, scene_id):
    """{output_base}_{size}px/s1_rtc_mpc/fid_{fid}/{window}/{scene_id}.tif"""
    return os.path.join(
        f"{output_base.rstrip('/')}_{size}px",
        PRODUCT_NAME,
        f"fid_{fid}",
        window,
        f"{scene_id}.tif",
    )


def process_task(catalog, output_base, sizes, fid, geometry, window, scene_id):
    """Download tiles for one (fid × window × scene). Returns # written."""
    out_paths = {s: out_path(output_base, s, fid, window, scene_id) for s in sizes}
    pending = {s: p for s, p in out_paths.items() if not os.path.exists(p)}
    if not pending:
        return 0

    item = get_rtc_item(catalog, scene_id)
    if item is None:
        logger.warning(f"No RTC for {scene_id} (fid={fid} {window})")
        return 0

    try:
        polygon = shape(geometry)
        results = write_tiles(item, polygon, list(pending.keys()), pending)
        return sum(1 for v in results.values() if v)
    except Exception as e:
        logger.error(f"Write failed for {scene_id} (fid={fid} {window}): {e}")
        return 0


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
    logger.info(f"Total tasks (row × window × scene): {len(tasks)}")

    if args.limit:
        tasks = tasks[: args.limit]
        logger.info(f"Limited to {len(tasks)} tasks")

    if args.dry_run:
        existing = pending = 0
        for fid, geometry, window, scene_id in tasks:
            for size in args.sizes:
                if os.path.exists(out_path(args.output, size, fid, window, scene_id)):
                    existing += 1
                else:
                    pending += 1
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
                    logger.info(f"Progress: {i + 1}/{len(futures)} tasks")
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
                logger.info(f"Progress: {i + 1}/{len(tasks)} tasks")

    logger.info(f"Done. {completed} tiles written, {failed} task errors.")


if __name__ == "__main__":
    main()
