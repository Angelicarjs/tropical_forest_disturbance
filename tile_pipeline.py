#!/usr/bin/env python3
"""
Pipeline to download Sentinel-1 and Sentinel-2 tiles from Google Earth Engine
for multiple foundation models (CROMA, TerraFM, Clay).

Downloads tiles at configurable size (default 120x120 @ 10m). Re-run the
pipeline with a different --tile-size to produce datasets sized for each
foundation model (e.g. 224 for TerraFM, 256 for Clay).

Products downloaded per image ID:
  - S2 L2A  (COPERNICUS/S2_SR_HARMONIZED)  — all models
  - S1 GRD  (COPERNICUS/S1_GRD)            — CROMA

Output structure (output dir is auto-suffixed with _{N}px):
  tiles_120px/
  ├── s2_l2a/
  │   └── fid_{fid}/
  │       ├── evt/{image_id}/tile_{n}.tif   (12 bands @ 10m, NxN)
  │       ├── bef/{image_id}/tile_{n}.tif
  │       └── aft/{image_id}/tile_{n}.tif
  └── s1_grd/ ...

Deduplication:
  Merges v1, v2, v3 CSVs and deduplicates (fid, image_id, window, sensor) tuples
  so each image is downloaded only once even if it appears in multiple filter versions.

  Border-straddle granules: a polygon crossing an MGRS boundary produces one S2
  image id per granule for the same date (e.g. ..._T20LMR and ..._T20LNR). These
  are grouped by acquisition (datatake) and, per tile, only the granule with the
  least nodata is kept. A tile is discarded (never written) when even the best
  granule is more than --max-empty nodata (default 0.5), which drops the 71% /
  100%-empty border tiles outright.

Usage:
  python tile_pipeline.py                          # Process all FIDs at 120x120 (sequential)
  python tile_pipeline.py --tile-size 224          # Download at 224x224 (TerraFM native)
  python tile_pipeline.py --tile-size 256          # Download at 256x256 (Clay native)
  python tile_pipeline.py --sample-pct 10          # 120x120, 10% of FIDs (default size)
  python tile_pipeline.py --workers 6              # 6 parallel workers
  python tile_pipeline.py --workers 6 --resume     # Resume parallel run
  python tile_pipeline.py --fids 193               # Process specific FIDs
  python tile_pipeline.py --fids 193 --first-only  # One image per category (test)
  python tile_pipeline.py --dry-run                # Preview download counts
  python tile_pipeline.py --products s2_l2a s1_grd # Only specific products
"""

import ee
import csv
import json
import os
import math
import time
import argparse
import logging
import multiprocessing as mp
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from rasterio.io import MemoryFile

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GEE_PROJECT = 'zinc-wares-316319'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, 'data_csv')

if _SCRIPT_DIR.startswith('/share/castor'):   # cluster IRISA
    _DEFAULT_CSV_DIR = _DATA_DIR
    _DEFAULT_OUTPUT = os.path.expanduser('~/thesis_tiles') 
else:                                          # Mac
    _DEFAULT_OUTPUT = os.path.expanduser('~/Desktop/Master/thesis/tiles_120px')

CSV_FILES = ['v1_images_s2_s1.csv', 'v2_images_s2_s1.csv', 'v3_images_s2_s1.csv']

CRS_CODE = 'EPSG:3857'
DEFAULT_TILE_SIZE_PX = 120  # Used as default for --tile-size; actual size is threaded through at runtime

# GEE collection IDs per product
COLLECTIONS = {
    's2_l2a': 'COPERNICUS/S2_SR_HARMONIZED',
    's1_grd': 'COPERNICUS/S1_GRD',
}

# Bands per product (download all bands relevant to any model)
PRODUCT_BANDS = {
    's2_l2a': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
    's1_grd': ['VV', 'VH'],
}

ALL_PRODUCTS = list(COLLECTIONS.keys())

# Which CSV columns map to which (sensor, window)
IMAGE_ID_COLUMNS = {
    ('s2', 'evt'): 'evtIdsS2',
    ('s2', 'bef'): 'befIdsS2',
    ('s2', 'aft'): 'aftIdsS2',
    ('s1', 'evt'): 'evtIdsS1',
    ('s1', 'bef'): 'befIdsS1',
    ('s1', 'aft'): 'aftIdsS1',
}

# Rate limiting
MAX_RETRIES = 5
BASE_DELAY = 2
REQUEST_DELAY = 0.3

# Discard a tile if more than this fraction of its pixels are nodata (all-band-zero)
DEFAULT_MAX_EMPTY = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

_LOG_PATH = os.path.join(_SCRIPT_DIR, 'tile_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_PATH),
    ]
)
logger = logging.getLogger(__name__)

# GDAL emits a cosmetic TIFF-tag warning for GEE's multi-band GeoTIFFs
# ("ExtraSamples doesn't match SamplesPerPixel"); the pixels read fine, so silence it.
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

transformer_4326_to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


def project_to_3857(geom_4326):
    return shapely_transform(transformer_4326_to_3857.transform, geom_4326)


def create_tile_grid(polygon_3857, tile_size_m):
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


def group_s2_by_datatake(image_ids):
    """Group S2 image ids that share an acquisition (datatake) but differ only
    in the MGRS granule token, e.g.
        20230604T142721_20230604T142716_T20LMR
        20230604T142721_20230604T142716_T20LNR
    -> both map to '20230604T142721_20230604T142716'. These are the
    border-straddle duplicates: same date, neighbouring granules. Distinct
    dates fall into distinct groups, so the time series is preserved.

    Returns {datatake_id: [granule_image_id, ...]}.
    """
    groups = {}
    for img_id in image_ids:
        head, _, tail = img_id.rpartition('_')
        # An MGRS granule token looks like 'T20LMR' (T + 5 alphanumerics)
        if head and tail.startswith('T') and len(tail) == 6:
            acq = head
        else:
            acq = img_id          # no recognisable granule token; treat as unique
        groups.setdefault(acq, []).append(img_id)
    return groups


def group_s1_by_datatake(image_ids):
    """Group S1 GRD ids that belong to the same acquisition (datatake) but are
    split into consecutive along-track slices, e.g.
        S1A_IW_GRDH_1SDV_20230322T094910_20230322T094935_047755_05BCAD_F524
        S1A_IW_GRDH_1SDV_20230322T094935_20230322T095000_047755_05BCAD_A5E4
    -> both map to datatake '05BCAD' (the second-to-last id field, shared by every
    slice of one pass). A polygon straddling the slice seam is covered by the union
    of the slices, so grouping lets fetch_tile pick the least-empty slice per tile
    instead of dropping the half each slice misses. Distinct datatakes (hence
    distinct dates) stay in distinct groups, so the time series is preserved.

    Returns {datatake_id: [granule_image_id, ...]}.
    """
    groups = {}
    for img_id in image_ids:
        parts = img_id.split('_')
        # S1 GRD id: <mission>_<mode>_<prod>_<pol>_<start>_<stop>_<orbit>_<datatake>_<unique>
        if len(parts) >= 9:
            acq = parts[-2]
        else:
            acq = img_id          # unrecognised format; treat as unique
        groups.setdefault(acq, []).append(img_id)
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# GEE download
# ──────────────────────────────────────────────────────────────────────────────

def fetch_tile(image, bands, tile_bounds_3857, tile_size_px):
    """Fetch one NxN tile at 10m from GEE without writing it.

    Returns (status, geotiff_bytes, empty_frac):
      - ('ok', bytes, frac)    successful fetch; frac = fraction of all-band-zero pixels
      - ('empty', None, 1.0)   GEE reported no valid pixels for this region
      - ('error', None, 1.0)   all retries failed

    The caller decides whether to keep the bytes: among the granules of one
    acquisition it keeps the lowest frac, and only if it is below the empty
    threshold.
    """
    minx, miny, maxx, maxy = tile_bounds_3857

    request = {
        'expression': image.select(bands),
        'fileFormat': 'GEO_TIFF',
        'grid': {
            'dimensions': {'width': tile_size_px, 'height': tile_size_px},
            'affineTransform': {
                'scaleX': 10, 'shearX': 0, 'translateX': minx,
                'shearY': 0, 'scaleY': -10, 'translateY': maxy,
            },
            'crsCode': CRS_CODE,
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            result = ee.data.computePixels(request)
            # Decode in memory to measure how much of the tile is nodata. A pixel
            # counts as empty only when ALL bands are 0 (the value GEE writes for
            # pixels outside the image/granule footprint).
            with MemoryFile(result) as mem, mem.open() as ds:
                arr = ds.read()
            empty_frac = float((arr == 0).all(axis=0).mean())
            time.sleep(REQUEST_DELAY)
            return 'ok', result, empty_frac
        except ee.ee_exception.EEException as e:
            error_msg = str(e)
            if 'No valid pixels' in error_msg or 'empty' in error_msg.lower():
                return 'empty', None, 1.0
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"GEE error (attempt {attempt+1}/{MAX_RETRIES}): {error_msg}")
            time.sleep(delay)
        except Exception as e:
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"Error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(delay)

    logger.error(f"Failed after {MAX_RETRIES} attempts")
    return 'error', None, 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ──────────────────────────────────────────────────────────────────────────────

def load_progress(output_base):
    progress_file = os.path.join(output_base, '.progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_progress(completed, output_base):
    progress_file = os.path.join(output_base, '.progress.json')
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(list(completed), f)


# ──────────────────────────────────────────────────────────────────────────────
# CSV parsing with deduplication across v1/v2/v3
# ──────────────────────────────────────────────────────────────────────────────

def parse_and_merge_csvs(csv_dir, first_only=False):
    """
    Parse v1, v2, v3 CSVs and merge them. For each FID, collect the union
    of all unique image IDs across filter versions to avoid re-downloading.

    Returns:
        list of records: [{'fid': str, 'geometry': dict, 'images': {
            ('s2', 'evt'): [id1, id2, ...],
            ('s1', 'bef'): [id1, ...],
            ...
        }, 'versions': set}]
    """
    # fid -> merged record
    merged = {}

    for csv_name in CSV_FILES:
        csv_path = os.path.join(csv_dir, csv_name)
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found, skipping: {csv_path}")
            continue

        version = csv_name.split('_')[0]  # 'v1', 'v2', 'v3'
        count = 0

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = row['fid'].strip()
                if not fid:
                    continue

                if fid not in merged:
                    geo = json.loads(row['.geo'])
                    merged[fid] = {
                        'fid': fid,
                        'geometry': geo,
                        'images': {},
                        'versions': set(),
                    }

                merged[fid]['versions'].add(version)

                for (sensor, window), col in IMAGE_ID_COLUMNS.items():
                    raw = row.get(col, '').strip()
                    if not raw:
                        continue
                    ids = [img_id.strip() for img_id in raw.split(',') if img_id.strip()]
                    if not ids:
                        continue

                    key = (sensor, window)
                    if key not in merged[fid]['images']:
                        merged[fid]['images'][key] = set()
                    merged[fid]['images'][key].update(ids)

                count += 1

        logger.info(f"Parsed {csv_name}: {count} rows")

    # Convert sets to sorted lists and apply first_only
    records = []
    total_unique_pairs = 0
    for fid, data in merged.items():
        for key in data['images']:
            id_list = sorted(data['images'][key])
            if first_only:
                id_list = id_list[:1]
            data['images'][key] = id_list
            total_unique_pairs += len(id_list)
        if data['images']:
            records.append(data)

    logger.info(f"Merged: {len(records)} FIDs, {total_unique_pairs} unique (fid, image, window) pairs")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Processing logic
# ──────────────────────────────────────────────────────────────────────────────

def process_fid(record, completed, output_base, products, tile_size_px, max_empty):
    """Download all tiles for a single FID across all requested products."""
    fid = record['fid']
    tile_size_m = tile_size_px * 10
    geom_4326 = shape(record['geometry'])
    geom_3857 = project_to_3857(geom_4326)
    tiles = create_tile_grid(geom_3857, tile_size_m)

    if not tiles:
        logger.warning(f"FID {fid}: no intersecting tiles")
        return 0

    total_downloads = 0

    for (sensor, window), image_ids in record['images'].items():
        # Determine which products to download for this sensor
        if sensor == 's2':
            sensor_products = [p for p in products if p.startswith('s2_')]
        else:
            sensor_products = [p for p in products if p.startswith('s1_')]

        if not sensor_products:
            continue

        # Group granules of the SAME acquisition. For S2 a polygon straddling an
        # MGRS boundary yields one image id per granule (..._T20LMR, ..._T20LNR)
        # for the same date -> duplicates of each other.
        if sensor == 's2':
            acq_groups = group_s2_by_datatake(image_ids)
        else:
            # S1 GRD is sliced along-track; group the slices of one datatake
            acq_groups = group_s1_by_datatake(image_ids)

        for product in sensor_products:
            collection_id = COLLECTIONS[product]
            bands = PRODUCT_BANDS[product]

            for acq_id, granule_ids in acq_groups.items():
                safe_acq = acq_id.replace('/', '_')

                for tile_idx, tile in enumerate(tiles):
                    # One completed key per (acquisition, tile), independent of
                    # which granule ends up winning.
                    tile_key = f"{product}|{fid}|{window}|{safe_acq}|tile_{tile_idx}"
                    if tile_key in completed:
                        continue

                    # Resume backstop: already written under any granule folder.
                    if any(os.path.exists(os.path.join(
                            output_base, product, f'fid_{fid}', window,
                            gid.replace('/', '_'), f'tile_{tile_idx}.tif'))
                           for gid in granule_ids):
                        completed.add(tile_key)
                        continue

                    # Fetch the tile from every granule of this acquisition and
                    # keep the one with the least nodata. Discard the tile if even
                    # the best is more than `max_empty` nodata (border gap / no
                    # coverage) -> this drops the 71% / 100%-empty tiles.
                    best = None          # (empty_frac, safe_granule_id, geotiff_bytes)
                    any_error = False
                    for granule_id in granule_ids:
                        try:
                            image = ee.Image(collection_id + '/' + granule_id)
                        except Exception as e:
                            logger.error(f"FID {fid}: failed to create image "
                                         f"{collection_id}/{granule_id}: {e}")
                            any_error = True
                            continue

                        status, result, frac = fetch_tile(
                            image, bands, tile.bounds, tile_size_px)
                        if status == 'ok' and frac <= max_empty:
                            if best is None or frac < best[0]:
                                best = (frac, granule_id.replace('/', '_'), result)
                            if frac <= 0.0:
                                break   # fully covered: cannot do better, skip the rest
                        elif status == 'error':
                            any_error = True

                    if best is not None:
                        _, safe_id, result = best
                        path = os.path.join(
                            output_base, product, f'fid_{fid}',
                            window, safe_id, f'tile_{tile_idx}.tif'
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, 'wb') as f:
                            f.write(result)
                        completed.add(tile_key)
                        total_downloads += 1
                    elif not any_error:
                        # Empty (or too empty) in every granule -> record as done
                        # so a --resume run does not keep retrying it.
                        completed.add(tile_key)
                        logger.warning(
                            f"FID {fid}: dropped empty tile "
                            f"{product} {window} {safe_acq} tile_{tile_idx}")

    return total_downloads


# ──────────────────────────────────────────────────────────────────────────────
# Parallel worker
# ──────────────────────────────────────────────────────────────────────────────

def worker_process(worker_id, records, resume, output_base, products, tile_size_px, max_empty):
    """Independent worker that processes a subset of FIDs."""
    worker_log = os.path.join(_SCRIPT_DIR, f'tile_pipeline_worker_{worker_id}.log')
    fh = logging.FileHandler(worker_log)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Worker {worker_id}: starting with {len(records)} FIDs, products: {products}, tile_size: {tile_size_px}px")

    ee.Initialize(project=GEE_PROJECT)

    progress_file = os.path.join(output_base, f'.progress_worker_{worker_id}.json')
    if resume and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed = set(json.load(f))
        logger.info(f"Worker {worker_id}: resumed with {len(completed)} completed")
    else:
        completed = set()

    def save_worker_progress():
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(list(completed), f)

    for idx, record in enumerate(records):
        logger.info(f"Worker {worker_id}: FID {record['fid']} ({idx+1}/{len(records)})")
        try:
            dl = process_fid(record, completed, output_base, products, tile_size_px, max_empty)
            logger.info(f"Worker {worker_id}: FID {record['fid']} done, {dl} downloads")
            save_worker_progress()
        except Exception as e:
            logger.error(f"Worker {worker_id}: FID {record['fid']} error: {e}", exc_info=True)
            save_worker_progress()

    save_worker_progress()
    logger.info(f"Worker {worker_id}: finished. {len(completed)} total completed.")
    return worker_id, len(completed)


# ──────────────────────────────────────────────────────────────────────────────
# Dry run
# ──────────────────────────────────────────────────────────────────────────────

def dry_run(records, completed, products, tile_size_px):
    total_per_product = {p: 0 for p in products}
    total_already_done = 0
    tile_size_m = tile_size_px * 10

    for record in records:
        geom_3857 = project_to_3857(shape(record['geometry']))
        n_tiles = len(create_tile_grid(geom_3857, tile_size_m))

        for (sensor, window), image_ids in record['images'].items():
            if sensor == 's2':
                sensor_products = [p for p in products if p.startswith('s2_')]
            else:
                sensor_products = [p for p in products if p.startswith('s1_')]

            for product in sensor_products:
                for image_id in image_ids:
                    safe_id = image_id.replace('/', '_')
                    for tile_idx in range(n_tiles):
                        key = f"{product}|{record['fid']}|{window}|{safe_id}|tile_{tile_idx}"
                        if key in completed:
                            total_already_done += 1
                        else:
                            total_per_product[product] += 1

    total = sum(total_per_product.values())

    logger.info(f"DRY RUN: {len(records)} FIDs at {tile_size_px}x{tile_size_px} px")
    logger.info(f"  Products requested: {products}")
    # Bytes per pixel measured on existing 534px tiles: S2 ~11.4, S1 ~14.5
    px = tile_size_px * tile_size_px
    bpp = {'s2_l2a': 11.4, 's1_grd': 14.5}
    for product, count in total_per_product.items():
        mb_per_tile = px * bpp.get(product, 12) / 1e6
        est_gb = count * mb_per_tile / 1000
        logger.info(f"  {product}: ~{count} downloads (~{mb_per_tile:.2f} MB/tile, ~{est_gb:.2f} GB total)")
    logger.info(f"  TOTAL new downloads: ~{total}")
    logger.info(f"  Already completed: {total_already_done}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download S1/S2 tiles (NxN @ 10m) for multiple foundation models'
    )
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Directory containing v1/v2/v3 CSV files (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output base directory (default: auto-detect). '
                             'Will be auto-suffixed with _{tile_size}px.')
    parser.add_argument('--tile-size', type=int, default=DEFAULT_TILE_SIZE_PX,
                        help=f'Tile size in pixels (default: {DEFAULT_TILE_SIZE_PX}). '
                             'Common choices: 120, 224, 256, 534. '
                             'Physical footprint = tile_size * 10m.')
    parser.add_argument('--products', nargs='+', choices=ALL_PRODUCTS, default=ALL_PRODUCTS,
                        help='Which products to download (default: all)')
    parser.add_argument('--fids', nargs='+', type=str,
                        help='Specific FIDs to process')
    parser.add_argument('--fid-list', type=str, default=None,
                        help='File with FIDs to process (one per line). '
                             'Takes precedence over --fids and --sample-pct.')
    parser.add_argument('--first-only', action='store_true',
                        help='Only process the first image per category (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from progress file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded')
    parser.add_argument('--sample-pct', type=float, default=100.0,
                        help='Percentage of FIDs to process (e.g., 10 for 10%%)')
    parser.add_argument('--sample-n', type=int, default=None,
                        help='Exact number of FIDs to sample (e.g., 5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, recommended: 4-8)')
    parser.add_argument('--max-empty', type=float, default=DEFAULT_MAX_EMPTY,
                        help=f'Discard a tile when its fraction of all-nodata pixels '
                             f'exceeds this (default: {DEFAULT_MAX_EMPTY}). Among duplicate '
                             f'granules of the same acquisition, the least-empty tile below '
                             f'this threshold is kept.')
    args = parser.parse_args()

    if args.tile_size <= 0:
        parser.error('--tile-size must be a positive integer')

    tile_size_px = args.tile_size

    csv_dir = args.csv_dir if args.csv_dir else _DEFAULT_CSV_DIR
    base_output = args.output if args.output else _DEFAULT_OUTPUT
    output_base = f"{base_output.rstrip('/')}_{tile_size_px}px"
    products = args.products

    logger.info(f"Parsing CSVs from: {csv_dir}")
    logger.info(f"Tile size: {tile_size_px}x{tile_size_px} px ({tile_size_px * 10} m footprint)")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Products: {products}")
    records = parse_and_merge_csvs(csv_dir, first_only=args.first_only)

    if args.fid_list:
        with open(args.fid_list, 'r') as f:
            fid_set = {line.strip() for line in f if line.strip()}
        records = [r for r in records if r['fid'] in fid_set]
        logger.info(f"Filtered to {len(records)} FIDs from {args.fid_list}")
    elif args.fids:
        fid_set = set(args.fids)
        records = [r for r in records if r['fid'] in fid_set]
        logger.info(f"Filtered to {len(records)} requested FIDs")

    if records and args.sample_pct < 100.0 and not args.fid_list:
        import random
        random.seed(42)
        n = max(1, int(len(records) * args.sample_pct / 100.0))
        records = random.sample(records, n)
        logger.info(f"Sampled {n} FIDs ({args.sample_pct}%)")

    if records and args.sample_n and not args.fid_list:
        import random
        random.seed(42)
        n = min(args.sample_n, len(records))
        records = random.sample(records, n)
        logger.info(f"Sampled {n} FIDs (--sample-n)")

    if not records:
        logger.error("No records to process. Check that CSVs exist in: " + csv_dir)
        return

    os.makedirs(output_base, exist_ok=True)

    if args.dry_run:
        logger.info("Initializing Google Earth Engine...")
        ee.Initialize(project=GEE_PROJECT)
        completed = load_progress(output_base) if args.resume else set()
        dry_run(records, completed, products, tile_size_px)
        return

    # ── Parallel mode ──
    if args.workers > 1:
        n_workers = min(args.workers, len(records))
        logger.info(f"Launching {n_workers} parallel workers for {len(records)} FIDs")

        chunks = [[] for _ in range(n_workers)]
        for i, record in enumerate(records):
            chunks[i % n_workers].append(record)

        for i, chunk in enumerate(chunks):
            logger.info(f"  Worker {i}: {len(chunk)} FIDs")

        with mp.Pool(processes=n_workers) as pool:
            worker_args = [
                (i, chunk, args.resume, output_base, products, tile_size_px, args.max_empty)
                for i, chunk in enumerate(chunks)
            ]
            results = pool.starmap(worker_process, worker_args)

        all_completed = set()
        for i in range(n_workers):
            pf = os.path.join(output_base, f'.progress_worker_{i}.json')
            if os.path.exists(pf):
                with open(pf, 'r') as f:
                    all_completed.update(json.load(f))
        save_progress(all_completed, output_base)

        total = sum(count for _, count in results)
        logger.info(f"All workers finished. {total} total completed downloads.")
        logger.info(f"Output: {output_base}")
        return

    # ── Sequential mode ──
    logger.info("Initializing Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT)

    completed = load_progress(output_base) if args.resume else set()
    if completed:
        logger.info(f"Resuming with {len(completed)} completed downloads")

    total_fids = len(records)
    for idx, record in enumerate(records):
        logger.info(f"Processing FID {record['fid']} ({idx+1}/{total_fids})")
        try:
            dl = process_fid(record, completed, output_base, products, tile_size_px, args.max_empty)
            logger.info(f"FID {record['fid']}: {dl} new downloads")
            save_progress(completed, output_base)
        except Exception as e:
            logger.error(f"FID {record['fid']}: unexpected error: {e}", exc_info=True)
            save_progress(completed, output_base)
            continue

    save_progress(completed, output_base)
    logger.info(f"Pipeline complete. {len(completed)} total downloads.")
    logger.info(f"Output: {output_base}")


if __name__ == '__main__':
    main()
