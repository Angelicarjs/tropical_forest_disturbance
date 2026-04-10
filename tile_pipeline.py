#!/usr/bin/env python3
"""
Pipeline to download Sentinel-1 and Sentinel-2 tiles from Google Earth Engine
for multiple foundation models (CROMA, TerraFM, Clay, SkySense).

Downloads master tiles at 534x534 pixels (5340m @ 10m), the largest input size
needed (TerraFM). Smaller crops for other models are done at embedding time.

Products downloaded per image ID:
  - S2 L2A  (COPERNICUS/S2_SR_HARMONIZED)  — all models
  - S2 L1C  (COPERNICUS/S2_HARMONIZED)     — TerraFM
  - S1 GRD  (COPERNICUS/S1_GRD)            — CROMA, SkySense
  - S1 RTC  (COPERNICUS/S1_RTC)            — TerraFM, Clay

Output structure:
  tiles/
  ├── s2_l2a/
  │   └── fid_{fid}/
  │       ├── evt/{image_id}/tile_{n}.tif   (13 bands @ 10m, 534x534)
  │       ├── bef/{image_id}/tile_{n}.tif
  │       └── aft/{image_id}/tile_{n}.tif
  ├── s2_l1c/
  │   └── fid_{fid}/
  │       ├── evt/{image_id}/tile_{n}.tif   (13 bands @ 10m, 534x534)
  │       ├── bef/{image_id}/tile_{n}.tif
  │       └── aft/{image_id}/tile_{n}.tif
  ├── s1_grd/
  │   └── fid_{fid}/
  │       ├── evt/{image_id}/tile_{n}.tif   (VV+VH @ 10m, 534x534)
  │       ├── bef/{image_id}/tile_{n}.tif
  │       └── aft/{image_id}/tile_{n}.tif
  └── s1_rtc/
      └── fid_{fid}/
          ├── evt/{image_id}/tile_{n}.tif   (VV+VH @ 10m, 534x534)
          ├── bef/{image_id}/tile_{n}.tif
          └── aft/{image_id}/tile_{n}.tif

Deduplication:
  Merges v1, v2, v3 CSVs and deduplicates (fid, image_id, window, sensor) tuples
  so each image is downloaded only once even if it appears in multiple filter versions.

Usage:
  python tile_pipeline.py                          # Process all FIDs (sequential)
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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GEE_PROJECT = 'zinc-wares-316319'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, 'data_csv')

if os.path.exists(os.path.expanduser('~/thesis_scripts')):
    _DEFAULT_CSV_DIR = _DATA_DIR  # data_csv/ next to the script
    _DEFAULT_OUTPUT = os.path.expanduser('~/thesis_tiles')
else:
    _DEFAULT_CSV_DIR = _DATA_DIR
    _DEFAULT_OUTPUT = '/Users/angelicamariamorenorojas/Desktop/Master/thesis/tiles'

CSV_FILES = ['v1_images_s2_s1.csv', 'v2_images_s2_s1.csv', 'v3_images_s2_s1.csv']

CRS_CODE = 'EPSG:3857'
TILE_SIZE_PX = 534
TILE_SIZE_M = TILE_SIZE_PX * 10  # 5340m at 10m resolution

# GEE collection IDs per product
COLLECTIONS = {
    's2_l2a': 'COPERNICUS/S2_SR_HARMONIZED',
    's2_l1c': 'COPERNICUS/S2_HARMONIZED',
    's1_grd': 'COPERNICUS/S1_GRD',
    's1_rtc': 'COPERNICUS/S1_RTC',
}

# Bands per product (download all bands relevant to any model)
PRODUCT_BANDS = {
    's2_l2a': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
    's2_l1c': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
    's1_grd': ['VV', 'VH'],
    's1_rtc': ['VV', 'VH'],
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

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

transformer_4326_to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


def project_to_3857(geom_4326):
    return shapely_transform(transformer_4326_to_3857.transform, geom_4326)


def create_tile_grid(polygon_3857):
    minx, miny, maxx, maxy = polygon_3857.bounds
    grid_minx = math.floor(minx / TILE_SIZE_M) * TILE_SIZE_M
    grid_miny = math.floor(miny / TILE_SIZE_M) * TILE_SIZE_M
    grid_maxx = math.ceil(maxx / TILE_SIZE_M) * TILE_SIZE_M
    grid_maxy = math.ceil(maxy / TILE_SIZE_M) * TILE_SIZE_M

    tiles = []
    x = grid_minx
    while x < grid_maxx:
        y = grid_miny
        while y < grid_maxy:
            tile = box(x, y, x + TILE_SIZE_M, y + TILE_SIZE_M)
            if tile.intersects(polygon_3857):
                tiles.append(tile)
            y += TILE_SIZE_M
        x += TILE_SIZE_M
    return tiles


# ──────────────────────────────────────────────────────────────────────────────
# GEE download
# ──────────────────────────────────────────────────────────────────────────────

def download_tile(image, bands, tile_bounds_3857, output_path):
    """Download a single 534x534 tile at 10m resolution."""
    if os.path.exists(output_path):
        return True

    minx, miny, maxx, maxy = tile_bounds_3857

    request = {
        'expression': image.select(bands),
        'fileFormat': 'GEO_TIFF',
        'grid': {
            'dimensions': {'width': TILE_SIZE_PX, 'height': TILE_SIZE_PX},
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
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(result)
            time.sleep(REQUEST_DELAY)
            return True
        except ee.ee_exception.EEException as e:
            error_msg = str(e)
            if 'No valid pixels' in error_msg or 'empty' in error_msg.lower():
                logger.warning(f"No valid pixels: {output_path}")
                return False
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"GEE error (attempt {attempt+1}/{MAX_RETRIES}): {error_msg}")
            time.sleep(delay)
        except Exception as e:
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"Error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(delay)

    logger.error(f"Failed after {MAX_RETRIES} attempts: {output_path}")
    return False


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

def process_fid(record, completed, output_base, products):
    """Download all tiles for a single FID across all requested products."""
    fid = record['fid']
    geom_4326 = shape(record['geometry'])
    geom_3857 = project_to_3857(geom_4326)
    tiles = create_tile_grid(geom_3857)

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

        for image_id in image_ids:
            safe_id = image_id.replace('/', '_')

            for product in sensor_products:
                collection_id = COLLECTIONS[product]
                bands = PRODUCT_BANDS[product]

                try:
                    image = ee.Image(collection_id + '/' + image_id)
                except Exception as e:
                    logger.error(f"FID {fid}: failed to create image {collection_id}/{image_id}: {e}")
                    continue

                for tile_idx, tile in enumerate(tiles):
                    key = f"{product}|{fid}|{window}|{safe_id}|tile_{tile_idx}"
                    if key in completed:
                        continue

                    path = os.path.join(
                        output_base, product, f'fid_{fid}',
                        window, safe_id, f'tile_{tile_idx}.tif'
                    )

                    if download_tile(image, bands, tile.bounds, path):
                        completed.add(key)
                        total_downloads += 1

    return total_downloads


# ──────────────────────────────────────────────────────────────────────────────
# Parallel worker
# ──────────────────────────────────────────────────────────────────────────────

def worker_process(worker_id, records, resume, output_base, products):
    """Independent worker that processes a subset of FIDs."""
    worker_log = os.path.join(_SCRIPT_DIR, f'tile_pipeline_worker_{worker_id}.log')
    fh = logging.FileHandler(worker_log)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Worker {worker_id}: starting with {len(records)} FIDs, products: {products}")

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
            dl = process_fid(record, completed, output_base, products)
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

def dry_run(records, completed, products):
    total_per_product = {p: 0 for p in products}
    total_already_done = 0

    for record in records:
        geom_3857 = project_to_3857(shape(record['geometry']))
        n_tiles = len(create_tile_grid(geom_3857))

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

    logger.info(f"DRY RUN: {len(records)} FIDs")
    logger.info(f"  Products requested: {products}")
    for product, count in total_per_product.items():
        est_mb = count * (15 if product.startswith('s2') else 2)
        logger.info(f"  {product}: ~{count} downloads (~{est_mb / 1000:.1f} GB)")
    logger.info(f"  TOTAL new downloads: ~{total}")
    logger.info(f"  Already completed: {total_already_done}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download S1/S2 tiles (534x534 @ 10m) for multiple foundation models'
    )
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Directory containing v1/v2/v3 CSV files (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output base directory (default: auto-detect)')
    parser.add_argument('--products', nargs='+', choices=ALL_PRODUCTS, default=ALL_PRODUCTS,
                        help='Which products to download (default: all)')
    parser.add_argument('--fids', nargs='+', type=str,
                        help='Specific FIDs to process')
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
    args = parser.parse_args()

    csv_dir = args.csv_dir if args.csv_dir else _DEFAULT_CSV_DIR
    output_base = args.output if args.output else _DEFAULT_OUTPUT
    products = args.products

    logger.info(f"Parsing CSVs from: {csv_dir}")
    logger.info(f"Products: {products}")
    records = parse_and_merge_csvs(csv_dir, first_only=args.first_only)

    if args.fids:
        fid_set = set(args.fids)
        records = [r for r in records if r['fid'] in fid_set]
        logger.info(f"Filtered to {len(records)} requested FIDs")

    if records and args.sample_pct < 100.0:
        import random
        random.seed(42)
        n = max(1, int(len(records) * args.sample_pct / 100.0))
        records = random.sample(records, n)
        logger.info(f"Sampled {n} FIDs ({args.sample_pct}%)")

    if records and args.sample_n:
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
        dry_run(records, completed, products)
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
                (i, chunk, args.resume, output_base, products)
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
            dl = process_fid(record, completed, output_base, products)
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
