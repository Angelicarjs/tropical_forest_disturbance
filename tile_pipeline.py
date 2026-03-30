#!/usr/bin/env python3
"""
Pipeline to download Sentinel-1 and Sentinel-2 tiles from Google Earth Engine,
clipped to deforestation polygons, in native resolution and CROMA-ready formats.

Output structure:
  tiles/
  ├── native/
  │   └── fid_{fid}/
  │       ├── s2_evt/{image_id}/{res}m/tile_{n}.tif
  │       ├── s2_bef/{image_id}/{res}m/tile_{n}.tif
  │       ├── s1_evt/{image_id}/10m/tile_{n}.tif
  │       └── s1_bef/{image_id}/10m/tile_{n}.tif
  └── croma/
      └── fid_{fid}/
          ├── s2_evt/{image_id}/tile_{n}.tif   (12 bands @ 10m, 120x120)
          ├── s2_bef/{image_id}/tile_{n}.tif
          ├── s1_evt/{image_id}/tile_{n}.tif   (symlink to native S1)
          └── s1_bef/{image_id}/tile_{n}.tif   (symlink to native S1)

Usage:
  python tile_pipeline.py                          # Process all FIDs (sequential)
  python tile_pipeline.py --workers 6              # 6 parallel workers
  python tile_pipeline.py --workers 6 --resume     # Resume parallel run
  python tile_pipeline.py --fids 193               # Process specific FIDs
  python tile_pipeline.py --fids 193 --first-only  # One image per category (test)
  python tile_pipeline.py --dry-run                # Preview download counts
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
from functools import partial
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GEE_PROJECT = 'zinc-wares-316319'

# Detect environment: cluster vs local (used as defaults, overridable via CLI)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.expanduser('~/thesis_scripts')):
    _DEFAULT_CSV = os.path.expanduser('~/thesis_scripts/s1_s2_images_thesis_v2.csv')
    _DEFAULT_OUTPUT = os.path.expanduser('~/thesis_tiles')
else:
    _DEFAULT_CSV = '/Users/angelicamariamorenorojas/Desktop/Master/thesis/data/s1_s2_images_thesis_v2.csv'
    _DEFAULT_OUTPUT = '/Users/angelicamariamorenorojas/Desktop/Master/thesis/tiles'
CRS_CODE = 'EPSG:3857'
TILE_SIZE_M = 1200  # 120px at 10m, matching CROMA's default

# Sentinel-2 band groups by native resolution
S2_NATIVE_BANDS = {
    10: ['B2', 'B3', 'B4', 'B8'],
    20: ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
    60: ['B1', 'B9'],  # B10 (cirrus) not available in S2_SR_HARMONIZED (L2A)
}

# CROMA S2: 12 bands at 10m (no B10/cirrus)
CROMA_S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

# Sentinel-1 bands (VV + VH, in dB from GEE)
S1_BANDS = ['VV', 'VH']

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

def download_tile(image, bands, tile_bounds_3857, resolution_m, output_path):
    if os.path.exists(output_path):
        return True

    minx, miny, maxx, maxy = tile_bounds_3857
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))

    request = {
        'expression': image.select(bands),
        'fileFormat': 'GEO_TIFF',
        'grid': {
            'dimensions': {'width': width, 'height': height},
            'affineTransform': {
                'scaleX': resolution_m, 'shearX': 0, 'translateX': minx,
                'shearY': 0, 'scaleY': -resolution_m, 'translateY': maxy,
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
# CSV parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_csv(csv_path, first_only=False):
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row['fid'].strip()
            geo = json.loads(row['.geo'])
            record = {'fid': fid, 'geometry': geo, 'images': {}}

            for category, col in [
                ('s2_evt', 'evtIdsS2'), ('s2_bef', 'befIdsS2'),
                ('s1_evt', 'evtIdsS1'), ('s1_bef', 'befIdsS1'),
            ]:
                raw = row.get(col, '').strip()
                if raw:
                    ids = [img_id.strip() for img_id in raw.split(',') if img_id.strip()]
                    if ids:
                        record['images'][category] = [ids[0]] if first_only else ids

            if record['images']:
                records.append(record)
    return records


def get_gee_image(image_id, sensor):
    if sensor == 's2':
        return ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + image_id)
    else:
        return ee.Image('COPERNICUS/S1_GRD/' + image_id)


# ──────────────────────────────────────────────────────────────────────────────
# Processing logic
# ──────────────────────────────────────────────────────────────────────────────

def process_tiles_for_image(fid, category, image_id, image, sensor,
                            geom_3857, completed, output_base,
                            croma_only=False):
    safe_id = image_id.replace('/', '_')
    downloads = 0
    tiles = create_tile_grid(geom_3857)

    for tile_idx, tile in enumerate(tiles):
        bounds = tile.bounds

        # ── Native resolution ──
        if not croma_only:
            if sensor == 's2':
                for res_m, bands in S2_NATIVE_BANDS.items():
                    key = f"native|{fid}|{category}|{safe_id}|{res_m}m|tile_{tile_idx}"
                    if key in completed:
                        continue
                    path = os.path.join(
                        output_base, 'native', f'fid_{fid}', category,
                        safe_id, f'{res_m}m', f'tile_{tile_idx}.tif'
                    )
                    if download_tile(image, bands, bounds, res_m, path):
                        completed.add(key)
                        downloads += 1
            else:
                key = f"native|{fid}|{category}|{safe_id}|10m|tile_{tile_idx}"
                if key not in completed:
                    path = os.path.join(
                        output_base, 'native', f'fid_{fid}', category,
                        safe_id, '10m', f'tile_{tile_idx}.tif'
                    )
                    if download_tile(image, S1_BANDS, bounds, 10, path):
                        completed.add(key)
                        downloads += 1

        # ── CROMA format (12 bands @ 10m for S2, direct download for S1) ──
        key = f"croma|{fid}|{category}|{safe_id}|tile_{tile_idx}"
        if key in completed:
            continue

        croma_path = os.path.join(
            output_base, 'croma', f'fid_{fid}', category,
            safe_id, f'tile_{tile_idx}.tif'
        )

        if sensor == 's2':
            if download_tile(image, CROMA_S2_BANDS, bounds, 10, croma_path):
                completed.add(key)
                downloads += 1
        else:
            if croma_only:
                # Download S1 directly (no native to symlink to)
                if download_tile(image, S1_BANDS, bounds, 10, croma_path):
                    completed.add(key)
                    downloads += 1
            else:
                # S1: identical to native — symlink instead of duplicate download
                native_path = os.path.join(
                    output_base, 'native', f'fid_{fid}', category,
                    safe_id, '10m', f'tile_{tile_idx}.tif'
                )
                if os.path.exists(native_path) and not os.path.exists(croma_path):
                    os.makedirs(os.path.dirname(croma_path), exist_ok=True)
                    os.symlink(os.path.abspath(native_path), croma_path)
                completed.add(key)
                downloads += 1

    return downloads


def process_fid(record, completed, output_base, croma_only=False):
    fid = record['fid']
    geom_4326 = shape(record['geometry'])
    geom_3857 = project_to_3857(geom_4326)
    tiles = create_tile_grid(geom_3857)

    logger.info(f"FID {fid}: {len(tiles)} tiles, "
                f"categories: {list(record['images'].keys())}")

    if not tiles:
        logger.warning(f"FID {fid}: no intersecting tiles")
        return

    total_downloads = 0
    for category, image_ids in record['images'].items():
        sensor = 's2' if 's2' in category else 's1'
        for image_id in image_ids:
            try:
                image = get_gee_image(image_id, sensor)
            except Exception as e:
                logger.error(f"FID {fid}: failed to load {image_id}: {e}")
                continue
            logger.info(f"  {category} | {image_id}")
            dl = process_tiles_for_image(
                fid, category, image_id, image, sensor, geom_3857, completed,
                output_base, croma_only=croma_only
            )
            total_downloads += dl
            save_progress(completed, output_base)

    logger.info(f"FID {fid}: {total_downloads} new downloads")


# ──────────────────────────────────────────────────────────────────────────────
# Parallel worker
# ──────────────────────────────────────────────────────────────────────────────

def worker_process(worker_id, records, resume, output_base, croma_only=False):
    """Independent worker that processes a subset of FIDs.
    Each worker initialises its own GEE session and keeps its own progress file.
    File-exists checks on disk prevent duplicate downloads across workers."""
    # Set up per-worker logging
    worker_log = os.path.join(_SCRIPT_DIR, f'tile_pipeline_worker_{worker_id}.log')
    fh = logging.FileHandler(worker_log)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Worker {worker_id}: starting with {len(records)} FIDs")

    # Each worker gets its own GEE session
    ee.Initialize(project=GEE_PROJECT)

    # Per-worker progress file
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
            fid = record['fid']
            geom_4326 = shape(record['geometry'])
            geom_3857 = project_to_3857(geom_4326)
            tiles = create_tile_grid(geom_3857)

            if not tiles:
                logger.warning(f"Worker {worker_id}: FID {fid} no intersecting tiles")
                continue

            total_downloads = 0
            for category, image_ids in record['images'].items():
                sensor = 's2' if 's2' in category else 's1'
                for image_id in image_ids:
                    try:
                        image = get_gee_image(image_id, sensor)
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: FID {fid} failed to load {image_id}: {e}")
                        continue
                    dl = process_tiles_for_image(
                        fid, category, image_id, image, sensor, geom_3857, completed,
                        output_base, croma_only=croma_only
                    )
                    total_downloads += dl
                    save_worker_progress()

            logger.info(f"Worker {worker_id}: FID {fid} done, {total_downloads} downloads")
        except Exception as e:
            logger.error(f"Worker {worker_id}: FID {record['fid']} error: {e}", exc_info=True)
            save_worker_progress()

    save_worker_progress()
    logger.info(f"Worker {worker_id}: finished. {len(completed)} total completed.")
    return worker_id, len(completed)


# ──────────────────────────────────────────────────────────────────────────────
# Dry run
# ──────────────────────────────────────────────────────────────────────────────

def dry_run(records, completed):
    total_native = 0
    total_croma = 0

    for record in records:
        geom_3857 = project_to_3857(shape(record['geometry']))
        n_tiles = len(create_tile_grid(geom_3857))

        for category, image_ids in record['images'].items():
            sensor = 's2' if 's2' in category else 's1'
            n_imgs = len(image_ids)
            if sensor == 's2':
                total_native += n_tiles * n_imgs * 3  # 3 resolution groups
                total_croma += n_tiles * n_imgs * 1   # 12-band at 10m
            else:
                total_native += n_tiles * n_imgs * 1  # VV+VH at 10m
                # S1 CROMA = symlink, no download

    total = total_native + total_croma
    logger.info(f"DRY RUN: {len(records)} FIDs")
    logger.info(f"  Native:  ~{total_native} downloads")
    logger.info(f"  CROMA S2: ~{total_croma} downloads (S1 = symlinks)")
    logger.info(f"  TOTAL:   ~{total} actual downloads")
    logger.info(f"  Already completed: {len(completed)}")
    logger.info(f"  Remaining: ~{max(0, total - len(completed))}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download S1/S2 tiles from GEE')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to input CSV file (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output base directory (default: auto-detect)')
    parser.add_argument('--croma-only', action='store_true',
                        help='Only download CROMA format tiles (skip native)')
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
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, recommended: 4-8)')
    args = parser.parse_args()

    csv_path = args.csv if args.csv else _DEFAULT_CSV
    output_base = args.output if args.output else _DEFAULT_OUTPUT

    logger.info(f"Parsing CSV: {csv_path}")
    records = parse_csv(csv_path, first_only=args.first_only)
    logger.info(f"Found {len(records)} FIDs with images")

    if args.fids:
        fid_set = set(args.fids)
        records = [r for r in records if r['fid'] in fid_set]
        logger.info(f"Filtered to {len(records)} requested FIDs")

    if args.sample_pct < 100.0:
        import random
        random.seed(42)  # reproducible sampling
        n = max(1, int(len(records) * args.sample_pct / 100.0))
        records = random.sample(records, n)
        logger.info(f"Sampled {n} FIDs ({args.sample_pct}%)")

    if not records:
        logger.error("No records to process")
        return

    if args.croma_only:
        logger.info("CROMA-only mode: skipping native resolution downloads")

    os.makedirs(output_base, exist_ok=True)

    if args.dry_run:
        logger.info("Initializing Google Earth Engine...")
        ee.Initialize(project=GEE_PROJECT)
        completed = load_progress(output_base) if args.resume else set()
        if completed:
            logger.info(f"Resuming with {len(completed)} completed downloads")
        dry_run(records, completed)
        return

    # ── Parallel mode ──
    if args.workers > 1:
        n_workers = min(args.workers, len(records))
        logger.info(f"Launching {n_workers} parallel workers for {len(records)} FIDs")

        # Distribute FIDs across workers (round-robin for balanced load)
        chunks = [[] for _ in range(n_workers)]
        for i, record in enumerate(records):
            chunks[i % n_workers].append(record)

        for i, chunk in enumerate(chunks):
            logger.info(f"  Worker {i}: {len(chunk)} FIDs")

        # Launch workers as separate processes
        with mp.Pool(processes=n_workers) as pool:
            worker_args = [
                (i, chunk, args.resume, output_base, args.croma_only)
                for i, chunk in enumerate(chunks)
            ]
            results = pool.starmap(worker_process, worker_args)

        # Merge per-worker progress files into main progress
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

    # ── Sequential mode (original) ──
    logger.info("Initializing Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT)

    completed = load_progress(output_base) if args.resume else set()
    if completed:
        logger.info(f"Resuming with {len(completed)} completed downloads")

    total_fids = len(records)
    for idx, record in enumerate(records):
        logger.info(f"Processing FID {record['fid']} ({idx+1}/{total_fids})")
        try:
            process_fid(record, completed, output_base, croma_only=args.croma_only)
        except Exception as e:
            logger.error(f"FID {record['fid']}: unexpected error: {e}", exc_info=True)
            save_progress(completed, output_base)
            continue

    save_progress(completed, output_base)
    logger.info(f"Pipeline complete. {len(completed)} total downloads.")
    logger.info(f"Output: {output_base}")


if __name__ == '__main__':
    main()
