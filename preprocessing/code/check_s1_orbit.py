#!/usr/bin/env python3
"""
Check the orbit direction (ASCENDING / DESCENDING) of the Sentinel-1 GRD
images that were downloaded by tile_pipeline.py.

The orbit pass is NOT stored in the downloaded .tif files; it only exists as
the `orbitProperties_pass` metadata property on the S1 image in Earth Engine.
This script recovers the downloaded image IDs and queries GEE for each one.

Image IDs can come from either:
  --tiles-dir  : scan the s1_grd/ folder tree (the images you actually downloaded)
  --csv-dir    : read evtIdsS1 / befIdsS1 / aftIdsS1 columns from the v1/v2/v3 CSVs

Usage (run on the cluster, where GEE is authenticated):
  python check_s1_orbit.py --tiles-dir ~/thesis_tiles_120px
  python check_s1_orbit.py --csv-dir ./data_csv
  python check_s1_orbit.py --tiles-dir ~/thesis_tiles_120px --csv out.csv
"""

import ee
import os
import csv
import glob
import argparse

GEE_PROJECT = 'zinc-wares-316319'
S1_COLLECTION = 'COPERNICUS/S1_GRD'
S1_ID_COLUMNS = ['evtIdsS1', 'befIdsS1', 'aftIdsS1']


def ids_from_tiles_dir(tiles_dir):
    """Recover downloaded S1 image IDs from the s1_grd/ folder structure.

    Layout: <tiles_dir>/s1_grd/fid_*/{evt,bef,aft}/<image_id>/tile_*.tif
    The pipeline stores the image_id as the folder name (slashes -> underscores,
    but S1 GRD IDs contain none, so the folder name is the image_id).
    """
    s1_root = os.path.join(tiles_dir, 's1_grd')
    if not os.path.isdir(s1_root):
        raise SystemExit(f"No s1_grd/ folder under {tiles_dir}")

    ids = set()
    pattern = os.path.join(s1_root, 'fid_*', '*', '*')
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            ids.add(os.path.basename(path))
    return ids


def ids_from_csv_dir(csv_dir):
    """Recover S1 image IDs from the evt/bef/aft S1 columns of the CSVs."""
    ids = set()
    for csv_path in glob.glob(os.path.join(csv_dir, 'v*_images_s2_s1.csv')):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for col in S1_ID_COLUMNS:
                    raw = (row.get(col) or '').strip()
                    for img_id in raw.split(','):
                        img_id = img_id.strip()
                        if img_id:
                            ids.add(img_id)
    return ids


def orbit_pass(image_id):
    """Return 'ASCENDING' / 'DESCENDING' (or 'ERROR:<msg>') for one S1 image."""
    try:
        img = ee.Image(f'{S1_COLLECTION}/{image_id}')
        return img.get('orbitProperties_pass').getInfo()
    except Exception as e:
        return f'ERROR:{e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-dir', help='Downloaded tiles dir (contains s1_grd/)')
    ap.add_argument('--csv-dir', help='Dir with v1/v2/v3 CSVs (alternative source of IDs)')
    ap.add_argument('--csv', help='Optional path to write a per-image CSV report')
    args = ap.parse_args()

    if not args.tiles_dir and not args.csv_dir:
        ap.error('Provide --tiles-dir (recommended) or --csv-dir')

    if args.tiles_dir:
        image_ids = ids_from_tiles_dir(args.tiles_dir)
        source = f'downloaded tiles in {args.tiles_dir}'
    else:
        image_ids = ids_from_csv_dir(args.csv_dir)
        source = f'CSVs in {args.csv_dir}'

    image_ids = sorted(image_ids)
    print(f'Found {len(image_ids)} unique S1 image IDs from {source}')
    if not image_ids:
        return

    ee.Initialize(project=GEE_PROJECT)

    counts = {}
    rows = []
    for i, image_id in enumerate(image_ids, 1):
        p = orbit_pass(image_id)
        counts[p] = counts.get(p, 0) + 1
        rows.append((image_id, p))
        print(f'  [{i}/{len(image_ids)}] {image_id} -> {p}')

    print('\n=== Summary ===')
    for p, n in sorted(counts.items()):
        pct = 100 * n / len(image_ids)
        print(f'  {p}: {n} ({pct:.1f}%)')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image_id', 'orbit_pass'])
            w.writerows(rows)
        print(f'\nWrote per-image report to {args.csv}')


if __name__ == '__main__':
    main()
