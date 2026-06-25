"""
Driver: generate CROMA *joint* (S1+S2) embeddings over the whole dataset
(or a FID subset), reusing the building blocks in make_embeddings.py.

For each (fid, window) it pairs S2 and S1 acquisitions by date (<= max-gap-days)
via iter_joint_pairs_from_csv, then runs CROMA's joint encoder on each matched
pair. The model and the global normalization stats are loaded ONCE and reused.
Already-computed tiles (.npy on disk) are skipped, so the job is resumable.

Run on a GPU node (see run_embeddings.sh):
    python embed_all_joint.py \
        --tiles-root ~/thesis_tiles_120px \
        --csv-dir   ./data_csv \
        [--fid-list ./data_csv/sample_10pct_stratified.txt] \
        [--max-gap-days 7] [--overwrite]
"""

import argparse
import time
from pathlib import Path

from make_embeddings import (
    load_model,
    load_or_compute_stats,
    embed_image_joint,
    iter_joint_pairs_from_csv,
    get_device,
)

CSV_FILES = ["v1_images_s2_s1.csv", "v2_images_s2_s1.csv", "v3_images_s2_s1.csv"]

# FIDs dropped from the dataset: every acquisition has S1 frame-edge nodata
# (S1 swath border), so they are excluded from embedding generation.
EXCLUDE_FIDS = {"12", "30", "31", "33", "58", "65", "71", "96",
                "153", "169", "276", "288", "310", "345"}


def main():
    ap = argparse.ArgumentParser(description="Generate CROMA joint (S1+S2) embeddings over the dataset.")
    ap.add_argument("--tiles-root", required=True,
                    help="Path to tiles_120px (the folder containing s2_l2a/ and s1_grd/).")
    ap.add_argument("--csv-dir", required=True,
                    help="Directory with v1/v2/v3_images_s2_s1.csv.")
    ap.add_argument("--fid-list", default=None,
                    help="Optional file with FIDs to keep (one per line). Default: all FIDs.")
    ap.add_argument("--max-gap-days", type=int, default=7,
                    help="Max S2<->S1 date gap (days) to form a joint pair (default: 7).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the .npy already exists.")
    args = ap.parse_args()

    tiles_root = Path(args.tiles_root).expanduser()
    csv_dir = Path(args.csv_dir).expanduser()

    # Optional FID filter (the 10% with 41 stratified sample)
    keep_fids = None
    if args.fid_list:
        with open(args.fid_list) as f:
            keep_fids = {line.strip() for line in f if line.strip()}
        print(f"[filter] restricting to {len(keep_fids)} FIDs from {args.fid_list}")

    device = get_device()
    print(f"[device] {device}")

    # Load the joint model and both sets of normalization stats ONCE.
    model = load_model("joint", device=device)
    s2_stats = load_or_compute_stats(tiles_root, "s2_l2a")
    s1_stats = load_or_compute_stats(tiles_root, "s1_grd")

    # Iterate pairs across all three CSVs; dedup identical pairs so the same
    # (fid, window, S2, S1) is not processed twice.
    seen = set()
    n_pairs = 0
    t0 = time.time()

    for csv_name in CSV_FILES:
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            print(f"[warn] missing {csv_path}, skipping")
            continue
        print(f"[csv] {csv_name}")
        for fid, win, s2_dir, s1_dir, gap in iter_joint_pairs_from_csv(
            str(csv_path), str(tiles_root), max_gap_days=args.max_gap_days,
        ):
            if keep_fids is not None and str(fid) not in keep_fids:
                continue
            if str(fid) in EXCLUDE_FIDS:
                continue
            key = (str(fid), win, Path(s2_dir).name, Path(s1_dir).name)
            if key in seen:
                continue
            seen.add(key)
            embed_image_joint(
                s2_dir, s1_dir, model=model,
                s2_stats=s2_stats, s1_stats=s1_stats,
                device=device, overwrite=args.overwrite,
            )
            n_pairs += 1

    dt = time.time() - t0
    print(f"\n[done] {n_pairs} unique (fid, window, S2, S1) pairs processed in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
