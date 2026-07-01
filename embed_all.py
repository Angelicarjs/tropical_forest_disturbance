"""
Driver: generate CROMA embeddings over the whole dataset (or a FID subset),
reusing the building blocks in make_embeddings.py.

Modes (--modality):
    joint    S1+S2 fused pairs (default), via iter_joint_pairs_from_csv.
    optical  S2-only, one embedding per S2 acquisition on disk.
    both     optical first, then joint.

The model(s) and the global normalization stats are loaded ONCE and reused.
Already-computed tiles (.npy on disk) are skipped, so the job is resumable.

Run on a GPU node (see run_embeddings.sh):
    python embed_all_joint.py \
        --tiles-root ~/thesis_tiles_120px \
        --csv-dir   ./data_csv \
        [--modality both] \
        [--fid-list ./data_csv/sample_10pct_stratified.txt] \
        [--max-gap-days 7] [--overwrite]
"""

import argparse
import time
from pathlib import Path

from make_embeddings import (
    load_model,
    load_or_compute_stats,
    embed_image,
    embed_image_joint,
    iter_joint_pairs_from_csv,
    get_device,
)

CSV_FILES = ["v1_images_s2_s1.csv", "v2_images_s2_s1.csv", "v3_images_s2_s1.csv"]


def run_optical(tiles_root, keep_fids, overwrite, device):
    """One optical (S2-only) embedding per S2 acquisition folder on disk."""
    model = load_model("optical", device=device)
    stats = load_or_compute_stats(tiles_root, "s2_l2a")
    product_root = tiles_root / "s2_l2a"

    n = 0
    # layout: s2_l2a/fid_<fid>/<window>/<image_id>/
    for image_dir in sorted(product_root.glob("fid_*/*/*")):
        if not image_dir.is_dir():
            continue
        fid = image_dir.parent.parent.name.replace("fid_", "")
        if keep_fids is not None and fid not in keep_fids:
            continue
        embed_image(str(image_dir), model=model, stats=stats,
                    device=device, overwrite=overwrite)
        n += 1
    return n


def run_joint(tiles_root, csv_dir, keep_fids, max_gap_days, overwrite, device):
    """CROMA joint (S1+S2) embedding per matched acquisition pair."""
    model = load_model("joint", device=device)
    s2_stats = load_or_compute_stats(tiles_root, "s2_l2a")
    s1_stats = load_or_compute_stats(tiles_root, "s1_grd")

    # Iterate pairs across all three CSVs; dedup identical pairs so the same
    # (fid, window, S2, S1) is not processed twice.
    seen = set()
    n = 0
    for csv_name in CSV_FILES:
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            print(f"[warn] missing {csv_path}, skipping")
            continue
        print(f"[csv] {csv_name}")
        for fid, win, s2_dir, s1_dir, gap in iter_joint_pairs_from_csv(
            str(csv_path), str(tiles_root), max_gap_days=max_gap_days,
        ):
            if keep_fids is not None and str(fid) not in keep_fids:
                continue
            key = (str(fid), win, Path(s2_dir).name, Path(s1_dir).name)
            if key in seen:
                continue
            seen.add(key)
            embed_image_joint(
                s2_dir, s1_dir, model=model,
                s2_stats=s2_stats, s1_stats=s1_stats,
                device=device, overwrite=overwrite,
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Generate CROMA embeddings over the dataset.")
    ap.add_argument("--tiles-root", required=True,
                    help="Path to tiles_120px (the folder containing s2_l2a/ and s1_grd/).")
    ap.add_argument("--csv-dir", required=True,
                    help="Directory with v1/v2/v3_images_s2_s1.csv.")
    ap.add_argument("--modality", choices=["joint", "optical", "both"], default="joint",
                    help="Which embeddings to generate (default: joint).")
    ap.add_argument("--fid-list", default=None,
                    help="Optional file with FIDs to keep (one per line). Default: all FIDs.")
    ap.add_argument("--max-gap-days", type=int, default=7,
                    help="Max S2<->S1 date gap (days) to form a joint pair (default: 7).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the .npy already exists.")
    args = ap.parse_args()

    tiles_root = Path(args.tiles_root).expanduser()
    csv_dir = Path(args.csv_dir).expanduser()

    # Optional FID filter (e.g. the 10% stratified sample)
    keep_fids = None
    if args.fid_list:
        with open(args.fid_list) as f:
            keep_fids = {line.strip() for line in f if line.strip()}
        print(f"[filter] restricting to {len(keep_fids)} FIDs from {args.fid_list}")

    device = get_device()
    print(f"[device] {device}")

    t0 = time.time()
    if args.modality in ("optical", "both"):
        n_opt = run_optical(tiles_root, keep_fids, args.overwrite, device)
        print(f"[optical] {n_opt} S2 acquisitions embedded")
    if args.modality in ("joint", "both"):
        n_joint = run_joint(tiles_root, csv_dir, keep_fids,
                            args.max_gap_days, args.overwrite, device)
        print(f"[joint] {n_joint} pairs embedded")

    dt = time.time() - t0
    print(f"\n[done] finished in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
