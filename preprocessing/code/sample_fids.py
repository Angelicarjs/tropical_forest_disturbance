#!/usr/bin/env python3
"""Stratified 10% sample of FIDs by CLASSNAME. Writes one FID per line."""

import argparse
import math
import os
import random

import geopandas as gpd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_PATH = os.path.join(SCRIPT_DIR, "data_shp", "label_polygons.shp")
OUT_PATH = os.path.join(SCRIPT_DIR, "data_csv", "sample_10pct_stratified.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    gdf = gpd.read_file(SHP_PATH)
    gdf["fid"] = gdf["fid"].astype(int).astype(str)

    rng = random.Random(args.seed)
    sampled = []
    print(f"{'CLASSNAME':<32} {'total':>6} {'sampled':>8}")
    print("-" * 50)
    for cls, sub in gdf.groupby("CLASSNAME"):
        fids = sorted(sub["fid"].tolist())
        n = max(1, math.ceil(len(fids) * args.pct / 100.0))
        chosen = rng.sample(fids, n)
        sampled.extend(chosen)
        print(f"{cls:<32} {len(fids):>6} {n:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<32} {len(gdf):>6} {len(sampled):>8}")

    sampled = sorted(sampled, key=int)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for fid in sampled:
            f.write(f"{fid}\n")
    print(f"\nWrote {len(sampled)} FIDs to {args.out}")


if __name__ == "__main__":
    main()
