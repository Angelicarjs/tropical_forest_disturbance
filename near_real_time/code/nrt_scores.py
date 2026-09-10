"""Per-acquisition scores, the input to the threshold selection.

`fid_curve` already returns one row per acquisition with `p_dist`, the mean
probability of not being forest over the polygon tokens, and `p_class`, the mean
probability of the class this polygon actually belongs to. This script only
stores those rows so that the threshold can be chosen once, offline, without
applying the classifier again every time the criterion changes.

The threshold uses `p_dist`: reading `p_class` requires knowing the class of the
polygon beforehand, which an operational system does not. `p_class` is kept for
the separate question of when the disturbance type becomes identifiable, and it
is written only for the multiclass modes.

One FID per invocation, so it can be array-parallelized exactly like nrt_fid.py.

Output: nrt_scores/<mode>_<model>_v<version>/fid_<FID>.csv

Usage:
    python nrt_scores.py --fid 28
    python nrt_scores.py --fid 28 --mode joint optical --model log_reg --version 3
"""
import argparse
import os

import geopandas as gpd
import joblib
import pandas as pd

from nrt_fid import DEFAULT_VERSION, MODEL_NAMES, MODES, SHP, fid_curve
from seg_dataset import CLASS_TO_ID

OUT = "near_real_time/results/nrt_scores_new"   # the reported scores live in nrt_scores/; run_scores.sh writes there


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", type=int, required=True)
    ap.add_argument("--mode", nargs="+", default=["all"],
                    help=f"any of {list(MODES)}, or 'all'")
    ap.add_argument("--model", nargs="+", default=["log_reg"],
                    help=f"any of {MODEL_NAMES}, or 'all'")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=["1", "2", "3"],
                    help=f"cloud filter (default: {DEFAULT_VERSION}, the strictest)")
    ap.add_argument("--out", default=OUT,
                    help="output root; defaults beside the reported scores, not onto them")
    ap.add_argument("--tag", default="",
                    help="suffix for the output folder, e.g. 'test'. Keeps the scores "
                         "of one split from landing next to those of another, which "
                         "would silently invalidate a threshold chosen on the first")
    args = ap.parse_args()

    modes = list(MODES) if "all" in args.mode else args.mode
    models = list(MODEL_NAMES) if "all" in args.model else args.model
    bad = [m for m in modes if m not in MODES] + [m for m in models if m not in MODEL_NAMES]
    if bad:
        ap.error(f"unknown mode/model: {bad}")

    gdf = gpd.read_file(SHP).to_crs("EPSG:3857")
    gdf["fid"] = gdf["fid"].astype(int).astype(str)
    sub = gdf[gdf["fid"] == str(args.fid)]
    if sub.empty:
        raise SystemExit(f"fid {args.fid} is not in {SHP}")
    polys = [(g, 1) for g in sub.geometry]
    view = pd.to_datetime(sub["VIEW_DATE"].iloc[0])
    cls_id = CLASS_TO_ID[sub["CLASSNAME"].iloc[0]]

    for model in models:
        for mode in modes:
            path = os.path.join(MODES[mode]["res"], model, "model.joblib")
            if not os.path.exists(path):
                print(f"[skip] {model} / {mode}: no model at {path}")
                continue

            # Loaded one at a time: a random forest is hundreds of MB in memory.
            clf = joblib.load(path)
            # p_class only exists in the multiclass label space: the binary one
            # holds forest and clear-cut alone, so a FID of any other class has
            # no column to read.
            curve = fid_curve(args.fid, clf, polys, MODES[mode]["emb"],
                              cls_id=None if MODES[mode]["binary"] else cls_id,
                              version=args.version)
            del clf
            if curve.empty:
                print(f"[skip] {model} / {mode}: no admitted acquisition")
                continue

            curve["fid"] = args.fid
            curve["mode"] = mode
            curve["model"] = model
            # VIEW_DATE is when DETER detected, not when the disturbance happened,
            # so the threshold selection needs to be able to drop the acquisitions
            # closest to it, where the label is least trustworthy.
            curve["view_date"] = view
            curve["days"] = (curve["date"] - view).dt.days
            name = f"{mode}_{model}_v{args.version}"
            d = os.path.join(args.out, f"{name}_{args.tag}" if args.tag else name)
            os.makedirs(d, exist_ok=True)
            dest = os.path.join(d, f"fid_{args.fid}.csv")
            curve.to_csv(dest, index=False)
            print(f"{model} / {mode}: {len(curve)} acquisitions -> {dest}")


if __name__ == "__main__":
    main()
