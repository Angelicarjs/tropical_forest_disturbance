"""Spatial maps of the classification, one row per disturbance class, at manuscript size.

`run_eval.py` already writes fidmaps.png, but with up to four tiles per polygon it comes
out too tall for a page, and the polygons it maps are the first four of the test split
rather than one per class. This script reuses the same plotting function on a saved
model, so nothing is refitted, and draws one row per class by default.

What the figure shows that the metrics cannot: where the predicted classes fall with
respect to the label polygon. The reference panel marks as background everything outside
the polygon, so a prediction of disturbance out there is either an error or a disturbance
the reference never mapped, and a confusion matrix cannot tell the two apart.

The polygon of each class is drawn at random with a fixed seed rather than chosen, since
picking the largest or the clearest one per class would flatter the model. State the seed
in the caption.

Usage, from the repository root:
    python classification_models/code/figure_fid_maps.py
    python classification_models/code/figure_fid_maps.py --fids 389 25 3
"""
import argparse
import os
import random

import joblib

import eval_plots as ep
from seg_dataset import DisturbanceSegDataset
from token_pipeline import FOREST_ROOT, ID_TO_CLASS, load_or_make_split


def one_fid_per_class(ds, seed):
    """One test polygon per disturbance class, drawn at random with a fixed seed.

    Only polygons with a tile on disk are eligible, since the others have nothing to
    draw. Classes are visited in label order so the rows are always in the same order.
    """
    fid_cls = {f: ds.fid_polys[f][0][1] for f in ds.fid_polys}
    test_fids, _ = load_or_make_split(fid_cls)
    on_disk = {s["fid"] for s in ds.samples}

    per_class = {}
    for fid, cls in fid_cls.items():
        if fid in test_fids and fid in on_disk:
            per_class.setdefault(cls, []).append(fid)

    rng = random.Random(seed)
    chosen = []
    for cls in sorted(per_class):
        fids = sorted(per_class[cls])
        pick = rng.choice(fids)
        chosen.append(pick)
        print(f"  {ID_TO_CLASS[cls]:<32} fid {pick}  (of {len(fids)} eligible)")
    missing = sorted(set(fid_cls.values()) - set(per_class))
    for cls in missing:
        print(f"  {ID_TO_CLASS[cls]:<32} no eligible test polygon, class omitted")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="classification_models/results/results_0",
                    help="directory run_eval.py wrote into, holding <model>/model.joblib")
    ap.add_argument("--model", default="log_reg", choices=["log_reg", "rf"])
    ap.add_argument("--fids", nargs="+", default=None,
                    help="polygons to map; omitted draws one per class")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed of the per class draw, reported in the caption")
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--embed-kind", default="joint",
                    help="embedding modality: joint | s2_l2a | s1_grd")
    ap.add_argument("--tiles-root", default=os.path.expanduser("~/thesis_tiles_120px"))
    ap.add_argument("--shp", default="data/data_shp/label_polygons.shp")
    ap.add_argument("--forest-root", default=FOREST_ROOT)
    ap.add_argument("--tiles-per-fid", type=int, default=1,
                    help="rows per polygon; 1 keeps the figure to one page")
    ap.add_argument("--row-height", type=float, default=1.6,
                    help="inches per row; lower it if the figure overflows the page")
    ap.add_argument("--no-forest-cells", action="store_true",
                    help="drop the green shading of the sampled forest training cells")
    ap.add_argument("--out", default=None,
                    help="output path; defaults to <results-root>/<model>/fidmaps_manuscript.pdf")
    args = ap.parse_args()

    path = os.path.join(args.results_root, args.model, "model.joblib")
    if not os.path.exists(path):
        raise SystemExit(f"no model at {path}; run run_eval.py first")

    ds = DisturbanceSegDataset(args.embeddings_root, args.tiles_root, args.shp,
                               embed_kind=args.embed_kind)
    clf = joblib.load(path)

    if args.fids:
        fids = args.fids
        available = {s["fid"] for s in ds.samples}
        missing = [f for f in fids if f not in available]
        if missing:
            print(f"[warn] no tile on disk for {missing}, they will be skipped")
    else:
        print(f"one test polygon per class, drawn with seed {args.seed}:")
        fids = one_fid_per_class(ds, args.seed)

    out = args.out or os.path.join(args.results_root, args.model,
                                   "fidmaps_manuscript.pdf")
    name = {"log_reg": "Logistic Regression", "rf": "Random Forest"}[args.model]
    ep.plot_fid_maps(ds, clf, fids, name, out=out,
                     max_tiles_per_fid=args.tiles_per_fid,
                     forest_root=args.forest_root,
                     show_forest=not args.no_forest_cells,
                     row_height=args.row_height)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
