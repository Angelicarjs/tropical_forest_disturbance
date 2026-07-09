"""
Single entry point for the per-token disturbance evaluation.

Loads the embeddings ONCE, then for each model (Random Forest, Logistic
Regression) it:
  - trains on the balanced train+val tokens (<=2500/class); test kept full
  - saves artifacts (model.joblib, pred.npy, y_test.npy, test_fids.npy)
  - saves figures: confusion.png, prf.png, fidmaps.png, learning_curve.png
into results/rf/ and results/log_reg/. A shared results/token_distribution.png
is written once. The notebook only reads these outputs.

Run on the cluster:  sbatch run_eval.sh
"""
import os
import argparse
import joblib
import numpy as np

from seg_dataset import DisturbanceSegDataset
from rand_forest import (FOREST_ROOT, load_or_make_split, build_pixel_dataset_forest,
                         balance_classes, make_rf, learning_curve)
from log_reg import make_lr
import eval_plots as ep

MODELS = [("RandomForest", make_rf, "results/rf"),
          ("LogisticRegression", make_lr, "results/log_reg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--tiles-root", default=os.path.expanduser("~/thesis_tiles_120px"))
    ap.add_argument("--shp", default="data_shp/label_polygons.shp")
    ap.add_argument("--forest-root", default=FOREST_ROOT)
    ap.add_argument("--n-repeats", type=int, default=3, help="learning-curve repeats")
    ap.add_argument("--fids", nargs="*", default=None,
                    help="test FIDs to map (default: first 4 test FIDs)")
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_root, exist_ok=True)

    # ---- load everything once (heavy) ----
    ds = DisturbanceSegDataset(args.embeddings_root, args.tiles_root, args.shp)
    X, y, fids = build_pixel_dataset_forest(ds, args.forest_root)
    fids_str = fids.astype(str)

    fid_cls = {f: ds.fid_polys[f][0][1] for f in set(fids.tolist()) if f in ds.fid_polys}
    test_fids, trainval_fids = load_or_make_split(fid_cls)
    tr = np.isin(fids_str, list(trainval_fids))
    te = np.isin(fids_str, list(test_fids))
    tr_bal = balance_classes(y, tr, per_class=2500)      # train balanced; test kept full
    print(f"[data] train {int(tr_bal.sum())} tok | test {int(te.sum())} tok (full)", flush=True)

    # shared, split-level figure (model-independent)
    ep.plot_token_distribution(y, tr, tr_bal, te,
                               out=os.path.join(args.results_root, "token_distribution.png"))

    # test FIDs to visualize (from the test set, so they are always valid)
    sample_fids = {s["fid"] for s in ds.samples}
    map_fids = args.fids or sorted((f for f in test_fids if f in sample_fids), key=int)[:4]
    print(f"[maps] FIDs: {map_fids}", flush=True)

    for name, make, outdir in MODELS:
        os.makedirs(outdir, exist_ok=True)
        print(f"[{name}] training...", flush=True)
        clf = make()
        clf.fit(X[tr_bal], y[tr_bal])
        pred = clf.predict(X[te])

        joblib.dump(clf, os.path.join(outdir, "model.joblib"))
        np.save(os.path.join(outdir, "pred.npy"), pred)
        np.save(os.path.join(outdir, "y_test.npy"), y[te])
        np.save(os.path.join(outdir, "test_fids.npy"), fids_str[te])

        ep.plot_confusion(y[te], pred, name, out=os.path.join(outdir, "confusion.png"))
        ep.plot_prf(y[te], pred, name, out=os.path.join(outdir, "prf.png"))
        ep.plot_fid_maps(ds, clf, map_fids, name, out=os.path.join(outdir, "fidmaps.png"))

        print(f"[{name}] learning curve...", flush=True)
        rows = learning_curve(args.embeddings_root, args.tiles_root, args.shp,
                              n_repeats=args.n_repeats, make_model=make,
                              model_name=name, data=(ds, X, y, fids))
        ep.plot_learning_curve(rows, name, out=os.path.join(outdir, "learning_curve.png"))
        print(f"[{name}] done -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
