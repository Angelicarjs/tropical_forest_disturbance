"""
Verify the CROMA joint embeddings.

A) Integrity — shape, NaN/Inf, all-zero, basic value stats over a sample.
B) Signal    — can a SIMPLE classifier (Random Forest) on mean-pooled embeddings
               separate the disturbance types? FID-split, per-class report.

If B separates the frequent classes well, the embeddings carry signal and a weak
segmentation is a model problem. If even B fails, the problem is upstream.
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
from seg_dataset import DisturbanceSegDataset, CLASS_TO_ID

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
ID_TO_CLASS[0] = "background"


def integrity(emb_root="embeddings", n=300):
    paths = glob.glob(str(Path(emb_root) / "joint" / "fid_*" / "*" / "*" / "tile_*.npy"))
    print(f"[A] total .npy en joint/: {len(paths)}")
    if not paths:
        return
    rng = random.Random(0)
    rng.shuffle(paths)
    bad_shape = nan = zero = 0
    #means, stds = [], []
    for p in paths[:n]:
        a = np.load(p)
        if a.shape != (15, 15, 768):
            bad_shape += 1
            continue
        if not np.isfinite(a).all():
            nan += 1
        if np.abs(a).sum() == 0:
            zero += 1
        #means.append(float(a.mean()))
        #stds.append(float(a.std()))
    print(f"    revisados {min(n, len(paths))}:  bad_shape={bad_shape}  nan/inf={nan}  all_zero={zero}")
    #if means:
        #print(f"    valores: mean≈{np.mean(means):.3f}  std≈{np.mean(stds):.3f}  "
              #f"(should be not 0)")


def signal(emb_root, tiles_root, shp):
    ds = DisturbanceSegDataset(emb_root, tiles_root, shp)
    X, y, fids = [], [], []
    for s in ds.samples:
        vec = np.load(s["npy"]).reshape(-1, 768).mean(0)   # mean-pool espacial -> 768
        X.append(vec)
        y.append(ds.fid_polys[s["fid"]][0][1])             # clase del FID
        fids.append(s["fid"])
    X, y, fids = np.array(X), np.array(y), np.array(fids)
    print(f"\n[B] {len(X)} tiles | {len(set(fids))} FIDs | clases={sorted(set(y.tolist()))}")

    # split por FID (sin fugas)
    ufids = sorted(set(fids.tolist()))
    random.Random(42).shuffle(ufids)
    val_fids = set(ufids[: max(1, len(ufids) // 5)])
    va = np.isin(fids, list(val_fids))
    tr = ~va
    print(f"    train: {tr.sum()} tiles | val: {va.sum()} tiles")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
    except ModuleNotFoundError:
        print("    sklearn no instalado -> pip install scikit-learn")
        return

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    labels = sorted(set(y.tolist()))
    names = [ID_TO_CLASS[c] for c in labels]
    print(f"    accuracy (val): {accuracy_score(y[va], pred):.3f}\n")
    print(classification_report(y[va], pred, labels=labels, target_names=names, zero_division=0))


if __name__ == "__main__":
    integrity("embeddings")
    signal("embeddings",
           os.path.expanduser("~/thesis_tiles_120px"),
           "data_shp/label_polygons.shp")
