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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
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
    print(f"    checked {min(n, len(paths))}:  bad_shape={bad_shape}  nan/inf={nan}  all_zero={zero}")
    #if means:
        #print(f"    values: mean≈{np.mean(means):.3f}  std≈{np.mean(stds):.3f}  "
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

    # fid -> clase (una por fid)
    fid_cls = {}
    for f, c in zip(fids.tolist(), y.tolist()):
        fid_cls.setdefault(f, c)
    
    # group fids per class
    cls_fids = defaultdict(list)
    for f, c in fid_cls.items():
        cls_fids[c].append(f)
    
    # 20% per class for val, rest for train. If a class has only 1 FID, put it all in train (0 val).
    rng = random.Random(42)
    val_fids = set()
    for c, fl in cls_fids.items():
        fl = sorted(fl)
        rng.shuffle(fl)
        n_val = max(1, len(fl) // 5) if len(fl) > 1 else 0
        val_fids.update(fl[:n_val])

    #val where in val_fids
    va = np.isin(fids, list(val_fids))
    #train where not in val
    tr = ~va
    print(f"train: {tr.sum()} tiles | val: {va.sum()} tiles")

    # 300 trees, balanced class weights (unbalanced classes), random_state=0 (reproducible), n_jobs=-1 (cpu in parallel).
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    labels = sorted(set(y.tolist()))
    names = [ID_TO_CLASS[c] for c in labels]
    print(f"accuracy (val): {accuracy_score(y[va], pred):.3f}\n")
    print(classification_report(y[va], pred, labels=labels, target_names=names, zero_division=0))


if __name__ == "__main__":
    integrity("embeddings")
    signal("embeddings",
           os.path.expanduser("~/thesis_tiles_120px"),
           "data_shp/label_polygons.shp")
