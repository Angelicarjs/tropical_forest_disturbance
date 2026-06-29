"""
Random forest at a pixel level

A) Integrity — shape, NaN/Inf, all-zero.
B) Random forest training and evaluation on a fixed 50% test / 50% train+val split (at FID level, stratified by class).
C) Learning curve: how test performance changes as the training set grows (N FIDs per
"""

import os
import glob
import random
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import defaultdict
import numpy as np
from seg_dataset import DisturbanceSegDataset, CLASS_TO_ID, NUM_CLASSES
import matplotlib.pyplot as plt

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
ID_TO_CLASS[0] = "background"


def integrity(emb_root="embeddings", n=300):
    paths = glob.glob(str(Path(emb_root) / "joint" / "fid_*" / "*" / "*" / "tile_*.npy"))
    print(f"[A] total .npy in joint/: {len(paths)}")
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


def load_or_make_split(fid_cls, split_dir="splits", seed=42):
    """
    50% test / 50% train+val split at the FID level, stratified by class.
    FIDs are saved to txt so the split is FIXED across runs (test and
    train+val are never mixed). If the txt files already exist, they are read.
    Returns (test_fids: set, trainval_fids: set).
    """
    split_dir = Path(split_dir)
    test_path = split_dir / "split_test_fids.txt"
    trainval_path = split_dir / "split_trainval_fids.txt"

    # if a saved split already exists -> read it and do not re-mix
    if test_path.exists() and trainval_path.exists():
        test_fids = {l.strip() for l in test_path.read_text().splitlines() if l.strip()}
        trainval_fids = {l.strip() for l in trainval_path.read_text().splitlines() if l.strip()}
        # warn if there are new FIDs not covered by the saved split
        new_fids = set(map(str, fid_cls.keys())) - test_fids - trainval_fids
        if new_fids:
            print(f"[WARN] {len(new_fids)} new FIDs are not in the saved split "
                  f"and will be ignored: {sorted(new_fids)[:5]}...")
        print(f"split loaded from {split_dir}/  (test={len(test_fids)} | trainval={len(trainval_fids)} FIDs)")
        return test_fids, trainval_fids

    # group fids per class
    cls_fids = defaultdict(list)
    for f, c in fid_cls.items():
        cls_fids[c].append(f)

    # 50% test / 50% train+val per class. If a class has only 1 FID -> train+val (0 test).
    rng = random.Random(seed)
    test_fids, trainval_fids = set(), set()
    for c, fl in cls_fids.items():
        fl = sorted(str(x) for x in fl)
        rng.shuffle(fl)
        n_test = len(fl) // 2 if len(fl) > 1 else 0
        test_fids.update(fl[:n_test])
        trainval_fids.update(fl[n_test:])

    split_dir.mkdir(parents=True, exist_ok=True)
    test_path.write_text("\n".join(sorted(test_fids)) + "\n")
    trainval_path.write_text("\n".join(sorted(trainval_fids)) + "\n")
    print(f"    split created and saved to {split_dir}/  (test={len(test_fids)} | trainval={len(trainval_fids)} FIDs)")
    return test_fids, trainval_fids


def majority_downsample(mask, out_h, out_w, n_classes):
    """
    Downsample a (H, W) integer class mask to (out_h, out_w) by majority vote.
    Each output cell covers an (H//out_h) x (W//out_w) block; the cell takes the
    most frequent class in that block (ties go to the lowest class id, e.g. background).
    """
    H, W = mask.shape
    fh, fw = H // out_h, W // out_w
    mask = mask[:out_h * fh, :out_w * fw]
    # (out_h, out_w, fh*fw): one row per output cell, holding all pixels of its block
    blocks = mask.reshape(out_h, fh, out_w, fw).transpose(0, 2, 1, 3).reshape(out_h, out_w, fh * fw)
    # per-class pixel counts within each block, then pick the argmax class
    counts = np.stack([(blocks == c).sum(-1) for c in range(n_classes)], axis=-1)
    return counts.argmax(-1)  # (out_h, out_w)


def build_pixel_dataset(ds):
    """
    PER-PIXEL: each embedding cell (15x15 grid) is one sample with 768 features.
    Its label comes from the per-pixel polygon mask (120x120) downsampled to the
    embedding grid by majority vote. Labels include background (0) + classes 1..6.
    Returns X (n_px, 768), y (n_px,), fids (n_px,) as numpy arrays.
    """
    X, y, fids = [], [], []
    for s in ds.samples:
        emb = np.load(s["npy"])                  # (15, 15, 768)
        gh, gw, _ = emb.shape
        px = emb.reshape(-1, 768)                # (gh*gw, 768): one row per cell
        mask = ds._build_mask(s)                 # (120, 120) per-pixel class mask
        lab = majority_downsample(mask, gh, gw, NUM_CLASSES).reshape(-1)  # (gh*gw,)
        X.append(px)
        y.append(lab)
        fids.append(np.full(lab.shape[0], s["fid"]))
    return np.concatenate(X, 0), np.concatenate(y, 0), np.concatenate(fids, 0)


def balance_classes(y, train_mask, per_class=2500, seed=0):
    """Class-balance the TRAINING tokens.

    Keeps at most `per_class` tokens per class, drawn at random from the tokens
    selected by `train_mask`. Classes with fewer than `per_class` tokens keep all
    of them. Tokens outside `train_mask` (e.g. the test set) are never touched.
    Returns a new boolean mask over the same array as `train_mask`.
    """
    rng = np.random.default_rng(seed)
    bal = np.zeros_like(train_mask)
    train_idx = np.flatnonzero(train_mask)
    for c in np.unique(y[train_idx]):
        idx_c = train_idx[y[train_idx] == c]
        if len(idx_c) > per_class:
            idx_c = rng.choice(idx_c, size=per_class, replace=False)
        bal[idx_c] = True
    return bal



def make_rf():
    # 1000 trees, balanced class weights (unbalanced classes), random_state=0 (reproducible), n_jobs=n_jobs (cpu in parallel).
    n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", -1)) #based on sh parallelization
    return RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0, n_jobs=n_jobs)


def signal(emb_root, tiles_root, shp, make_model=make_rf, model_name="RandomForest"):
    ds = DisturbanceSegDataset(emb_root, tiles_root, shp)
    X, y, fids = build_pixel_dataset(ds)
    print(f"\n[B] {model_name} | {len(X)} pixels | {len(set(fids.tolist()))} FIDs | "
          f"classes={sorted(set(y.tolist()))}")

    # fid -> class (the FID's disturbance class, used only to stratify the split)
    fid_cls = {f: ds.fid_polys[f][0][1] for f in set(fids.tolist())}

    # fixed, stratified split: 50% test / 50% train+val (at FID level, saved to txt)
    test_fids, trainval_fids = load_or_make_split(fid_cls)

    fids_str = fids.astype(str)
    # test where in test_fids
    te = np.isin(fids_str, list(test_fids))
    # train+val where in trainval_fids
    tr = np.isin(fids_str, list(trainval_fids))
    print(f"train+val: {tr.sum()} pixels | test: {te.sum()} pixels")

    # classes / names — computed once
    labels = sorted(set(y.tolist()))
    names = [ID_TO_CLASS[c] for c in labels]

    # balance training tokens: <= 2500 per class (random); classes with fewer keep all
    tr_bal = balance_classes(y, tr, per_class=2500)

    # per-class counts: train+val before, train+val after, test
    n_before = [int((y[tr] == c).sum()) for c in labels]
    n_after  = [int((y[tr_bal] == c).sum()) for c in labels]
    n_te     = [int((y[te] == c).sum()) for c in labels]

    # table
    print(f"\n{'class':<14} {'before':>9} {'after':>9} {'test':>8}")
    print("-" * 42)
    for name, a, b, t in zip(names, n_before, n_after, n_te):
        print(f"{name:<14} {a:>9} {b:>9} {t:>8}")
    print("-" * 42)
    print(f"{'TOTAL':<14} {int(tr.sum()):>9} {int(tr_bal.sum()):>9} {int(te.sum()):>8}")

    # histogram: before vs after (+ test for reference)
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w, n_before, w, label="train+val (before)", color="tab:blue")
    b2 = ax.bar(x,     n_after,  w, label="train+val (after)",  color="tab:green")
    b3 = ax.bar(x + w, n_te,     w, label="test",               color="tab:orange")
    ax.axhline(2500, color="red", ls="--", lw=1.5, label="2500 cap")
    for b in (b1, b2, b3):
        ax.bar_label(b, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("tokens"); ax.set_title("Token distribution per class — before vs after balancing")
    ax.legend(); plt.tight_layout()
    plt.savefig("token_distribution.png", dpi=150, bbox_inches="tight")
    print("saved token_distribution.png")

    # train on the balanced set
    tr = tr_bal
    print(f"\nafter balancing -> {int(tr.sum())} train tokens")

    clf = make_model()
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    print(f"accuracy (test): {accuracy_score(y[te], pred):.3f}\n")
    print(classification_report(y[te], pred, labels=labels, target_names=names, zero_division=0))


def learning_curve(emb_root, tiles_root, shp, n_repeats=3,
                   make_model=make_rf, model_name="RandomForest"):
    """
    Learning curve: how test performance changes as the training set grows.

    The TEST set stays fixed (the 50% test split). On the train+val side we take
    N FIDs per class (with all their pixels), with N growing on a log schedule
    (2, 4, 8, ... up to the largest per-class FID count). For each N we train a
    fresh model, evaluate on the fixed test set, and average over `n_repeats`
    random FID draws to smooth the noise???. Results are printed only (nothing is
    saved). The FID draws are seeded independently of the model, so calling this
    with different models reuses the EXACT same FIDs -> the curves are comparable.
    """
    ds = DisturbanceSegDataset(emb_root, tiles_root, shp)
    X, y, fids = build_pixel_dataset(ds)
    fids_str = fids.astype(str)

    # fid -> disturbance class; fixed test / train+val split (FID level)
    fid_cls = {f: ds.fid_polys[f][0][1] for f in set(fids.tolist())}
    test_fids, trainval_fids = load_or_make_split(fid_cls)
    te = np.isin(fids_str, list(test_fids))

    # train+val FIDs grouped per disturbance class
    cls_fids = defaultdict(list)
    for f in trainval_fids:
        if f in fid_cls:
            cls_fids[fid_cls[f]].append(f)
    max_n = max(len(v) for v in cls_fids.values())

    # log schedule: 2, 4, 8, ... up to max_n (always include max_n itself)
    ns, k = [], 2
    while k < max_n:
        ns.append(k)
        k *= 2
    ns.append(max_n)
    ns = sorted(set(n for n in ns if n >= 1))

    dist_labels = sorted(CLASS_TO_ID.values())        # 1..6 (disturbance only)
    all_labels = sorted(set(y[te].tolist()))          # incl. background, for context

    print(f"\n[C] learning curve | {model_name} | max FIDs/class={max_n} | "
          f"N schedule={ns} | repeats={n_repeats}")
    print(f"    {'N/class':>8} {'train_px':>9} {'acc':>14} {'f1_dist':>14} {'f1_all':>14}")

    rows = []
    for n in ns:
        accs, f1d, f1a, npx = [], [], [], []
        for r in range(n_repeats):
            rng = random.Random(1000 + r)
            selected = []
            for c, fl in cls_fids.items():
                fl = sorted(fl)
                rng.shuffle(fl)
                selected.extend(fl[:min(n, len(fl))])
            tr = np.isin(fids_str, selected)
            clf = make_model()
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            f1d.append(f1_score(y[te], pred, labels=dist_labels, average="macro", zero_division=0))
            f1a.append(f1_score(y[te], pred, labels=all_labels, average="macro", zero_division=0))
            npx.append(int(tr.sum()))
        row = {
            "n_fids_per_class": n,
            "train_pixels_mean": float(np.mean(npx)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_dist_mean": float(np.mean(f1d)), "f1_dist_std": float(np.std(f1d)),
            "f1_all_mean": float(np.mean(f1a)), "f1_all_std": float(np.std(f1a)),
        }
        rows.append(row)
        print(f"    {n:>8} {row['train_pixels_mean']:>9.0f} "
              f"{row['acc_mean']:>7.3f}±{row['acc_std']:<5.3f} "
              f"{row['f1_dist_mean']:>7.3f}±{row['f1_dist_std']:<5.3f} "
              f"{row['f1_all_mean']:>7.3f}±{row['f1_all_std']:<5.3f}")


if __name__ == "__main__":
    integrity("embeddings")
    signal("embeddings",
           os.path.expanduser("~/thesis_tiles_120px"),
           "data_shp/label_polygons.shp")
    learning_curve("embeddings",
                   os.path.expanduser("~/thesis_tiles_120px"),
                   "data_shp/label_polygons.shp")
