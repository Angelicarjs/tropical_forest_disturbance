"""
Binary separability probe: forest (0) vs. Clear-cut bare soil (1).

Mirrors run_eval.py but keeps ONLY the two classes {0, 1}, so the classifier
answers a single question: forest vs. clear-cut bare soil. It reuses the exact
same FID-level split, the same balanced training and the same fixed full test
set as the multiclass pipeline. Data/model helpers are imported from
rand_forest.py / log_reg.py; the 2-class plotting is defined here so the
multiclass pipeline is left untouched.

For each model (Random Forest, Logistic Regression) it:
  - trains on the balanced train+val tokens; test kept full
  - saves artifacts (model.joblib, pred.npy, y_test.npy, test_fids.npy)
  - saves figures: confusion.png, prf.png, fidmaps.png, learning_curve.png
into results_binary/rf/ and results_binary/log_reg/.
"""
import os
import argparse
import random

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

from seg_dataset import DisturbanceSegDataset, CLASS_TO_ID
from rand_forest import (FOREST_ROOT, load_or_make_split,
                         build_pixel_dataset_forest, balance_classes, make_rf)
from log_reg import make_lr
import eval_plots as ep

# --- binary label space -----------------------------------------------------
NEG_ID = 0                                   # forest (curated stable-forest tokens)
POS_ID = CLASS_TO_ID["Clear-cut bare soil"]  # 1
BIN_LABELS = [NEG_ID, POS_ID]
BIN_NAMES = ["forest", "clear-cut bare soil"]

MODELS = [("RandomForest", make_rf, "results_binary/rf"),
          ("LogisticRegression", make_lr, "results_binary/log_reg")]


def plot_confusion_binary(y_true, y_pred, model_name, out):
    """Row-normalized (recall) and column-normalized (precision) 2x2 heatmaps."""
    cm = confusion_matrix(y_true, y_pred, labels=BIN_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, axis, title in [(axes[0], 1, "row-normalized (recall)"),
                            (axes[1], 0, "column-normalized (precision)")]:
        denom = cm.sum(axis=axis, keepdims=True)
        cmn = np.zeros_like(cm, dtype=float)
        np.divide(cm, denom, out=cmn, where=denom != 0)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(BIN_LABELS))); ax.set_xticklabels(BIN_NAMES, rotation=20, ha="right")
        ax.set_yticks(range(len(BIN_LABELS))); ax.set_yticklabels(BIN_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{model_name} - {title}")
        for i in range(len(BIN_LABELS)):
            for j in range(len(BIN_LABELS)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_prf_binary(y_true, y_pred, model_name, out):
    """Per-class precision / recall / F1 bars for the two classes."""
    p = precision_score(y_true, y_pred, labels=BIN_LABELS, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=BIN_LABELS, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=BIN_LABELS, average=None, zero_division=0)
    x = np.arange(len(BIN_LABELS)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w, p, w, label="precision")
    ax.bar(x,     r, w, label="recall")
    ax.bar(x + w, f, w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(BIN_NAMES)
    ax.set_ylim(0, 1.02); ax.set_ylabel("score")
    ax.set_title(f"{model_name} - per-class precision / recall / F1")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def learning_curve_binary(X, y, fids_str, trainval_fids, te,
                          make_model, model_name, n_repeats=3):
    """Binary training-size sensitivity. The fixed test set stays full; on the
    train+val side we take N FIDs per binary class (all their tokens), with N on
    a log schedule, train a fresh model, and average over `n_repeats` FID draws.
    Reported: accuracy and the precision / recall / F1 of the clear-cut (positive)
    class. FID draws are seeded so both models see the exact same FIDs.
    """
    tv = np.isin(fids_str, list(trainval_fids))
    cls_fids = {c: sorted(set(fids_str[tv & (y == c)].tolist())) for c in BIN_LABELS}
    max_n = max(len(v) for v in cls_fids.values())

    ns, k = [], 2
    while k < max_n:
        ns.append(k); k *= 2
    ns.append(max_n)
    ns = sorted(set(n for n in ns if n >= 1))

    print(f"\n[C] binary training-size sensitivity | {model_name} | "
          f"max FIDs/class={max_n} | N schedule={ns} | repeats={n_repeats}")
    print(f"    {'N/class':>8} {'train_tok':>9} {'acc':>14} "
          f"{'prec_cc':>14} {'rec_cc':>14} {'f1_cc':>14}")

    rows = []
    for n in ns:
        accs, prec, rec, f1s, npx = [], [], [], [], []
        for r in range(n_repeats):
            rng = random.Random(1000 + r)
            selected = []
            for c in BIN_LABELS:                       # fixed class order -> reproducible
                fl = list(cls_fids[c]); rng.shuffle(fl)
                selected.extend(fl[:min(n, len(fl))])
            tr = np.isin(fids_str, selected)
            clf = make_model()
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            prec.append(precision_score(y[te], pred, pos_label=POS_ID, zero_division=0))
            rec.append(recall_score(y[te], pred, pos_label=POS_ID, zero_division=0))
            f1s.append(f1_score(y[te], pred, pos_label=POS_ID, zero_division=0))
            npx.append(int(tr.sum()))
        row = {
            "n_fids_per_class": n,
            "train_tokens_mean": float(np.mean(npx)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "prec_cc_mean": float(np.mean(prec)), "prec_cc_std": float(np.std(prec)),
            "rec_cc_mean": float(np.mean(rec)), "rec_cc_std": float(np.std(rec)),
            "f1_cc_mean": float(np.mean(f1s)), "f1_cc_std": float(np.std(f1s)),
        }
        rows.append(row)
        print(f"    {n:>8} {row['train_tokens_mean']:>9.0f} "
              f"{row['acc_mean']:>7.3f}±{row['acc_std']:<5.3f} "
              f"{row['prec_cc_mean']:>7.3f}±{row['prec_cc_std']:<5.3f} "
              f"{row['rec_cc_mean']:>7.3f}±{row['rec_cc_std']:<5.3f} "
              f"{row['f1_cc_mean']:>7.3f}±{row['f1_cc_std']:<5.3f}")
    return rows


def plot_learning_curve_binary(rows, model_name, out):
    x = [r["train_tokens_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, lbl in [("acc", "accuracy"),
                     ("prec_cc", "precision (clear-cut)"),
                     ("rec_cc", "recall (clear-cut)"),
                     ("f1_cc", "F1 (clear-cut)")]:
        m = [r[f"{key}_mean"] for r in rows]
        s = [r[f"{key}_std"] for r in rows]
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("train tokens (mean)"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_name} (binary: forest vs. clear-cut) - training-size sensitivity")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--embed-kind", default="joint",
                    help="embedding modality subdir: joint | s2_l2a | s1_grd")
    ap.add_argument("--tiles-root", default=os.path.expanduser("~/thesis_tiles_120px"))
    ap.add_argument("--shp", default="data_shp/label_polygons.shp")
    ap.add_argument("--forest-root", default=FOREST_ROOT)
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="repeats for the training-size sensitivity analysis")
    ap.add_argument("--fids", nargs="*", default=None,
                    help="test FIDs to map (default: first 4 test FIDs)")
    ap.add_argument("--results-root", default="results_binary")
    args = ap.parse_args()

    os.makedirs(args.results_root, exist_ok=True)

    # ---- load everything once (heavy) ----
    ds = DisturbanceSegDataset(args.embeddings_root, args.tiles_root, args.shp,
                               embed_kind=args.embed_kind)
    X, y, fids = build_pixel_dataset_forest(ds, args.forest_root)

    # ---- keep ONLY forest (0) and clear-cut bare soil (1) ----
    keep = np.isin(y, BIN_LABELS)
    X, y, fids = X[keep], y[keep], fids[keep]
    fids_str = fids.astype(str)
    print(f"[binary] kept {len(y)} tokens | forest={int((y == NEG_ID).sum())} "
          f"| clear-cut={int((y == POS_ID).sum())}", flush=True)

    # ---- fixed FID-level split (reuses the SAME split file as run_eval.py) ----
    fid_cls = {f: ds.fid_polys[f][0][1] for f in set(fids.tolist()) if f in ds.fid_polys}
    test_fids, trainval_fids = load_or_make_split(fid_cls)
    tr = np.isin(fids_str, list(trainval_fids))
    te = np.isin(fids_str, list(test_fids))
    tr_bal = balance_classes(y, tr, per_class=2500)      # train balanced; test kept full
    print(f"[data] train {int(tr_bal.sum())} tok | test {int(te.sum())} tok (full)", flush=True)

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

        # printed report on the full test set
        print(f"\n[{name}] accuracy (test): {accuracy_score(y[te], pred):.3f}")
        print(f"[{name}] precision (clear-cut): "
              f"{precision_score(y[te], pred, pos_label=POS_ID, zero_division=0):.3f}")
        print(classification_report(y[te], pred, labels=BIN_LABELS,
                                    target_names=BIN_NAMES, zero_division=0), flush=True)

        joblib.dump(clf, os.path.join(outdir, "model.joblib"))
        np.save(os.path.join(outdir, "pred.npy"), pred)
        np.save(os.path.join(outdir, "y_test.npy"), y[te])
        np.save(os.path.join(outdir, "test_fids.npy"), fids_str[te])

        plot_confusion_binary(y[te], pred, name, os.path.join(outdir, "confusion.png"))
        plot_prf_binary(y[te], pred, name, os.path.join(outdir, "prf.png"))
        ep.plot_fid_maps(ds, clf, map_fids, name, out=os.path.join(outdir, "fidmaps.png"))

        print(f"[{name}] learning curve...", flush=True)
        rows = learning_curve_binary(X, y, fids_str, trainval_fids, te,
                                     make_model=make, model_name=name,
                                     n_repeats=args.n_repeats)
        plot_learning_curve_binary(rows, name, os.path.join(outdir, "learning_curve.png"))
        print(f"[{name}] done -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
