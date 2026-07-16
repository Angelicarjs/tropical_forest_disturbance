"""
Shared plotting helpers for the visual evaluation of the per-token classifiers.

Used both by run_eval.py (saves PNGs, backend Agg) and by eval_results.ipynb
(renders inline). Functions take an optional `out` path (save) and `show` flag
(display inline); the script passes out=..., the notebook passes show=True.
"""
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from seg_dataset import NUM_CLASSES
from rand_forest import ID_TO_CLASS, majority_downsample

# fixed color per model class id (0..6); 0 = forest for the MODEL
CLASS_COLORS = {
    0: "#1b7837",   # forest
    1: "#a6611a",   # clear-cut bare soil
    2: "#d7191c",   # fire scar
    3: "#fdae61",   # degradation
    4: "#7b3294",   # disorderly selective logging
    5: "#a6d96a",   # clear-cut vegetation
    6: "#2c7bb6",   # geometric selective logging
}
CMAP = ListedColormap([CLASS_COLORS[c] for c in range(NUM_CLASSES)])
NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES, 1), CMAP.N)
NAMES = [ID_TO_CLASS[c] for c in range(NUM_CLASSES)]
LABELS = list(range(NUM_CLASSES))

# maps only: dedicated 'background' id, distinct from the model's forest class.
# GT outside the polygon is unlabeled background; the model has no background class.
MAP_BG_ID = NUM_CLASSES
MAP_COLORS = [CLASS_COLORS[c] for c in range(NUM_CLASSES)] + ["#d9d9d9"]  # background = grey
MAP_NAMES = NAMES + ["background (unlabeled)"]
MAP_CMAP = ListedColormap(MAP_COLORS)
MAP_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 1, 1), MAP_CMAP.N)


def _finish(fig, out, show):
    """Save to `out` if given, display if `show`, otherwise close to free memory."""
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion(y_true, y_pred, model_name, out=None, show=False):
    """Two heatmaps: row-normalized (recall) and column-normalized (precision)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, axis, title in [(axes[0], 1, "row-normalized (recall)"),
                            (axes[1], 0, "column-normalized (precision)")]:
        denom = cm.sum(axis=axis, keepdims=True)
        cmn = np.zeros_like(cm, dtype=float)
        np.divide(cm, denom, out=cmn, where=denom != 0)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(LABELS); ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(LABELS); ax.set_yticklabels(NAMES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{model_name} - {title}", fontsize=10)
        for i in LABELS:
            for j in LABELS:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _finish(fig, out, show)


def plot_prf(y_true, y_pred, model_name, out=None, show=False):
    """Per-class precision / recall / F1 bars; headline macro over disturbance (1..6)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    x = np.arange(len(LABELS)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, p, w, label="precision", color="tab:blue")
    ax.bar(x,     r, w, label="recall",    color="tab:orange")
    ax.bar(x + w, f1, w, label="F1",       color="tab:green")
    for arr, dx, col in [(p, -w, "tab:blue"), (r, 0, "tab:orange"), (f1, w, "tab:green")]:
        for xi, v in zip(x + dx, arr):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=col)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    macro = (p[1:].mean(), r[1:].mean(), f1[1:].mean())   # disturbance classes only
    ax.set_title(f"{model_name} - per-class scores (full test)  |  "
                 f"macro(1-6): P={macro[0]:.2f}  R={macro[1]:.2f}  F1={macro[2]:.2f}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    _finish(fig, out, show)


def _load_rgb(ds, s, bands=(4, 3, 2), p_lo=2, p_hi=98):
    """True-color RGB (B4/B3/B2) for a tile, percentile-stretched to [0, 1].

    Band order in the s2_l2a tiles is the CROMA order
    [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12] (1-based), so
    R=B4 (band 4), G=B3 (band 3), B=B2 (band 2). Returns None if the tif
    is missing (e.g. embeddings present but tiles not synced).
    """
    tif = (ds.tiles_root / "s2_l2a" / f"fid_{s['fid']}"
           / s["window"] / s["s2_id"] / f"{s['tile']}.tif")
    if not tif.exists():
        return None
    with rasterio.open(tif) as src:
        rgb = np.stack([src.read(b).astype(np.float32) for b in bands], axis=-1)  # (H,W,3)
    lo, hi = np.percentile(rgb, [p_lo, p_hi])
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def _fmt_date(s2_id):
    """Extract YYYYMMDD from a scene id like '20230515T142711_...' -> '2023-05-15'."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})", s2_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else s2_id


def _outline_polygon(ax, fine, gh, gw):
    """Draw the true polygon boundary (from the fine 120x120 mask) on top of a
    gh x gw token-grid axis, aligned with imshow cell centers."""
    if not (fine > 0).any():
        return
    H, W = fine.shape
    Xg = np.linspace(-0.5, gw - 0.5, W)          # map the 120 px onto the token axis
    Yg = np.linspace(-0.5, gh - 0.5, H)
    ax.contour(Xg, Yg, (fine > 0).astype(float), levels=[0.5],
               colors="red", linewidths=1.0)


def plot_fid_maps(ds, clf, fids, model_name, out=None, show=False,
                  max_tiles_per_fid=4, outline_polygons=True):
    """For each FID: RGB image vs ground-truth vs predicted class grid, one row per tile.

    Ground truth uses a dedicated 'background (unlabeled)' color for everything
    outside the disturbance polygon, kept distinct from the model's 'forest' class.
    When outline_polygons is True, the true polygon boundary is drawn (red) on top
    of the ground-truth and predicted panels.
    """
    rows = []
    for fid in fids:
        samples = [s for s in ds.samples if s["fid"] == fid][:max_tiles_per_fid]
        rows += [(fid, s) for s in samples]
    if not rows:
        print("[maps] no samples for the requested FIDs")
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(8.2, 2.7 * n), squeeze=False)
    for k, (fid, s) in enumerate(rows):
        emb = np.load(s["npy"])                          # (gh, gw, 768)
        gh, gw, _ = emb.shape
        pred = clf.predict(emb.reshape(-1, 768)).reshape(gh, gw)   # 0=forest .. 6
        fine = ds._build_mask(s)                         # 120x120 rasterized polygons
        gt = majority_downsample(fine, gh, gw, NUM_CLASSES)
        gt_disp = np.where(gt == 0, MAP_BG_ID, gt)       # bg (outside polygon) -> grey

        rgb = _load_rgb(ds, s)                           # (H, W, 3) or None
        if rgb is not None:
            axes[k][0].imshow(rgb, interpolation="nearest")
        else:
            axes[k][0].text(0.5, 0.5, "no RGB", ha="center", va="center", fontsize=8)
        dist = "/".join(sorted({ID_TO_CLASS[c] for _, c in ds.fid_polys.get(fid, [])})) or "?"
        axes[k][0].set_title(
            f"fid {fid} · {s['tile']} · {_fmt_date(s['s2_id'])} · {s['window']}\n{dist}",
            fontsize=7)
        axes[k][1].imshow(gt_disp, cmap=MAP_CMAP, norm=MAP_NORM, interpolation="nearest")
        axes[k][2].imshow(pred, cmap=MAP_CMAP, norm=MAP_NORM, interpolation="nearest")
        if outline_polygons:
            _outline_polygon(axes[k][1], fine, gh, gw)
            _outline_polygon(axes[k][2], fine, gh, gw)
        axes[k][1].set_title("ground truth", fontsize=8)
        axes[k][2].set_title(model_name, fontsize=8)
        for ax in axes[k]:
            ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=MAP_COLORS[c]) for c in range(len(MAP_NAMES))]
    fig.legend(handles, MAP_NAMES, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _finish(fig, out, show)


def plot_training_size_sensitivity(rows, model_name, out=None, show=False):
    """Plot the rows returned by rand_forest.training_size_sensitivity()."""
    x = [r["train_tokens_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, lbl in [("acc", "accuracy"),
                     ("prec_dist", "macro-precision (disturbance 1-6)"),
                     ("prec_dist_w", "weighted-precision (disturbance 1-6)"),
                     ("f1_dist", "macro-F1 (disturbance 1-6)"),
                     ("f1_all", "macro-F1 (all)")]:
        m = [r[f"{key}_mean"] for r in rows]
        s = [r[f"{key}_std"] for r in rows]
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("train tokens (mean)"); ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_name} - sensitivity to training-set size")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, out, show)


def plot_token_distribution(y, tr, tr_bal, te, cap=2500, out=None, show=False):
    """Tokens per class: train full vs train balanced vs test (full, not subsampled)."""
    n_tr_before = [int((y[tr] == c).sum()) for c in LABELS]
    n_tr_after = [int((y[tr_bal] == c).sum()) for c in LABELS]
    n_te = [int((y[te] == c).sum()) for c in LABELS]
    x = np.arange(len(LABELS)); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, n_tr_before, w, label="train full", color="tab:blue")
    b2 = ax.bar(x,     n_tr_after,  w, label="train balanced", color="tab:green")
    b3 = ax.bar(x + w, n_te,        w, label="test (full)", color="tab:orange")
    ax.axhline(cap, color="red", ls="--", lw=1.2, label=f"{cap} cap")
    for b in (b1, b2, b3):
        ax.bar_label(b, fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_ylabel("tokens"); ax.set_title("Token distribution per class")
    ax.legend()
    fig.tight_layout()
    _finish(fig, out, show)
