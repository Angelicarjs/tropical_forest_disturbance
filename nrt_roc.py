"""Choose the probability threshold of the near real time rule.

The rule flags an acquisition when p_dist >= tau and confirms it when a second
acquisition within X days is also above tau. This script fixes tau.

The label of an acquisition comes from the window it was retrieved in: `bef` is
intact forest (negative) and `aft` is disturbed (positive). The event window is
dropped, since it spans the reference date and its acquisitions fall on both
sides of the disturbance.

Two caveats are worth stating wherever these numbers are reported. The labels
assume the disturbance happened on VIEW_DATE, so an early detection in the
before window is counted as a false alarm; the false alarm rate measured here is
therefore an upper bound. And the classes are imbalanced, roughly six before
acquisitions per after one, which is why the precision recall curve is shown
next to the ROC.

Usage:
    python nrt_roc.py --model log_reg
    python nrt_roc.py --model log_reg --fpr 0.05
    python nrt_roc.py --model rf --mode joint optical --version 3
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

SCORES = "nrt_scores"
# Color carries the modality and line style the label space, as in nrt_fid.py.
STYLE = {
    "joint":          ("#2a78d6", "-"),
    "joint_binary":   ("#2a78d6", "--"),
    "optical":        ("#eb6834", "-"),
    "optical_binary": ("#eb6834", "--"),
}


def load(mode, model, version, buffer=0):
    """Pool the per-FID score files of one (mode, model, version).

    `buffer` drops the before acquisitions that fall within that many days of
    VIEW_DATE. The disturbance may well have occurred during the before window,
    since VIEW_DATE is a detection date and not an occurrence date, so the
    acquisitions closest to it are the least trustworthy negatives.
    """
    files = sorted(glob.glob(f"{SCORES}/{mode}_{model}_v{version}/fid_*.csv"))
    if not files:
        return None
    d = pd.concat(map(pd.read_csv, files), ignore_index=True)
    d = d[d.win != "evt"]                      # straddles the reference date
    if buffer:
        if "days" not in d.columns:
            raise SystemExit("the score files carry no 'days' column: "
                             "re-run nrt_scores.py to use --buffer")
        n = len(d)
        d = d[(d.win == "aft") | (d.days <= -buffer)]
        print(f"[{mode}] buffer {buffer} d: {n - len(d)} before acquisitions dropped")
    d["y"] = (d.win == "aft").astype(int)
    return d


def pick_tau(fpr, tpr, thr, target_fpr=None):
    """Youden by default: the threshold furthest from the diagonal."""
    if target_fpr is None:
        i = int(np.argmax(tpr - fpr))
    else:
        i = int(np.searchsorted(fpr, target_fpr))
        i = min(i, len(thr) - 1)
    return i, float(thr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["all"])
    ap.add_argument("--model", default="log_reg")
    ap.add_argument("--version", default="3", choices=["1", "2", "3"])
    ap.add_argument("--fpr", type=float, default=None,
                    help="fix the false alarm rate instead of using Youden")
    ap.add_argument("--buffer", type=int, default=0, metavar="DAYS",
                    help="drop before acquisitions within DAYS of VIEW_DATE, where "
                         "the disturbance may already have happened (default: 0)")
    ap.add_argument("--out", default=".", help="directory for the figure and the table")
    args = ap.parse_args()

    modes = list(STYLE) if "all" in args.mode else args.mode

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.bbox": "tight"})

    text_w = 160 / 25.4
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(text_w, text_w * 0.48))
    rows = []

    for mode in modes:
        d = load(mode, args.model, args.version, args.buffer)
        if d is None or d.y.nunique() < 2:
            print(f"[skip] {mode}: no scores, or only one class present")
            continue

        fpr, tpr, thr = roc_curve(d.y, d.p_dist)
        i, tau = pick_tau(fpr, tpr, thr, args.fpr)
        prec, rec, _ = precision_recall_curve(d.y, d.p_dist)

        color, ls = STYLE.get(mode, ("0.4", "-"))
        ax_roc.plot(fpr, tpr, color=color, ls=ls, label=mode.replace("_", " "))
        ax_roc.plot(fpr[i], tpr[i], "o", color=color, ms=5)
        ax_pr.plot(rec, prec, color=color, ls=ls, label=mode.replace("_", " "))

        rows.append({
            "mode": mode, "model": args.model, "version": args.version,
            "buffer_days": args.buffer,
            "polygons": int(d.fid.nunique()), "acquisitions": int(len(d)),
            "positives": round(float(d.y.mean()), 3),
            "auc": round(float(roc_auc_score(d.y, d.p_dist)), 3),
            "avg_precision": round(float(average_precision_score(d.y, d.p_dist)), 3),
            "tau": round(tau, 3),
            "detection": round(float(tpr[i]), 3),
            "false_alarm": round(float(fpr[i]), 3),
        })

    if not rows:
        raise SystemExit(f"nothing to plot: is {SCORES}/ populated?")

    ax_roc.plot([0, 1], [0, 1], ls=":", lw=0.8, color="0.6")
    ax_roc.set_xlabel("False alarm rate")
    ax_roc.set_ylabel("Detection rate")
    ax_roc.set_aspect("equal")
    ax_roc.legend(frameon=False, loc="lower right")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_aspect("equal")

    os.makedirs(args.out, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig_path = os.path.join(args.out, f"roc_pr_{args.model}_v{args.version}.pdf")
    fig.savefig(fig_path)

    table = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, f"thresholds_{args.model}_v{args.version}.csv")
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))
    print(f"\nsaved {fig_path}\nsaved {csv_path}")


if __name__ == "__main__":
    main()
