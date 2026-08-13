"""Apply the near real time rule to each polygon and measure its timeliness.

The rule, following Reiche et al. (2021): read the series of a polygon in
chronological order, flag the first acquisition whose score reaches the
threshold, and confirm the flag when a second acquisition within `confirm_days`
also reaches it. The date reported is that of the first flagged acquisition, not
of the confirming one, so confirmation costs confidence but not timeliness.

The threshold comes from the training split, chosen in nrt_threshold.ipynb, and is
passed in here, never recomputed.

Delay is measured against VIEW_DATE, the date DETER recorded the event. A
negative delay means the alert precedes it. VIEW_DATE is a detection date and
not an occurrence date, so a negative delay is not by itself proof of an early
detection; it only shows the alert came first.

Usage:
    python nrt_timing.py --tau 0.842
    python nrt_timing.py --tau 0.842 --mode joint --model rf --confirm-days 28
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SCORES = "nrt_scores"


def load(mode, model, version, split_file, tag=""):
    name = f"{mode}_{model}_v{version}"
    folder = f"{name}_{tag}" if tag else name
    files = sorted(glob.glob(f"{SCORES}/{folder}/fid_*.csv"))
    if not files:
        raise SystemExit(f"no scores in {SCORES}/{folder}/")
    d = pd.concat(map(pd.read_csv, files), ignore_index=True)
    if split_file:
        import re
        keep = {int(x) for x in re.findall(r"[0-9]+", open(split_file).read())}
        d = d[d.fid.isin(keep)]
    # fid_curve converts the date before writing, so the CSV carries it as ISO
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["fid", "date"])


def first_alert(series, tau, confirm_days):
    """(alert_date, confirmed) for one polygon, or (None, False) if never flagged."""
    over = series[series.p_dist >= tau]
    if over.empty:
        return None, False
    for _, row in over.iterrows():
        later = over[over.date > row.date]
        window = later[(later.date - row.date).dt.days <= confirm_days]
        if not window.empty:
            return row.date, True
    # flagged but never confirmed: report the first flag, marked unconfirmed
    return over.iloc[0].date, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True,
                    help="threshold fixed on the training split")
    ap.add_argument("--mode", nargs="+", default=["optical", "joint"])
    ap.add_argument("--model", default="log_reg")
    ap.add_argument("--version", default="3")
    ap.add_argument("--confirm-days", type=int, default=90)
    ap.add_argument("--split", default="data/splits/split_test_fids.txt",
                    help="restrict to these polygons; empty string for all")
    ap.add_argument("--tag", default="test",
                    help="suffix of the score folders, as passed to nrt_scores.py")
    ap.add_argument("--match", action="store_true",
                    help="keep only the acquisitions present in every mode, so the "
                         "comparison isolates the representation instead of the "
                         "number of dates each mode has")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = {m: load(m, args.model, args.version, args.split, args.tag)
            for m in args.mode}

    if args.match and len(data) > 1:
        # Matched on the Sentinel-2 image, not on the date: a joint pair is dated by
        # the later of its two acquisitions, so the same optical image carries a
        # different date in the two modes and a join on dates would miss most pairs.
        shared = set.intersection(*(set(zip(x.fid, x.s2_id)) for x in data.values()))
        for m, x in data.items():
            kept = x[[k in shared for k in zip(x.fid, x.s2_id)]]
            print(f"[{m}] matched: {len(kept)} of {len(x)} acquisitions, "
                  f"{kept.fid.nunique()} polygons")
            data[m] = kept

    summary = []
    for mode, d in data.items():
        rows = []
        for fid, g in d.groupby("fid"):
            view = pd.to_datetime(g.view_date.iloc[0])
            date, confirmed = first_alert(g, args.tau, args.confirm_days)
            rows.append({
                "fid": fid, "mode": mode, "model": args.model,
                "n_acq": len(g), "view_date": view.date(),
                "alert_date": None if date is None else date.date(),
                "confirmed": confirmed,
                "delay_days": None if date is None else (date - view).days,
            })
        per_fid = pd.DataFrame(rows)
        dest = os.path.join(args.out, f"timing_{mode}_{args.model}_v{args.version}.csv")
        per_fid.to_csv(dest, index=False)

        conf = per_fid[per_fid.confirmed]
        delays = conf.delay_days.dropna().astype(int)
        summary.append({
            "mode": mode, "model": args.model, "tau": args.tau,
            "confirm_days": args.confirm_days,
            "polygons": len(per_fid),
            "confirmed": len(conf),
            "confirmed_pct": round(len(conf) / len(per_fid), 3),
            "delay_median": None if delays.empty else int(np.median(delays)),
            "delay_q25": None if delays.empty else int(np.percentile(delays, 25)),
            "delay_q75": None if delays.empty else int(np.percentile(delays, 75)),
            "before_view_pct": None if delays.empty else round((delays < 0).mean(), 3),
        })
        print(f"saved {dest}")

    table = pd.DataFrame(summary)
    dest = os.path.join(args.out, f"timing_summary_{args.model}_v{args.version}.csv")
    table.to_csv(dest, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved {dest}")
    print("\ndelay_days is measured against VIEW_DATE; negative means the alert came first.")


if __name__ == "__main__":
    main()
