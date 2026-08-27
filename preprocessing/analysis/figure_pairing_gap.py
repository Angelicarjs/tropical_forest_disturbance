"""Distribution of the day gap between each Sentinel-2 acquisition and its closest
Sentinel-1 scene, with the median, the 90th percentile and the adopted 7 day threshold.

Same computation as the "Pairing analysis" cell of histogram_day_gap.ipynb, exported as
a publication figure for the manuscript.
"""

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # same style as the other manuscript figures
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except ImportError:  # fall back to an equivalent inline style
    plt.rcParams.update({
        "font.family": "serif",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.8,
    })

DATA_DIR = os.path.expanduser(
    "~/Desktop/Master/thesis/tropical_forest_disturbance/data/data_csv"
)
FIGDIR = os.path.expanduser(
    "~/Desktop/Master/thesis/manuscript/manuscript_tropical_forest_disturbance/figures"
)
CSVS = ["v1_images_s2_s1.csv", "v2_images_s2_s1.csv", "v3_images_s2_s1.csv"]
WINDOWS = [("befDatS2", "befDatS1"), ("evtDatS2", "evtDatS1"), ("aftDatS2", "aftDatS1")]
THRESHOLD = 7

TEXT_W = 160 / 25.4  # text width of the manuscript, in inches
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
})


def parse_doy_dates(cell):
    """Dates in the CSVs are stored as "YYYY-?-DOY" (year, unused, day of year)."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    dates = []
    for entry in str(cell).split(","):
        parts = entry.strip().split("-")
        try:
            year, doy = int(parts[0]), int(parts[2])
            dates.append(datetime(year, 1, 1) + timedelta(days=doy - 1))
        except (IndexError, ValueError):
            pass
    return dates


def collect_gaps():
    """For every Sentinel-2 acquisition, the gap to the closest Sentinel-1 scene."""
    gaps = []
    for name in CSVS:
        df = pd.read_csv(os.path.join(DATA_DIR, name))
        for _, row in df.iterrows():
            for col_s2, col_s1 in WINDOWS:
                if col_s2 not in df.columns or col_s1 not in df.columns:
                    continue
                s1_dates = parse_doy_dates(row[col_s1])
                if not s1_dates:
                    continue
                for s2 in parse_doy_dates(row[col_s2]):
                    gaps.append(min(abs((s2 - s1).days) for s1 in s1_dates))
    return np.array(gaps)


gaps = collect_gaps()
median = int(np.median(gaps))
p90 = int(np.percentile(gaps, 90))
retained = 100 * (gaps <= THRESHOLD).mean()
print(f"n = {len(gaps)} pairs | median {median} d | P90 {p90} d | max {gaps.max()} d")
print(f"within {THRESHOLD} days: {retained:.1f}%")

edges = np.arange(0, gaps.max() + 2) - 0.5
counts, _ = np.histogram(gaps, bins=edges)
centers = np.arange(0, gaps.max() + 1)

fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 0.5))
ax.bar(centers, counts, width=0.85, color="steelblue")
ax.set_xlabel("Sentinel-1 to Sentinel-2 gap (days)")
ax.set_ylabel("Number of pairs")
ax.set_xticks(centers)
ax.set_xlim(-0.7, gaps.max() + 0.7)
ax.set_ylim(0, counts.max() * 1.20)

# the three markers sit close together, so each label is offset away from its neighbor
for x, color, label, ha, dx in ((median, "darkorange", f"Median = {median} d", "right", -5),
                                (p90, "firebrick", f"P90 = {p90} d", "right", -5),
                                (THRESHOLD, "black", f"Threshold = {THRESHOLD} d", "left", 5)):
    ax.axvline(x, color=color, ls="--" if x == THRESHOLD else ":", lw=1.2)
    ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(dx, -6), textcoords="offset points",
                ha=ha, va="top", color=color, fontsize=9)

out = os.path.join(FIGDIR, "pairing_gap_distribution.pdf")
fig.savefig(out)
print("written", out)
