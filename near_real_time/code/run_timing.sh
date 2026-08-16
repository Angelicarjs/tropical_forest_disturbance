#!/usr/bin/env bash
# Timeliness of the near real time rule, at the three operating points of the manuscript.
#
# Each mode gets its own threshold, as sec:workflow_nrt states: the same probability
# does not mean the same thing in two feature spaces, so a shared tau would compare
# the modes at different false alarm rates. nrt_timing.py takes a single --tau, hence
# one invocation per (operating point, mode).
#
# The taus come from the Youden point and the 5 % / 10 % false alarm points of the
# unmatched ROC curves of the TRAINING split, printed by nrt_threshold.ipynb.
#
# Run from the repository root.
set -euo pipefail
export PYTHONPATH="$PWD"

RES="near_real_time/results"

run () {   # run <outdir> <mode> <tau>
  python near_real_time/code/nrt_timing.py --tau "$3" --mode "$2" --out "$RES/$1"
}

#            outdir          mode      tau
run timing_fa05_optical      optical   0.970
run timing_fa05_joint        joint     0.972
run timing_fa10_optical      optical   0.943
run timing_fa10_joint        joint     0.948
run timing_youden_optical    optical   0.842
run timing_youden_joint      joint     0.851

echo
echo "summaries:"
cat "$RES"/timing_{fa05,fa10,youden}_{optical,joint}/timing_summary_log_reg_v3.csv \
  | awk 'NR==1 || !/^mode,/'
