#!/bin/bash
#SBATCH --job-name=nrt
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results_nrt/nrt_%A_%a.log

# One array task per FID: the FIDs share no state, so they run side by side
# instead of looping inside a single job.
#
# Run it, do not sbatch it. The first pass submits itself as an array sized to
# the FID list, so no range is ever kept in sync by hand:
#
#   ./run_nrt.sh                 # the FIDS list below
#   ./run_nrt.sh 28 63           # ad-hoc list
#   THROTTLE=2 ./run_nrt.sh      # same list, at most 2 tasks at a time

FIDS=(28 214 389 25)

if [ -z "$SLURM_JOB_ID" ]; then                 # login node: submit, do not compute
    [ $# -gt 0 ] && FIDS=("$@")
    mkdir -p results_nrt                        # slurm opens the log before the task runs
    RANGE="0-$((${#FIDS[@]} - 1))${THROTTLE:+%$THROTTLE}"
    echo "submitting ${#FIDS[@]} FIDs (${FIDS[*]}) as array $RANGE"
    exec sbatch --array="$RANGE" "$0" "${FIDS[@]}"
fi

FIDS=("$@")                                     # the list handed over by the submit pass
FID=${FIDS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$FID" ]; then
    echo "no FID at array index $SLURM_ARRAY_TASK_ID: run ./run_nrt.sh directly, it submits itself"
    exit 1
fi

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # matplotlib without GUI
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in RF
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | FID: $FID | Start: $(date)"
python nrt_fid.py --fid "$FID" --mode all --model all --save results_nrt
echo "End: $(date)"
