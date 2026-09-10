#!/bin/bash
#SBATCH --job-name=nrtscores
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=near_real_time/results/nrt_scores/log_%A_%a.log

# One array task per FID, same pattern as run_nrt.sh: the FIDs share no state.
# The FID list comes from a split file, so the threshold is never chosen on
# polygons the classifier was evaluated on.
#
# Run it, do not sbatch it. The first pass submits itself as an array sized to
# the split.
#
#   ./run_scores.sh                        # trainval, multiclass modes, both models
#   MODEL=log_reg ./run_scores.sh          # only the linear probe
#   MODE=all ./run_scores.sh               # add the binary label spaces
#   SPLIT=data/splits/split_test_fids.txt TAG=test ./run_scores.sh
#   THROTTLE=10 ./run_scores.sh            # at most 10 tasks at a time
#
# MODE and MODEL are passed unquoted on purpose: each accepts several values.

MODE=${MODE:-"joint optical"}
MODEL=${MODEL:-"log_reg rf"}
VERSION=${VERSION:-3}
SPLIT=${SPLIT:-data/splits/split_trainval_fids.txt}
TAG=${TAG:-}          # suffix for the output folder; set TAG=test with the test split

if [ -z "$SLURM_JOB_ID" ]; then                 # login node: submit, do not compute
    mapfile -t FIDS < <(grep -Eo '[0-9]+' "$SPLIT")
    if [ ${#FIDS[@]} -eq 0 ]; then
        echo "no FID found in $SPLIT"; exit 1
    fi
    mkdir -p near_real_time/results/nrt_scores          # slurm opens the log before the task runs
    RANGE="0-$((${#FIDS[@]} - 1))${THROTTLE:+%$THROTTLE}"
    echo "submitting ${#FIDS[@]} FIDs from $SPLIT as array $RANGE"
    echo "  mode=$MODE model=$MODEL cloud filter=v$VERSION tag=${TAG:-none}"
    exec sbatch --array="$RANGE" \
         --export=ALL,MODE="$MODE",MODEL="$MODEL",VERSION="$VERSION",TAG="$TAG" \
         "$0" "${FIDS[@]}"
fi

FIDS=("$@")                                     # the list handed over by the submit pass
FID=${FIDS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$FID" ]; then
    echo "no FID at array index $SLURM_ARRAY_TASK_ID: run ./run_scores.sh directly"
    exit 1
fi

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

# sbatch is launched from the repository root, and SLURM records that directory here,
# so the job runs from the root on any account. Falls back to $PWD outside SLURM.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONPATH="$PWD"   # shared modules (seg_dataset, token_pipeline, obs_date...) live at the repo root
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in RF
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Node: $(hostname) | FID: $FID | $MODEL | v$VERSION | Start: $(date)"
python near_real_time/code/nrt_scores.py --fid "$FID" --mode $MODE --model $MODEL \
       --version "$VERSION" --tag "$TAG" --out near_real_time/results/nrt_scores
echo "End: $(date)"
