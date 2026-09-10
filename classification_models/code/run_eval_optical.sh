#!/bin/bash
#SBATCH --job-name=eval_opt
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=classification_models/results/results_optical/eval_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

# sbatch is launched from the repository root, and SLURM records that directory here,
# so the job runs from the root on any account. Falls back to $PWD outside SLURM.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p classification_models/results/results_optical   # so slurm can write the log there

export PYTHONPATH="$PWD"   # shared modules (seg_dataset, token_pipeline, obs_date...) live at the repo root
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # matplotlib without GUI
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in RF / LogisticRegressionCV
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"
python classification_models/code/run_eval.py \
    --embed-kind s2_l2a \
    --forest-root embeddings/s2_l2a_forest \
    --results-root classification_models/results/results_optical \
    --align-with joint s2_l2a \
    --fids 83 389 25 3
echo "End: $(date)"
