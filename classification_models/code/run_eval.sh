#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/eval_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

mkdir -p results   # so slurm can write results/eval_%j.log

export PYTHONPATH="$PWD"   # shared modules (seg_dataset, token_pipeline, obs_date...) live at the repo root
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # matplotlib without GUI
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in RF / LogisticRegressionCV
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"
python classification_models/code/run_eval.py --align-with joint s2_l2a --fids 83 389 25 3
echo "End: $(date)"
