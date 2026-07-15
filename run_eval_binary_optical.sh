#!/bin/bash
#SBATCH --job-name=eval_bin_opt
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results_binary_optical/eval_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

mkdir -p results_binary_optical   # so slurm can write results_binary_optical/eval_%j.log

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # matplotlib without GUI
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in RF / LogisticRegressionCV
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"
python run_eval_binary.py \
    --embed-kind s2_l2a \
    --forest-root embeddings/s2_l2a_forest \
    --results-root results_binary_optical
echo "End: $(date)"
