#!/bin/bash
#SBATCH --job-name=log_reg
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_log_reg_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1   # due to n_jobs parallelism in LogisticRegressionCV
export PYTHONWARNINGS="ignore::FutureWarning"  # silence sklearn 1.10 deprecation warnings (incl. worker processes)
echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"
python log_reg.py
echo "End: $(date)"