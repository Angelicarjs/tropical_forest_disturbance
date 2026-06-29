#!/bin/bash
#SBATCH --job-name=rand_forest
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_rand_forest_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

export PYTHONUNBUFFERED=1 #does not accumulate output in buffer, but prints it immediately
export MPLBACKEND=Agg #use of matplotlib without GUI
export OMP_NUM_THREADS=1 # due to jobs in RF

echo "Node: $(hostname) | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"
python rand_forest.py
echo "End: $(date)" 