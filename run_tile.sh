#!/bin/bash
#SBATCH --job-name=tiles
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_tiles_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd /share/castor/home/e2406749/tropical_forest_disturbance

echo "Nodo: $(hostname) | Inicio: $(date)"
python tile_pipeline.py --sample-pct 10 --tile-size 224 --workers 6 --resume --products s2_l2a
echo "Fin: $(date)"
