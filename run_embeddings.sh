#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_embed_%j.log

# --- paths (adjust if yours differ) ---
PROJ=/share/castor/home/e2406749/tropical_forest_disturbance
TILES=$HOME/thesis_tiles_120px
CSV=$PROJ/data_csv

# --- conda ---
source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd "$PROJ"

echo "Nodo: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | Inicio: $(date)"

# TEST: only the 41-FID stratified sample. Remove --fid-list to run on ALL FIDs.
python embed_all_joint.py \
    --tiles-root "$TILES" \
    --csv-dir "$CSV" \
    --fid-list "$CSV/sample_10pct_stratified.txt"

echo "Fin: $(date)"
