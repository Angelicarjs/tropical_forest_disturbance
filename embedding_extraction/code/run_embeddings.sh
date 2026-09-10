#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_embed_%j.log

# --- paths (adjust if yours differ) ---
PROJ="${SLURM_SUBMIT_DIR:-$PWD}"   # sbatch is launched from the repository root
TILES=$HOME/thesis_tiles_120px
CSV=$PROJ/data/data_csv

# --- conda ---
source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd "$PROJ"
export PYTHONPATH="$PWD"   # shared modules live at the repo root

echo "Nodo: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | Inicio: $(date)"

python embedding_extraction/code/embed_all.py \
    --tiles-root "$TILES" \
    --csv-dir "$CSV" \
    --modality "optical" \

echo "Fin: $(date)"
