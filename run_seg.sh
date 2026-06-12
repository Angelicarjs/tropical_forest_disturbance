#!/bin/bash
#SBATCH --job-name=train_seg
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_seg_%j.log

PROJ=/share/castor/home/e2406749/tropical_forest_disturbance

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

cd "$PROJ"
export PYTHONUNBUFFERED=1   # live logs (no buffering)

echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | Start: $(date)"

python train_seg.py \
    --tiles-root "$HOME/thesis_tiles_120px" \
    --epochs 40 \
    --batch-size 16

echo "Fin: $(date)"
