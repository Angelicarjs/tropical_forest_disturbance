#!/bin/bash
#SBATCH --job-name=tiles
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_tiles_%j.log

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate croma_viz

# sbatch is launched from the repository root, and SLURM records that directory here,
# so the job runs from the root on any account. Falls back to $PWD outside SLURM.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"   # shared modules live at the repo root

echo "Nodo: $(hostname) | Inicio: $(date)"
python preprocessing/code/tile_pipeline.py --workers 6 --resume
echo "Fin: $(date)"
