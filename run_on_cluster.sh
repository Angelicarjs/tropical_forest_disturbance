#!/bin/bash
# Launch the tile pipeline on the IRISA cluster (dmis) with parallel workers.
#
# Usage (run from your local machine):
#   ./run_on_cluster.sh              # Upload scripts + data, launch with 6 workers
#   ./run_on_cluster.sh --resume     # Resume a previous run
#   ./run_on_cluster.sh --dry-run    # Preview what would be downloaded
#
# To run manually on the cluster:
#   ssh -J e2406749@cluster-irisa.univ-ubs.fr e2406749@dmis
#   eval "$(/share/common/anaconda/condabin/conda shell.bash hook)"
#   conda activate gee_tiles
#   nohup python ~/thesis_scripts/tile_pipeline.py --workers 6 --resume > ~/thesis_scripts/pipeline_output.log 2>&1 &

set -e

REMOTE_USER="e2406749"
JUMP_HOST="${REMOTE_USER}@cluster-irisa.univ-ubs.fr"
TARGET_HOST="${REMOTE_USER}@dmis"
LOCAL_SCRIPTS="/Users/angelicamariamorenorojas/Desktop/Master/thesis/scripts"
LOCAL_DATA="/Users/angelicamariamorenorojas/Desktop/Master/thesis/data"
REMOTE_SCRIPTS="~/thesis_scripts"

EXTRA_ARGS="${@}"
WORKERS=6  # 6 parallel workers — safe for GEE rate limits

echo "=== Step 1: Upload pipeline script + V2 CSV to cluster ==="
rsync -avz --progress \
    -e "ssh -J ${JUMP_HOST}" \
    "${LOCAL_SCRIPTS}/tile_pipeline.py" \
    "${LOCAL_DATA}/s1_s2_images_thesis_v2.csv" \
    "${TARGET_HOST}:${REMOTE_SCRIPTS}/"

echo ""
echo "=== Step 2: Launch pipeline on cluster ==="
ssh -J "$JUMP_HOST" "$TARGET_HOST" bash -l << REMOTE_EOF
    # Activate conda environment
    conda activate gee_tiles

    # Kill any existing pipeline processes
    pkill -f "tile_pipeline.py" 2>/dev/null || true
    sleep 1

    echo "Starting pipeline with ${WORKERS} workers..."
    nohup python ${REMOTE_SCRIPTS}/tile_pipeline.py \
        --workers ${WORKERS} \
        --resume \
        ${EXTRA_ARGS} \
        > ${REMOTE_SCRIPTS}/pipeline_output.log 2>&1 &

    PID=\$!
    echo "Pipeline started with PID \${PID}"
    echo "PID saved to ${REMOTE_SCRIPTS}/pipeline.pid"
    echo \${PID} > ${REMOTE_SCRIPTS}/pipeline.pid

    # Show first few lines of output
    sleep 3
    echo ""
    echo "=== Initial output ==="
    head -20 ${REMOTE_SCRIPTS}/pipeline_output.log 2>/dev/null || echo "(waiting for output...)"
REMOTE_EOF

echo ""
echo "=== Pipeline launched ==="
echo ""
echo "Monitor progress:"
echo "  ssh -J ${JUMP_HOST} ${TARGET_HOST} 'tail -f ~/thesis_scripts/pipeline_output.log'"
echo ""
echo "Check if running:"
echo "  ssh -J ${JUMP_HOST} ${TARGET_HOST} 'ps aux | grep tile_pipeline'"
echo ""
echo "Stop pipeline:"
echo "  ssh -J ${JUMP_HOST} ${TARGET_HOST} 'kill \$(cat ~/thesis_scripts/pipeline.pid)'"
echo ""
echo "Check disk usage:"
echo "  ssh -J ${JUMP_HOST} ${TARGET_HOST} 'du -sh ~/thesis_tiles/'"
