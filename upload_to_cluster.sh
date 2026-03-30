#!/bin/bash
# Upload tiles to the IRISA cluster (dmis) via SSH jump host
#
# Usage:
#   ./upload_to_cluster.sh                  # Upload entire tiles/ directory
#   ./upload_to_cluster.sh native           # Upload only native/ subdirectory
#   ./upload_to_cluster.sh croma            # Upload only croma/ subdirectory

TILES_DIR="/Users/angelicamariamorenorojas/Desktop/Master/thesis/tiles"
REMOTE_USER="e2406749"
JUMP_HOST="${REMOTE_USER}@cluster-irisa.univ-ubs.fr"
TARGET_HOST="${REMOTE_USER}@dmis"
REMOTE_DIR="~/thesis_tiles"

SUBDIR="${1:-}"

if [ -n "$SUBDIR" ]; then
    SOURCE="${TILES_DIR}/${SUBDIR}/"
    DEST="${REMOTE_DIR}/${SUBDIR}/"
else
    SOURCE="${TILES_DIR}/"
    DEST="${REMOTE_DIR}/"
fi

echo "Creating remote directory..."
ssh -J "$JUMP_HOST" "$TARGET_HOST" "mkdir -p ${DEST}"

echo "Uploading ${SOURCE} -> ${TARGET_HOST}:${DEST}"
echo "Using jump host: ${JUMP_HOST}"

rsync -avz --progress \
    -e "ssh -J ${JUMP_HOST}" \
    "${SOURCE}" \
    "${TARGET_HOST}:${DEST}"

echo "Upload complete."
echo "Remote path: ${DEST}"
