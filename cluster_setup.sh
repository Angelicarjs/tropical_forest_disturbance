#!/bin/bash
# Run this ON the cluster (dmis) to set up the pipeline environment
# Usage: bash ~/thesis_scripts/cluster_setup.sh

set -e

echo "=== Setting up tile pipeline on cluster ==="

# Create working directories
mkdir -p ~/thesis_tiles
mkdir -p ~/thesis_scripts

# Create conda environment with required packages
echo "Creating conda environment 'gee_tiles'..."
conda create -n gee_tiles python=3.11 -y
conda activate gee_tiles

echo "Installing Python packages..."
pip install earthengine-api shapely pyproj

# Clean local cache to save home directory space (shared cache is in /share/common/anaconda/pkgs)
echo "Cleaning local conda cache..."
conda clean --all -y

# Authenticate GEE (will print a URL — open it in your browser)
echo ""
echo "=== GEE Authentication ==="
echo "This will print a URL. Copy it, open in your browser, log in,"
echo "and paste the authorization code back here."
echo ""
python -c "import ee; ee.Authenticate(auth_mode='paste')"

# Test GEE connection
echo ""
echo "=== Testing GEE connection ==="
python -c "
import ee
ee.Initialize(project='zinc-wares-316319')
fc = ee.FeatureCollection('projects/zinc-wares-316319/assets/images_polygons')
print('Asset accessible, features:', fc.size().getInfo())
print('GEE ready!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "This environment covers the download stage only. Embedding extraction and"
echo "everything downstream run in the separate 'croma_viz' environment, which"
echo "carries torch and torchgeo."
echo ""
echo "To run the tiling pipeline, from the repository root:"
echo "  conda activate gee_tiles"
echo "  export PYTHONPATH=\$PWD"
echo "  nohup python preprocessing/code/tile_pipeline.py --workers 6 --resume \\"
echo "        > preprocessing/logs/tile_pipeline.log 2>&1 &"
echo ""
echo "To monitor:"
echo "  tail -f preprocessing/logs/tile_pipeline.log"
