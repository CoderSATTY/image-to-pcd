#!/bin/bash
# Depth Anything V2 - Setup Script
# Downloads model weights and sets up all dependencies

set -e

echo "=========================================="
echo "  Depth Anything V2 - Setup"
echo "=========================================="

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy opencv-python Pillow tqdm open3d

# Download model weights
echo "[2/3] Downloading model weights (~1.3GB)..."
mkdir -p checkpoints
if [ ! -f "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth" ]; then
    wget -q --show-progress -O checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true"
else
    echo "Model already downloaded."
fi

# Clone model code
echo "[3/3] Setting up model code..."
if [ ! -d "depth_anything_v2" ]; then
    git clone --depth 1 -q https://github.com/DepthAnything/Depth-Anything-V2.git temp_da2
    cp -r temp_da2/depth_anything_v2 ./
    cp -r temp_da2/metric_depth/depth_anything_v2/* ./depth_anything_v2/
    rm -rf temp_da2
else
    echo "Model code already exists."
fi

# Create directories
mkdir -p input output

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  1. Put images in ./input/"
echo "  2. Run: python depth_to_pointcloud.py \\"
echo "       --encoder vitl \\"
echo "       --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \\"
echo "       --max-depth 20 \\"
echo "       --img-path ./input \\"
echo "       --outdir ./output"
echo ""
