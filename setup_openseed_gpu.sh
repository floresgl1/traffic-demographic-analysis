#!/bin/bash
# Setup script for OpenSeed with GPU support
# Run this on a machine with NVIDIA GPU and CUDA

set -e

echo "========================================="
echo "OpenSeed GPU Setup for Traffic Analysis"
echo "========================================="
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are you sure you have an NVIDIA GPU?"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "CUDA version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found, defaulting to CUDA 11.8"
    CUDA_VERSION="11.8"
fi

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo ""
echo "[3/5] Installing PyTorch with CUDA support..."
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing for CUDA 12.x..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing for CUDA 11.x..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing default PyTorch (should work with most CUDA versions)..."
    pip install torch torchvision
fi

# Install base requirements
echo ""
echo "[4/5] Installing project requirements..."
pip install -r requirements.txt

# Install detectron2 with CUDA support
echo ""
echo "[5/5] Installing detectron2 with CUDA support..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo ""
echo "========================================="
echo "✓ Installation complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To verify GPU is working, run:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}\")'"
echo ""
echo "To test the OpenSeed installation, run:"
echo "  python scripts/test_openseed.py"
