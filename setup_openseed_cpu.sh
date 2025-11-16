#!/bin/bash
# Setup script for OpenSeed on CPU
# Run this to install all dependencies

set -e

echo "========================================="
echo "OpenSeed CPU Setup for Traffic Analysis"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Create virtual environment (recommended)
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

# Install PyTorch CPU version
echo ""
echo "[3/5] Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install base requirements
echo ""
echo "[4/5] Installing project requirements..."
pip install -r requirements.txt

# Install detectron2 (CPU version)
echo ""
echo "[5/5] Installing detectron2 (CPU)..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo ""
echo "========================================="
echo "✓ Installation complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python scripts/test_openseed.py"
echo ""
echo "NOTE: CPU inference will be slow. For production use,"
echo "consider running on a machine with GPU."
