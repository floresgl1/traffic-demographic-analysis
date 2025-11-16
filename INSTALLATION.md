# OpenSeed Installation Guide

This guide will help you set up the OpenSeed model for vehicle detection on your local machine.

## System Requirements

### Minimum Requirements (CPU)
- Python 3.8 or higher
- 8GB RAM
- 10GB disk space
- Linux, macOS, or Windows

### Recommended Requirements (GPU)
- Python 3.8 or higher
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- CUDA 11.7+ or 12.1+
- 16GB RAM
- 20GB disk space
- Linux or Windows with WSL2

## Quick Start

### Option A: Automated Setup (Linux/macOS)

#### For CPU-only:
```bash
chmod +x setup_openseed_cpu.sh
./setup_openseed_cpu.sh
```

#### For GPU:
```bash
chmod +x setup_openseed_gpu.sh
./setup_openseed_gpu.sh
```

After installation:
```bash
# Activate the virtual environment
source venv/bin/activate

# Test the installation
python scripts/test_openseed.py

# Try the demo
python scripts/demo_openseed.py --sample
```

### Option B: Manual Setup

#### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

#### Step 2: Install PyTorch

Choose based on your hardware:

**CPU Only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU with CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU with CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Install Detectron2

This is the most critical step. Detectron2 can be tricky to install.

**Option 1: From source (recommended)**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Option 2: Pre-built wheels (Linux only)**
```bash
# For CUDA 11.8 and PyTorch 1.13
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html

# For CUDA 12.1 and PyTorch 2.0
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
```

**Option 3: Conda (alternative)**
```bash
conda install -c conda-forge detectron2
```

Verify detectron2 installation:
```bash
python -c "import detectron2; print(f'Detectron2 {detectron2.__version__}')"
```

#### Step 5: Test Installation

```bash
python scripts/test_openseed.py
```

This will:
1. Check all package imports
2. Test the OpenSeed analyzer class
3. (Optional) Download and initialize the model
4. (Optional) Run a simple inference test

## Alternative Setup Options

### Option C: Google Colab (No Local Setup Required)

Perfect for testing without local setup:

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Run the following:

```python
# Install dependencies
!pip install transformers timm opencv-python pillow
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/traffic-demographic-analysis.git
%cd traffic-demographic-analysis

# Test it
!python scripts/demo_openseed.py --sample
```

### Option D: Docker (Advanced)

Coming soon - we'll provide a Dockerfile for easy deployment.

### Option E: Conda Environment

```bash
# Create conda environment
conda create -n traffic-analysis python=3.10
conda activate traffic-analysis

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install detectron2
conda install -c conda-forge detectron2
# OR
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Troubleshooting

### Issue: Detectron2 Won't Install

**Solution 1: Install build dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
```

**Solution 2: Try a specific version**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

**Solution 3: Use conda**
```bash
conda install -c conda-forge detectron2
```

### Issue: CUDA Out of Memory

**Solution 1: Use CPU instead**
```python
analyzer = OpenSeedTrafficAnalyzer(device='cpu')
```

**Solution 2: Reduce image size**
```python
import cv2
img = cv2.imread('large_image.jpg')
img = cv2.resize(img, (960, 540))  # Half resolution
```

**Solution 3: Clear CUDA cache**
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Model Download Fails

**Solution 1: Check internet connection**

**Solution 2: Use a proxy**
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

**Solution 3: Download manually**
```python
from transformers import AutoProcessor, AutoModelForUniversalSegmentation

# This will download to default cache
processor = AutoProcessor.from_pretrained("facebook/openseed-vitl")
model = AutoModelForUniversalSegmentation.from_pretrained("facebook/openseed-vitl")
```

### Issue: Slow Inference on CPU

This is expected. OpenSeed is a large model:
- CPU inference: ~2-5 seconds per image
- GPU inference: ~100-200ms per image

**Recommendations:**
1. Use a GPU if possible
2. Process fewer frames (e.g., 1 FPS instead of 30 FPS)
3. Consider using a lighter model like YOLOv8 for real-time needs

### Issue: Import Errors

**Check Python version:**
```bash
python --version  # Should be 3.8+
```

**Reinstall in clean environment:**
```bash
# Remove old environment
rm -rf venv

# Create fresh environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Reinstall everything
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Performance Benchmarks

### Inference Time (Single Image, 1920x1080)

| Hardware | Model | Time | Throughput |
|----------|-------|------|------------|
| RTX 4090 | ViT-L | ~80ms | 12.5 FPS |
| RTX 3090 | ViT-L | ~150ms | 6.7 FPS |
| RTX 3060 | ViT-L | ~250ms | 4 FPS |
| CPU i9 | ViT-L | ~2s | 0.5 FPS |
| RTX 3090 | Swin-L | ~200ms | 5 FPS |

### Memory Usage

| Model | GPU Memory | RAM |
|-------|------------|-----|
| ViT-L | ~6GB | ~4GB |
| Swin-L | ~8GB | ~5GB |

## Next Steps

Once installation is complete:

1. **Run the demo:**
   ```bash
   python scripts/demo_openseed.py --sample
   ```

2. **Try with your own image:**
   ```bash
   python scripts/demo_openseed.py path/to/traffic_image.jpg
   ```

3. **Explore the notebook:**
   ```bash
   jupyter notebook notebooks/01_openseed_vehicle_detection.ipynb
   ```

4. **Read the documentation:**
   - Model documentation: `src/models/README.md`
   - API reference in the docstrings
   - Example usage in notebooks

5. **Start building your pipeline:**
   - Camera feed integration
   - Demographic data collection
   - Spatial analysis

## Support

If you encounter issues not covered here:

1. Check the [OpenSeed GitHub](https://github.com/IDEA-Research/OpenSeeD) for model-specific issues
2. Check the [Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
3. Open an issue in this repository with:
   - Your Python version (`python --version`)
   - Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Your CUDA version (if using GPU)
   - Complete error message

## Additional Resources

- [OpenSeed Paper](https://arxiv.org/abs/2303.08131)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
