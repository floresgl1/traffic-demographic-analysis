# Scripts

Utility scripts for testing and using the OpenSeed vehicle detection system.

## Available Scripts

### test_openseed.py

Comprehensive test script to verify your OpenSeed installation.

**Usage:**
```bash
python scripts/test_openseed.py
```

**What it does:**
1. Checks all required package imports
2. Verifies OpenSeed analyzer class loads correctly
3. (Optional) Downloads and initializes the model
4. (Optional) Runs a simple inference test

**When to use:**
- After initial installation
- After updating dependencies
- When troubleshooting issues
- To verify GPU is working correctly

### demo_openseed.py

Demo script showing vehicle detection on real images.

**Usage:**
```bash
# Use a sample traffic image
python scripts/demo_openseed.py --sample

# Use your own image
python scripts/demo_openseed.py path/to/image.jpg

# Download from URL
python scripts/demo_openseed.py --url https://example.com/traffic.jpg

# Advanced options
python scripts/demo_openseed.py image.jpg \
  --confidence 0.6 \
  --device cuda \
  --model facebook/openseed-swinl \
  --output results/annotated.jpg
```

**Options:**
- `--sample`: Download and use a sample traffic image
- `--url URL`: Download image from URL
- `--confidence FLOAT`: Set confidence threshold (default: 0.5)
- `--device {cpu,cuda,auto}`: Choose device (default: auto)
- `--model {vitl,swinl}`: Choose model variant (default: vitl)
- `--output PATH`: Save annotated image to custom path

**Output:**
- Detection statistics (counts, ratios)
- List of individual detections
- Annotated image saved to `outputs/detections/`

**Examples:**

Test with high confidence threshold (fewer detections):
```bash
python scripts/demo_openseed.py --sample --confidence 0.7
```

Use CPU and save to specific location:
```bash
python scripts/demo_openseed.py my_image.jpg --device cpu --output my_result.jpg
```

## Workflow Examples

### 1. First-Time Setup Verification

```bash
# Run the test script
python scripts/test_openseed.py

# If successful, try the demo
python scripts/demo_openseed.py --sample
```

### 2. Testing on Your Traffic Footage

```bash
# Extract a frame from your video first
ffmpeg -i traffic_video.mp4 -ss 00:01:00 -vframes 1 frame.jpg

# Run detection
python scripts/demo_openseed.py frame.jpg --confidence 0.5
```

### 3. Batch Processing Multiple Images

```bash
# Process all images in a directory
for img in data/raw/*.jpg; do
    python scripts/demo_openseed.py "$img" --output "outputs/detections/$(basename $img)"
done
```

### 4. Quick Performance Test

```bash
# Time the inference
time python scripts/demo_openseed.py --sample

# Compare CPU vs GPU
time python scripts/demo_openseed.py --sample --device cpu
time python scripts/demo_openseed.py --sample --device cuda
```

## Development Scripts

Want to add more scripts? Follow these patterns:

**Template for new scripts:**
```python
#!/usr/bin/env python3
"""
Brief description of what this script does
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import OpenSeedTrafficAnalyzer

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

## Common Issues

### Script can't find modules

Make sure you run from the project root:
```bash
# Wrong - from scripts directory
cd scripts
python test_openseed.py  # Will fail

# Right - from project root
cd /path/to/traffic-demographic-analysis
python scripts/test_openseed.py  # Works
```

### Permission denied

Make scripts executable:
```bash
chmod +x scripts/*.py
```

### Model download hangs

The model is large (~2-3GB). First download can take several minutes depending on your internet speed. Be patient!

## Next Steps

After running these scripts successfully:

1. Explore the Jupyter notebooks in `notebooks/`
2. Read the full documentation in `src/models/README.md`
3. Start building your data collection pipeline
4. Integrate with Caltrans camera feeds

## Contributing

When adding new scripts:
1. Add clear docstrings
2. Include usage examples in this README
3. Follow the template pattern above
4. Make them executable (`chmod +x`)
5. Test on both CPU and GPU
