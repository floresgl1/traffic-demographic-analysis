# Models Module

This module contains the OpenSeed model wrapper and related utilities for vehicle detection and classification in traffic camera footage.

## Overview

The OpenSeed Traffic Analyzer provides a high-level interface for detecting and classifying vehicles in images and video streams using the OpenSeed panoptic segmentation model.

## Features

- **Vehicle Detection**: Detect cars, trucks, buses, and motorcycles
- **Batch Processing**: Process multiple video frames efficiently
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Configurable Thresholds**: Adjust confidence thresholds for different use cases
- **Statistical Analysis**: Get counts, ratios, and temporal aggregations
- **Visualization**: Annotate images with bounding boxes and labels

## Installation

### Prerequisites

1. Install PyTorch with CUDA support (recommended):
   ```bash
   # For CUDA 11.7
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install project requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Detectron2 (required for OpenSeed):
   ```bash
   # Option 1: From source (recommended)
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

   # Option 2: Using conda
   conda install -c conda-forge detectron2
   ```

### Verify Installation

```python
from src.models import OpenSeedTrafficAnalyzer

# This will download the model on first use (~2-3GB)
analyzer = OpenSeedTrafficAnalyzer()
print(analyzer.get_model_info())
```

## Quick Start

### Basic Usage

```python
from src.models import OpenSeedTrafficAnalyzer

# Initialize analyzer
analyzer = OpenSeedTrafficAnalyzer(
    model_name="facebook/openseed-vitl",
    confidence_threshold=0.5
)

# Detect vehicles in an image
results = analyzer.detect_vehicles(
    'path/to/traffic_image.jpg',
    return_visualizations=True
)

# Print results
print(f"Total Vehicles: {results['total_vehicles']}")
print(f"Cars: {results['counts']['car']}")
print(f"Trucks: {results['counts']['truck']}")
print(f"Car/Truck Ratio: {results['car_truck_ratio']:.2f}")
```

### Process Video Frames

```python
import cv2

# Extract frames from video
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Process frames
frames = extract_frames('traffic_video.mp4')
aggregated_stats = analyzer.process_video_frames(frames, aggregate=True)

print(f"Avg Vehicles per Frame: {aggregated_stats['avg_vehicles_per_frame']:.1f}")
print(f"Total Counts: {aggregated_stats['total_counts']}")
```

### Save Annotated Images

```python
results = analyzer.detect_vehicles(
    'traffic_image.jpg',
    return_visualizations=True
)

if 'annotated_image' in results:
    analyzer.save_annotated_image(
        results['annotated_image'],
        'output/annotated_traffic.jpg'
    )
```

## Configuration

### Model Options

Two pre-trained models are available:

1. **facebook/openseed-vitl** (default)
   - Vision Transformer Large backbone
   - Good balance of speed and accuracy
   - Recommended for most use cases

2. **facebook/openseed-swinl**
   - Swin Transformer Large backbone
   - Higher accuracy but slower
   - Best for offline analysis

```python
# Use Swin Transformer model
analyzer = OpenSeedTrafficAnalyzer(
    model_name="facebook/openseed-swinl",
    confidence_threshold=0.6
)
```

### Confidence Thresholds

Adjust the confidence threshold based on your requirements:

```python
# Higher threshold = fewer false positives
analyzer = OpenSeedTrafficAnalyzer(confidence_threshold=0.7)

# Lower threshold = detect more vehicles (may include false positives)
analyzer = OpenSeedTrafficAnalyzer(confidence_threshold=0.3)
```

Recommended thresholds:
- **0.3-0.4**: Maximum recall (catch all possible vehicles)
- **0.5**: Balanced (default)
- **0.6-0.8**: High precision (only confident detections)

### Device Selection

```python
# Force CPU usage
analyzer = OpenSeedTrafficAnalyzer(device='cpu')

# Force GPU usage
analyzer = OpenSeedTrafficAnalyzer(device='cuda')

# Auto-detect (default)
analyzer = OpenSeedTrafficAnalyzer(device=None)
```

## API Reference

### OpenSeedTrafficAnalyzer

#### Constructor

```python
OpenSeedTrafficAnalyzer(
    model_name: str = "facebook/openseed-vitl",
    confidence_threshold: float = 0.5,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
)
```

**Parameters:**
- `model_name`: Name of the OpenSeed model to use
- `confidence_threshold`: Minimum confidence score (0-1) for detections
- `device`: Computing device ('cuda', 'cpu', or None for auto-detect)
- `cache_dir`: Directory to cache downloaded models

#### Methods

##### detect_vehicles()

```python
detect_vehicles(
    image: Union[np.ndarray, Image.Image, str],
    return_visualizations: bool = False
) -> Dict
```

Detect and classify vehicles in a single image.

**Returns:**
- `detections`: List of detected vehicles with bbox, class, confidence
- `counts`: Dictionary of vehicle counts by type
- `total_vehicles`: Total number of vehicles detected
- `car_truck_ratio`: Ratio of cars to trucks
- `annotated_image`: (optional) Image with bounding boxes drawn

##### process_video_frames()

```python
process_video_frames(
    frames: List[np.ndarray],
    aggregate: bool = True
) -> Union[Dict, List[Dict]]
```

Process multiple video frames in batch.

**Parameters:**
- `frames`: List of frames as numpy arrays
- `aggregate`: If True, return aggregated statistics; else per-frame results

**Returns:**
- If `aggregate=True`: Dictionary with aggregated statistics
- If `aggregate=False`: List of per-frame results

##### get_model_info()

```python
get_model_info() -> Dict
```

Get information about the loaded model.

**Returns:**
- `model_name`: Name of the model
- `device`: Device being used
- `confidence_threshold`: Current threshold
- `vehicle_classes`: List of detectable vehicle classes

## Vehicle Classes

The analyzer detects the following vehicle types:

| Class | COCO ID | Description |
|-------|---------|-------------|
| car | 2 | Passenger cars, sedans, SUVs |
| motorcycle | 3 | Motorcycles, scooters |
| bus | 5 | Buses, coaches |
| truck | 7 | Trucks, vans, commercial vehicles |

## Performance Considerations

### GPU Memory

- **VIT-L model**: Requires ~6-8GB VRAM
- **Swin-L model**: Requires ~8-10GB VRAM

For limited GPU memory:
- Reduce image resolution before processing
- Process frames in smaller batches
- Use the VIT-L model instead of Swin-L

### Processing Speed

Approximate inference times (single image, 1920x1080):

| Hardware | VIT-L | Swin-L |
|----------|-------|--------|
| RTX 3090 | ~150ms | ~200ms |
| RTX 3060 | ~250ms | ~350ms |
| CPU (i9) | ~2s | ~3s |

### Batch Processing

For best performance when processing videos:
- Use GPU if available
- Process frames in batches of 8-16
- Sample frames rather than processing every frame (e.g., 1 FPS)

## Troubleshooting

### Model Download Issues

If the model fails to download:
```python
# Specify a custom cache directory
analyzer = OpenSeedTrafficAnalyzer(
    cache_dir="/path/to/cache"
)
```

### CUDA Out of Memory

If you get OOM errors:
```python
# 1. Reduce image size
import cv2
resized = cv2.resize(image, (960, 540))  # Half resolution

# 2. Use CPU
analyzer = OpenSeedTrafficAnalyzer(device='cpu')

# 3. Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Detectron2 Installation Issues

If detectron2 fails to install:
```bash
# Install build dependencies
sudo apt-get install python3-dev build-essential

# Install from source with specific CUDA version
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html
```

## Examples

See the [example notebook](../../notebooks/01_openseed_vehicle_detection.ipynb) for complete working examples.

## References

- [OpenSeed Paper](https://arxiv.org/abs/2303.08131)
- [OpenSeed GitHub](https://github.com/IDEA-Research/OpenSeeD)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
