"""
Configuration settings for OpenSeed and other models
"""
import os
from pathlib import Path

# Model configurations
OPENSEED_MODELS = {
    'vitl': {
        'name': 'facebook/openseed-vitl',
        'description': 'Vision Transformer Large backbone',
        'recommended_for': 'Balance of speed and accuracy'
    },
    'swinl': {
        'name': 'facebook/openseed-swinl',
        'description': 'Swin Transformer Large backbone',
        'recommended_for': 'Higher accuracy, slower inference'
    }
}

# Default model
DEFAULT_MODEL = 'vitl'

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_CONFIDENCE_THRESHOLD = 0.3
MAX_CONFIDENCE_THRESHOLD = 0.95

# Cache directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_CACHE_DIR = os.path.join(str(Path.home()), '.cache', 'traffic_analysis', 'models')
DATA_CACHE_DIR = os.path.join(str(PROJECT_ROOT), 'data', 'cache')

# Performance settings
BATCH_SIZE = 8  # Number of frames to process in parallel
MAX_IMAGE_SIZE = (1920, 1080)  # Max resolution for processing
ENABLE_GPU = True  # Use GPU if available

# Vehicle detection settings
VEHICLE_CLASSES_TO_DETECT = ['car', 'truck', 'bus', 'motorcycle']

# Video processing settings
DEFAULT_FPS_SAMPLE_RATE = 2  # Extract 1 frame every N seconds
MAX_FRAMES_PER_VIDEO = 300  # Limit frames per video for memory

# Output settings
SAVE_ANNOTATED_IMAGES = True
SAVE_DETECTION_JSON = True
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), 'outputs', 'detections')

# Ensure output directories exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
