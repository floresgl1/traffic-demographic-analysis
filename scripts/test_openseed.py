#!/usr/bin/env python3
"""
Test script to verify OpenSeed installation and model loading
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import detectron2
        print(f"✓ Detectron2 {detectron2.__version__}")
    except ImportError as e:
        print(f"✗ Detectron2 import failed: {e}")
        print("  Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        return False

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        from PIL import Image
        print(f"✓ Pillow (PIL)")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_model_class():
    """Test that the OpenSeed analyzer class can be imported"""
    print("Testing OpenSeed analyzer class...")

    try:
        from src.models import OpenSeedTrafficAnalyzer
        print("✓ OpenSeedTrafficAnalyzer class imported")

        # Test class attributes
        print(f"  - Vehicle classes: {list(OpenSeedTrafficAnalyzer.VEHICLE_CLASSES.keys())}")
        print(f"  - COCO IDs: {list(OpenSeedTrafficAnalyzer.VEHICLE_CLASSES.values())}")

        return True
    except ImportError as e:
        print(f"✗ Failed to import OpenSeedTrafficAnalyzer: {e}")
        return False


def test_model_initialization():
    """Test model initialization (this will download the model if not cached)"""
    print("\nTesting model initialization...")
    print("WARNING: This will download ~2-3GB of model weights on first run.")
    print("The model will be cached for future use.\n")

    response = input("Proceed with model download/initialization? (y/n): ")
    if response.lower() != 'y':
        print("Skipping model initialization test.")
        return None

    try:
        from src.models import OpenSeedTrafficAnalyzer
        import torch

        print("\nInitializing OpenSeed analyzer...")
        print("(This may take a few minutes on first run)\n")

        analyzer = OpenSeedTrafficAnalyzer(
            model_name="facebook/openseed-vitl",
            confidence_threshold=0.5,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print("✓ Model initialized successfully!")

        # Print model info
        info = analyzer.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  - {key}: {value}")

        return analyzer
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_inference(analyzer):
    """Test inference on a simple synthetic image"""
    print("\nTesting inference on synthetic image...")

    try:
        import numpy as np
        from PIL import Image

        # Create a simple test image (solid color)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)

        print("Running inference...")
        results = analyzer.detect_vehicles(pil_image, return_visualizations=False)

        print("✓ Inference completed!")
        print(f"\nResults:")
        print(f"  - Total vehicles detected: {results['total_vehicles']}")
        print(f"  - Counts by type: {results['counts']}")

        if results['total_vehicles'] == 0:
            print("\n  Note: No vehicles detected (expected for random image)")

        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("OpenSeed Installation Test")
    print("=" * 60)
    print()

    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed. Please install missing dependencies.")
        sys.exit(1)

    # Test model class
    if not test_model_class():
        print("\n✗ Model class test failed.")
        sys.exit(1)

    # Test model initialization (optional)
    analyzer = test_model_initialization()

    if analyzer is None:
        print("\nTests completed (model initialization skipped).")
        print("\nTo complete the setup, run this script again and initialize the model.")
        sys.exit(0)
    elif analyzer is False:
        print("\n✗ Model initialization failed.")
        sys.exit(1)

    # Test inference
    if not test_simple_inference(analyzer):
        print("\n✗ Inference test failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ All tests passed! OpenSeed is ready to use.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. See notebooks/01_openseed_vehicle_detection.ipynb for examples")
    print("  2. Run scripts/demo_openseed.py to test on a real image")
    print("  3. Start building your traffic analysis pipeline!")


if __name__ == "__main__":
    main()
