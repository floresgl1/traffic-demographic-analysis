#!/usr/bin/env python3
"""
Demo script showing how to use OpenSeed for vehicle detection on a real image

Usage:
  python scripts/demo_openseed.py <image_path>
  python scripts/demo_openseed.py --url <image_url>
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import OpenSeedTrafficAnalyzer
import cv2
import numpy as np
from PIL import Image


def download_sample_image():
    """Download a sample traffic image for testing"""
    import requests
    from io import BytesIO

    print("Downloading sample traffic image...")

    # Pexels free stock image of traffic
    sample_urls = [
        "https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg?auto=compress&cs=tinysrgb&w=1260",
        "https://images.pexels.com/photos/2199293/pexels-photo-2199293.jpeg?auto=compress&cs=tinysrgb&w=1260",
    ]

    for url in sample_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                print(f"✓ Downloaded sample image: {image.size}")
                return image
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    raise Exception("Failed to download sample image")


def main():
    parser = argparse.ArgumentParser(description="OpenSeed vehicle detection demo")
    parser.add_argument('image_path', nargs='?', help='Path to traffic image')
    parser.add_argument('--url', help='Download image from URL')
    parser.add_argument('--sample', action='store_true', help='Use a sample traffic image')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', help='Device to use')
    parser.add_argument('--output', help='Path to save annotated image')
    parser.add_argument('--model', default='facebook/openseed-vitl',
                        choices=['facebook/openseed-vitl', 'facebook/openseed-swinl'],
                        help='Model to use')

    args = parser.parse_args()

    # Determine image source
    if args.sample:
        print("Using sample traffic image...")
        image = download_sample_image()
    elif args.url:
        print(f"Downloading image from: {args.url}")
        import requests
        from io import BytesIO
        response = requests.get(args.url, timeout=10)
        image = Image.open(BytesIO(response.content))
    elif args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            sys.exit(1)
        print(f"Loading image: {args.image_path}")
        image = args.image_path
    else:
        print("Error: Please provide an image path, --url, or --sample")
        print("Usage: python scripts/demo_openseed.py <image_path>")
        print("   or: python scripts/demo_openseed.py --sample")
        sys.exit(1)

    # Initialize analyzer
    print(f"\nInitializing OpenSeed analyzer...")
    print(f"  - Model: {args.model}")
    print(f"  - Confidence threshold: {args.confidence}")
    print(f"  - Device: {args.device}")

    device = None if args.device == 'auto' else args.device

    try:
        analyzer = OpenSeedTrafficAnalyzer(
            model_name=args.model,
            confidence_threshold=args.confidence,
            device=device
        )
    except Exception as e:
        print(f"\nError initializing analyzer: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        print("  pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        sys.exit(1)

    # Print model info
    info = analyzer.get_model_info()
    print(f"\n✓ Model loaded on: {info['device']}")

    # Run detection
    print("\nRunning vehicle detection...")
    print("(This may take a few seconds...)\n")

    results = analyzer.detect_vehicles(image, return_visualizations=True)

    # Print results
    print("=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"\nTotal Vehicles Detected: {results['total_vehicles']}")
    print(f"\nBreakdown by Type:")
    for vehicle_type, count in results['counts'].items():
        if count > 0:
            print(f"  {vehicle_type.capitalize():12s}: {count}")

    if results['car_truck_ratio'] is not None:
        print(f"\nCar/Truck Ratio: {results['car_truck_ratio']:.2f}")

    # Show individual detections
    if results['detections']:
        print(f"\nIndividual Detections ({len(results['detections'])}):")
        for i, det in enumerate(results['detections'][:10], 1):  # Show first 10
            bbox = det.get('bbox', 'N/A')
            print(f"  {i}. {det['class_name'].capitalize():12s} "
                  f"(confidence: {det['confidence']:.2f}) "
                  f"bbox: {bbox}")

        if len(results['detections']) > 10:
            print(f"  ... and {len(results['detections']) - 10} more")

    # Save annotated image
    if 'annotated_image' in results:
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output path
            os.makedirs('outputs/detections', exist_ok=True)
            output_path = 'outputs/detections/annotated_result.jpg'

        analyzer.save_annotated_image(results['annotated_image'], output_path)
        print(f"\n✓ Annotated image saved to: {output_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
