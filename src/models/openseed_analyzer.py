"""
OpenSeed Traffic Analyzer

This module provides a wrapper around the OpenSeed model for vehicle detection
and classification in traffic camera footage. OpenSeed is a unified approach for
panoptic, instance, and semantic segmentation.

Key Features:
- Vehicle detection and classification (car, truck, bus, motorcycle)
- Batch processing for video frames
- Configurable confidence thresholds
- Support for GPU acceleration
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenSeedTrafficAnalyzer:
    """
    OpenSeed-based vehicle detection and classification for traffic analysis.

    This class wraps the OpenSeed model to provide specialized functionality for
    detecting and classifying vehicles in traffic camera footage.

    Attributes:
        model: The loaded OpenSeed model
        device: Computing device (cuda or cpu)
        confidence_threshold: Minimum confidence score for detections
        vehicle_classes: List of vehicle class names to detect
    """

    # COCO class mapping for vehicles (OpenSeed uses COCO classes)
    VEHICLE_CLASSES = {
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7
    }

    # Reverse mapping
    CLASS_ID_TO_NAME = {v: k for k, v in VEHICLE_CLASSES.items()}

    def __init__(
        self,
        model_name: str = "facebook/openseed-vitl",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the OpenSeed traffic analyzer.

        Args:
            model_name: Name or path of the OpenSeed model
            confidence_threshold: Minimum confidence score (0-1) for detections
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.cache_dir = cache_dir or os.path.join(str(Path.home()), '.cache', 'openseed')

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing OpenSeed analyzer on {self.device}")

        # Initialize model
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """
        Load the OpenSeed model and processor.

        Note: OpenSeed requires transformers and detectron2. This method handles
        model loading from HuggingFace or local cache.
        """
        try:
            from transformers import AutoProcessor, AutoModelForUniversalSegmentation

            logger.info(f"Loading OpenSeed model: {self.model_name}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            self.model = AutoModelForUniversalSegmentation.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)

            self.model.eval()  # Set to evaluation mode
            logger.info("OpenSeed model loaded successfully")

        except ImportError as e:
            logger.error(
                "Required libraries not found. Please install: "
                "pip install transformers detectron2"
            )
            raise ImportError(
                "OpenSeed requires transformers and detectron2. "
                "Install with: pip install transformers detectron2"
            ) from e
        except Exception as e:
            logger.error(f"Error loading OpenSeed model: {e}")
            raise

    def detect_vehicles(
        self,
        image: Union[np.ndarray, Image.Image, str],
        return_visualizations: bool = False
    ) -> Dict:
        """
        Detect and classify vehicles in an image.

        Args:
            image: Input image as numpy array, PIL Image, or file path
            return_visualizations: If True, return annotated image

        Returns:
            Dictionary containing:
                - detections: List of detected vehicles with bbox, class, confidence
                - counts: Dictionary of vehicle counts by type
                - total_vehicles: Total number of vehicles detected
                - car_truck_ratio: Ratio of cars to trucks
                - annotated_image: (optional) Image with bounding boxes drawn
        """
        # Load and preprocess image
        pil_image = self._load_image(image)

        # Run inference
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

        # Post-process outputs to get vehicle detections
        detections = self._post_process_outputs(outputs, pil_image.size)

        # Calculate statistics
        result = self._calculate_statistics(detections)
        result['detections'] = detections

        # Optionally add visualization
        if return_visualizations:
            result['annotated_image'] = self._visualize_detections(
                pil_image, detections
            )

        return result

    def process_video_frames(
        self,
        frames: List[np.ndarray],
        aggregate: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Process multiple video frames in batch.

        Args:
            frames: List of frames as numpy arrays
            aggregate: If True, return aggregated statistics; else per-frame results

        Returns:
            Aggregated statistics or list of per-frame results
        """
        results = []

        logger.info(f"Processing {len(frames)} frames...")

        for i, frame in enumerate(frames):
            try:
                result = self.detect_vehicles(frame, return_visualizations=False)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(frames)} frames")

            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")
                continue

        if aggregate:
            return self._aggregate_results(results)
        else:
            return results

    def _load_image(self, image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _post_process_outputs(
        self,
        outputs,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Post-process model outputs to extract vehicle detections.

        Args:
            outputs: Raw model outputs
            image_size: Original image size (width, height)

        Returns:
            List of detection dictionaries
        """
        detections = []

        # Extract predictions (this depends on OpenSeed's output format)
        # Note: The exact post-processing depends on OpenSeed's API
        # This is a template that should be adjusted based on actual model outputs

        try:
            # Get segmentation masks and class predictions
            # OpenSeed outputs semantic/panoptic/instance segmentation
            # We need to extract instance-level detections for vehicles

            pred_masks = outputs.get('pred_masks', None)
            pred_classes = outputs.get('pred_classes', None)
            pred_scores = outputs.get('pred_scores', None)

            if pred_classes is not None and pred_scores is not None:
                for idx, (cls, score) in enumerate(zip(pred_classes, pred_scores)):
                    cls_id = cls.item() if torch.is_tensor(cls) else cls
                    score_val = score.item() if torch.is_tensor(score) else score

                    # Check if it's a vehicle class and meets confidence threshold
                    if cls_id in self.CLASS_ID_TO_NAME and score_val >= self.confidence_threshold:
                        detection = {
                            'class_id': cls_id,
                            'class_name': self.CLASS_ID_TO_NAME[cls_id],
                            'confidence': score_val,
                            'bbox': None  # Will be extracted from mask if available
                        }

                        # Extract bounding box from mask if available
                        if pred_masks is not None and idx < len(pred_masks):
                            bbox = self._mask_to_bbox(pred_masks[idx], image_size)
                            detection['bbox'] = bbox

                        detections.append(detection)

        except Exception as e:
            logger.warning(f"Error in post-processing: {e}")

        return detections

    def _mask_to_bbox(self, mask: torch.Tensor, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert segmentation mask to bounding box."""
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Find contours
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        coords = np.column_stack(np.where(mask_uint8 > 0))

        if len(coords) == 0:
            return (0, 0, 0, 0)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate traffic statistics from detections."""
        counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0
        }

        for det in detections:
            class_name = det['class_name']
            counts[class_name] += 1

        total_vehicles = sum(counts.values())

        # Calculate car/truck ratio
        car_truck_ratio = None
        if counts['truck'] > 0:
            car_truck_ratio = counts['car'] / counts['truck']

        return {
            'counts': counts,
            'total_vehicles': total_vehicles,
            'car_truck_ratio': car_truck_ratio
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate statistics across multiple frames."""
        if not results:
            return {
                'total_frames': 0,
                'avg_vehicles_per_frame': 0,
                'total_counts': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
                'avg_car_truck_ratio': None
            }

        total_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        total_vehicles = 0
        ratios = []

        for result in results:
            for vehicle_type, count in result['counts'].items():
                total_counts[vehicle_type] += count
            total_vehicles += result['total_vehicles']

            if result['car_truck_ratio'] is not None:
                ratios.append(result['car_truck_ratio'])

        avg_ratio = np.mean(ratios) if ratios else None

        return {
            'total_frames': len(results),
            'avg_vehicles_per_frame': total_vehicles / len(results),
            'total_counts': total_counts,
            'avg_counts_per_frame': {
                k: v / len(results) for k, v in total_counts.items()
            },
            'avg_car_truck_ratio': avg_ratio
        }

    def _visualize_detections(
        self,
        image: Image.Image,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Color mapping for different vehicle types
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),    # Blue
            'bus': (0, 165, 255),    # Orange
            'motorcycle': (255, 255, 0)  # Cyan
        }

        for det in detections:
            if det['bbox'] is None:
                continue

            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_cv,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            cv2.putText(
                img_cv,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return img_cv

    def save_annotated_image(self, annotated_image: np.ndarray, output_path: str):
        """Save annotated image to file."""
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Saved annotated image to {output_path}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'vehicle_classes': list(self.VEHICLE_CLASSES.keys()),
            'cache_dir': self.cache_dir
        }


# Convenience function for quick testing
def quick_detect(image_path: str, confidence_threshold: float = 0.5) -> Dict:
    """
    Quick vehicle detection on a single image.

    Args:
        image_path: Path to image file
        confidence_threshold: Minimum confidence for detections

    Returns:
        Detection results dictionary
    """
    analyzer = OpenSeedTrafficAnalyzer(confidence_threshold=confidence_threshold)
    return analyzer.detect_vehicles(image_path, return_visualizations=True)
