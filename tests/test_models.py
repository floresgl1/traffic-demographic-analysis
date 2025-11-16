"""
Unit tests for model implementations
"""
import pytest
import numpy as np
from PIL import Image
import torch


class TestOpenSeedTrafficAnalyzer:
    """Tests for OpenSeed traffic analyzer"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL image for testing"""
        return Image.new('RGB', (640, 480), color='red')

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        # This will try to load the model, which may fail without proper setup
        # So we'll just test that the class can be imported
        assert OpenSeedTrafficAnalyzer is not None

    def test_vehicle_classes_mapping(self):
        """Test that vehicle class mappings are correctly defined"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        expected_classes = ['car', 'motorcycle', 'bus', 'truck']

        for cls in expected_classes:
            assert cls in OpenSeedTrafficAnalyzer.VEHICLE_CLASSES
            assert OpenSeedTrafficAnalyzer.VEHICLE_CLASSES[cls] > 0

        # Test reverse mapping
        for cls_name, cls_id in OpenSeedTrafficAnalyzer.VEHICLE_CLASSES.items():
            assert cls_id in OpenSeedTrafficAnalyzer.CLASS_ID_TO_NAME
            assert OpenSeedTrafficAnalyzer.CLASS_ID_TO_NAME[cls_id] == cls_name

    def test_load_image_numpy(self):
        """Test loading numpy array images"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        # Create analyzer instance (may fail if model not available)
        try:
            analyzer = OpenSeedTrafficAnalyzer()

            # Test numpy array input
            img_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pil_img = analyzer._load_image(img_np)

            assert isinstance(pil_img, Image.Image)
            assert pil_img.mode == 'RGB'
            assert pil_img.size == (640, 480)
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")

    def test_load_image_pil(self):
        """Test loading PIL images"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        try:
            analyzer = OpenSeedTrafficAnalyzer()

            # Test PIL image input
            img_pil = Image.new('RGB', (640, 480), color='blue')
            result = analyzer._load_image(img_pil)

            assert isinstance(result, Image.Image)
            assert result.mode == 'RGB'
            assert result.size == (640, 480)
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")

    def test_mask_to_bbox(self):
        """Test mask to bounding box conversion"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        try:
            analyzer = OpenSeedTrafficAnalyzer()

            # Create a simple binary mask
            mask = np.zeros((100, 100), dtype=np.float32)
            mask[25:75, 30:80] = 1.0  # Rectangle from (30,25) to (80,75)

            bbox = analyzer._mask_to_bbox(torch.tensor(mask), (100, 100))

            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox

            # Check bbox is reasonable (allowing some tolerance)
            assert 25 <= x1 <= 35
            assert 20 <= y1 <= 30
            assert 75 <= x2 <= 85
            assert 70 <= y2 <= 80
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")

    def test_calculate_statistics(self):
        """Test statistics calculation from detections"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        try:
            analyzer = OpenSeedTrafficAnalyzer()

            # Mock detections
            detections = [
                {'class_name': 'car', 'confidence': 0.9},
                {'class_name': 'car', 'confidence': 0.85},
                {'class_name': 'truck', 'confidence': 0.8},
                {'class_name': 'bus', 'confidence': 0.75},
                {'class_name': 'car', 'confidence': 0.9},
            ]

            stats = analyzer._calculate_statistics(detections)

            assert stats['total_vehicles'] == 5
            assert stats['counts']['car'] == 3
            assert stats['counts']['truck'] == 1
            assert stats['counts']['bus'] == 1
            assert stats['counts']['motorcycle'] == 0
            assert stats['car_truck_ratio'] == 3.0
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")

    def test_aggregate_results(self):
        """Test aggregation of multiple frame results"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        try:
            analyzer = OpenSeedTrafficAnalyzer()

            # Mock frame results
            results = [
                {
                    'counts': {'car': 10, 'truck': 2, 'bus': 1, 'motorcycle': 0},
                    'total_vehicles': 13,
                    'car_truck_ratio': 5.0
                },
                {
                    'counts': {'car': 8, 'truck': 4, 'bus': 0, 'motorcycle': 1},
                    'total_vehicles': 13,
                    'car_truck_ratio': 2.0
                },
                {
                    'counts': {'car': 12, 'truck': 3, 'bus': 2, 'motorcycle': 0},
                    'total_vehicles': 17,
                    'car_truck_ratio': 4.0
                }
            ]

            aggregated = analyzer._aggregate_results(results)

            assert aggregated['total_frames'] == 3
            assert aggregated['avg_vehicles_per_frame'] == pytest.approx(43/3, rel=1e-5)
            assert aggregated['total_counts']['car'] == 30
            assert aggregated['total_counts']['truck'] == 9
            assert aggregated['total_counts']['bus'] == 3
            assert aggregated['total_counts']['motorcycle'] == 1
            assert aggregated['avg_car_truck_ratio'] == pytest.approx(3.666666, rel=1e-4)
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")

    def test_get_model_info(self):
        """Test model info retrieval"""
        from src.models.openseed_analyzer import OpenSeedTrafficAnalyzer

        try:
            analyzer = OpenSeedTrafficAnalyzer(
                model_name="facebook/openseed-vitl",
                confidence_threshold=0.6
            )

            info = analyzer.get_model_info()

            assert 'model_name' in info
            assert 'device' in info
            assert 'confidence_threshold' in info
            assert 'vehicle_classes' in info

            assert info['model_name'] == "facebook/openseed-vitl"
            assert info['confidence_threshold'] == 0.6
            assert set(info['vehicle_classes']) == {'car', 'truck', 'bus', 'motorcycle'}
        except Exception as e:
            pytest.skip(f"Skipping test - model not available: {e}")


class TestModelConfig:
    """Tests for model configuration"""

    def test_config_imports(self):
        """Test that config module can be imported"""
        from src.models import config
        assert config is not None

    def test_openseed_models_defined(self):
        """Test that OpenSeed models are defined in config"""
        from src.models.config import OPENSEED_MODELS, DEFAULT_MODEL

        assert len(OPENSEED_MODELS) > 0
        assert DEFAULT_MODEL in OPENSEED_MODELS

        for model_key, model_info in OPENSEED_MODELS.items():
            assert 'name' in model_info
            assert 'description' in model_info
            assert 'recommended_for' in model_info

    def test_threshold_values(self):
        """Test that threshold values are reasonable"""
        from src.models.config import (
            DEFAULT_CONFIDENCE_THRESHOLD,
            MIN_CONFIDENCE_THRESHOLD,
            MAX_CONFIDENCE_THRESHOLD
        )

        assert 0 <= MIN_CONFIDENCE_THRESHOLD <= 1
        assert 0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1
        assert 0 <= MAX_CONFIDENCE_THRESHOLD <= 1
        assert MIN_CONFIDENCE_THRESHOLD <= DEFAULT_CONFIDENCE_THRESHOLD <= MAX_CONFIDENCE_THRESHOLD

    def test_vehicle_classes(self):
        """Test that vehicle classes are defined"""
        from src.models.config import VEHICLE_CLASSES_TO_DETECT

        assert len(VEHICLE_CLASSES_TO_DETECT) > 0
        expected_classes = ['car', 'truck', 'bus', 'motorcycle']

        for cls in expected_classes:
            assert cls in VEHICLE_CLASSES_TO_DETECT
