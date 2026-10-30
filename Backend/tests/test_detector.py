"""
Tests for the Detection dataclass and SecurityDetector.
Detector inference is mocked — we test the data structures and filtering logic.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.detector import Detection, TARGET_CLASSES, CLASS_IDS


# ═══════════════════════════════════════════════════════════
#  Detection Dataclass
# ═══════════════════════════════════════════════════════════

class TestDetection:
    def test_center(self):
        det = Detection(bbox=(10.0, 20.0, 50.0, 80.0), confidence=0.9, class_id=0, label="person")
        cx, cy = det.center()
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_area(self):
        det = Detection(bbox=(0.0, 0.0, 100.0, 200.0), confidence=0.8, class_id=0, label="person")
        assert det.area() == pytest.approx(20000.0)

    def test_zero_area(self):
        det = Detection(bbox=(50.0, 50.0, 50.0, 50.0), confidence=0.5, class_id=0, label="person")
        assert det.area() == pytest.approx(0.0)

    def test_is_vehicle_person(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.9, class_id=0, label="person")
        assert det.is_vehicle() is False

    def test_is_vehicle_car(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.9, class_id=2, label="car")
        assert det.is_vehicle() is True

    def test_is_vehicle_motorcycle(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.9, class_id=3, label="motorcycle")
        assert det.is_vehicle() is True

    def test_is_vehicle_bus(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.9, class_id=5, label="bus")
        assert det.is_vehicle() is True

    def test_is_vehicle_truck(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.9, class_id=7, label="truck")
        assert det.is_vehicle() is True


class TestTargetClasses:
    def test_person_class_id(self):
        assert 0 in TARGET_CLASSES
        assert TARGET_CLASSES[0] == "person"

    def test_all_vehicle_classes_present(self):
        for cls_id in [2, 3, 5, 7]:
            assert cls_id in TARGET_CLASSES

    def test_class_ids_list(self):
        assert sorted(CLASS_IDS) == [0, 2, 3, 5, 7]


class TestSecurityDetector:
    @patch("core.detector.YOLO")
    def test_init_loads_model(self, mock_yolo_cls):
        from core.detector import SecurityDetector
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        det = SecurityDetector(model_path="test.pt", device="cpu", imgsz=640)

        mock_yolo_cls.assert_called_once_with("test.pt")
        mock_model.to.assert_called_once_with("cpu")
        assert det.imgsz == 640
        assert det.high_conf == 0.40
        assert det.low_conf == 0.10

    @patch("core.detector.YOLO")
    def test_custom_thresholds(self, mock_yolo_cls):
        from core.detector import SecurityDetector
        mock_yolo_cls.return_value = MagicMock()

        det = SecurityDetector(
            model_path="test.pt", device="cpu",
            high_conf=0.50, low_conf=0.05, min_area=200.0,
            nms_iou=0.80, imgsz=1280,
        )
        assert det.high_conf == 0.50
        assert det.low_conf == 0.05
        assert det.min_area == 200.0
        assert det.nms_iou == 0.80
        assert det.imgsz == 1280

    @patch("core.detector.YOLO")
    def test_detect_splits_high_low(self, mock_yolo_cls):
        """High-conf dets go to first list, low-conf to second."""
        from core.detector import SecurityDetector

        # Create mock boxes
        mock_box_high = MagicMock()
        mock_box_high.xyxy = [MagicMock()]
        mock_box_high.xyxy[0].cpu.return_value.tolist.return_value = [10.0, 20.0, 100.0, 200.0]
        mock_box_high.conf.cpu.return_value = 0.85
        mock_box_high.cls.cpu.return_value.item.return_value = 0

        mock_box_low = MagicMock()
        mock_box_low.xyxy = [MagicMock()]
        mock_box_low.xyxy[0].cpu.return_value.tolist.return_value = [200.0, 200.0, 280.0, 400.0]
        mock_box_low.conf.cpu.return_value = 0.25
        mock_box_low.cls.cpu.return_value.item.return_value = 0

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_high, mock_box_low]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        det = SecurityDetector(model_path="test.pt", device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        high, low = det.detect(frame)

        assert len(high) == 1
        assert len(low) == 1
        assert high[0].confidence == 0.85
        assert low[0].confidence == 0.25

    @patch("core.detector.YOLO")
    def test_detect_filters_small_area(self, mock_yolo_cls):
        """Detections below min_area should be discarded."""
        from core.detector import SecurityDetector

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.tolist.return_value = [10.0, 10.0, 15.0, 15.0]  # area=25
        mock_box.conf.cpu.return_value = 0.90
        mock_box.cls.cpu.return_value.item.return_value = 0

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_model

        det = SecurityDetector(model_path="test.pt", device="cpu", min_area=120.0)
        high, low = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(high) == 0
        assert len(low) == 0

    @patch("core.detector.YOLO")
    def test_batch_detect_returns_per_frame(self, mock_yolo_cls):
        from core.detector import SecurityDetector

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.tolist.return_value = [10.0, 20.0, 100.0, 200.0]
        mock_box.conf.cpu.return_value = 0.90
        mock_box.cls.cpu.return_value.item.return_value = 0

        mock_result1 = MagicMock()
        mock_result1.boxes = [mock_box]
        mock_result2 = MagicMock()
        mock_result2.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result1, mock_result2]
        mock_yolo_cls.return_value = mock_model

        det = SecurityDetector(model_path="test.pt", device="cpu")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
        results = det.batch_detect(frames)

        assert len(results) == 2
        assert len(results[0][0]) == 1  # Frame 1: 1 high-conf
        assert len(results[1][0]) == 0  # Frame 2: 0 detections

    @patch("core.detector.YOLO")
    def test_batch_detect_empty_list(self, mock_yolo_cls):
        from core.detector import SecurityDetector
        mock_yolo_cls.return_value = MagicMock()
        det = SecurityDetector(model_path="test.pt", device="cpu")
        assert det.batch_detect([]) == []
