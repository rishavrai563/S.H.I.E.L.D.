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

        mock_result = _fake_result([[10.0, 20.0, 100.0, 200.0, 0.85, 0],
                                    [200.0, 200.0, 280.0, 400.0, 0.25, 0]])

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

        mock_result = _fake_result([[10.0, 10.0, 15.0, 15.0, 0.90, 0]])  # area=25
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

        mock_result1 = _fake_result([[10.0, 20.0, 100.0, 200.0, 0.90, 0]])
        mock_result2 = _fake_result([])

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



def _fake_result(rows):
    """YOLO-like result whose boxes.data is an (N, 6) array [x1, y1, x2, y2, conf, cls]."""
    r = MagicMock()
    r.boxes.data = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    return r


class TestShapeGrouping:
    @patch("core.detector.YOLO")
    def test_mixed_shapes_run_per_shape_and_keep_order(self, mock_yolo_cls):
        from core.detector import SecurityDetector
        model = MagicMock()
        # one predict call per shape group; tag each result with the frame width it saw
        model.predict.side_effect = lambda imgs, **kw: [
            _fake_result([[0, 0, 50, 50, 0.9, 0]] * (1 if im.shape[1] == 640 else 2)) for im in imgs]
        mock_yolo_cls.return_value = model
        det = SecurityDetector(model_path="t.pt", device="cpu")
        a, b = np.zeros((480, 640, 3), np.uint8), np.zeros((1080, 960, 3), np.uint8)
        out = det.batch_detect([a, b, a])
        assert model.predict.call_count == 2
        assert [len(h) for h, _ in out] == [1, 2, 1]
