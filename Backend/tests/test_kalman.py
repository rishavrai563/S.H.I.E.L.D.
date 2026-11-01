"""
Tests for the 8-state Kalman filter and geometry helpers.
Pure math — no external dependencies beyond numpy.
"""
import pytest
import numpy as np
from core.kalman import (
    KalmanFilter8,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    clamp_bbox,
    bbox_iou,
    bbox_center,
    bbox_area,
)


# ═══════════════════════════════════════════════════════════
#  Geometry Helpers
# ═══════════════════════════════════════════════════════════

class TestXyxyToCxcywh:
    def test_basic_conversion(self):
        cx, cy, w, h = xyxy_to_cxcywh((10.0, 20.0, 50.0, 80.0))
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)
        assert w == pytest.approx(40.0)
        assert h == pytest.approx(60.0)

    def test_minimum_size_enforced(self):
        """Zero-area bbox should clamp to minimum 2px size."""
        cx, cy, w, h = xyxy_to_cxcywh((100.0, 100.0, 100.0, 100.0))
        assert w >= 2.0
        assert h >= 2.0

    def test_roundtrip(self, sample_bbox):
        cx, cy, w, h = xyxy_to_cxcywh(sample_bbox)
        x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, w, h)
        assert x1 == pytest.approx(sample_bbox[0], abs=0.1)
        assert y1 == pytest.approx(sample_bbox[1], abs=0.1)
        assert x2 == pytest.approx(sample_bbox[2], abs=0.1)
        assert y2 == pytest.approx(sample_bbox[3], abs=0.1)


class TestClampBbox:
    def test_within_bounds_unchanged(self):
        result = clamp_bbox(10.0, 20.0, 100.0, 200.0, 1920, 1080)
        assert result == (10.0, 20.0, 100.0, 200.0)

    def test_clamps_to_frame_edges(self):
        x1, y1, x2, y2 = clamp_bbox(-50.0, -30.0, 2000.0, 1200.0, 1920, 1080)
        assert x1 >= 0.0
        assert y1 >= 0.0
        assert x2 <= 1919.0
        assert y2 <= 1079.0

    def test_degenerate_box_gets_min_size(self):
        x1, y1, x2, y2 = clamp_bbox(500.0, 500.0, 500.0, 500.0, 1920, 1080)
        assert x2 > x1
        assert y2 > y1


class TestBboxIou:
    def test_identical_boxes(self):
        box = (10.0, 20.0, 100.0, 200.0)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 50.0, 50.0)
        b = (100.0, 100.0, 200.0, 200.0)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 100.0, 100.0)
        b = (50.0, 50.0, 150.0, 150.0)
        iou = bbox_iou(a, b)
        assert 0.0 < iou < 1.0
        # Intersection: 50×50=2500, Union: 10000+10000-2500=17500
        assert iou == pytest.approx(2500.0 / 17500.0, abs=0.01)

    def test_contained_box(self):
        outer = (0.0, 0.0, 200.0, 200.0)
        inner = (50.0, 50.0, 100.0, 100.0)
        iou = bbox_iou(outer, inner)
        # Inner area = 2500, Outer area = 40000, Union = 40000
        assert iou == pytest.approx(2500.0 / 40000.0, abs=0.01)


class TestBboxCenter:
    def test_center(self):
        cx, cy = bbox_center((10.0, 20.0, 50.0, 80.0))
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)


class TestBboxArea:
    def test_positive_area(self):
        assert bbox_area((0.0, 0.0, 100.0, 200.0)) == pytest.approx(20000.0)

    def test_zero_area(self):
        assert bbox_area((50.0, 50.0, 50.0, 50.0)) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════
#  Kalman Filter
# ═══════════════════════════════════════════════════════════

class TestKalmanFilter8:
    def test_initialization(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        assert kf.x.shape == (8,)
        # Initial velocity should be zero
        assert np.allclose(kf.x[4:], 0.0)
        # Position should match input bbox center
        cx, cy, w, h = xyxy_to_cxcywh(sample_bbox)
        assert kf.x[0] == pytest.approx(cx)
        assert kf.x[1] == pytest.approx(cy)

    def test_predict_moves_by_velocity(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        # Set known velocity
        kf.x[4] = 5.0   # vx
        kf.x[5] = -3.0  # vy
        old_cx, old_cy = kf.x[0], kf.x[1]
        kf.predict()
        assert kf.x[0] == pytest.approx(old_cx + 5.0, abs=1.0)
        assert kf.x[1] == pytest.approx(old_cy - 3.0, abs=1.0)

    def test_predict_increases_uncertainty(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        P_before = np.trace(kf.P)
        kf.predict()
        P_after = np.trace(kf.P)
        assert P_after > P_before

    def test_update_reduces_uncertainty(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        kf.predict()
        P_before_update = np.trace(kf.P)
        kf.update(sample_bbox)
        P_after_update = np.trace(kf.P)
        assert P_after_update < P_before_update

    def test_update_corrects_toward_measurement(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        kf.predict()
        # Shift bbox right by 20px
        shifted = (sample_bbox[0] + 20, sample_bbox[1], sample_bbox[2] + 20, sample_bbox[3])
        kf.update(shifted)
        # Center should have shifted toward the measurement
        cx_shifted = (shifted[0] + shifted[2]) / 2
        assert kf.x[0] > (sample_bbox[0] + sample_bbox[2]) / 2

    def test_predicted_bbox_output(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        kf.predict()
        pred = kf.predicted_bbox()
        assert len(pred) == 4
        x1, y1, x2, y2 = pred
        assert x2 > x1
        assert y2 > y1

    def test_mahalanobis_self_is_small(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        kf.predict()
        d = kf.mahalanobis(sample_bbox)
        assert d < 20.0  # Should be very small for self-match

    def test_mahalanobis_far_is_large(self, sample_bbox):
        kf = KalmanFilter8(sample_bbox)
        kf.predict()
        far_bbox = (900.0, 900.0, 1000.0, 1000.0)
        d = kf.mahalanobis(far_bbox)
        assert d > 100.0

    def test_convergence_over_multiple_updates(self):
        """Kalman filter should converge to true trajectory over time."""
        true_x, true_y = 100.0, 200.0
        vx, vy = 3.0, 1.5
        kf = KalmanFilter8((true_x - 20, true_y - 40, true_x + 20, true_y + 40))

        for t in range(30):
            true_x += vx
            true_y += vy
            bbox = (true_x - 20, true_y - 40, true_x + 20, true_y + 40)
            kf.predict()
            kf.update(bbox)

        # After convergence, velocity estimate should be close to truth
        assert kf.x[4] == pytest.approx(vx, abs=1.0)
        assert kf.x[5] == pytest.approx(vy, abs=1.0)
