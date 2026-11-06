"""
Tests for SingleCameraTracker — the 8-step tracking pipeline.
Uses synthetic Detection objects and embeddings.
"""
import pytest
import numpy as np
from core.detector import Detection
from core.tracker import (
    SingleCameraTracker,
    Track,
    TrackState,
    TrackEvent,
    LostBuffer,
    _cosine_sim,
    _nms_dedup,
)
from core.kalman import bbox_iou


def _det(x1, y1, x2, y2, conf=0.9, cls=0):
    """Shortcut to create a person Detection."""
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls, label="person")


def _emb(seed):
    """Create a deterministic L2-normalized 512-dim embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ═══════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════

class TestCosineSim:
    def test_identical(self):
        a = _emb(42)
        assert _cosine_sim(a, a) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal(self):
        a = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(512, dtype=np.float32)
        b[1] = 1.0
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=0.01)

    def test_none_returns_zero(self):
        assert _cosine_sim(None, _emb(42)) == 0.0
        assert _cosine_sim(_emb(42), None) == 0.0
        assert _cosine_sim(None, None) == 0.0


class TestNmsDedup:
    def test_no_overlap(self):
        dets = [_det(0, 0, 50, 50), _det(200, 200, 300, 300)]
        kept = _nms_dedup(dets, iou_threshold=0.50)
        assert len(kept) == 2

    def test_overlapping_suppressed(self):
        dets = [_det(0, 0, 100, 100, conf=0.9), _det(5, 5, 105, 105, conf=0.7)]
        kept = _nms_dedup(dets, iou_threshold=0.50)
        assert len(kept) == 1
        assert kept[0].confidence == 0.9  # Higher conf kept

    def test_single_detection(self):
        kept = _nms_dedup([_det(0, 0, 50, 50)])
        assert len(kept) == 1

    def test_empty_list(self):
        assert _nms_dedup([]) == []


# ═══════════════════════════════════════════════════════════
#  Track Object
# ═══════════════════════════════════════════════════════════

class TestTrack:
    def test_init(self):
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        assert t.track_id == 1
        assert t.state == TrackState.TENTATIVE
        assert t.age == 1
        assert t.hit_streak == 1
        assert t.miss_streak == 0

    def test_avg_confidence(self):
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        t.update_confidence(0.8)
        t.update_confidence(0.9)
        t.update_confidence(1.0)
        assert t.avg_confidence == pytest.approx(0.9, abs=0.01)

    def test_quality_score(self):
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        for _ in range(50):
            t.update_confidence(0.8)
            t.age += 1
        q = t.quality
        assert 0.0 < q <= 1.0

    def test_embedding_pool(self):
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        t._pool_k = 3
        for i in range(5):
            t.add_embedding(_emb(i), 0.5 + i * 0.1)
        # Only top 3 should be kept
        assert len(t._emb_heap) == 3
        assert t.pooled_embedding is not None
        assert t.pooled_embedding.shape == (512,)

    def test_best_embedding_tracked(self):
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        e1 = _emb(1)
        e2 = _emb(2)
        t.add_embedding(e1, 0.7)
        t.add_embedding(e2, 0.9)
        assert t.best_embedding_conf == 0.9
        assert np.allclose(t.best_embedding, e2)


# ═══════════════════════════════════════════════════════════
#  Lost Buffer
# ═══════════════════════════════════════════════════════════

class TestLostBuffer:
    def test_add_and_get(self):
        lb = LostBuffer(max_lost_sec=5.0)
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        lb.add(t, current_time=1.0)
        assert len(lb) == 1
        assert lb.get_all()[0].track_id == 1

    def test_remove(self):
        lb = LostBuffer(max_lost_sec=5.0)
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        lb.add(t, current_time=1.0)
        removed = lb.remove(1)
        assert removed.track_id == 1
        assert len(lb) == 0

    def test_remove_nonexistent(self):
        lb = LostBuffer(max_lost_sec=5.0)
        assert lb.remove(999) is None

    def test_evict_expired(self):
        lb = LostBuffer(max_lost_sec=2.0)
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        lb.add(t, current_time=1.0)
        expired = lb.evict_expired(current_time=4.0)  # 3s > 2s limit
        assert len(expired) == 1
        assert expired[0].track_id == 1

    def test_not_expired_yet(self):
        lb = LostBuffer(max_lost_sec=5.0)
        t = Track(track_id=1, bbox=(100, 200, 180, 400), camera_id="cam1", timestamp=0.0)
        lb.add(t, current_time=1.0)
        expired = lb.evict_expired(current_time=2.0)  # Only 1s < 5s limit
        assert len(expired) == 0
        assert len(lb) == 1


# ═══════════════════════════════════════════════════════════
#  SingleCameraTracker — Integration
# ═══════════════════════════════════════════════════════════

class TestSingleCameraTracker:
    def _make_tracker(self, **kwargs):
        defaults = dict(
            camera_id="cam1", frame_w=640, frame_h=480,
            n_confirm=2, max_miss=5, lost_buffer_sec=3.0,
        )
        defaults.update(kwargs)
        return SingleCameraTracker(**defaults)

    def test_init(self):
        tracker = self._make_tracker()
        assert tracker.camera_id == "cam1"
        assert len(tracker.tracks) == 0
        assert len(tracker.active_ids) == 0

    def test_single_detection_creates_tentative(self):
        tracker = self._make_tracker()
        dets = [_det(100, 200, 180, 400)]
        events = tracker.update(dets, [], {0: _emb(1)}, 0.0)
        assert len(tracker.tentative_ids) == 1
        assert len(tracker.active_ids) == 0

    def test_tentative_promoted_to_active(self):
        tracker = self._make_tracker(n_confirm=2)
        dets = [_det(100, 200, 180, 400)]
        embs = {0: _emb(1)}

        tracker.update(dets, [], embs, 0.0)  # Frame 1: tentative
        assert len(tracker.active_ids) == 0

        events = tracker.update(dets, [], embs, 0.033)  # Frame 2: promoted
        assert len(tracker.active_ids) == 1
        # Should emit TRACK_ACTIVATED event
        activated = [e for e in events if e.type == "TRACK_ACTIVATED"]
        assert len(activated) == 1

    def test_track_follows_moving_detection(self):
        tracker = self._make_tracker(n_confirm=2)
        emb = _emb(42)

        # Frame 1-2: Create and promote
        tracker.update([_det(100, 200, 180, 400)], [], {0: emb}, 0.0)
        tracker.update([_det(100, 200, 180, 400)], [], {0: emb}, 0.033)

        # Frame 3-5: Move detection rightward
        for i in range(3):
            x_offset = (i + 1) * 10
            d = _det(100 + x_offset, 200, 180 + x_offset, 400)
            tracker.update([d], [], {0: emb}, 0.066 + i * 0.033)

        # Track should still exist
        assert len(tracker.active_ids) == 1
        tid = list(tracker.active_ids)[0]
        # Track bbox should have moved rightward
        assert tracker.tracks[tid].bbox[0] > 100

    def test_missed_track_eventually_removed(self):
        tracker = self._make_tracker(n_confirm=1, max_miss=3, lost_buffer_sec=0.1)
        emb = _emb(42)

        tracker.update([_det(100, 200, 180, 400)], [], {0: emb}, 0.0)
        assert len(tracker.active_ids) == 1

        # No detections for several frames
        for i in range(5):
            tracker.update([], [], {}, 0.5 + i * 0.033)

        assert len(tracker.active_ids) == 0

    def test_two_separate_people(self):
        tracker = self._make_tracker(n_confirm=2)
        dets = [
            _det(50, 100, 130, 300),    # Person A
            _det(400, 100, 480, 300),   # Person B
        ]
        embs = {0: _emb(1), 1: _emb(2)}

        tracker.update(dets, [], embs, 0.0)
        tracker.update(dets, [], embs, 0.033)

        assert len(tracker.active_ids) == 2

    def test_set_global_id(self):
        tracker = self._make_tracker(n_confirm=1)
        tracker.update([_det(100, 200, 180, 400)], [], {0: _emb(1)}, 0.0)
        tid = list(tracker.active_ids)[0]

        tracker.set_global_id(tid, 42, "Alice")
        assert tracker.local_to_global[tid] == 42
        assert tracker.tracks[tid].global_id == 42
        assert tracker.tracks[tid].display_name == "Alice"

    def test_get_active_tracks(self):
        tracker = self._make_tracker(n_confirm=1)
        tracker.update([_det(100, 200, 180, 400)], [], {0: _emb(1)}, 0.0)
        active = tracker.get_active_tracks()
        assert len(active) == 1

    def test_low_conf_keeps_track_alive(self):
        """Low-conf detections should prevent track from dying."""
        tracker = self._make_tracker(n_confirm=1, max_miss=3)
        emb = _emb(42)

        # Create active track
        tracker.update([_det(100, 200, 180, 400)], [], {0: emb}, 0.0)
        assert len(tracker.active_ids) == 1

        # Next frames: only low-conf detection nearby
        low_dets = [_det(105, 205, 185, 405, conf=0.15)]
        for i in range(3):
            tracker.update([], low_dets, {}, 0.033 + i * 0.033)

        # Track should still be alive
        assert len(tracker.active_ids) >= 1

    def test_finalize_all(self):
        tracker = self._make_tracker(n_confirm=1, min_export_frames=1)
        tracker.update([_det(100, 200, 180, 400)], [], {0: _emb(1)}, 0.0)

        events = tracker.finalize_all(1.0)
        ended = [e for e in events if e.type == "TRACK_ENDED"]
        assert len(ended) >= 1
        assert len(tracker.tracks) == 0


class TestWideBoxBirth:
    """Wide boxes (crawling / lying / sideways-camera people) must still become tracks;
    only a wide box that covers two separately-detected people is treated as merged."""

    @staticmethod
    def _emb(seed):
        v = np.random.RandomState(seed).randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def _run(self, high, low=()):
        t = SingleCameraTracker("c", frame_w=1920, frame_h=1080, n_confirm=3)
        embs = {i: self._emb(i) for i in range(len(high))}
        for k in range(6):
            t.update(list(high), list(low), embs, k * 0.2)
        return t

    def test_lone_wide_person_is_tracked(self):
        crawler = _det(400, 700, 1000, 850)          # 600 x 150, ratio 4
        assert len(self._run([crawler]).active_ids) == 1

    def test_wide_and_upright_people_both_tracked(self):
        lying, sitting = _det(270, 40, 835, 200), _det(400, 195, 645, 395)
        assert len(self._run([lying, sitting]).active_ids) == 2

    def test_merged_box_over_two_people_rejected(self):
        a, b = _det(100, 300, 300, 500), _det(320, 300, 520, 500)   # two crouching people
        merged = _det(90, 290, 530, 510, conf=0.5)   # one wide box spanning both
        t = self._run([a, b, merged])
        assert len(t.active_ids) == 2
