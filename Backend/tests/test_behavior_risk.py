"""
Tests for the Behavioral Risk Analyzer.
Uses synthetic track data — no ML dependencies.
"""
import pytest
from core.behavior_risk import (
    RiskAnalyzer,
    TrackBehaviorState,
    TemporalSmoother,
    TrackStateManager,
    InteractionDetector,
)


# ═══════════════════════════════════════════════════════════
#  TemporalSmoother
# ═══════════════════════════════════════════════════════════

class TestTemporalSmoother:
    def test_moving_average_empty(self):
        from collections import deque
        s = TemporalSmoother()
        assert s.moving_average(deque()) == 0.0

    def test_moving_average_values(self):
        from collections import deque
        s = TemporalSmoother()
        vals = deque([10.0, 20.0, 30.0])
        assert s.moving_average(vals) == pytest.approx(20.0)

    def test_stable_label_same(self):
        s = TemporalSmoother(min_switch_frames=3)
        st = TrackBehaviorState(
            id=1, positions_history=None, velocity_history=None,
            last_seen_frame=0, state_label="NORMAL",
        )
        label, dur = s.stable_label(st, "NORMAL")
        assert label == "NORMAL"
        assert dur >= 1

    def test_stable_label_requires_dwell(self):
        """Label shouldn't switch until min_switch_frames is met."""
        s = TemporalSmoother(min_switch_frames=3)
        st = TrackBehaviorState(
            id=1, positions_history=None, velocity_history=None,
            last_seen_frame=0, state_label="NORMAL",
        )
        # Propose SUSPICIOUS for fewer than 3 frames
        label1, _ = s.stable_label(st, "SUSPICIOUS_MOTION")
        assert label1 == "NORMAL"  # Not yet switched

        label2, _ = s.stable_label(st, "SUSPICIOUS_MOTION")
        assert label2 == "NORMAL"  # Still waiting

        label3, _ = s.stable_label(st, "SUSPICIOUS_MOTION")
        assert label3 == "SUSPICIOUS_MOTION"  # Now switches


# ═══════════════════════════════════════════════════════════
#  TrackStateManager
# ═══════════════════════════════════════════════════════════

class TestTrackStateManager:
    def test_new_track_created(self):
        tsm = TrackStateManager()
        tracks = [{"id": 1, "bbox": [10, 20, 50, 80]}]
        states, exits = tsm.update_tracks(0, tracks)
        assert 1 in states
        assert exits == []

    def test_track_velocity_computed(self):
        tsm = TrackStateManager()
        tsm.update_tracks(0, [{"id": 1, "bbox": [10, 20, 50, 80]}])
        tsm.update_tracks(1, [{"id": 1, "bbox": [20, 25, 60, 85]}])
        st = tsm.states[1]
        assert len(st.velocity_history) == 2
        assert st.velocity_history[-1] > 0  # Should have moved

    def test_disappeared_track_exit(self):
        tsm = TrackStateManager()
        tsm.update_tracks(0, [{"id": 1, "bbox": [10, 20, 50, 80]}])
        _, exits = tsm.update_tracks(1, [])  # Track 1 disappeared
        assert 1 in exits
        assert 1 not in tsm.states

    def test_multiple_tracks(self):
        tsm = TrackStateManager()
        tracks = [
            {"id": 1, "bbox": [10, 20, 50, 80]},
            {"id": 2, "bbox": [200, 200, 280, 400]},
        ]
        states, _ = tsm.update_tracks(0, tracks)
        assert len(states) == 2


# ═══════════════════════════════════════════════════════════
#  InteractionDetector
# ═══════════════════════════════════════════════════════════

class TestInteractionDetector:
    def test_no_interaction_when_far(self):
        det = InteractionDetector(proximity_px=100.0, hold_frames=3)
        tracks = [
            {"id": 1, "bbox": [0, 0, 50, 100]},
            {"id": 2, "bbox": [500, 500, 550, 600]},
        ]
        result = det.update(tracks)
        assert len(result) == 0

    def test_interaction_after_hold(self):
        det = InteractionDetector(proximity_px=200.0, hold_frames=2)
        tracks = [
            {"id": 1, "bbox": [100, 100, 150, 200]},
            {"id": 2, "bbox": [140, 100, 190, 200]},
        ]
        det.update(tracks)  # Frame 1
        det.update(tracks)  # Frame 2 — hold met
        result = det.update(tracks)  # Frame 3
        # Tracks should be interacting now
        assert 1 in result or 2 in result

    def test_single_track_no_interaction(self):
        det = InteractionDetector()
        tracks = [{"id": 1, "bbox": [100, 100, 200, 300]}]
        result = det.update(tracks)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════
#  RiskAnalyzer (Full Pipeline)
# ═══════════════════════════════════════════════════════════

class TestRiskAnalyzer:
    def test_init(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080)
        assert ra.frame_width == 1920
        assert ra.frame_height == 1080
        assert ra.virtual_fence_line is not None

    def test_empty_tracks(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080)
        result = ra.update(frame_idx=0, tracks=[])
        assert result["tracks"] == []
        assert result["scene_status"] == "CALM"
        assert "scene_density" in result

    def test_single_normal_track(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080, fps=30.0)
        tracks = [{"id": 1, "bbox": [100, 200, 180, 400]}]
        result = ra.update(frame_idx=0, tracks=tracks)
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["id"] == 1
        assert "label" in result["tracks"][0]
        assert "confidence" in result["tracks"][0]
        assert "speed" in result["tracks"][0]

    def test_new_entry_label(self):
        """Brand new track should get NEW_ENTRY label."""
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080, fps=30.0, new_entry_seconds=2.0)
        tracks = [{"id": 1, "bbox": [100, 200, 180, 400]}]
        result = ra.update(frame_idx=0, tracks=tracks)
        # New tracks start as NEW_ENTRY (unless higher priority fires)
        label = result["tracks"][0]["label"]
        assert label in ("NEW_ENTRY", "NORMAL", "ISOLATED")

    def test_scene_density_increases_with_crowd(self):
        ra = RiskAnalyzer(frame_width=640, frame_height=480, fps=30.0)
        few_tracks = [{"id": i, "bbox": [i * 50, 100, i * 50 + 40, 300]} for i in range(2)]
        r1 = ra.update(frame_idx=0, tracks=few_tracks)

        ra2 = RiskAnalyzer(frame_width=640, frame_height=480, fps=30.0)
        many_tracks = [{"id": i, "bbox": [i * 30, 100, i * 30 + 25, 300]} for i in range(15)]
        r2 = ra2.update(frame_idx=0, tracks=many_tracks)

        assert r2["scene_density"] > r1["scene_density"]

    def test_update_fence(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080)
        old_coords = ra.virtual_fence_coords
        new_coords = [(100, 200), (800, 200)]
        ra.update_fence(new_coords)
        assert ra.virtual_fence_coords == new_coords
        assert ra.virtual_fence_coords != old_coords

    def test_priority_ordering(self):
        """INTRUSION should have highest priority."""
        assert RiskAnalyzer.PRIORITY["INTRUSION"] > RiskAnalyzer.PRIORITY["SUSPICIOUS_MOTION"]
        assert RiskAnalyzer.PRIORITY["SUSPICIOUS_MOTION"] > RiskAnalyzer.PRIORITY["NORMAL"]
        assert RiskAnalyzer.PRIORITY["NORMAL"] == 0

    def test_track_over_time(self):
        """Simulate a track across multiple frames."""
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080, fps=30.0)
        for frame in range(30):
            x = 100 + frame * 5
            tracks = [{"id": 1, "bbox": [x, 200, x + 80, 400]}]
            result = ra.update(frame_idx=frame, tracks=tracks)

        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["speed"] > 0

    def test_exits_reported(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080)
        ra.update(frame_idx=0, tracks=[{"id": 1, "bbox": [100, 200, 180, 400]}])
        result = ra.update(frame_idx=1, tracks=[])  # Track 1 exits
        assert 1 in result["exits"]


class TestLoiteringAndFence:
    def test_loitering_after_dwell(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080, fps=10.0, loiter_seconds=5.0,
                          loiter_displacement_px=30.0, min_switch_frames=2)
        labels = []
        for i in range(80):  # 8 s standing still (small jitter)
            jitter = (i % 3) * 2
            res = ra.update(frame_idx=i, tracks=[{"id": 1, "bbox": [500 + jitter, 300, 560 + jitter, 450]}],
                            timestamp=i * 0.1)
            labels.append(res["tracks"][0]["label"])
        assert "LOITERING" in labels[55:]
        assert res["tracks"][0]["dwell_sec"] >= 5.0

    def test_moving_person_never_loiters(self):
        ra = RiskAnalyzer(frame_width=1920, frame_height=1080, fps=10.0, loiter_seconds=3.0, min_switch_frames=2)
        for i in range(80):
            res = ra.update(frame_idx=i, tracks=[{"id": 1, "bbox": [100 + i * 10, 300, 160 + i * 10, 450]}],
                            timestamp=i * 0.1)
            assert res["tracks"][0]["label"] != "LOITERING"

    def test_fence_crossing_flags_intrusion_and_cleans_up(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000, fps=10.0, min_switch_frames=1)
        ra.update_fence([(0, 500), (1000, 500)])
        for i, y2 in enumerate([450, 480, 520, 560, 600, 640, 680, 700]):
            res = ra.update(frame_idx=i, tracks=[{"id": 7, "bbox": [400, y2 - 150, 460, y2]}], timestamp=i * 0.1)
        assert 7 in ra.intruders
        assert res["tracks"][0]["label"] == "INTRUSION"
        assert res["scene_status"] == "BREACH_DETECTED"
        ra.update(frame_idx=99, tracks=[], timestamp=9.9)   # track leaves
        assert 7 not in ra.intruders


class TestDrawnFences:
    @staticmethod
    def _walk(ra, xs, ys, tid=7):
        res = None
        for i, (x, y) in enumerate(zip(xs, ys)):
            res = ra.update(frame_idx=i, tracks=[{"id": tid, "bbox": [x - 30, y - 150, x + 30, y]}], timestamp=i * 0.1)
        return res

    def test_zone_flags_while_inside_and_clears_on_exit(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000, fps=10.0, min_switch_frames=6)
        assert ra.update_fence([(300, 300), (700, 300), (700, 700), (300, 700)], "zone")
        res = self._walk(ra, [100, 200, 400], [500] * 3)              # steps into the square
        assert 7 in ra.zone_occupants and 7 not in ra.intruders
        assert res["tracks"][0]["label"] == "RESTRICTED_ZONE"         # instantly, despite smoothing
        assert res["tracks"][0]["in_zone"] is True
        res = self._walk(ra, [800], [500])                             # one step out
        assert 7 not in ra.zone_occupants
        assert res["tracks"][0]["label"] != "RESTRICTED_ZONE"         # cleared the moment they leave

    def test_irregular_polygon_zone(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000)
        assert ra.update_fence([(100, 100), (900, 100), (500, 400), (900, 900), (100, 900)], "zone")  # notched
        self._walk(ra, [800, 800], [400, 400])     # inside the notch = outside the zone
        assert 7 not in ra.zone_occupants
        self._walk(ra, [200, 200], [500, 500])
        assert 7 in ra.zone_occupants

    def test_tripwire_label_is_immediate(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000, fps=10.0, min_switch_frames=6)
        ra.update_fence([(0, 500), (1000, 500)], "line")
        res = self._walk(ra, [500, 500], [450, 550])
        assert res["tracks"][0]["label"] == "INTRUSION"

    def test_multi_segment_tripwire(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000)
        assert ra.update_fence([(0, 600), (400, 600), (600, 300), (1000, 300)], "line")
        self._walk(ra, [800, 800, 800], [200, 280, 360])   # crosses the 3rd segment
        assert 7 in ra.intruders

    def test_invalid_shapes_rejected_and_disable(self):
        ra = RiskAnalyzer(frame_width=1000, frame_height=1000)
        assert ra.update_fence([(0, 0), (10, 10)], "zone") is False
        assert ra.update_fence([(0, 0), (100, 0), (200, 0)], "zone") is False   # no area
        assert ra.update_fence([], "line") is True
        self._walk(ra, [500, 500, 500], [100, 500, 900])
        assert not ra.intruders and ra.fence_dict()["points"] == []
