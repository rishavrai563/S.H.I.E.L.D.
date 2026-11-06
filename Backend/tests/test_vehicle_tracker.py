"""Tests for the IoU vehicle tracker used by ANPR voting and vehicle classification."""
from dataclasses import dataclass

from core.vehicle_tracker import VehicleTracker, iou


@dataclass
class Det:
    bbox: tuple
    confidence: float = 0.9
    label: str = "car"


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_stable_ids_across_frames():
    vt = VehicleTracker()
    seen, _ = vt.update([Det((0, 0, 100, 60)), Det((300, 0, 400, 60))], 0.0)
    ids = sorted(t.track_id for t in seen)
    seen, _ = vt.update([Det((8, 2, 108, 62)), Det((305, 1, 405, 61))], 0.1)
    assert sorted(t.track_id for t in seen) == ids
    assert all(t.confirmed for t in seen)


def test_fast_vehicle_center_fallback():
    vt = VehicleTracker()
    first, _ = vt.update([Det((0, 0, 100, 60))], 0.0)
    second, _ = vt.update([Det((70, 0, 170, 60))], 0.1)  # IoU ~0.18 but centre close
    assert second[0].track_id == first[0].track_id


def test_class_majority_vote():
    vt = VehicleTracker()
    for i, lbl in enumerate(["truck", "truck", "bus", "truck"]):
        seen, _ = vt.update([Det((0, 0, 200, 120), label=lbl)], i * 0.1)
    assert seen[0].label == "truck"


def test_tracks_expire():
    vt = VehicleTracker(max_age_sec=1.0)
    vt.update([Det((0, 0, 100, 60))], 0.0)
    _, ended = vt.update([], 0.5)
    assert ended == []
    _, ended = vt.update([], 2.0)
    assert len(ended) == 1 and vt.tracks == {}


def test_to_dict():
    vt = VehicleTracker()
    seen, _ = vt.update([Det((0, 0, 100, 60), label="motorcycle")], 0.0)
    d = seen[0].to_dict()
    assert d["class_name"] == "motorcycle" and d["plate"] == ""
