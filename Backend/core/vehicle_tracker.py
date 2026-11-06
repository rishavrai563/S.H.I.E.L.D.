"""
Lightweight multi-vehicle tracker (per camera).

Gives each detected vehicle a stable ID across frames so that:
  - ANPR can vote across frames per vehicle (temporal consensus),
  - each vehicle is reported once instead of once per frame,
  - vehicle class is stabilised by majority vote (car / motorcycle / bus / truck).

Association: Hungarian assignment on (1 - IoU), with a centre-distance
fallback for fast vehicles whose boxes stop overlapping between frames.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


@dataclass
class VehicleTrack:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    class_votes: Counter = field(default_factory=Counter)
    plate: str = ""
    watchlist_tag: str = ""

    @property
    def label(self) -> str:
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "vehicle"

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    def to_dict(self) -> Dict:
        return {"id": self.track_id, "class_name": self.label, "bbox": [round(float(v), 1) for v in self.bbox],
                "confidence": round(float(self.confidence), 3), "plate": self.plate,
                "watchlist_tag": self.watchlist_tag, "dwell_sec": round(self.last_seen - self.first_seen, 1)}


class VehicleTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age_sec: float = 2.0,
                 center_gate_ratio: float = 1.0):
        self.iou_threshold = float(iou_threshold)
        self.max_age_sec = float(max_age_sec)
        self.center_gate_ratio = float(center_gate_ratio)
        self.tracks: Dict[int, VehicleTrack] = {}
        self._next_id = 1

    def _cost(self, track: VehicleTrack, det_bbox) -> float:
        ov = iou(track.bbox, det_bbox)
        if ov >= self.iou_threshold:
            return 1.0 - ov
        # centre-distance fallback, normalised by track diagonal
        tx, ty = (track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2
        dx, dy = (det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2
        diag = max(1.0, ((track.bbox[2] - track.bbox[0]) ** 2 + (track.bbox[3] - track.bbox[1]) ** 2) ** 0.5)
        dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 / diag
        return 1.0 + dist if dist <= self.center_gate_ratio else 1e6

    def update(self, detections: List, timestamp: float) -> Tuple[List[VehicleTrack], List[VehicleTrack]]:
        """
        Args:
            detections: objects with .bbox (x1,y1,x2,y2), .confidence, .label
            timestamp:  seconds
        Returns:
            (active_tracks_seen_this_frame, ended_tracks)
        """
        track_ids = list(self.tracks.keys())
        matched_tracks, matched_dets = set(), set()
        seen: List[VehicleTrack] = []

        if track_ids and detections:
            cost = np.array([[self._cost(self.tracks[t], d.bbox) for d in detections] for t in track_ids])
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(cost)
                pairs = list(zip(rows, cols))
            else:  # greedy fallback
                pairs, used_r, used_c = [], set(), set()
                for r, c in sorted(np.ndindex(cost.shape), key=lambda rc: cost[rc]):
                    if r not in used_r and c not in used_c:
                        pairs.append((r, c)); used_r.add(r); used_c.add(c)
            for r, c in pairs:
                if cost[r, c] >= 1e5:
                    continue
                tr, det = self.tracks[track_ids[r]], detections[c]
                tr.bbox = tuple(float(v) for v in det.bbox)
                tr.confidence = float(det.confidence)
                tr.last_seen = timestamp
                tr.hits += 1
                tr.misses = 0
                tr.class_votes[det.label] += 1
                matched_tracks.add(tr.track_id)
                matched_dets.add(c)
                seen.append(tr)

        for c, det in enumerate(detections):
            if c in matched_dets:
                continue
            tr = VehicleTrack(track_id=self._next_id, bbox=tuple(float(v) for v in det.bbox),
                              confidence=float(det.confidence), first_seen=timestamp, last_seen=timestamp)
            tr.class_votes[det.label] += 1
            self.tracks[tr.track_id] = tr
            self._next_id += 1
            seen.append(tr)

        ended = []
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if tid not in matched_tracks and tr.last_seen != timestamp:
                tr.misses += 1
            if timestamp - tr.last_seen > self.max_age_sec:
                ended.append(self.tracks.pop(tid))
        return seen, ended
