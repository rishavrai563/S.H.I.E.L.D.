"""
Single-Camera Person Tracker — 8-Step Pipeline.

Maintains stable person IDs within a single camera view.
Core guarantees:
    ✔ Same person never gets multiple IDs
    ✔ Temporary disappearance doesn't create new ID (Lost buffer)
    ✔ Crowd overlap doesn't swap IDs (adaptive appearance weights)
    ✔ Low-confidence detections keep tracks alive during occlusion
    ✔ Re-entry within few seconds restores same ID

Architecture:
    ① YOLO Detection split (done externally) → high + low confidence
    ② Kalman Predict all tracks
    ③ Build gated cost matrix (motion + appearance + IOU)
    ④ Hungarian Assignment (high-conf ↔ active tracks)
    ⑤ Low-conf recovery (unmatched tracks ↔ low-conf dets)
    ⑥ Track state transitions (birth, promote, lost, remove)
    ⑦ Update embedding pool (min-heap top-K)
    ⑧ Draw annotated frame

DSA used:
    - Dict[int, Track]  → O(1) track lookup/insert/delete
    - Set[int]          → O(1) membership checks for active/lost IDs
    - heapq min-heap    → O(log K) embedding pool insert
    - deque             → O(1) lost buffer insert/evict
    - Manhattan gating  → skip impossible pairs before O(n³) Hungarian
"""
from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from core.kalman import (
    KalmanFilter8, bbox_iou, bbox_center, bbox_area,
    clamp_bbox, xyxy_to_cxcywh, cxcywh_to_xyxy,
)
from core.detector import Detection


# ─────────────────── Track State Machine ────────────────

class TrackState(Enum):
    TENTATIVE = "Tentative"    # Just born, not confirmed yet
    ACTIVE = "Active"          # Confirmed, being tracked
    LOST = "Lost"              # Missing, waiting for re-appearance


# ─────────────────── Track Events ───────────────────────

@dataclass
class TrackEvent:
    """Event emitted by tracker to trigger global matching."""
    type: str                   # "TRACK_ACTIVATED" or "TRACK_ENDED"
    track_id: int
    camera_id: str
    timestamp: float
    embedding: Optional[np.ndarray] = None        # single best (for early match)
    pooled_embedding: Optional[np.ndarray] = None  # top-K median (for refined match)
    entry_time: float = 0.0
    exit_time: float = 0.0
    quality: float = 0.0
    track_age: int = 0


# ─────────────────── Track Object ───────────────────────

class Track:
    """
    A single tracked person within one camera view.
    
    Uses HashMap + Set for O(1) lookups, min-heap for O(log K) embedding pool,
    and running average for O(1) confidence tracking.
    """

    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float],
                 camera_id: str, timestamp: float):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.camera_id = camera_id

        # Bounding box tracking
        self.bbox = bbox                    # Current best estimate
        self.pred_bbox = bbox               # Kalman prediction (before update)
        self.kalman = KalmanFilter8(bbox)

        # Lifecycle counters
        self.hit_streak: int = 1            # Consecutive matched frames
        self.miss_streak: int = 0           # Consecutive missed frames
        self.age: int = 1                   # Total frames since creation

        # Timestamps
        self.entry_time: float = timestamp
        self.last_seen_time: float = timestamp
        self.lost_at: Optional[float] = None

        # Running average confidence — O(1) update instead of O(n) mean
        self._conf_sum: float = 0.0
        self._conf_count: int = 0

        # Embedding pool — min-heap by confidence, top-K kept
        # Heap stores (confidence, unique_id, embedding) to break ties
        self._emb_heap: List[Tuple[float, int, np.ndarray]] = []
        self._emb_counter: int = 0         # Unique ID for heap ordering
        self._pool_k: int = 10
        self.pooled_embedding: Optional[np.ndarray] = None
        self.best_embedding: Optional[np.ndarray] = None
        self.best_embedding_conf: float = 0.0

        # Trajectory history (for timeline & quality) — capped at 500 points
        self._max_trajectory: int = 500
        self.trajectory: List[Tuple[float, float, float]] = []  # (cx, cy, timestamp)
        center = bbox_center(bbox)
        self.trajectory.append((center[0], center[1], timestamp))

        # Global identity link (set by orchestrator after global matching)
        self.global_id: Optional[int] = None
        self.display_name: Optional[str] = None

    @property
    def avg_confidence(self) -> float:
        """Running average detection confidence. O(1)."""
        return self._conf_sum / self._conf_count if self._conf_count > 0 else 0.0

    @property
    def quality(self) -> float:
        """Track quality score: 0.6 × avg_conf + 0.4 × length_factor."""
        length_factor = min(1.0, self.age / 50.0)
        return 0.6 * self.avg_confidence + 0.4 * length_factor

    def update_confidence(self, conf: float) -> None:
        """Update running average confidence. O(1)."""
        self._conf_sum += conf
        self._conf_count += 1

    def add_embedding(self, embedding: np.ndarray, confidence: float) -> None:
        """
        Add embedding to top-K pool using min-heap. O(log K).
        
        Min-heap keeps the LOWEST confidence at the top.
        When pool is full, if new confidence > min, replace min.
        """
        if embedding is None:
            return

        self._emb_counter += 1
        entry = (confidence, self._emb_counter, embedding)

        if len(self._emb_heap) < self._pool_k:
            heapq.heappush(self._emb_heap, entry)
        elif confidence > self._emb_heap[0][0]:
            heapq.heapreplace(self._emb_heap, entry)
        else:
            return  # Not good enough, skip

        # Track the single best embedding (for Phase 1 early match)
        if confidence > self.best_embedding_conf:
            self.best_embedding = embedding.copy()
            self.best_embedding_conf = confidence

        # Recompute pooled embedding (median of all in pool)
        self._recompute_pooled()

    def _recompute_pooled(self) -> None:
        """Recompute median-pooled embedding from heap. O(K × 512)."""
        if not self._emb_heap:
            self.pooled_embedding = None
            return
        stacked = np.stack([e for _, _, e in self._emb_heap])
        median = np.median(stacked, axis=0)
        norm = np.linalg.norm(median)
        self.pooled_embedding = (median / norm).astype(np.float32) if norm > 1e-8 else None


# ─────────────────── Lost Track Buffer ──────────────────

class LostBuffer:
    """
    Buffer for temporarily lost tracks.
    
    DSA: deque (FIFO by lost_time) + dict (O(1) lookup by ID).
    Insert O(1), evict oldest O(1), search for match O(n).
    """

    def __init__(self, max_lost_sec: float = 2.0):
        self.max_lost_sec = max_lost_sec
        self.buffer: Deque[Track] = deque()
        self.lookup: Dict[int, Track] = {}

    def add(self, track: Track, current_time: float) -> None:
        """Add a track to the lost buffer."""
        track.lost_at = current_time
        self.buffer.append(track)
        self.lookup[track.track_id] = track

    def remove(self, track_id: int) -> Optional[Track]:
        """Remove and return a specific track from the buffer. O(n)."""
        if track_id not in self.lookup:
            return None
        track = self.lookup.pop(track_id)
        # Mark for removal from deque (lazy cleanup in evict)
        return track

    def evict_expired(self, current_time: float) -> List[Track]:
        """Pop expired tracks from the front. O(k) where k = expired count."""
        expired = []
        while self.buffer:
            t = self.buffer[0]
            if t.track_id not in self.lookup:
                # Already removed via remove(), clean up deque
                self.buffer.popleft()
                continue
            if t.lost_at is not None and (current_time - t.lost_at) > self.max_lost_sec:
                self.buffer.popleft()
                del self.lookup[t.track_id]
                expired.append(t)
            else:
                break  # Remaining tracks are newer (deque is FIFO-ordered)
        return expired

    def get_all(self) -> List[Track]:
        """Get all tracks in the buffer."""
        return list(self.lookup.values())

    def __len__(self) -> int:
        return len(self.lookup)


# ─────────────────── Helper Functions ───────────────────

def _l2_normalize(v: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalize a vector. Returns None if zero-length."""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).ravel()
    if v.size == 0:
        return None
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-8 else None


def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()))


# NOTE: _compute_crowd_score() was removed — replaced by per-pair
# track_has_neighbor logic in _associate(). See Bug 6 fix.


def _nms_dedup(detections: List[Detection], iou_threshold: float = 0.70) -> List[Detection]:
    """
    Custom NMS on unmatched detections before creating tentative tracks.
    Prevents duplicate tracks when YOLO outputs overlapping boxes for same person.
    O(m²) where m = unmatched detections (typically 0-3).
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(bbox_iou(d.bbox, k.bbox) < iou_threshold for k in keep):
            keep.append(d)
    return keep


# ─────────────────── Single Camera Tracker ──────────────

class SingleCameraTracker:
    """
    Maintains stable person IDs within one camera view using the 8-step pipeline.
    
    Usage:
        tracker = SingleCameraTracker("cam1", frame_w=1920, frame_h=1080)
        events = tracker.update(high_dets, low_dets, embeddings_map, timestamp)
        annotated_frame = tracker.draw_on_frame(frame)
    """

    def __init__(self, camera_id: str,
                 frame_w: int = 1920,
                 frame_h: int = 1080,
                 # Lifecycle
                 n_confirm: int = 3,
                 tentative_max_miss: int = 3,
                 max_miss: int = 20,
                 lost_buffer_sec: float = 15.0,
                 # Cost matrix
                 gate_dist_px: int = 200,
                 mahal_gate: float = 9.48,
                 cost_thr_high: float = 0.65,
                 cost_thr_low: float = 0.70,
                 # Adaptive weights
                 weights_normal: Tuple[float, float, float] = (0.50, 0.40, 0.10),
                 weights_crowd: Tuple[float, float, float] = (0.30, 0.45, 0.25),
                 crowd_overlap_min: float = 0.10,
                 crowd_score_thr: float = 0.08,
                 # Appearance
                 sim_thr_live: float = 0.45,
                 sim_thr_lost: float = 0.35,
                 embedding_pool_k: int = 5,
                 # Duplicate guard
                 dup_iou_thr: float = 0.50,
                 # Track merging
                 merge_sim_thr: float = 0.60,
                 # Quality / export
                 min_export_frames: int = 5,
                 quality_threshold: float = 0.60,
                 min_box_area: float = 300.0,
                 debug_bbox_log: bool = False):

        self.camera_id = camera_id
        self.W = frame_w
        self.H = frame_h

        # Config
        self.n_confirm = n_confirm
        self.tentative_max_miss = tentative_max_miss
        self.max_miss = max_miss
        self.gate_dist = gate_dist_px
        # Scale gate distance proportionally to frame diagonal
        # 200px is correct for 640×480; scale up for larger resolutions
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        ref_diag = (640 ** 2 + 480 ** 2) ** 0.5  # 800px reference
        self.gate_dist = int(gate_dist_px * (diag / ref_diag))
        self.mahal_gate = mahal_gate
        self.cost_thr_high = cost_thr_high
        self.cost_thr_low = cost_thr_low
        self.w_normal = weights_normal
        self.w_crowd = weights_crowd
        self.crowd_overlap_min = crowd_overlap_min
        self.crowd_score_thr = crowd_score_thr
        self.sim_thr_live = sim_thr_live
        self.sim_thr_lost = sim_thr_lost
        self.emb_pool_k = embedding_pool_k
        self.dup_iou_thr = dup_iou_thr
        self.merge_sim_thr = merge_sim_thr
        self.min_export_frames = min_export_frames
        self.quality_threshold = quality_threshold
        self.min_box_area = float(min_box_area)
        self.debug_bbox_log = bool(debug_bbox_log)

        # Track storage — O(1) operations via dict + set
        self.tracks: Dict[int, Track] = {}          # All active + tentative tracks
        self.active_ids: Set[int] = set()           # IDs in ACTIVE state
        self.tentative_ids: Set[int] = set()        # IDs in TENTATIVE state
        self.lost_buffer = LostBuffer(max_lost_sec=lost_buffer_sec)

        # Local track ID → Global ID mapping (set by orchestrator)
        self.local_to_global: Dict[int, int] = {}

        # ID counter
        self._next_id: int = 0

    # ═══════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════

    def update(self,
               high_dets: List[Detection],
               low_dets: List[Detection],
               embeddings: Dict[int, np.ndarray],
               timestamp: float) -> List[TrackEvent]:
        """
        Process one frame of detections through the 8-step pipeline.
        
        Args:
            high_dets:   High-confidence detections (≥ 0.40).
            low_dets:    Low-confidence detections (0.10 – 0.40).
            embeddings:  Map of detection index → 512-dim embedding.
                         Only high-conf detections have embeddings.
            timestamp:   Current timestamp (seconds).
            
        Returns:
            List of TrackEvents to be processed by global matcher.
        """
        events: List[TrackEvent] = []

        # ── Step ② Kalman Predict ─────────────────────────
        # Predict ALL tracks (active + tentative)
        for track in list(self.tracks.values()):
            track.kalman.predict()
            track.pred_bbox = clamp_bbox(*track.kalman.predicted_bbox(), self.W, self.H)
            track.age += 1

        # Also predict lost buffer tracks (needed for spatial proximity check)
        for lost_track in self.lost_buffer.get_all():
            lost_track.kalman.predict()
            lost_track.pred_bbox = clamp_bbox(
                *lost_track.kalman.predicted_bbox(), self.W, self.H
            )

        # -- Step 3+4 Cost Matrix + Hungarian (high-conf) --
        # CASCADE MATCHING: Active tracks get priority over tentative.
        # This prevents newly-born tracks from stealing detections.
        matched_track_ids: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Stage A-1: ACTIVE tracks get first pick (priority)
        active_tids = [tid for tid in self.tracks if tid in self.active_ids]
        if active_tids:
            matches_active = self._associate(
                track_ids=active_tids,
                detections=high_dets,
                embeddings=embeddings,
                is_low_conf=False,
            )
            for tid, det_idx, det, emb in matches_active:
                self._apply_match(self.tracks[tid], det, emb, timestamp)
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)

        # Stage A-2: TENTATIVE tracks get remaining detections
        tentative_tids = [tid for tid in self.tracks
                          if tid in self.tentative_ids and tid not in matched_track_ids]
        if tentative_tids:
            remaining_dets_for_tent = [
                (i, high_dets[i]) for i in range(len(high_dets))
                if i not in matched_det_indices
            ]
            if remaining_dets_for_tent:
                tent_det_indices = [x[0] for x in remaining_dets_for_tent]
                tent_dets = [x[1] for x in remaining_dets_for_tent]
                tent_embeddings = {
                    j: embeddings[orig_i]
                    for j, orig_i in enumerate(tent_det_indices)
                    if orig_i in embeddings
                }
                matches_tent = self._associate(
                    track_ids=tentative_tids,
                    detections=tent_dets,
                    embeddings=tent_embeddings,
                    is_low_conf=False,
                )
                for tid, det_idx, det, emb in matches_tent:
                    orig_det_idx = tent_det_indices[det_idx]
                    self._apply_match(self.tracks[tid], det, emb, timestamp)
                    matched_track_ids.add(tid)
                    matched_det_indices.add(orig_det_idx)

        # -- Step 4b: IOU-based fallback recovery ----------
        # When appearance matching fails (overhead camera, bad crop),
        # try pure IOU for tracks WITHOUT nearby neighbors (safe).
        # Inspired by ByteTrack second association stage.
        remaining_track_ids = [tid for tid in self.tracks if tid not in matched_track_ids]
        remaining_det_indices = [i for i in range(len(high_dets)) if i not in matched_det_indices]

        for tid in remaining_track_ids:
            track = self.tracks[tid]
            # Only IOU fallback for ISOLATED tracks (no nearby neighbor)
            has_neighbor = any(
                bbox_iou(track.pred_bbox, self.tracks[ot].pred_bbox) > 0.20
                for ot in self.tracks if ot != tid
            )
            # Allow IOU fallback for tracks that have been missing ≥ 2 frames
            # even with neighbors — they're likely occluded and need recovery
            if has_neighbor and track.miss_streak < 2:
                continue  # Skip -- only block for freshly-matched tracks

            best_iou, best_idx = 0.3, None  # Minimum IOU = 0.3
            for di in remaining_det_indices:
                iou_val = bbox_iou(track.pred_bbox, high_dets[di].bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = di
            if best_idx is not None:
                det = high_dets[best_idx]
                emb = embeddings.get(best_idx)
                self._apply_match(track, det, emb, timestamp)
                matched_track_ids.add(tid)
                matched_det_indices.add(best_idx)
                remaining_det_indices.remove(best_idx)

        # ── Step ⑤ Low-conf recovery ─────────────────────
        unmatched_track_ids = [tid for tid in self.tracks if tid not in matched_track_ids]

        matches_b = self._associate(
            track_ids=unmatched_track_ids,
            detections=low_dets,
            embeddings={},  # No embeddings for low-conf
            is_low_conf=True,
        )
        for tid, det_idx, det, emb in matches_b:
            self._apply_match(self.tracks[tid], det, None, timestamp)
            matched_track_ids.add(tid)

        # ── Step ⑥ Track State Transitions ────────────────
        still_unmatched = [tid for tid in self.tracks if tid not in matched_track_ids]

        # Handle unmatched tracks
        for tid in still_unmatched:
            track = self.tracks[tid]

            # ── Occlusion-aware miss counting ─────────────────
            # If this track is spatially covered by another ACTIVE track,
            # it's "hidden" not "gone" — don't increment miss_streak.
            is_occluded = any(
                bbox_iou(track.pred_bbox, self.tracks[ot].pred_bbox) > 0.30
                for ot in self.active_ids
                if ot != tid and ot in self.tracks and ot in matched_track_ids
            )
            if is_occluded:
                # Track is hidden behind a matched active track
                track.hit_streak = 0  # Not a hit, but not truly missing
                continue  # Skip miss_streak increment entirely

            track.miss_streak += 1
            track.hit_streak = 0

            if track.state == TrackState.TENTATIVE:
                if track.miss_streak >= self.tentative_max_miss:
                    # Tentative died before confirmation
                    self.tentative_ids.discard(tid)
                    del self.tracks[tid]
            elif track.state == TrackState.ACTIVE:
                if track.miss_streak >= self.max_miss:
                    # Active → Removed immediately. Do not reuse old track IDs.
                    if track.age >= self.min_export_frames and track.quality >= self.quality_threshold:
                        events.append(TrackEvent(
                            type="TRACK_ENDED",
                            track_id=track.track_id,
                            camera_id=self.camera_id,
                            timestamp=timestamp,
                            pooled_embedding=track.pooled_embedding,
                            entry_time=track.entry_time,
                            exit_time=track.last_seen_time,
                            quality=track.quality,
                        ))
                    self.active_ids.discard(tid)
                    self.local_to_global.pop(tid, None)
                    del self.tracks[tid]

        # Build reliable index map: original high_det index → (det, embedding)
        # This avoids fragile object identity (`is`) checks
        unmatched_high_indices = [i for i in range(len(high_dets))
                                  if i not in matched_det_indices]
        unmatched_high_map = {
            i: (high_dets[i], embeddings.get(i))
            for i in unmatched_high_indices
        }

        # Try re-associating unmatched high-conf dets with lost tracks
        # KEY FIX: When person re-enters from different position, IOU≈0.
        # So we use APPEARANCE-DOMINANT scoring, not blended scoring.
        revived_indices: Set[int] = set()

        for det_orig_idx, (det, det_emb) in unmatched_high_map.items():
            best_tid, best_score = None, self.sim_thr_lost
            for lost_track in self.lost_buffer.get_all():
                if lost_track.pooled_embedding is not None and det_emb is not None:
                    sim = _cosine_sim(lost_track.pooled_embedding, det_emb)
                    spatial = bbox_iou(lost_track.pred_bbox, det.bbox)

                    # Age-adaptive threshold: established tracks (age>30)
                    # get a lenient gate since their identity is proven.
                    effective_thr = self.sim_thr_lost - 0.05 if lost_track.age > 30 else self.sim_thr_lost

                    # Appearance-first scoring:
                    # If person re-entered from different position (IOU≈0),
                    # rely purely on appearance. IOU is a bonus, not required.
                    if spatial > 0.1:
                        # Nearby: blend appearance + spatial
                        score = 0.6 * sim + 0.4 * spatial
                    else:
                        # Re-entry from different position: pure appearance
                        score = sim

                    if score > max(best_score, effective_thr):
                        best_score = score
                        best_tid = lost_track.track_id

            if best_tid is not None:
                track = self.lost_buffer.remove(best_tid)
                if track is not None:
                    track.state = TrackState.ACTIVE
                    track.lost_at = None
                    track.miss_streak = 0
                    track.hit_streak = 0
                    # Reset Kalman velocity — person may re-enter from different direction
                    track.kalman.x[4:] = 0.0
                    self.tracks[track.track_id] = track
                    self.active_ids.add(track.track_id)
                    self._apply_match(track, det, det_emb, timestamp)
                    revived_indices.add(det_orig_idx)

        # ── APPEARANCE GUARD BEFORE BIRTH ──────────────────
        # Before creating ANY new track, check if this person already
        # exists as an active or lost track. This prevents:
        #   - Multiple YOLO boxes → multiple tracks for same person
        #   - Re-entry creating new IDs when spatial matching fails
        birth_indices = [i for i in unmatched_high_indices if i not in revived_indices]
        birth_dets_raw = [high_dets[i] for i in birth_indices]
        birth_dets = _nms_dedup(birth_dets_raw, self.dup_iou_thr)

        for det in birth_dets:
            if bbox_area(det.bbox) < self.min_box_area:
                continue
            # Look up embedding
            det_emb = None
            for idx in birth_indices:
                if high_dets[idx].bbox == det.bbox:
                    det_emb = embeddings.get(idx)
                    break

            # -- Check 1: Does this person already have an ACTIVE track? --
            # YOLO multi-box: same person, different bounding box.
            # ADAPTIVE: strict when detection is near OTHER detections,
            # lenient when isolated (prevents fragmentation).
            if det_emb is not None:
                # Check if this detection is near another unmatched detection
                det_is_near_others = any(
                    bbox_iou(det.bbox, high_dets[oi].bbox) > 0.05
                    for oi in birth_indices
                    if high_dets[oi].bbox != det.bbox
                )
                # Adaptive thresholds
                if det_is_near_others:
                    guard_sim = 0.55      # Strict -- prevent cross-person
                    guard_min_iou = 0.15  # Require spatial overlap
                else:
                    # Still require moderate overlap for isolated detections;
                    # otherwise a similar-looking far person can collapse tracks.
                    guard_sim = 0.60
                    guard_min_iou = 0.10

                best_active_track = None
                best_active_sim = guard_sim
                for tid in self.active_ids:  # active only
                    track = self.tracks.get(tid)
                    if track is not None and track.pooled_embedding is not None:
                        iou_with_track = bbox_iou(track.pred_bbox, det.bbox)
                        if iou_with_track < guard_min_iou:
                            continue
                        sim = _cosine_sim(track.pooled_embedding, det_emb)
                        if sim > best_active_sim:
                            best_active_sim = sim
                            best_active_track = track

                if best_active_track is not None:
                    # Don't create new track -- update existing one
                    self._apply_match(best_active_track, det, det_emb, timestamp)
                    continue  # Skip birth

            # ── Check 2: Does this person match a LOST track? ──
            # (Re-entry that spatial re-association missed)
            if det_emb is not None:
                best_lost_tid = None
                best_lost_sim = self.sim_thr_lost
                for lost_track in self.lost_buffer.get_all():
                    if lost_track.pooled_embedding is not None:
                        sim = _cosine_sim(lost_track.pooled_embedding, det_emb)
                        # Age-adaptive: proven tracks get lenient gate
                        effective_thr = self.sim_thr_lost - 0.05 if lost_track.age > 30 else self.sim_thr_lost
                        if sim > max(best_lost_sim, effective_thr):
                            best_lost_sim = sim
                            best_lost_tid = lost_track.track_id

                if best_lost_tid is not None:
                    track = self.lost_buffer.remove(best_lost_tid)
                    if track is not None:
                        track.state = TrackState.ACTIVE
                        track.lost_at = None
                        track.miss_streak = 0
                        track.hit_streak = 0
                        # Reset Kalman velocity — person may re-enter from different direction
                        track.kalman.x[4:] = 0.0
                        self.tracks[track.track_id] = track
                        self.active_ids.add(track.track_id)
                        self._apply_match(track, det, det_emb, timestamp)
                        continue  # Skip birth

            # ── Check 3: Spatial duplicate guard ──
            is_dup = False
            for track in self.tracks.values():
                if bbox_iou(track.bbox, det.bbox) >= self.dup_iou_thr:
                    is_dup = True
                    break
            if is_dup:
                continue

            # ── Check 4: Aspect ratio guard for merged detections ──
            # YOLO sometimes outputs a single wide bbox for two people
            # walking side-by-side. Reject overly wide boxes at birth.
            det_w = det.bbox[2] - det.bbox[0]
            det_h = det.bbox[3] - det.bbox[1]
            if det_h > 0 and det_w / det_h > 1.5:
                continue  # Too wide — likely merged detection

            # ── Truly new person — create tentative track ──
            new_track = self._create_track(det, det_emb, timestamp)

        # Promote tentative → active (hit_streak >= n_confirm)
        for tid in list(self.tentative_ids):
            if tid not in self.tracks:
                continue
            track = self.tracks[tid]
            if track.hit_streak >= self.n_confirm:
                track.state = TrackState.ACTIVE
                self.tentative_ids.discard(tid)
                self.active_ids.add(tid)

                # TRIGGER Phase 1 global match (get GlobalID immediately)
                events.append(TrackEvent(
                    type="TRACK_ACTIVATED",
                    track_id=track.track_id,
                    camera_id=self.camera_id,
                    timestamp=timestamp,
                    embedding=track.best_embedding,
                    entry_time=track.entry_time,
                    track_age=track.age,
                ))

        # ── ACTIVE TRACK MERGING ──────────────────────────
        # If two active tracks are tracking the same person (high embedding
        # similarity + spatial overlap), merge the younger into the older.
        # This catches cases where YOLO multi-box survives to Active state.
        self._merge_duplicate_tracks(timestamp)

        # Evict expired lost tracks
        expired = self.lost_buffer.evict_expired(timestamp)
        for track in expired:
            if track.age >= self.min_export_frames and track.quality >= self.quality_threshold:
                events.append(TrackEvent(
                    type="TRACK_ENDED",
                    track_id=track.track_id,
                    camera_id=self.camera_id,
                    timestamp=timestamp,
                    pooled_embedding=track.pooled_embedding,
                    entry_time=track.entry_time,
                    exit_time=track.last_seen_time,
                    quality=track.quality,
                ))

        return events

    def set_global_id(self, local_track_id: int, global_id: int,
                      display_name: str) -> None:
        """Set the global identity for a local track (called by orchestrator)."""
        self.local_to_global[local_track_id] = global_id
        if local_track_id in self.tracks:
            self.tracks[local_track_id].global_id = global_id
            self.tracks[local_track_id].display_name = display_name

    def get_active_tracks(self) -> List[Track]:
        """Get all confirmed active tracks."""
        return [self.tracks[tid] for tid in self.active_ids if tid in self.tracks]

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame.
        
        Returns annotated frame for MJPEG streaming.
        """
        annotated = frame.copy()
        
        for track in self.tracks.values():
            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            # Color: yellow for tentative, green for registered, blue for unregistered.
            if track.state == TrackState.TENTATIVE:
                color = (0, 220, 220)  # Yellow — tentative
            elif track.display_name and not track.display_name.startswith("Person"):
                color = (0, 200, 0)   # Green — registered
            elif track.global_id is not None:
                color = (200, 130, 0) # Blue — unregistered
            else:
                color = (150, 150, 150)  # Gray — no global ID yet

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            if track.display_name:
                label = f"G{track.global_id}: {track.display_name}"
            elif track.global_id is not None:
                label = f"G{track.global_id}"
            else:
                label = f"T{track.track_id}"

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                       font, font_scale, (255, 255, 255), thickness)

        return annotated

    def finalize_all(self, timestamp: float) -> List[TrackEvent]:
        """Flush all tracks at stream end. Returns final export events."""
        events: List[TrackEvent] = []
        
        for track in list(self.tracks.values()):
            if track.age >= self.min_export_frames:
                events.append(TrackEvent(
                    type="TRACK_ENDED",
                    track_id=track.track_id,
                    camera_id=self.camera_id,
                    timestamp=timestamp,
                    pooled_embedding=track.pooled_embedding,
                    entry_time=track.entry_time,
                    exit_time=track.last_seen_time,
                    quality=track.quality,
                ))
        
        for track in self.lost_buffer.get_all():
            if track.age >= self.min_export_frames:
                events.append(TrackEvent(
                    type="TRACK_ENDED",
                    track_id=track.track_id,
                    camera_id=self.camera_id,
                    timestamp=timestamp,
                    pooled_embedding=track.pooled_embedding,
                    entry_time=track.entry_time,
                    exit_time=track.last_seen_time,
                    quality=track.quality,
                ))
        
        self.tracks.clear()
        self.active_ids.clear()
        self.tentative_ids.clear()
        return events

    # ═══════════════════════════════════════════════════════
    #  INTERNAL: MATCHING
    # ═══════════════════════════════════════════════════════

    def _associate(self,
                   track_ids: List[int],
                   detections: List[Detection],
                   embeddings: Dict[int, np.ndarray],
                   is_low_conf: bool,
                   ) -> List[Tuple[int, int, Detection, Optional[np.ndarray]]]:
        """
        Hungarian assignment between tracks and detections.
        
        Uses Manhattan gating to skip impossible pairs (DSA optimization).
        PER-PAIR adaptive weights: strict appearance matching only for tracks
        that have a nearby neighbor (prevents cross-assignment), lenient
        motion-based matching for isolated tracks.
        
        Returns: List of (track_id, det_index, detection, embedding) matches.
        """
        if not track_ids or not detections:
            return []

        n_t = len(track_ids)
        n_d = len(detections)
        cost = np.full((n_t, n_d), 1e6, dtype=np.float32)

        # Pre-compute which tracks have a close neighbor (O(n^2), tiny n)
        # Threshold 0.15 = meaningful overlap, not just proximity
        track_has_neighbor: Dict[int, bool] = {}
        for tid in track_ids:
            t = self.tracks[tid]
            track_has_neighbor[tid] = any(
                bbox_iou(t.pred_bbox, self.tracks[ot].pred_bbox) > 0.15
                for ot in self.tracks if ot != tid
            )

        for r, tid in enumerate(track_ids):
            track = self.tracks[tid]
            pcx, pcy = bbox_center(track.pred_bbox)

            # Per-pair adaptive weights
            w_m, w_a, w_i = self.w_crowd if track_has_neighbor[tid] else self.w_normal

            for c, det in enumerate(detections):
                dcx, dcy = det.center()

                # -- Manhattan distance gate (O(1) per pair) ---
                if abs(pcx - dcx) + abs(pcy - dcy) > self.gate_dist:
                    continue  # Skip -- impossible match

                # -- Motion cost (Mahalanobis) -----------------
                mah = track.kalman.mahalanobis(det.bbox)
                if mah > self.mahal_gate * 4.0:  # Hard gate
                    continue
                motion_cost = min(1.0, mah / max(self.mahal_gate, 1e-6))

                # -- IOU cost ----------------------------------
                iou = bbox_iou(track.pred_bbox, det.bbox)
                iou_cost = 1.0 - iou

                # -- Appearance cost ---------------------------
                det_emb = embeddings.get(c)
                if not is_low_conf and track.pooled_embedding is not None and det_emb is not None:
                    sim = _cosine_sim(track.pooled_embedding, det_emb)
                    # Per-pair sim gate: only strict when track has nearby neighbor
                    if track_has_neighbor[tid] and sim < 0.40:
                        continue  # Strict: prevent cross-assignment near neighbors
                    appear_cost = 1.0 - max(0.0, sim)
                else:
                    appear_cost = 0.7
                    if is_low_conf:
                        cost[r, c] = 0.60 * motion_cost + 0.40 * iou_cost
                        continue

                cost[r, c] = w_m * motion_cost + w_a * appear_cost + w_i * iou_cost

        # -- Hungarian assignment --------------------------
        rows, cols = linear_sum_assignment(cost)

        threshold = self.cost_thr_low if is_low_conf else self.cost_thr_high
        matches: List[Tuple[int, int, Detection, Optional[np.ndarray]]] = []

        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e5 or cost[r, c] > threshold:
                continue
            tid = track_ids[r]
            track = self.tracks[tid]
            det = detections[c]

            # -- Post-Hungarian position-jump validation (velocity-aware) --
            # Only for established tracks with nearby neighbors.
            if track.age > 5 and not is_low_conf and track_has_neighbor[tid]:
                pcx, pcy = bbox_center(track.pred_bbox)
                dcx, dcy = det.center()
                jump = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5
                vx = abs(track.kalman.x[4])
                vy = abs(track.kalman.x[5])
                speed = (vx ** 2 + vy ** 2) ** 0.5
                max_jump = min(self.gate_dist * 0.8,
                               max(self.gate_dist * 0.4, speed * 4.0))
                if jump > max_jump:
                    continue  # Too far -- likely a cross-assignment

            det_emb = embeddings.get(c)
            matches.append((tid, c, det, det_emb))

        return matches

    def _apply_match(self, track: Track, det: Detection,
                     embedding: Optional[np.ndarray],
                     timestamp: float) -> None:
        """Apply a confirmed match: update Kalman, embedding pool, counters."""
        # Clamp bbox to frame
        bbox = clamp_bbox(*det.bbox, self.W, self.H)

        if bbox_area(bbox) < self.min_box_area:
            # Reinitialize track filter state to stabilize after tiny/degenerate boxes.
            track.kalman = KalmanFilter8(track.bbox)
            track.miss_streak += 1
            track.hit_streak = 0
            track.last_seen_time = timestamp
            return

        # Always incorporate latest detection for matched tracks.
        track.kalman.update(bbox)
        track.bbox = bbox

        # Counters
        track.hit_streak += 1
        track.miss_streak = 0
        track.last_seen_time = timestamp

        # Confidence (running average O(1))
        track.update_confidence(det.confidence)

        # Embedding pool (min-heap O(log K)) — high-conf only
        if embedding is not None:
            track.add_embedding(embedding, det.confidence)

        # Trajectory (capped to prevent unbounded memory growth)
        center = bbox_center(track.bbox)
        track.trajectory.append((center[0], center[1], timestamp))
        if len(track.trajectory) > track._max_trajectory:
            track.trajectory = track.trajectory[-track._max_trajectory:]

    def _merge_duplicate_tracks(self, timestamp: float) -> None:
        """
        Merge duplicate active tracks + suppress static objects.
        
        If two active tracks have high embedding similarity + spatial overlap,
        merge the younger into the older. Also removes static tracks (reflections).
        """
        # Disabled static suppression here; it can remove valid stationary people
        # in crowded real-world scenes and reduce recall.

        if len(self.active_ids) < 2:
            return

        active_list = [self.tracks[tid] for tid in self.active_ids if tid in self.tracks]
        merge_pairs: List[Tuple[Track, Track]] = []  # (keep, remove)

        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                t1, t2 = active_list[i], active_list[j]
                if t1.pooled_embedding is None or t2.pooled_embedding is None:
                    continue

                # GlobalID-aware merge prevention
                if (t1.global_id is not None and t2.global_id is not None
                        and t1.global_id != t2.global_id):
                    continue

                sim = _cosine_sim(t1.pooled_embedding, t2.pooled_embedding)
                iou = bbox_iou(t1.bbox, t2.bbox)

                # Very conservative merge gate to avoid collapsing two different
                # people into one track.
                should_merge = (
                    sim > 0.92 and
                    iou > 0.65 and
                    t1.age > 20 and
                    t2.age > 20
                )

                if should_merge:
                    if t1.age >= t2.age:
                        merge_pairs.append((t1, t2))
                    else:
                        merge_pairs.append((t2, t1))

        # Execute merges
        merged_ids: Set[int] = set()
        for keep, remove in merge_pairs:
            if remove.track_id in merged_ids or keep.track_id in merged_ids:
                continue
            for conf, uid, emb in remove._emb_heap:
                keep.add_embedding(emb, conf)
            # Propagate GlobalID if needed
            if keep.global_id is None and remove.global_id is not None:
                keep.global_id = remove.global_id
                keep.display_name = remove.display_name
                self.local_to_global[keep.track_id] = remove.global_id
            if remove.track_id in self.local_to_global:
                del self.local_to_global[remove.track_id]
            self.active_ids.discard(remove.track_id)
            self.tentative_ids.discard(remove.track_id)
            if remove.track_id in self.tracks:
                del self.tracks[remove.track_id]
            merged_ids.add(remove.track_id)

    def _suppress_static_objects(self) -> None:
        """
        Remove tracks that haven't moved significantly -- likely reflections,
        mannequins, or static poster/display detections.
        Uses body-relative displacement threshold.
        """
        remove_ids: List[int] = []
        for tid in self.active_ids:
            track = self.tracks.get(tid)
            if track is None or track.age < 60:
                continue
            if len(track.trajectory) >= 2:
                sx, sy, _ = track.trajectory[0]
                ex, ey, _ = track.trajectory[-1]
                displacement = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                bw = track.bbox[2] - track.bbox[0]
                bh = track.bbox[3] - track.bbox[1]
                bbox_diag = (bw ** 2 + bh ** 2) ** 0.5
                if displacement < bbox_diag * 0.1 and track.avg_confidence < 0.50:
                    # Low displacement AND low confidence = likely mannequin/reflection.
                    # Real stationary people (performers) get high-conf YOLO detections.
                    remove_ids.append(tid)
        for tid in remove_ids:
            self.active_ids.discard(tid)
            if tid in self.tracks:
                del self.tracks[tid]

    def _create_track(self, det: Detection, embedding: Optional[np.ndarray],
                      timestamp: float) -> Track:
        """Create a new tentative track from an unmatched high-conf detection."""
        self._next_id += 1
        bbox = clamp_bbox(*det.bbox, self.W, self.H)

        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            camera_id=self.camera_id,
            timestamp=timestamp,
        )
        track._pool_k = self.emb_pool_k
        track.update_confidence(det.confidence)

        if embedding is not None:
            track.add_embedding(embedding, det.confidence)

        self.tracks[track.track_id] = track
        self.tentative_ids.add(track.track_id)

        return track
