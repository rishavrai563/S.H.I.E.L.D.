"""
Global Matcher — Cross-camera person identity matching.

Assigns a consistent GlobalID to each person across all cameras.
Uses a two-phase matching strategy:

  Phase 1 (Early Match): Called when a track becomes Active.
    - Uses single best embedding for fast GlobalID assignment.
    - Checks registered persons first, then gallery, then creates new ID.

  Phase 2 (Refined Match): Called when a track ends (goes permanently lost).
    - Uses pooled embedding (median of top-10) for robust association.
    - May correct Phase 1's assignment if pooled embedding is more accurate.
    - Updates the gallery prototype via conservative EMA.

Thread-safe: entire process() is wrapped in a Lock — takes <1ms, negligible contention.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GlobalIdentity:
    """Represents a unique person across all cameras."""
    global_id: int
    prototype: np.ndarray                   # 512-dim L2-normalized embedding
    display_name: str                       # "Rahul" or "Person #5"
    is_registered: bool = False

    # Tracking info
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    first_seen_camera: str = ""
    last_seen_camera: str = ""
    times_seen: int = 0                     # How many tracklets matched

    # Best thumbnail (stored as JPEG bytes later — for now just quality score)
    best_quality: float = 0.0

    # Movement history
    camera_history: List[Tuple[str, float, float]] = field(default_factory=list)
    # List of (camera_id, entry_time, exit_time)


@dataclass
class MatchResult:
    """Result of a global matching operation."""
    global_id: int
    display_name: str
    confidence: float                       # Match score
    is_new: bool                            # True if new GlobalID was created
    phase: int                              # 1 or 2
    is_registered: bool = False


class GlobalMatcher:
    """
    Cross-camera person identity matcher.

    Usage:
        matcher = GlobalMatcher()

        # When track becomes Active (fast):
        result = matcher.match_phase1(embedding, camera_id, timestamp)

        # When track ends (accurate):
        result = matcher.match_phase2(pooled_embedding, camera_id, entry_time, exit_time, quality)
    """

    def __init__(self,
                 # Matching thresholds
                 registered_threshold: float = 0.70,
                 gallery_threshold: float = 0.60,
                 margin_gate: float = 0.08,
                 # Prototype update
                 ema_alpha: float = 0.10,
                 ema_min_confidence: float = 0.65,
                 # Time plausibility
                 same_camera_min_gap: float = 1.0,
                 diff_camera_default_min: float = 2.0,
                 # Transit times from topology
                 transit_times: Optional[Dict[str, Tuple[float, float]]] = None):

        self.registered_thr = registered_threshold
        self.gallery_thr = gallery_threshold
        self.margin_gate = margin_gate
        self.ema_alpha = ema_alpha
        self.ema_min_conf = ema_min_confidence
        self.same_cam_gap = same_camera_min_gap
        self.diff_cam_min = diff_camera_default_min
        self.transit_times = transit_times or {}

        # Gallery storage
        self._identities: Dict[int, GlobalIdentity] = {}
        self._registered: Dict[int, GlobalIdentity] = {}  # Subset: registered persons
        self._next_global_id: int = 0

        # Gallery matrix (vectorized matching) — rebuilt on changes
        self._gallery_embeddings: Optional[np.ndarray] = None  # (N, 512)
        self._gallery_ids: List[int] = []
        self._gallery_dirty: bool = True

        # Thread safety
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════

    def match_phase1(self, embedding: np.ndarray,
                     camera_id: str, timestamp: float) -> MatchResult:
        """
        Phase 1 — Early Match. Called when track becomes Active.
        Uses single best embedding for fast GlobalID assignment.

        Returns MatchResult with assigned GlobalID.
        """
        if embedding is None:
            return self._create_new_identity(None, camera_id, timestamp, phase=1)

        with self._lock:
            # Step 1: Check registered persons first
            reg_result = self._match_registered(embedding, camera_id, timestamp)
            if reg_result is not None:
                return reg_result

            # Step 2: Check existing gallery
            gallery_result = self._match_gallery(
                embedding, camera_id, timestamp, phase=1
            )
            if gallery_result is not None:
                return gallery_result

            # Step 3: No match → create new GlobalID
            result = self._create_new_identity(embedding, camera_id, timestamp, phase=1)
            return result

    def match_phase2(self, pooled_embedding: np.ndarray,
                     camera_id: str, entry_time: float,
                     exit_time: float, quality: float,
                     current_global_id: Optional[int] = None) -> MatchResult:
        """
        Phase 2 — Refined Match. Called when track ends.
        Uses pooled embedding (more robust) for accurate cross-camera association.

        Args:
            pooled_embedding: Median-pooled top-K embedding (512-dim).
            camera_id: Camera where the track ended.
            entry_time: When the track was first seen.
            exit_time: When the track was last seen.
            quality: Track quality score.
            current_global_id: GlobalID assigned by Phase 1 (for correction).

        Returns MatchResult with (possibly corrected) GlobalID.
        """
        if pooled_embedding is None:
            if current_global_id is not None and current_global_id in self._identities:
                ident = self._identities[current_global_id]
                return MatchResult(
                    global_id=current_global_id,
                    display_name=ident.display_name,
                    confidence=0.0,
                    is_new=False,
                    phase=2,
                    is_registered=ident.is_registered,
                )
            return self._create_new_identity(None, camera_id, exit_time, phase=2)

        with self._lock:
            # Step 0: If Phase 1 already assigned a GlobalID, check it FIRST
            # This prevents Phase 2 from creating a new ID when it should confirm Phase 1
            if current_global_id is not None and current_global_id in self._identities:
                current_ident = self._identities[current_global_id]
                if current_ident.prototype is not None and np.any(current_ident.prototype):
                    current_sim = float(np.dot(pooled_embedding, current_ident.prototype))
                    # If current assignment is reasonable, confirm it
                    if current_sim >= (self.gallery_thr - 0.10):
                        # Check if gallery has a SIGNIFICANTLY better match
                        # Exclude current identity from search — we're looking for DIFFERENT matches
                        gallery_result = self._match_gallery(
                            pooled_embedding, camera_id, exit_time, phase=2,
                            exclude_gid=current_global_id
                        )
                        if gallery_result is not None and gallery_result.confidence > current_sim + 0.10:
                            # Gallery found much better match — correct Phase 1
                            self._update_prototype(gallery_result.global_id, pooled_embedding,
                                                   gallery_result.confidence)
                            self._log_movement(gallery_result.global_id, camera_id,
                                               entry_time, exit_time)
                            return gallery_result
                        else:
                            # Confirm Phase 1 assignment — always update prototype
                            # since we're confirming an existing match
                            self._update_prototype(current_global_id, pooled_embedding,
                                                   max(current_sim, self.ema_min_conf))
                            self._log_movement(current_global_id, camera_id,
                                               entry_time, exit_time)
                            current_ident.last_seen_time = exit_time
                            current_ident.last_seen_camera = camera_id
                            return MatchResult(
                                global_id=current_global_id,
                                display_name=current_ident.display_name,
                                confidence=current_sim,
                                is_new=False,
                                phase=2,
                                is_registered=current_ident.is_registered,
                            )

            # Step 1: Check registered persons
            reg_result = self._match_registered(pooled_embedding, camera_id, exit_time)
            if reg_result is not None:
                reg_result.phase = 2
                self._update_prototype(reg_result.global_id, pooled_embedding,
                                       reg_result.confidence)
                self._log_movement(reg_result.global_id, camera_id,
                                   entry_time, exit_time)
                return reg_result

            # Step 2: Check gallery
            gallery_result = self._match_gallery(
                pooled_embedding, camera_id, exit_time, phase=2
            )
            if gallery_result is not None:
                self._update_prototype(gallery_result.global_id, pooled_embedding,
                                       gallery_result.confidence)
                self._log_movement(gallery_result.global_id, camera_id,
                                   entry_time, exit_time)
                return gallery_result

            # Step 3: No match → create new GlobalID
            result = self._create_new_identity(
                pooled_embedding, camera_id, exit_time, phase=2
            )
            self._log_movement(result.global_id, camera_id, entry_time, exit_time)
            return result

    def register_person(self, name: str, prototype: np.ndarray) -> int:
        """
        Register a known person with their name and embedding prototype.

        Returns the assigned GlobalID.
        """
        with self._lock:
            self._next_global_id += 1
            gid = self._next_global_id

            identity = GlobalIdentity(
                global_id=gid,
                prototype=prototype.copy(),
                display_name=name,
                is_registered=True,
                first_seen_time=time.time(),
            )
            self._identities[gid] = identity
            self._registered[gid] = identity
            self._gallery_dirty = True
            return gid

    def get_identity(self, global_id: int) -> Optional[GlobalIdentity]:
        """Get a GlobalIdentity by ID."""
        return self._identities.get(global_id)

    def get_all_identities(self) -> List[GlobalIdentity]:
        """Get all known identities."""
        return list(self._identities.values())

    def get_journey(self, global_id: int) -> Optional[List[Tuple[str, float, float]]]:
        """Get movement history for a person: List of (camera_id, entry, exit)."""
        ident = self._identities.get(global_id)
        if ident:
            return ident.camera_history.copy()
        return None

    def update_gallery(self, global_id: int, embedding: np.ndarray,
                       camera_id: str, timestamp: float) -> None:
        """
        Intermediate gallery update for active tracks.
        
        Called periodically by the orchestrator while a track is still active,
        so the gallery prototype improves before TRACK_ENDED fires.
        This helps cross-camera matching when the person walks to another camera
        while still tracked on the first.
        """
        with self._lock:
            identity = self._identities.get(global_id)
            if identity is None:
                return
            # Use stronger EMA (alpha=0.2) since pooled embeddings are high-quality
            alpha = 0.20
            updated = alpha * embedding + (1 - alpha) * identity.prototype
            norm = np.linalg.norm(updated)
            if norm > 1e-6:
                identity.prototype = (updated / norm).astype(np.float32)
                identity.last_seen_time = timestamp
                identity.last_seen_camera = camera_id
                self._gallery_dirty = True

    @property
    def gallery_size(self) -> int:
        return len(self._identities)

    # ═══════════════════════════════════════════════════════
    #  INTERNAL: MATCHING LOGIC
    # ═══════════════════════════════════════════════════════

    def _match_registered(self, embedding: np.ndarray,
                          camera_id: str, timestamp: float
                          ) -> Optional[MatchResult]:
        """
        Check registered persons first (priority over gallery).
        Returns MatchResult if match found, None otherwise.
        """
        if not self._registered:
            return None

        best_gid = None
        best_sim = self.registered_thr

        for gid, identity in self._registered.items():
            if identity.prototype is None:
                continue
            sim = float(np.dot(embedding, identity.prototype))
            if sim > best_sim:
                if self._time_plausible(identity, camera_id, timestamp):
                    best_sim = sim
                    best_gid = gid

        if best_gid is not None:
            identity = self._identities[best_gid]
            identity.last_seen_time = timestamp
            identity.last_seen_camera = camera_id
            identity.times_seen += 1
            return MatchResult(
                global_id=best_gid,
                display_name=identity.display_name,
                confidence=best_sim,
                is_new=False,
                phase=1,
                is_registered=True,
            )

        return None

    def _match_gallery(self, embedding: np.ndarray,
                       camera_id: str, timestamp: float,
                       phase: int,
                       exclude_gid: Optional[int] = None) -> Optional[MatchResult]:
        """
        Vectorized gallery matching with margin gate.

        Steps:
          1. Compute similarities with all gallery entries (matrix multiply)
          2. Sort by descending similarity
          3. Check margin: top1 - top2 >= margin_gate (avoid ambiguous matches)
          4. Check time plausibility
          5. Return match or None
        
        Args:
            exclude_gid: If set, skip this GlobalID in the search (used when
                         guided Phase 2 already knows its current identity and
                         only wants to find DIFFERENT better matches).
        """
        self._rebuild_gallery_if_dirty()

        if self._gallery_embeddings is None or len(self._gallery_ids) == 0:
            return None

        # Vectorized cosine similarity: gallery_matrix @ query → (N,) scores
        similarities = self._gallery_embeddings @ embedding  # (N,)

        # If excluding a GID, mask it out
        if exclude_gid is not None and exclude_gid in self._gallery_ids:
            exclude_idx = self._gallery_ids.index(exclude_gid)
            similarities[exclude_idx] = -1.0  # Force below any threshold

        # Sort descending
        sorted_indices = np.argsort(-similarities)

        top1_idx = sorted_indices[0]
        top1_score = float(similarities[top1_idx])
        top1_gid = self._gallery_ids[top1_idx]

        # Margin gate: ensure the match is unambiguous
        if len(sorted_indices) > 1:
            top2_score = float(similarities[sorted_indices[1]])
            margin = top1_score - top2_score
        else:
            margin = 1.0  # Only one entry — no ambiguity

        if top1_score >= self.gallery_thr and margin >= self.margin_gate:
            identity = self._identities[top1_gid]

            # Time plausibility check
            if self._time_plausible(identity, camera_id, timestamp):
                identity.last_seen_time = timestamp
                identity.last_seen_camera = camera_id
                identity.times_seen += 1
                return MatchResult(
                    global_id=top1_gid,
                    display_name=identity.display_name,
                    confidence=top1_score,
                    is_new=False,
                    phase=phase,
                    is_registered=identity.is_registered,
                )

        return None

    def _create_new_identity(self, embedding: Optional[np.ndarray],
                             camera_id: str, timestamp: float,
                             phase: int) -> MatchResult:
        """Create a new GlobalID for an unrecognized person."""
        self._next_global_id += 1
        gid = self._next_global_id

        identity = GlobalIdentity(
            global_id=gid,
            prototype=embedding.copy() if embedding is not None else np.zeros(512, dtype=np.float32),
            display_name=f"Person #{gid}",
            is_registered=False,
            first_seen_time=timestamp,
            last_seen_time=timestamp,
            first_seen_camera=camera_id,
            last_seen_camera=camera_id,
            times_seen=1,
        )
        self._identities[gid] = identity
        self._gallery_dirty = True

        return MatchResult(
            global_id=gid,
            display_name=identity.display_name,
            confidence=0.0,
            is_new=True,
            phase=phase,
        )

    def _update_prototype(self, global_id: int,
                          new_embedding: np.ndarray,
                          confidence: float) -> None:
        """
        Conservative EMA update of gallery prototype.
        Only updates on confident matches to prevent identity drift.

        new_prototype = L2_normalize(α * new + (1 - α) * old)
        """
        if confidence < self.ema_min_conf:
            return

        identity = self._identities.get(global_id)
        if identity is None or identity.prototype is None:
            return

        # EMA blend
        updated = self.ema_alpha * new_embedding + (1 - self.ema_alpha) * identity.prototype

        # L2 normalize
        norm = np.linalg.norm(updated)
        if norm > 1e-6:
            updated = updated / norm

        identity.prototype = updated.astype(np.float32)
        self._gallery_dirty = True

    def _log_movement(self, global_id: int,
                      camera_id: str, entry_time: float,
                      exit_time: float) -> None:
        """Log a camera visit to the person's movement history."""
        identity = self._identities.get(global_id)
        if identity is not None:
            identity.camera_history.append((camera_id, entry_time, exit_time))

    # ═══════════════════════════════════════════════════════
    #  INTERNAL: UTILITIES
    # ═══════════════════════════════════════════════════════

    def _time_plausible(self, identity: GlobalIdentity,
                        camera_id: str, timestamp: float) -> bool:
        """
        Check if it's physically plausible for this person to be at
        this camera at this time, given their last known location.

        If person was last seen on the SAME camera:
          - Requires at least 1.0s gap (prevents self-duplicate)

        If person was last seen on a DIFFERENT camera:
          - Uses topology transit times if configured
          - Otherwise requires at least 5s gap (walking between cameras)
        """
        if identity.last_seen_time == 0:
            return True  # First sighting — always plausible

        # Use signed delta: timestamp should be >= last_seen_time
        # abs() would mask bugs where timestamps go backwards
        dt = timestamp - identity.last_seen_time
        if dt < 0:
            return True  # Clock skew / out-of-order event — allow

        if camera_id == identity.last_seen_camera:
            # Same camera: require gap to prevent same-track from matching itself
            return dt > self.same_cam_gap
        else:
            # Different camera: check topology transit times
            key = self._transit_key(identity.last_seen_camera, camera_id)
            if key in self.transit_times:
                min_t, max_t = self.transit_times[key]
                return min_t <= dt <= max_t * 3  # 3× slack for variability
            else:
                # No topology info: default minimum gap
                return dt >= self.diff_cam_min

    def _transit_key(self, cam_a: str, cam_b: str) -> str:
        """Generate a canonical key for transit time lookup."""
        if cam_a <= cam_b:
            return f"{cam_a}_{cam_b}"
        else:
            return f"{cam_b}_{cam_a}"

    def _rebuild_gallery_if_dirty(self) -> None:
        """Rebuild the gallery embedding matrix for vectorized matching."""
        if not self._gallery_dirty:
            return

        if not self._identities:
            self._gallery_embeddings = None
            self._gallery_ids = []
            self._gallery_dirty = False
            return

        ids = []
        embeddings = []
        for gid, identity in self._identities.items():
            if identity.prototype is not None and np.any(identity.prototype):
                ids.append(gid)
                embeddings.append(identity.prototype)

        if embeddings:
            self._gallery_embeddings = np.stack(embeddings, axis=0)  # (N, 512)
            self._gallery_ids = ids
        else:
            self._gallery_embeddings = None
            self._gallery_ids = []

        self._gallery_dirty = False
