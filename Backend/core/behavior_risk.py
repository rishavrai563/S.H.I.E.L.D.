from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple


@dataclass
class TrackBehaviorState:
    id: int
    positions_history: Deque[Tuple[float, float]]
    velocity_history: Deque[float]
    last_seen_frame: int
    state_label: str = "NORMAL"
    state_duration: int = 0
    interaction_history: Dict[int, int] = field(default_factory=dict)
    first_seen_frame: int = 0
    candidate_label: str = "NORMAL"
    candidate_duration: int = 0
    suspicious_active: bool = False
    isolated_active: bool = False
    confidence: float = 0.0
    speed: float = 0.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class TemporalSmoother:
    """Moving-average smoothing + dwell-time stabilization."""

    def __init__(self, window_size: int = 20, min_switch_frames: int = 6):
        self.window_size = max(1, int(window_size))
        self.min_switch_frames = max(1, int(min_switch_frames))

    def moving_average(self, values: Deque[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def stable_label(self, st: TrackBehaviorState, proposed_label: str) -> Tuple[str, int]:
        if st.state_label == proposed_label:
            st.state_duration += 1
            st.candidate_label = proposed_label
            st.candidate_duration = 0
            return st.state_label, st.state_duration

        if st.candidate_label != proposed_label:
            st.candidate_label = proposed_label
            st.candidate_duration = 1
        else:
            st.candidate_duration += 1

        if st.candidate_duration >= self.min_switch_frames:
            st.state_label = proposed_label
            st.state_duration = 1
            st.candidate_duration = 0
            return st.state_label, st.state_duration

        st.state_duration += 1
        return st.state_label, st.state_duration


class TrackStateManager:
    """Keeps lifecycle state per track id."""

    def __init__(self, history_size: int = 30):
        self.history_size = max(2, int(history_size))
        self.states: Dict[int, TrackBehaviorState] = {}
        self._last_exits: List[int] = []

    @staticmethod
    def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update_tracks(self, frame_idx: int, tracks: List[Dict]) -> Tuple[Dict[int, TrackBehaviorState], List[int]]:
        self._last_exits = []
        seen_ids: Set[int] = set()

        for tr in tracks:
            tid = int(tr["id"])
            bbox = tuple(float(v) for v in tr["bbox"])
            seen_ids.add(tid)
            cx, cy = self._center(bbox)

            if tid not in self.states:
                self.states[tid] = TrackBehaviorState(
                    id=tid,
                    positions_history=deque(maxlen=self.history_size),
                    velocity_history=deque(maxlen=self.history_size),
                    last_seen_frame=frame_idx,
                    first_seen_frame=frame_idx,
                )

            st = self.states[tid]
            prev_center = st.positions_history[-1] if st.positions_history else None
            st.positions_history.append((cx, cy))
            st.bbox = bbox
            st.last_seen_frame = frame_idx

            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                st.velocity_history.append((dx * dx + dy * dy) ** 0.5)
            else:
                st.velocity_history.append(0.0)

        existing_ids = set(self.states.keys())
        disappeared = existing_ids - seen_ids
        for tid in disappeared:
            self._last_exits.append(tid)
            del self.states[tid]

        return self.states, self._last_exits


class InteractionDetector:
    """Detects sustained close-range interactions between people."""

    def __init__(self, proximity_px: float = 120.0, hold_frames: int = 12):
        self.proximity_px = float(proximity_px)
        self.hold_frames = max(1, int(hold_frames))
        self.pair_close_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    @staticmethod
    def _pair(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _diag(bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return (w * w + h * h) ** 0.5

    def update(self, tracks: List[Dict]) -> Dict[int, Set[int]]:
        active: Dict[int, Set[int]] = defaultdict(set)
        seen_pairs: Set[Tuple[int, int]] = set()

        for i in range(len(tracks)):
            ti = tracks[i]
            ci = self._center(tuple(ti["bbox"]))
            di = self._diag(tuple(ti["bbox"]))
            for j in range(i + 1, len(tracks)):
                tj = tracks[j]
                cj = self._center(tuple(tj["bbox"]))
                dj = self._diag(tuple(tj["bbox"]))
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                dist = (dx * dx + dy * dy) ** 0.5
                dyn_thresh = max(self.proximity_px, 0.25 * (di + dj))
                pair = self._pair(int(ti["id"]), int(tj["id"]))
                seen_pairs.add(pair)

                if dist <= dyn_thresh:
                    self.pair_close_counts[pair] += 1
                else:
                    self.pair_close_counts[pair] = max(0, self.pair_close_counts[pair] - 1)

                if self.pair_close_counts[pair] >= self.hold_frames:
                    a, b = pair
                    active[a].add(b)
                    active[b].add(a)

        stale = [p for p in self.pair_close_counts if p not in seen_pairs]
        for p in stale:
            self.pair_close_counts[p] = max(0, self.pair_close_counts[p] - 2)
            if self.pair_close_counts[p] == 0:
                del self.pair_close_counts[p]

        return active


class RiskAnalyzer:
    """
    Temporal, track-level risk classifier for Border Security (IBVAP).
    Features Mathematical Virtual Fence (Tripwire) using Shapely line intersections.
    """

    PRIORITY = {
        "INTRUSION": 8,           # Tripwire crossed (highest priority)
        "RESTRICTED_ZONE": 7,     # Inside a restricted zone right now (suspicious until they leave)
        "LOITERING": 6,           # Near-stationary dwell beyond loiter_seconds
        "SUSPICIOUS_MOTION": 5,
        "INTERACTION": 4,
        "NEW_ENTRY": 3,
        "ISOLATED": 2,
        "HIGH_DENSITY": 1,
        "NORMAL": 0,
    }

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 30.0,
        window_size: int = 20,
        min_switch_frames: int = 6,
        new_entry_seconds: float = 2.0,
        suspicious_enter_speed: float = 350.0,
        suspicious_exit_speed: float = 140.0,
        isolation_distance_px: float = 400.0,
        isolation_enter_frames: int = 12,
        isolation_exit_frames: int = 6,
        density_enter: float = 5e-6,
        density_exit: float = 3.5e-6,
        density_hold_frames: int = 12,
        interaction_distance_px: float = 120.0,
        interaction_hold_frames: int = 12,
        suspicious_min_frames: int = 8,
        loiter_seconds: float = 30.0,
        loiter_displacement_px: float = 30.0,
    ):
        from shapely.geometry import LineString # Imported here for strict math

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.visible_area = float(max(1, self.frame_width * self.frame_height))
        self.fps = max(1e-3, float(fps))

        self.new_entry_frames = max(1, int(new_entry_seconds * self.fps))
        self.suspicious_enter = float(suspicious_enter_speed)
        self.suspicious_exit = float(suspicious_exit_speed)
        self.suspicious_min_frames = max(1, int(suspicious_min_frames))
        self.isolation_distance_px = float(isolation_distance_px)
        self.isolation_enter_frames = max(1, int(isolation_enter_frames))
        self.isolation_exit_frames = max(1, int(isolation_exit_frames))
        self.density_enter = float(density_enter)
        self.density_exit = float(density_exit)
        self.density_hold_frames = max(1, int(density_hold_frames))

        self.smoother = TemporalSmoother(window_size=window_size, min_switch_frames=min_switch_frames)
        self.state_manager = TrackStateManager(history_size=window_size)
        self.interaction_detector = InteractionDetector(
            proximity_px=interaction_distance_px,
            hold_frames=interaction_hold_frames,
        )

        self.loiter_seconds = float(loiter_seconds)
        self.loiter_displacement_px = float(loiter_displacement_px)
        # track id -> (anchor_x, anchor_y, anchor_time) for dwell measurement
        self._loiter_anchor: Dict[int, Tuple[float, float, float]] = {}
        self._prev_bottom: Dict[int, Tuple[float, float]] = {}

        self._isolation_counter: Dict[int, int] = defaultdict(int)
        self._density_counter = 0
        self._high_density_active = False
        self._last_scene_status = "CALM"

        # --- NEW: Virtual Fence Configuration ---
        # Default virtual fence line (can be dynamically updated via API later)
        # Formatted as [(x1, y1), (x2, y2)]
        self.virtual_fence_coords = [(50, self.frame_height - 50), (self.frame_width - 50, self.frame_height - 50)]
        self.virtual_fence_line = LineString(self.virtual_fence_coords)
        # "line" = tripwire (open polyline, any number of segments): crossing it = intrusion
        # "zone" = restricted area (closed polygon, any shape): standing inside it = intrusion
        self.fence_type = "line"
        self.fence_zone = None            # shapely Polygon when fence_type == "zone"
        self.intruders = set() # Store track IDs that breached the fence
        self.zone_occupants: Set[int] = set()        # track IDs inside the restricted zone now
        self._zone_entered_at: Dict[int, float] = {}

    # -- geometry helpers, shared with vehicle tracking in the orchestrator --
    def crossed_tripwire(self, prev_pt, curr_pt) -> bool:
        """True if the movement prev_pt -> curr_pt crosses the tripwire."""
        from shapely.geometry import LineString
        if self.virtual_fence_line is None or prev_pt is None or prev_pt == curr_pt:
            return False
        return self.virtual_fence_line.intersects(LineString([prev_pt, curr_pt]))

    def in_zone(self, pt) -> bool:
        """True if the point is inside the restricted zone."""
        from shapely.geometry import Point
        return self.fence_zone is not None and self.fence_zone.contains(Point(pt))

    def update_fence(self, new_coords: List[Tuple[float, float]], fence_type: str = "line") -> bool:
        """Set the fence from the dashboard editor. An empty list disables it.

        line: >= 2 points, crossing any segment flags INTRUSION (stays flagged).
        zone: >= 3 points, a person whose feet are inside is flagged while inside.
        """
        from shapely.geometry import LineString, Polygon
        coords = [(float(x), float(y)) for x, y in new_coords]
        if not coords:
            self.virtual_fence_coords, self.virtual_fence_line, self.fence_zone = [], None, None
            self.fence_type = fence_type
            self.intruders.clear(); self.zone_occupants.clear(); self._zone_entered_at.clear()
            print("🚧 Virtual fence disabled")
            return True
        if fence_type == "zone":
            if len(coords) < 3:
                return False
            poly = Polygon(coords)
            if not poly.is_valid:            # self-intersecting drawing -> repair
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 1.0:
                return False
            self.fence_zone, self.virtual_fence_line = poly, None
        else:
            if len(coords) < 2:
                return False
            fence_type = "line"
            self.virtual_fence_line, self.fence_zone = LineString(coords), None
        self.fence_type = fence_type
        self.virtual_fence_coords = coords
        self.intruders.clear(); self.zone_occupants.clear(); self._zone_entered_at.clear()
        print(f"🚧 Virtual fence ({fence_type}, {len(coords)} points) updated")
        return True

    def fence_dict(self) -> dict:
        return {"type": self.fence_type, "points": [[x, y] for x, y in self.virtual_fence_coords],
                "frame": {"width": self.frame_width, "height": self.frame_height}}

    def _compute_scene_density(self, people_count: int) -> float:
        return float(people_count) / self.visible_area

    def _with_priority(self, active_flags: Dict[str, bool]) -> str:
        best = "NORMAL"
        best_pri = -1
        for label, enabled in active_flags.items():
            if enabled:
                pri = self.PRIORITY.get(label, -1)
                if pri > best_pri:
                    best_pri = pri
                    best = label
        return best

    def _track_confidence(self, label: str, speed_px_s: float, interactions: int, state_duration: int) -> float:
        if label == "INTRUSION":
            return 1.0 # 100% confidence upon mathematical intersection
        if label == "SUSPICIOUS_MOTION":
            return max(0.0, min(1.0, (speed_px_s - self.suspicious_exit) / max(1.0, self.suspicious_enter - self.suspicious_exit)))
        if label == "INTERACTION":
            return min(1.0, 0.5 + 0.1 * interactions + 0.02 * state_duration)
        if label == "NEW_ENTRY":
            return max(0.0, min(1.0, 1.0 - (state_duration / max(1, self.new_entry_frames))))
        if label == "ISOLATED":
            return min(1.0, 0.4 + 0.03 * state_duration)
        if label == "HIGH_DENSITY":
            return min(1.0, 0.5 + 0.02 * state_duration)
        return min(1.0, 0.4 + 0.01 * state_duration)

    def _scene_status(self, suspicious_count: int, interaction_count: int, total_tracks: int, intrusion_count: int) -> str:
        if intrusion_count > 0:
            return "BREACH_DETECTED"
        if suspicious_count >= 2 or interaction_count >= 3 or total_tracks >= 10:
            return "HIGH_ACTIVITY"
        if suspicious_count >= 1 or interaction_count >= 1 or total_tracks >= 5:
            return "ACTIVE"
        return "CALM"

    def _update_loiter(self, tid: int, point: Tuple[float, float], now: float) -> float:
        """Return seconds the track has stayed within loiter_displacement_px of its anchor."""
        anchor = self._loiter_anchor.get(tid)
        if anchor is None or ((point[0] - anchor[0]) ** 2 + (point[1] - anchor[1]) ** 2) ** 0.5 > self.loiter_displacement_px:
            self._loiter_anchor[tid] = (point[0], point[1], now)
            return 0.0
        return now - anchor[2]

    def update(self, frame_idx: int, tracks: List[Dict], timestamp: Optional[float] = None) -> Dict:
        """
        Args:
            frame_idx: monotonically increasing frame counter.
            tracks:    [{"id": int, "bbox": [x1, y1, x2, y2]}, ...]
            timestamp: wall/stream time in seconds. Used for loitering dwell;
                       falls back to frame_idx / fps when omitted.
        """
        from shapely.geometry import LineString, Point

        now = float(timestamp) if timestamp is not None else frame_idx / self.fps
        states, exits = self.state_manager.update_tracks(frame_idx, tracks)
        for tid in exits:
            self.intruders.discard(tid)
            self.zone_occupants.discard(tid)
            self._zone_entered_at.pop(tid, None)
            self._loiter_anchor.pop(tid, None)
            self._prev_bottom.pop(tid, None)
            self._isolation_counter.pop(tid, None)
        interactions = self.interaction_detector.update(tracks)

        # Scene density calculation
        density = self._compute_scene_density(len(tracks))
        if not self._high_density_active:
            self._density_counter = self._density_counter + 1 if density >= self.density_enter else max(0, self._density_counter - 1)
            if self._density_counter >= self.density_hold_frames:
                self._high_density_active = True
        else:
            self._density_counter = self._density_counter + 1 if density >= self.density_exit else max(0, self._density_counter - 2)
            if density < self.density_exit and self._density_counter == 0:
                self._high_density_active = False

        track_payload = []
        suspicious_count = 0
        interacting_count = 0
        intrusion_count = 0

        centers = {}
        for tr in tracks:
            x1, y1, x2, y2 = [float(v) for v in tr["bbox"]]
            centers[int(tr["id"])] = (0.5 * (x1 + x2), y2) # Bottom-center coordinate for foot-level crossing

        for tr in tracks:
            tid = int(tr["id"])
            bbox = [float(v) for v in tr["bbox"]]
            st = states[tid]
            smooth_speed_px_per_frame = self.smoother.moving_average(st.velocity_history)
            speed_px_s = smooth_speed_px_per_frame * self.fps
            st.speed = speed_px_s

            # --- Virtual fence: foot-point movement vector vs tripwire ---
            x1, y1, x2, y2 = bbox
            curr_bottom_center = ((x1 + x2) / 2, y2)
            prev_bottom_center = self._prev_bottom.get(tid)
            # Restricted zone: flagged while the feet are inside, cleared the moment they leave
            if self.in_zone(curr_bottom_center):
                if tid not in self.zone_occupants:
                    self.zone_occupants.add(tid)
                    self._zone_entered_at[tid] = now
            else:
                self.zone_occupants.discard(tid)
                self._zone_entered_at.pop(tid, None)
            # Tripwire: crossing it flags the track (stays flagged while tracked)
            if self.crossed_tripwire(prev_bottom_center, curr_bottom_center):
                self.intruders.add(tid)
            self._prev_bottom[tid] = curr_bottom_center

            is_intruding = tid in self.intruders
            in_zone = tid in self.zone_occupants
            zone_dwell = now - self._zone_entered_at[tid] if in_zone else 0.0
            dwell_sec = self._update_loiter(tid, curr_bottom_center, now)
            is_loitering = dwell_sec >= self.loiter_seconds

            # Existing Anomaly Checks
            if not st.suspicious_active and speed_px_s >= self.suspicious_enter:
                st.suspicious_active = True
            elif st.suspicious_active and speed_px_s <= self.suspicious_exit:
                st.suspicious_active = False

            others = [oid for oid in centers.keys() if oid != tid]
            if others:
                cx, cy = centers[tid]
                nearest = min(((cx - centers[oid][0]) ** 2 + (cy - centers[oid][1]) ** 2) ** 0.5 for oid in others)
            else:
                nearest = 1e9

            few_people_scene = len(tracks) <= 3
            isolation_signal = few_people_scene and nearest >= self.isolation_distance_px
            if isolation_signal:
                self._isolation_counter[tid] += 1
            else:
                self._isolation_counter[tid] = max(0, self._isolation_counter[tid] - 1)

            if not st.isolated_active and self._isolation_counter[tid] >= self.isolation_enter_frames:
                st.isolated_active = True
            elif st.isolated_active and self._isolation_counter[tid] <= self.isolation_exit_frames:
                st.isolated_active = False

            interacting_ids = sorted(list(interactions.get(tid, set())))
            for iid in interacting_ids:
                st.interaction_history[iid] = st.interaction_history.get(iid, 0) + 1

            is_new_entry = (frame_idx - st.first_seen_frame) < self.new_entry_frames
            
            # Prioritized Flags
            active_flags = {
                "INTRUSION": is_intruding,
                "RESTRICTED_ZONE": in_zone,
                "LOITERING": is_loitering,
                "SUSPICIOUS_MOTION": st.suspicious_active,
                "INTERACTION": len(interacting_ids) > 0,
                "NEW_ENTRY": is_new_entry,
                "ISOLATED": st.isolated_active,
                "HIGH_DENSITY": self._high_density_active,
                "NORMAL": True,
            }
            proposed = self._with_priority(active_flags)
            if proposed in ("INTRUSION", "RESTRICTED_ZONE") or (
                    st.state_label == "RESTRICTED_ZONE" and not in_zone):
                # Fence events are geometric facts: apply instantly, no label smoothing
                if st.state_label != proposed:
                    st.state_label, st.state_duration = proposed, 0
                st.candidate_label, st.candidate_duration = proposed, 0
                st.state_duration += 1
                final_label, duration = proposed, st.state_duration
            else:
                final_label, duration = self.smoother.stable_label(st, proposed)

            if final_label == "INTRUSION":
                intrusion_count += 1
            if final_label in ("SUSPICIOUS_MOTION", "LOITERING", "RESTRICTED_ZONE"):
                suspicious_count += 1
            if final_label == "INTERACTION":
                interacting_count += 1

            conf = self._track_confidence(final_label, speed_px_s, len(interacting_ids), duration)
            if final_label == "LOITERING":
                conf = min(1.0, 0.75 + 0.25 * (dwell_sec - self.loiter_seconds) / max(1.0, self.loiter_seconds))
            
            # Hard gate for sustained motion
            if final_label == "SUSPICIOUS_MOTION":
                sustained = duration >= self.suspicious_min_frames
                high_conf = conf >= 0.90
                if not (sustained and high_conf):
                    final_label = "NORMAL"
                    conf = self._track_confidence("NORMAL", speed_px_s, len(interacting_ids), duration)
            st.confidence = conf

            track_payload.append({
                "id": tid,
                "bbox": bbox,
                "label": final_label,
                "confidence": round(conf, 3),
                "speed": round(speed_px_s, 2),
                "interaction_with": interacting_ids,
                "duration_in_state": int(duration),
                "dwell_sec": round(dwell_sec, 1),
                "in_zone": in_zone,
                "zone_dwell_sec": round(zone_dwell, 1),
            })

        scene_status = self._scene_status(suspicious_count, interacting_count, len(tracks), intrusion_count)
        self._last_scene_status = scene_status
        return {
            "tracks": track_payload,
            "exits": exits,
            "scene_status": scene_status,
            "scene_density": density,
        }
