"""
Orchestrator — Main multi-camera processing loop.

Wires together all Phase 1 + Phase 2 components into a single pipeline:
  StreamManager → PersonDetector → ReIDEncoder → SingleCameraTracker(s) → GlobalMatcher

Processing flow (each iteration):
  1. Grab latest frames from all cameras (non-blocking)
  2. Batch all frames through YOLO (single GPU call)
  3. Crop high-conf detections, batch through OSNet (single GPU call)
  4. Route detections + embeddings to per-camera trackers
  5. Process track events through GlobalMatcher
  6. Generate annotated frames for display/streaming

Supports:
  - Multiple webcams, RTSP streams, and video files
  - Batched GPU inference across all cameras
  - Real-time display with OpenCV windows or headless mode
"""
from __future__ import annotations

import sys
import os
import time
import csv
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import PersonDetector, Detection
from core.reid_encoder import ReIDEncoder
from core.tracker import SingleCameraTracker, TrackEvent, TrackState
from core.stream_manager import StreamManager, CameraConfig
from core.global_matcher import GlobalMatcher


class Orchestrator:
    """
    Main multi-camera tracking orchestrator.

    Usage:
        orch = Orchestrator(cameras=[...], display=True)
        orch.run()   # Blocking — runs until quit
    """

    def __init__(self,
                 cameras: List[CameraConfig],
                 # Model paths
                 yolo_path: str = "models/yolo11m.pt",
                 osnet_path: str = "models/osnet_x1_0_msmt17.pth",
                 device: str = "cuda",
                 # Display options
                 display: bool = True,
                 # Detector config
                 detector_config: Optional[Dict] = None,
                 # Tracker config
                 tracker_config: Optional[Dict] = None,
                 # Global matcher config
                 matcher_config: Optional[Dict] = None,
                 # Runtime config
                 runtime_config: Optional[Dict] = None,
                 # Transit times from topology
                 transit_times: Optional[Dict[str, Tuple[float, float]]] = None,
                 # Risk engine config
                 risk_config: Optional[Dict] = None):

        self.cameras = cameras
        self.display = display
        self.device = device

        # ── Initialize Models ──
        print("\n" + "=" * 60)
        print("  MULTI-CAMERA TRACKING SYSTEM — Phase 2")
        print("=" * 60)

        print(f"\n[1/4] Loading YOLOv11m...")
        d_cfg = detector_config or {}
        self.detector = PersonDetector(
            model_path=d_cfg.get("model_path", yolo_path),
            device=device,
            high_conf=d_cfg.get("high_conf", 0.40),
            low_conf=d_cfg.get("low_conf", 0.10),
            min_area=d_cfg.get("min_area", 100),
            nms_iou=d_cfg.get("nms_iou", 0.60),
            imgsz=d_cfg.get("imgsz", 640),
        )
        print("       ✓ Detector ready")

        print("[2/4] Loading OSNet_x1_0...")
        self.encoder = ReIDEncoder(
            model_path=osnet_path,
            device=device,
        )
        print("       ✓ ReID Encoder ready")

        # ── Initialize Global Matcher ──
        print("[3/4] Initializing Global Matcher...")
        m_cfg = matcher_config or {}
        self.global_matcher = GlobalMatcher(
            registered_threshold=m_cfg.get(
                "registered_threshold",
                m_cfg.get("registration_threshold", 0.70),
            ),
            gallery_threshold=m_cfg.get(
                "gallery_threshold",
                m_cfg.get("match_threshold", 0.60),
            ),
            margin_gate=m_cfg.get(
                "margin_gate",
                m_cfg.get("margin_threshold", 0.08),
            ),
            ema_alpha=m_cfg.get("ema_alpha", 0.10),
            ema_min_confidence=m_cfg.get("ema_min_confidence", 0.65),
            same_camera_min_gap=m_cfg.get("same_camera_min_gap", 1.0),
            diff_camera_default_min=m_cfg.get(
                "diff_camera_default_min",
                m_cfg.get("cross_camera_min_gap", 2.0),
            ),
            transit_times=transit_times,
        )
        print("       ✓ Global Matcher ready")

        # ── Initialize Streams + Trackers ──
        print(f"[4/4] Setting up {len(cameras)} cameras...")
        self.stream_manager = StreamManager()
        self.trackers: Dict[str, SingleCameraTracker] = {}

        t_cfg = tracker_config or {}
        for cam_cfg in cameras:
            # Add to stream manager
            self.stream_manager.add_camera(cam_cfg)

            # Create per-camera tracker (frame size will be updated on first frame)
            tracker = SingleCameraTracker(
                camera_id=cam_cfg.camera_id,
                frame_w=t_cfg.get("frame_w", 640),
                frame_h=t_cfg.get("frame_h", 480),
                n_confirm=t_cfg.get("n_confirm", 3),
                tentative_max_miss=t_cfg.get("tentative_max_miss", 3),
                max_miss=t_cfg.get("max_miss", 20),
                lost_buffer_sec=t_cfg.get("lost_buffer_sec", 15.0),
                gate_dist_px=t_cfg.get("gate_dist_px", 200),
                mahal_gate=t_cfg.get("mahal_gate", 9.48),
                cost_thr_high=t_cfg.get("cost_thr_high", t_cfg.get("cost_threshold_high", 0.65)),
                cost_thr_low=t_cfg.get("cost_thr_low", t_cfg.get("cost_threshold_low", 0.70)),
                weights_normal=tuple(t_cfg.get("weights_normal", (0.50, 0.40, 0.10))),
                weights_crowd=tuple(t_cfg.get("weights_crowd", (0.30, 0.45, 0.25))),
                crowd_overlap_min=t_cfg.get("crowd_overlap_min", 0.10),
                crowd_score_thr=t_cfg.get("crowd_score_thr", 0.08),
                sim_thr_live=t_cfg.get("sim_thr_live", t_cfg.get("sim_threshold_live", 0.45)),
                sim_thr_lost=t_cfg.get("sim_thr_lost", t_cfg.get("sim_threshold_lost", 0.35)),
                embedding_pool_k=t_cfg.get("embedding_pool_k", 5),
                dup_iou_thr=t_cfg.get("dup_iou_thr", 0.50),
                merge_sim_thr=t_cfg.get("merge_sim_thr", 0.60),
                min_export_frames=t_cfg.get("min_export_frames", 5),
                quality_threshold=t_cfg.get("quality_threshold", 0.60),
            )
            self.trackers[cam_cfg.camera_id] = tracker

        # Frame size per camera (updated on first frame)
        self._frame_sizes: Dict[str, Tuple[int, int]] = {}
        self._frame_sizes_initialized: Dict[str, bool] = {}

        # Processing resolution: resize large frames to this max dimension
        # e.g. 720p -> 360p style downscale for higher FPS
        runtime_cfg = runtime_config or {}
        self._max_process_dim = int(runtime_cfg.get("process_max_dim", 640))
        self._reid_every_n_frames = max(1, int(runtime_cfg.get("reid_every_n_frames", 1)))
        self._detection_event_cooldown_sec = float(
            runtime_cfg.get("detection_event_cooldown_sec", 0.0)
        )
        self._last_detection_event_ts: Dict[Tuple[str, int], float] = {}
        self._global_match_threshold = float(runtime_cfg.get("global_match_threshold", 0.70))
        self._global_min_track_age = int(runtime_cfg.get("global_min_track_age", 4))
        self._global_reid_update_every = max(1, int(runtime_cfg.get("global_reid_update_every", 4)))
        self._global_identity_ttl_sec = float(runtime_cfg.get("global_identity_ttl_sec", 180.0))
        self._global_embedding_history = int(runtime_cfg.get("global_embedding_history", 8))


        # Lightweight global identity store.
        self.global_db: Dict[int, Dict[str, Any]] = {}
        self.next_global_id = 1

        # Metrics
        self._frame_count = 0
        self._start_time = 0.0
        self._fps_display = 0.0

        # ── Risk Engine State ──
        r_cfg = risk_config or {}
        self._risk_alert_threshold = float(r_cfg.get("alert_threshold", 90.0))
        self._risk_decay_per_sec = float(r_cfg.get("decay_per_sec", 2.5))
        self._loiter_seconds = float(r_cfg.get("loiter_seconds", 30.0))
        self._loiter_displacement_px = float(r_cfg.get("loiter_displacement_px", 30.0))

        self._odd_start_hour = int(r_cfg.get("odd_hours_start", 1))
        self._odd_end_hour = int(r_cfg.get("odd_hours_end", 4))
        self._high_velocity_px_s = float(r_cfg.get("high_velocity_px_s", 300.0))
        self._high_velocity_sustain = int(r_cfg.get("high_velocity_sustain_frames", 5))
        self._crowd_threshold = int(r_cfg.get("crowd_threshold", 6))
        self._frequent_window_sec = float(r_cfg.get("entry_exit_window_sec", 120.0))
        self._frequent_count_thr = int(r_cfg.get("frequent_entry_exit_count", 6))
        self._min_track_age_sec = float(r_cfg.get("min_track_age_sec", 15.0))
        self._min_concurrent_flags = int(r_cfg.get("min_concurrent_flags", 2))
        self._alerts_enabled = bool(runtime_cfg.get("alerts_enabled", True))

        # Per-camera, per-track risk state: {cam_id: {track_id: state_dict}}
        self._risk_state: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # Per global-ID activity history
        self._global_activity: Dict[int, Dict[str, Any]] = {}



        # CSV event log
        self._log_path = os.path.join("logs", "events.csv")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "camera_id", "global_id", "local_track_id",
                            "event", "risk", "details"])
        print("       ✓ Risk engine initialized (strict mode)")

        # Frame dedup disabled: process latest available detections every loop.

    def run(self) -> None:
        """
        Main processing loop. Runs until user presses Q or all streams end.
        """
        # Start camera streams
        self.stream_manager.start_all()
        self._start_time = time.time()

        print("\n🎬 Starting multi-camera tracking... Press Q to quit.\n")

        try:
            while True:
                # ── 1. Grab latest frames from all cameras ──
                frame_data = self.stream_manager.grab_latest()

                if not frame_data:
                    if self.stream_manager.all_stopped():
                        print("\n📹 All video streams ended.")
                        break
                    time.sleep(0.01)
                    continue

                # Keep all latest frames (no timestamp-based skipping).
                self._frame_count += 1
                loop_ts = time.time() - self._start_time

                # ── 2. Batch all frames through YOLO ──
                cam_ids_ordered = list(frame_data.keys())

                # Resize oversized frames for processing speed
                processed_frames: Dict[str, np.ndarray] = {}
                frame_scales: Dict[str, Tuple[float, float]] = {}
                for cam_id in cam_ids_ordered:
                    original_frame = frame_data[cam_id][0]
                    frame = original_frame
                    h, w = original_frame.shape[:2]
                    max_dim = max(h, w)
                    scale = 1.0
                    if max_dim > self._max_process_dim:
                        scale = self._max_process_dim / max_dim
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                    processed_frames[cam_id] = frame
                    frame_scales[cam_id] = (1.0 / scale, 1.0 / scale)

                    # Update tracker dimensions on first frame (or if resized)
                    if cam_id not in self._frame_sizes_initialized:
                        ph, pw = original_frame.shape[:2]
                        self._frame_sizes[cam_id] = (pw, ph)
                        self._frame_sizes_initialized[cam_id] = True
                        self.trackers[cam_id].W = pw
                        self.trackers[cam_id].H = ph
                        diag = (pw ** 2 + ph ** 2) ** 0.5
                        ref_diag = (640 ** 2 + 480 ** 2) ** 0.5
                        self.trackers[cam_id].gate_dist = int(200 * (diag / ref_diag))

                frames_list = [processed_frames[cid] for cid in cam_ids_ordered]
                all_detections = self.detector.batch_detect(frames_list)
                # Scale detection boxes back to ORIGINAL frame coordinates.
                for cam_idx, cam_id in enumerate(cam_ids_ordered):
                    sx, sy = frame_scales[cam_id]
                    high_dets, low_dets = all_detections[cam_idx]
                    scaled_high: List[Detection] = []
                    scaled_low: List[Detection] = []
                    for det in high_dets:
                        x1, y1, x2, y2 = det.bbox
                        scaled_high.append(
                            Detection(bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy), confidence=det.confidence)
                        )
                    for det in low_dets:
                        x1, y1, x2, y2 = det.bbox
                        scaled_low.append(
                            Detection(bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy), confidence=det.confidence)
                        )
                    all_detections[cam_idx] = (scaled_high, scaled_low)

                # ── 3. Batch all high-conf crops through OSNet ──
                # Collect all crops across all cameras for a single GPU call
                all_crops: List[np.ndarray] = []
                crop_map: List[Tuple[int, int]] = []  # (cam_index, det_index) for each crop
                do_reid_this_frame = (self._frame_count % self._reid_every_n_frames == 0)

                for cam_idx, cam_id in enumerate(cam_ids_ordered):
                    high_dets, low_dets = all_detections[cam_idx]
                    frame = frame_data[cam_id][0]
                    fh, fw = frame.shape[:2]
                    tracker = self.trackers[cam_id]

                    for det_idx, det in enumerate(high_dets):
                        # ReID policy:
                        # - periodic updates every N frames
                        # - immediate for likely new tracks (low IOU vs existing tracks)
                        is_likely_new_track = True
                        for existing_track in tracker.tracks.values():
                            inter_x1 = max(det.bbox[0], existing_track.bbox[0])
                            inter_y1 = max(det.bbox[1], existing_track.bbox[1])
                            inter_x2 = min(det.bbox[2], existing_track.bbox[2])
                            inter_y2 = min(det.bbox[3], existing_track.bbox[3])
                            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                a1 = max(1.0, (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))
                                a2 = max(1.0, (existing_track.bbox[2] - existing_track.bbox[0]) * (existing_track.bbox[3] - existing_track.bbox[1]))
                                iou = inter / max(1.0, a1 + a2 - inter)
                                if iou >= 0.20:
                                    is_likely_new_track = False
                                    break
                        if not do_reid_this_frame and not is_likely_new_track:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in det.bbox]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(fw, x2), min(fh, y2)
                        if x2 > x1 + 2 and y2 > y1 + 2:
                            crop = frame[y1:y2, x1:x2]

                            # ── Crop clipping: reduce ReID contamination ──
                            # If this detection overlaps with another, clip the
                            # crop on the overlap side so OSNet sees primarily
                            # THIS person's features, not the neighbor's.
                            for other_idx, other_det in enumerate(high_dets):
                                if other_idx == det_idx:
                                    continue
                                ox1, oy1, ox2, oy2 = other_det.bbox
                                # Check if bboxes overlap
                                inter_x1 = max(x1, ox1)
                                inter_y1 = max(y1, oy1)
                                inter_x2 = min(x2, ox2)
                                inter_y2 = min(y2, oy2)
                                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                                    # Overlap exists — clip on the overlap side
                                    other_cx = (ox1 + ox2) / 2
                                    my_cx = (x1 + x2) / 2
                                    cw = crop.shape[1]
                                    if cw > 20:  # Only clip if crop is wide enough
                                        if other_cx > my_cx:
                                            # Other person is to the RIGHT → keep left 70%
                                            crop = crop[:, :int(cw * 0.7)]
                                        else:
                                            # Other person is to the LEFT → keep right 70%
                                            crop = crop[:, int(cw * 0.3):]
                                    break  # Only clip once per detection

                            all_crops.append(crop)
                            crop_map.append((cam_idx, det_idx))

                # Single batch ReID extraction across all cameras
                all_embeddings = None
                if all_crops:
                    all_embeddings = self.encoder.batch_extract(all_crops)

                # ── 4. Route to per-camera trackers ──
                all_events: List[Tuple[str, TrackEvent]] = []

                for cam_idx, cam_id in enumerate(cam_ids_ordered):
                    high_dets, low_dets = all_detections[cam_idx]

                    # Build embedding map for this camera
                    cam_embeddings: Dict[int, np.ndarray] = {}
                    if all_embeddings is not None:
                        for crop_idx, (ci, di) in enumerate(crop_map):
                            if ci == cam_idx and crop_idx < len(all_embeddings):
                                cam_embeddings[di] = all_embeddings[crop_idx]

                    # Run tracker update using per-camera frame capture timestamp
                    frame_ts = frame_data[cam_id][1] - self._start_time
                    if frame_ts < 0:
                        frame_ts = loop_ts
                    events = self.trackers[cam_id].update(
                        high_dets, low_dets, cam_embeddings, frame_ts
                    )

                    for ev in events:
                        all_events.append((cam_id, ev))

                # ── 5. Process events through Global Matcher ──
                self._process_events(all_events, loop_ts)

                # ── 5.5. Process behavioral risk ──
                for cam_id, tracker in self.trackers.items():
                    self._update_risk_engine(cam_id, tracker, loop_ts)

                # Periodic embedding refresh for already mapped active tracks.
                if self._frame_count % self._global_reid_update_every == 0:
                    for cam_id, tracker in self.trackers.items():
                        for track in tracker.get_active_tracks():
                            gid = tracker.local_to_global.get(track.track_id)
                            if gid is None or track.pooled_embedding is None:
                                continue
                            norm_pool = self._normalize_embedding(track.pooled_embedding)
                            if norm_pool is None:
                                continue
                            self._update_global_embedding(gid, norm_pool, loop_ts)

                # Periodic cleanup for stale global identities.
                if self._frame_count % 30 == 0:
                    self._prune_global_db(loop_ts)

                # ── 6. Display ──
                if self.display:
                    # Draw on original frames (same coordinate space as tracker bboxes)
                    display_data = {
                        cam_id: (frame_data[cam_id][0], frame_data[cam_id][1])
                        for cam_id in cam_ids_ordered
                        if cam_id in frame_data
                    }
                    self._display_frames(display_data, cam_ids_ordered)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    elif key in (ord('s'), ord('S')):
                        self._save_screenshots(display_data, cam_ids_ordered)

        except KeyboardInterrupt:
            print("\n\n⏹  Interrupted by user")

        # ── Cleanup ──
        final_ts = loop_ts if self._frame_count > 0 else time.time() - self._start_time
        self._finalize(final_ts)

    def _process_events(self, events: List[Tuple[str, TrackEvent]],
                        timestamp: float) -> None:
        """Route track events to lightweight global identity database."""
        for cam_id, ev in events:
            tracker = self.trackers[cam_id]

            if ev.type == "TRACK_ACTIVATED":
                # Always log activation attempts for debugging visibility.
                print(
                    f"  👀 [{cam_id}] Track #{ev.track_id} ACTIVATED "
                    f"(age={ev.track_age})"
                )

                # Prefer event embedding, fallback to current track pooled/best embedding.
                candidate_embedding = ev.embedding
                track_obj = tracker.tracks.get(ev.track_id)
                if candidate_embedding is None and track_obj is not None:
                    candidate_embedding = (
                        track_obj.best_embedding
                        if track_obj.best_embedding is not None
                        else track_obj.pooled_embedding
                    )

                if candidate_embedding is None:
                    print(
                        f"  ⚠️ [{cam_id}] Track #{ev.track_id}: embedding unavailable, "
                        "skipping global match"
                    )
                    continue
                if ev.track_age < self._global_min_track_age:
                    print(
                        f"  ⏳ [{cam_id}] Track #{ev.track_id}: waiting for stable age "
                        f"(need >= {self._global_min_track_age}, got {ev.track_age})"
                    )
                    continue

                normalized = self._normalize_embedding(candidate_embedding)
                if normalized is None:
                    print(
                        f"  ⚠️ [{cam_id}] Track #{ev.track_id}: invalid embedding norm, "
                        "skipping global match"
                    )
                    continue
                gid, score, is_new = self._match_or_create_global_id(
                    normalized, timestamp
                )
                event_key = (cam_id, gid)
                last_emit = self._last_detection_event_ts.get(event_key, -1e9)
                if (timestamp - last_emit) < self._detection_event_cooldown_sec:
                    print(
                        f"  💤 [{cam_id}] Track #{ev.track_id}: cooldown active for G#{gid}"
                    )
                    continue
                self._last_detection_event_ts[event_key] = timestamp
                tracker.set_global_id(ev.track_id, gid, f"Person #{gid}")
                
                # Initialize risk state for this local track
                self._risk_state.setdefault(cam_id, {})[ev.track_id] = {
                    "risk": 0.0,
                    "last_update": timestamp,
                    "pos_hist": deque(maxlen=240),
                    "loitering": False,

                    "last_center": None,
                    "vel_hist": deque(maxlen=10),
                    "last_flags": "",
                    "entry_time": timestamp,
                }
                
                action = "NEW PERSON DETECTED" if is_new else "EXISTING PERSON MATCHED"
                print(
                    f"  🌍 [{cam_id}] Track #{ev.track_id} → Global #{gid} "
                    f"[{action}, sim={score:.2f}]"
                )
                print(
                    f"  🧪 DBG cam={cam_id} track={ev.track_id} global={gid} conf={score:.3f}"
                )
                if is_new:
                    # New person baseline risk
                    if ev.track_id in self._risk_state.setdefault(cam_id, {}):
                        self._risk_state[cam_id][ev.track_id]["risk"] = min(
                            100.0, self._risk_state[cam_id][ev.track_id]["risk"] + 10.0
                        )

                    # Initialize global activity if new
                    if gid not in self._global_activity:
                        now_wall = time.time()
                        self._global_activity[gid] = {
                            "entries": deque(maxlen=50),
                            "first_seen": now_wall,
                            "last_seen": now_wall,
                        }
                    self._global_activity[gid]["entries"].append(time.time())
                    self._global_activity[gid]["last_seen"] = time.time()

                    with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            cam_id,
                            gid,
                            ev.track_id,
                            "TRACK_ACTIVATED",
                            f"{self._risk_state.get(cam_id, {}).get(ev.track_id, {}).get('risk', 0.0):.1f}",
                            action,
                        ])

            elif ev.type == "TRACK_ENDED":
                gid = tracker.local_to_global.get(ev.track_id)
                if gid is not None:
                    identity = self.global_db.get(gid)
                    if identity is not None:
                        identity["last_seen"] = timestamp
                    if ev.pooled_embedding is not None:
                        norm_pool = self._normalize_embedding(ev.pooled_embedding)
                        if norm_pool is not None:
                            self._update_global_embedding(gid, norm_pool, timestamp)
                    duration = ev.exit_time - ev.entry_time
                    print(
                        f"  ⏹️  [{cam_id}] Track #{ev.track_id} ended "
                        f"(G#{gid}, {duration:.1f}s, q={ev.quality:.2f})"
                    )
                tracker.local_to_global.pop(ev.track_id, None)
                if cam_id in self._risk_state:
                    self._risk_state[cam_id].pop(ev.track_id, None)

    def _normalize_embedding(self, emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if emb is None:
            return None
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return None
        return vec / norm

    def _match_or_create_global_id(self, embedding: np.ndarray,
                                   timestamp: float) -> Tuple[int, float, bool]:
        best_gid: Optional[int] = None
        best_score = -1.0
        for gid, data in self.global_db.items():
            embs: List[np.ndarray] = data.get("embeddings", [])
            if not embs:
                continue
            sims = [float(np.dot(embedding, e)) for e in embs]
            score = max(sims)
            if score > best_score:
                best_score = score
                best_gid = gid
        if best_gid is not None and best_score >= self._global_match_threshold:
            self._update_global_embedding(best_gid, embedding, timestamp)
            return best_gid, best_score, False
        gid = self.next_global_id
        self.next_global_id += 1
        self.global_db[gid] = {
            "embeddings": [embedding.copy()],
            "last_seen": timestamp,
        }
        return gid, 0.0, True

    def _update_global_embedding(self, gid: int, embedding: np.ndarray, timestamp: float) -> None:
        if gid not in self.global_db:
            return
        data = self.global_db[gid]
        embs: List[np.ndarray] = data.setdefault("embeddings", [])
        embs.append(embedding.copy())
        if len(embs) > self._global_embedding_history:
            del embs[:-self._global_embedding_history]
        data["last_seen"] = timestamp

    def _prune_global_db(self, timestamp: float) -> None:
        to_remove = [
            gid for gid, data in self.global_db.items()
            if (timestamp - float(data.get("last_seen", 0.0))) > self._global_identity_ttl_sec
        ]
        for gid in to_remove:
            self.global_db.pop(gid, None)



    def _update_risk_engine(self, cam_id: str, tracker: SingleCameraTracker, timestamp: float) -> None:
        """Process behavioral risk analysis for all active tracks on a camera."""
        active_tracks = tracker.get_active_tracks()
        people_count = len(active_tracks)

        for t in active_tracks:
            tid = t.track_id
            gid = tracker.local_to_global.get(tid)
            if gid is None:
                continue

            st = self._risk_state.setdefault(cam_id, {}).setdefault(tid, {
                "risk": 0.0,
                "last_update": timestamp,
                "pos_hist": deque(maxlen=240),
                "loitering": False,
                "last_center": None,
                "vel_hist": deque(maxlen=10),
                "last_flags": "",
                "entry_time": timestamp,
            })

            dt = max(1e-3, float(timestamp - st["last_update"]))
            st["last_update"] = timestamp
            # Faster decay to clear transient noise
            st["risk"] = max(0.0, min(100.0, st["risk"] - self._risk_decay_per_sec * dt))

            x1, y1, x2, y2 = t.bbox
            cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
            st["pos_hist"].append((timestamp, cx, cy))
            flags = []

            # 1. Strict Loitering
            if len(st["pos_hist"]) >= 2:
                t0, x0, y0 = st["pos_hist"][0]
                disp = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
                if (timestamp - t0) >= self._loiter_seconds and disp < self._loiter_displacement_px:
                    if not st["loitering"]:
                        st["risk"] = max(0.0, min(100.0, st["risk"] + 20.0))
                        st["loitering"] = True
                    flags.append("LOITERING")
                else:
                    st["loitering"] = False

            # 2. Odd hours
            hr = datetime.now().hour
            if self._odd_start_hour <= hr < self._odd_end_hour:
                st["risk"] = max(0.0, min(100.0, st["risk"] + 10.0 * dt / 5.0))
                flags.append("ODD-HOUR")

            # 4. Motion anomaly (strict: high velocity sustained)
            if st["last_center"] is not None:
                px, py = st["last_center"]
                vel = (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5) / dt
                st["vel_hist"].append(vel)
                if len(st["vel_hist"]) >= self._high_velocity_sustain:
                    recent_vels = list(st["vel_hist"])[-self._high_velocity_sustain:]
                    if all(v > self._high_velocity_px_s for v in recent_vels):
                        st["risk"] = max(0.0, min(100.0, st["risk"] + 15.0))
                        flags.append("FAST")
            st["last_center"] = (cx, cy)

            # 5. Frequent entry/exit
            acts = self._global_activity.get(gid, {})
            entries = acts.get("entries", deque())
            recent = [t0 for t0 in entries if (time.time() - t0) <= self._frequent_window_sec]
            if len(recent) >= self._frequent_count_thr:
                st["risk"] = max(0.0, min(100.0, st["risk"] + 15.0))
                flags.append("FREQ-ENTRY")

            # 6. Crowd density
            if people_count > self._crowd_threshold:
                st["risk"] = max(0.0, min(100.0, st["risk"] + 8.0))
                flags.append("CROWD")

            # Familiar person baseline decrease
            if len(recent) > 8:
                st["risk"] = max(0.0, min(100.0, st["risk"] - 5.0))

            st["last_flags"] = "|".join(sorted(set(flags))) if flags else "NORMAL"

            # Alert evaluation (strict AND-logic) — log only
            track_age = timestamp - st.get("entry_time", timestamp)
            if (self._alerts_enabled and
                st["risk"] >= self._risk_alert_threshold and
                track_age >= self._min_track_age_sec and
                len(set(flags)) >= self._min_concurrent_flags):
                with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        cam_id, gid, tid, "RISK_ALERT",
                        f"{st['risk']:.1f}", st["last_flags"]
                    ])

    def _display_frames(self, frame_data: Dict[str, Tuple[np.ndarray, float]],
                        cam_ids: List[str]) -> None:
        """Draw annotated frames with improved bounding boxes and risk scores."""
        # Update FPS every 10 frames
        if self._frame_count % 10 == 0:
            elapsed = time.time() - self._start_time
            self._fps_display = self._frame_count / elapsed if elapsed > 0 else 0

        annotated_frames = {}
        for cam_id in cam_ids:
            frame = frame_data[cam_id][0]
            tracker = self.trackers[cam_id]
            annotated = frame.copy()  # Draw directly — no tracker.draw_on_frame



            active = tracker.get_active_tracks()

            # Draw ALL tracks (tentative + active) with risk overlays
            for t in tracker.tracks.values():
                tid = t.track_id
                gid = tracker.local_to_global.get(tid)
                x1, y1, x2, y2 = [int(v) for v in t.bbox]
                bw, bh = x2 - x1, y2 - y1

                # Determine risk state
                st = self._risk_state.get(cam_id, {}).get(tid, {})
                risk_val = float(st.get("risk", 0.0))
                status = st.get("last_flags", "NORMAL")

                # Color based on state + risk

                if t.state == TrackState.TENTATIVE:
                    color = (0, 220, 220)   # Yellow — tentative
                    border_thickness = 1
                elif risk_val >= 70:
                    color = (0, 0, 255)     # Red — high risk
                    border_thickness = 3
                elif risk_val >= 40:
                    color = (0, 180, 255)   # Orange — medium risk
                    border_thickness = 2
                elif risk_val >= 15:
                    color = (0, 255, 255)   # Yellow — low risk
                    border_thickness = 2
                else:
                    color = (0, 220, 0)     # Green — safe
                    border_thickness = 2

                # Draw corner brackets instead of full rectangle for a modern look
                corner_len = max(12, min(bw, bh) // 4)
                # Top-left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, border_thickness)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, border_thickness)
                # Top-right
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, border_thickness)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, border_thickness)
                # Bottom-left
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, border_thickness)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, border_thickness)
                # Bottom-right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, border_thickness)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, border_thickness)
                # Thin full rectangle outline
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

                # Build label text
                if gid is not None:
                    id_label = f"G#{gid}"
                else:
                    id_label = f"T#{tid}"
                risk_label = f"R:{risk_val:.0f}"

                # --- Top label bar: ID + Risk Score ---
                top_text = f"{id_label}  {risk_label}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.50
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(top_text, font, font_scale, thickness)
                label_h = th + baseline + 8
                # Background pill
                cv2.rectangle(annotated, (x1, max(0, y1 - label_h)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, top_text, (x1 + 5, max(th + 2, y1 - baseline - 3)),
                            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                # --- Bottom label bar: status flags (only when non-NORMAL) ---
                if status and status != "NORMAL" and gid is not None:
                    flag_text = status
                    (fw, fh), fb = cv2.getTextSize(flag_text, font, 0.40, 1)
                    flag_h = fh + fb + 6
                    flag_bg_color = (0, 0, 180) if risk_val >= 70 else (0, 120, 180)
                    cv2.rectangle(annotated, (x1, y2), (x1 + fw + 8, y2 + flag_h), flag_bg_color, -1)
                    cv2.putText(annotated, flag_text, (x1 + 4, y2 + fh + 2),
                                font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Risk bar (thin horizontal bar under the top label)
                if gid is not None:
                    bar_w = max(1, x2 - x1)
                    fill_w = int(bar_w * min(risk_val, 100.0) / 100.0)
                    bar_y = max(0, y1 - label_h - 4)
                    cv2.rectangle(annotated, (x1, bar_y), (x1 + bar_w, bar_y + 3), (80, 80, 80), -1)
                    if fill_w > 0:
                        bar_color = (0, 0, 255) if risk_val >= 70 else (0, 180, 255) if risk_val >= 40 else (0, 220, 0)
                        cv2.rectangle(annotated, (x1, bar_y), (x1 + fill_w, bar_y + 3), bar_color, -1)

            # HUD overlay (top-left info panel)
            hud = [
                f"[{cam_id}] FPS: {self._fps_display:.1f}",
                f"Persons: {len(active)}",
                f"Global IDs: {len(self.global_db)}",
                f"Crowd: {'ALERT' if len(active) > self._crowd_threshold else 'OK'}",
            ]
            # Semi-transparent HUD background
            hud_h = 24 * len(hud) + 12
            hud_overlay = annotated.copy()
            cv2.rectangle(hud_overlay, (4, 4), (230, hud_h), (0, 0, 0), -1)
            cv2.addWeighted(hud_overlay, 0.55, annotated, 0.45, 0, annotated)
            for i, line in enumerate(hud):
                txt_color = (0, 200, 255) if i == 0 else (200, 200, 200)
                cv2.putText(annotated, line, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, txt_color, 1, cv2.LINE_AA)

            annotated_frames[cam_id] = annotated

        # Display: tile cameras in a grid if multiple
        if len(annotated_frames) == 1:
            cam_id = list(annotated_frames.keys())[0]
            cv2.imshow(f"Multi-Camera Tracking", annotated_frames[cam_id])
        else:
            grid = self._make_grid(annotated_frames, cam_ids)
            cv2.imshow("Multi-Camera Tracking", grid)

    def _make_grid(self, frames: Dict[str, np.ndarray],
                   cam_ids: List[str]) -> np.ndarray:
        """Tile multiple camera views into a 2×N grid."""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Target size for each tile
        tile_w, tile_h = 640, 480
        n = len(cam_ids)

        # Determine grid layout
        if n <= 2:
            cols, rows = 2, 1
        elif n <= 4:
            cols, rows = 2, 2
        else:
            cols = 3
            rows = (n + cols - 1) // cols

        grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, cam_id in enumerate(cam_ids):
            if cam_id not in frames:
                continue
            r, c = divmod(idx, cols)
            frame = frames[cam_id]
            fh, fw = frame.shape[:2]
            scale = min(tile_w / max(fw, 1), tile_h / max(fh, 1))
            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))
            resized = cv2.resize(frame, (new_w, new_h))
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            off_x = (tile_w - new_w) // 2
            off_y = (tile_h - new_h) // 2
            tile[off_y:off_y + new_h, off_x:off_x + new_w] = resized
            grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

        return grid

    def _save_screenshots(self, frame_data: Dict, cam_ids: List[str]) -> None:
        """Save screenshots of all cameras."""
        for cam_id in cam_ids:
            frame = frame_data[cam_id][0]
            annotated = self.trackers[cam_id].draw_on_frame(frame)
            fname = f"screenshot_{cam_id}_{self._frame_count}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"  📸 [{cam_id}] Screenshot: {fname}")

    def _finalize(self, timestamp: float) -> None:
        """Clean up all resources and print summary."""
        # Finalize all trackers
        for cam_id, tracker in self.trackers.items():
            events = tracker.finalize_all(timestamp)
            for ev in events:
                if ev.type == "TRACK_ENDED":
                    current_gid = tracker.local_to_global.get(ev.track_id)
                    if current_gid is not None and ev.pooled_embedding is not None:
                        norm_pool = self._normalize_embedding(ev.pooled_embedding)
                        if norm_pool is not None:
                            self._update_global_embedding(current_gid, norm_pool, timestamp)
                    duration = ev.exit_time - ev.entry_time
                    print(f"  ⏹  [{cam_id}] Track #{ev.track_id} finalized "
                          f"(G#{current_gid if current_gid is not None else 'N/A'}, {duration:.1f}s)")
                    tracker.local_to_global.pop(ev.track_id, None)

        # Stop streams
        self.stream_manager.stop_all()
        cv2.destroyAllWindows()

        # Summary
        elapsed = time.time() - self._start_time
        print(f"\n{'=' * 60}")
        print(f"  Session Summary")
        print(f"{'=' * 60}")
        print(f"  Duration:        {elapsed:.1f}s")
        print(f"  Frames:          {self._frame_count}")
        print(f"  Avg FPS:         {self._frame_count / elapsed:.1f}" if elapsed > 0 else "  Avg FPS: N/A")
        print(f"  Cameras:         {len(self.cameras)}")
        print(f"  Global persons:  {len(self.global_db)}")

        # Per-camera stats
        for cam_id, tracker in self.trackers.items():
            print(f"  [{cam_id}] local tracks: {tracker._next_id}")

        # Print all identities
        if self.global_db:
            print(f"\n  Known Persons:")
            for gid, data in sorted(self.global_db.items()):
                print(f"    G#{gid}: Person #{gid} (embeddings={len(data.get('embeddings', []))})")
        print(f"{'=' * 60}")


def load_cameras_from_config(cameras_yaml: str = "config/cameras.yaml",
                             sources: Optional[Dict[str, object]] = None
                             ) -> Tuple[List[CameraConfig], Dict[str, Tuple[float, float]]]:
    """
    Load camera configurations from YAML file and merge with sources.

    Args:
        cameras_yaml: Path to cameras.yaml
        sources: Override sources dict {cam_id: source} — cam_id → int/str

    Returns:
        (List of CameraConfigs, transit_times dict)
    """
    configs: List[CameraConfig] = []
    transit: Dict[str, Tuple[float, float]] = {}

    if os.path.exists(cameras_yaml):
        with open(cameras_yaml, "r") as f:
            data = yaml.safe_load(f) or {}

        if data and "cameras" in data:
            cam_meta: Dict[str, Dict[str, Any]] = data["cameras"]

            if sources:
                # Include only cameras explicitly requested via CLI mapping.
                for cam_id, source in sources.items():
                    cam_data = cam_meta.get(cam_id, {})
                    configs.append(CameraConfig(
                        camera_id=cam_id,
                        source=source,
                        name=cam_data.get("name", cam_id),
                        building=cam_data.get("building", ""),
                        floor=cam_data.get("floor", 0),
                        gps_lat=cam_data.get("gps_lat", 0.0),
                        gps_lng=cam_data.get("gps_lng", 0.0),
                    ))
            else:
                # No source overrides: keep all cameras from YAML.
                for cam_id, cam_data in cam_meta.items():
                    source = cam_data.get("source", 0)
                    configs.append(CameraConfig(
                        camera_id=cam_id,
                        source=source,
                        name=cam_data.get("name", cam_id),
                        building=cam_data.get("building", ""),
                        floor=cam_data.get("floor", 0),
                        gps_lat=cam_data.get("gps_lat", 0.0),
                        gps_lng=cam_data.get("gps_lng", 0.0),
                    ))

        if data and "transit_times" in data:
            for key, times in data["transit_times"].items():
                if isinstance(times, list) and len(times) == 2:
                    transit[key] = (float(times[0]), float(times[1]))

    # Fallback if YAML missing/empty.
    if not configs and sources:
        for cam_id, source in sources.items():
            configs.append(CameraConfig(camera_id=cam_id, source=source, name=cam_id))

    return configs, transit


def load_system_config(config_yaml: str = "config/config.yaml") -> Dict[str, Any]:
    """Load detector/tracker/matcher/system settings from YAML."""
    if not os.path.exists(config_yaml):
        return {}
    with open(config_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def main():
    """
    Entry point for multi-camera tracking.

    Usage:
        python -m core.orchestrator                           # Default: 1 webcam
        python -m core.orchestrator --cameras 0 1             # 2 webcams
        python -m core.orchestrator --cameras rtsp://a rtsp://b  # 2 RTSP
        python -m core.orchestrator --cameras video1.mp4 video2.mp4
    """
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking System")
    parser.add_argument("--cameras", nargs="+", default=["0"],
                        help="Camera sources: int (webcam), rtsp://..., or video.mp4")
    parser.add_argument("--config", default="config/cameras.yaml",
                        help="Camera topology config file (camera metadata + transit times)")
    parser.add_argument("--system-config", default="config/config.yaml",
                        help="System config file (detector/tracker/matcher settings)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no OpenCV windows)")
    parser.add_argument("--device", default=None,
                        help="Device override: cuda or cpu")
    parser.add_argument("--stream", default=None,
                        help="Stream output target (e.g. frontend, rtsp, hls)")
    args = parser.parse_args()

    # Parse camera sources
    sources = {}
    for i, src in enumerate(args.cameras):
        cam_id = f"cam{i + 1}"
        try:
            sources[cam_id] = int(src)
        except ValueError:
            sources[cam_id] = src

    # Load topology from config
    configs, transit_times = load_cameras_from_config(args.config, sources)
    # Load system-level runtime settings
    system_cfg = load_system_config(args.system_config)
    detector_cfg = system_cfg.get("detection", {}) if isinstance(system_cfg, dict) else {}
    reid_cfg = system_cfg.get("reid", {}) if isinstance(system_cfg, dict) else {}
    tracker_cfg = system_cfg.get("tracker", {}) if isinstance(system_cfg, dict) else {}
    matcher_cfg = system_cfg.get("global_matcher", {}) if isinstance(system_cfg, dict) else {}
    runtime_cfg = system_cfg.get("runtime", {}) if isinstance(system_cfg, dict) else {}
    sys_cfg = system_cfg.get("system", {}) if isinstance(system_cfg, dict) else {}

    device = args.device or sys_cfg.get("device", "cuda")
    yolo_path = detector_cfg.get("model_path", "models/yolo11m.pt")
    osnet_path = reid_cfg.get("model_path", "models/osnet_x1_0_msmt17.pth")

    # Run orchestrator
    orch = Orchestrator(
        cameras=configs,
        yolo_path=yolo_path,
        osnet_path=osnet_path,
        device=device,
        display=not args.no_display,
        detector_config=detector_cfg,
        tracker_config=tracker_cfg,
        matcher_config=matcher_cfg,
        runtime_config=runtime_cfg,
        transit_times=transit_times,
    )
    orch.run()


if __name__ == "__main__":
    main()
