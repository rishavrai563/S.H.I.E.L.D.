"""
SHIELD Master Orchestrator
--------------------------
One processing pipeline shared by the CLI (OpenCV windows) and the API server.

Per iteration:
  1. Grab latest frames from all cameras (non-blocking, night-enhanced)
  2. Batch all frames through YOLOv11m (persons + 4 vehicle classes)
  3. Vehicles  → VehicleTracker → ANPR (multi-pass OCR + temporal voting)
                 → plate watchlist matching
  4. Persons   → OSNet Re-ID → SingleCameraTracker → GlobalMatcher (cross-camera)
                 → Facial Recognition against the enrolled suspect gallery
  5. Behaviour → RiskAnalyzer (virtual fence, loitering, suspicious motion …)
  6. Alerts    → AlertManager (cooldowns, events.jsonl, optional C&C webhook)
  7. Telemetry → callback (WebSocket) + annotated frames
"""
from __future__ import annotations

import collections
import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.detector import SecurityDetector
from core.kalman import bbox_iou
from core.device import resolve_device
from core.reid_encoder import ReIDEncoder
from core.tracker import SingleCameraTracker
from core.stream_manager import StreamManager, CameraConfig
from core.global_matcher import GlobalMatcher
from core.behavior_risk import RiskAnalyzer
from core.anpr import ANPREngine
from core.face_auth import FaceRecognitionEngine, THREAT_CATEGORIES
from core.vehicle_tracker import VehicleTracker
from core.alerts import AlertManager

# Behaviour labels that raise operator alerts, with their severity.
BEHAVIOUR_ALERTS = {
    "INTRUSION": ("CRITICAL", "Tripwire crossed"),
    "LOITERING": ("WARNING", "Loitering detected"),
    "SUSPICIOUS_MOTION": ("WARNING", "Suspicious motion"),
}
CRITICAL_PLATE_TAGS = {"BLACKLISTED", "STOLEN", "WANTED", "RESTRICTED", "SMUGGLING"}


class Orchestrator:
    def __init__(self,
                 cameras: List[CameraConfig],
                 device: str = "cuda",
                 display: bool = True,
                 telemetry_callback: Optional[Callable[[Dict], None]] = None,
                 config: Optional[Dict] = None):
        device = resolve_device(device)
        self.cameras = cameras
        self.display = display
        self.device = device
        self.telemetry_callback = telemetry_callback
        self.config = config or {}

        det_cfg = self.config.get("detection", {})
        reid_cfg = self.config.get("reid", {})
        tracker_cfg = self.config.get("tracker", {})
        gm_cfg = self.config.get("global_matcher", {})
        risk_cfg = self.config.get("risk", {})
        stream_cfg = self.config.get("stream", {})
        runtime_cfg = self.config.get("runtime", {})
        face_cfg = self.config.get("face_recognition", {})
        anpr_cfg = self.config.get("anpr", {})
        alert_cfg = self.config.get("alerts", {})
        log_cfg = self.config.get("logging", {})
        integ_cfg = self.config.get("integrations", {})

        print("\n" + "=" * 60)
        print("  SHIELD: TACTICAL COMMAND & CONTROL")
        print("=" * 60)

        # ── Detection (YOLOv11m) ──────────────────────────────
        self.detector = SecurityDetector(
            model_path=det_cfg.get("model_path", "models/yolo11m.pt"),
            device=device,
            high_conf=det_cfg.get("high_conf", 0.40),
            low_conf=det_cfg.get("low_conf", 0.10),
            min_area=det_cfg.get("min_area", 120.0),
            nms_iou=det_cfg.get("nms_iou", 0.60),
            imgsz=det_cfg.get("imgsz", 640),
        )

        # ── Re-ID Encoder (OSNet_x1_0) ────────────────────────
        self.encoder = ReIDEncoder(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            model_path=reid_cfg.get("model_path", "models/osnet_x1_0_msmt17.pth"),
            device=device,
            input_width=reid_cfg.get("input_width", 128),
            input_height=reid_cfg.get("input_height", 256),
        )

        # ── ANPR Engine ───────────────────────────────────────
        self.anpr_enabled = bool(anpr_cfg.get("enabled", True))
        self.anpr_max_vehicles = int(anpr_cfg.get("max_vehicles_per_frame", 3))
        self.anpr_engine = ANPREngine(
            use_gpu=(device == "cuda"),
            min_plate_chars=anpr_cfg.get("min_plate_chars", 5),
            max_plate_chars=anpr_cfg.get("max_plate_chars", 11),
            ocr_confidence_threshold=anpr_cfg.get("ocr_confidence_threshold", 0.20),
            temporal_votes_required=anpr_cfg.get("temporal_votes_required", 2),
            temporal_window_size=anpr_cfg.get("temporal_window_size", 12),
            ocr_interval_sec=anpr_cfg.get("ocr_interval_sec", 0.25),
            min_vehicle_width_px=anpr_cfg.get("min_vehicle_width_px", 60),
            early_accept_confidence=anpr_cfg.get("early_accept_confidence", 0.60),
            aspect_ratio_range=tuple(anpr_cfg.get("plate_aspect_ratio", [2.5, 5.0])),
            plate_format=anpr_cfg.get("plate_format", "IN"),
            require_valid_format=anpr_cfg.get("require_valid_format", False),
            watchlist_max_distance=anpr_cfg.get("watchlist_max_edit_distance", 1.0),
            watchlist_path=anpr_cfg.get("watchlist_path", "data/plate_watchlist.json"),
        ) if self.anpr_enabled else None

        # Background OCR worker: EasyOCR is the slowest stage, so on a GPU box it runs
        # off the main loop and results are applied when ready (video keeps flowing).
        self._anpr_async = bool(anpr_cfg.get("async_ocr", False)) and self.anpr_engine is not None
        self._anpr_jobs: "queue.Queue" = queue.Queue(maxsize=int(anpr_cfg.get("async_queue_size", 8)))
        self._anpr_results: "queue.Queue" = queue.Queue()
        self._anpr_pending: set = set()
        self._anpr_pending_lock = threading.Lock()
        self._ocr_rate = float(anpr_cfg.get("max_ocr_per_sec", 8))
        self._ocr_max_attempts = int(anpr_cfg.get("max_attempts_per_vehicle", 24))
        self._ocr_tokens = self._ocr_rate
        self._ocr_refill_at = time.time()
        self._stop_event = threading.Event()  # created here so the worker can see it
        if self._anpr_async:
            threading.Thread(target=self._anpr_worker, name="anpr-ocr", daemon=True).start()
            print("   ✅ ANPR OCR running on background worker thread")

        # ── Facial Recognition System ─────────────────────────
        self.face_auth = FaceRecognitionEngine(
            backend=face_cfg.get("backend", "auto"),
            detector_model=face_cfg.get("detector_model", "models/face_detection_yunet_2023mar.onnx"),
            recognizer_model=face_cfg.get("recognizer_model", "models/face_recognition_sface_2021dec.onnx"),
            gallery_path=face_cfg.get("gallery_path", "data/face_gallery.json"),
            match_distance_threshold=face_cfg.get("match_distance_threshold", 0.60),
            margin=face_cfg.get("margin", 0.05),
            min_face_px=face_cfg.get("min_face_px", 28),
            detector_score_threshold=face_cfg.get("detector_score_threshold", 0.80),
            check_interval_sec=face_cfg.get("check_interval_sec", 0.5),
            confirm_hits=face_cfg.get("confirm_hits", 2),
            recheck_sec=face_cfg.get("recheck_sec", 30.0),
            device=device,
            enabled=face_cfg.get("enabled", True),
        )

        # ── Global Matcher ────────────────────────────────────
        self.global_matcher = GlobalMatcher(
            registered_threshold=gm_cfg.get("registration_threshold", 0.70),
            gallery_threshold=gm_cfg.get("match_threshold", 0.60),
            margin_gate=gm_cfg.get("margin_threshold", 0.08),
            ema_alpha=gm_cfg.get("ema_alpha", 0.10),
            ema_min_confidence=gm_cfg.get("ema_min_confidence", 0.65),
            same_camera_min_gap=gm_cfg.get("same_camera_min_gap", 1.0),
            diff_camera_default_min=gm_cfg.get("cross_camera_min_gap", 2.0),
        )

        # ── Streams & per-camera components ───────────────────
        self.stream_manager = StreamManager(night_vision=self.config.get("night_vision", {}))
        self.trackers: Dict[str, SingleCameraTracker] = {}
        self.risk_analyzers: Dict[str, RiskAnalyzer] = {}
        self.vehicle_trackers: Dict[str, VehicleTracker] = {}
        self.camera_names: Dict[str, str] = {}

        target_fps = stream_cfg.get("target_fps", 30.0)

        for cam_cfg in cameras:
            self.stream_manager.add_camera(cam_cfg)
            self.camera_names[cam_cfg.camera_id] = cam_cfg.name
            frame_w, frame_h = self._probe_resolution(cam_cfg.source)
            if getattr(cam_cfg, "width", 0) and getattr(cam_cfg, "height", 0) and                     str(cam_cfg.source).strip().isdigit():
                frame_w, frame_h = cam_cfg.width, cam_cfg.height  # webcam opened at requested size
            if int(getattr(cam_cfg, "rotate", 0) or 0) % 180 == 90:
                frame_w, frame_h = frame_h, frame_w
            print(f"  [INIT] {cam_cfg.camera_id} ({cam_cfg.name}): {frame_w}×{frame_h}")

            self.trackers[cam_cfg.camera_id] = SingleCameraTracker(
                camera_id=cam_cfg.camera_id,
                frame_w=frame_w,
                frame_h=frame_h,
                n_confirm=tracker_cfg.get("n_confirm", 3),
                tentative_max_miss=tracker_cfg.get("tentative_max_miss", 3),
                max_miss=tracker_cfg.get("max_miss", 20),
                lost_buffer_sec=tracker_cfg.get("lost_buffer_sec", 15.0),
                gate_dist_px=tracker_cfg.get("gate_dist_px", 200),
                mahal_gate=tracker_cfg.get("mahal_gate", 9.48),
                cost_thr_high=tracker_cfg.get("cost_threshold_high", 0.65),
                cost_thr_low=tracker_cfg.get("cost_threshold_low", 0.70),
                weights_normal=tuple(tracker_cfg.get("weights_normal", [0.50, 0.40, 0.10])),
                weights_crowd=tuple(tracker_cfg.get("weights_crowd", [0.30, 0.45, 0.25])),
                crowd_overlap_min=tracker_cfg.get("crowd_overlap_min", 0.10),
                crowd_score_thr=tracker_cfg.get("crowd_score_thr", 0.08),
                sim_thr_live=tracker_cfg.get("sim_threshold_live", 0.45),
                sim_thr_lost=tracker_cfg.get("sim_threshold_lost", 0.35),
                embedding_pool_k=tracker_cfg.get("embedding_pool_k", 5),
                dup_iou_thr=tracker_cfg.get("dup_iou_thr", 0.50),
                min_export_frames=tracker_cfg.get("min_export_frames", 5),
                quality_threshold=tracker_cfg.get("quality_threshold", 0.60),
            )

            self.risk_analyzers[cam_cfg.camera_id] = RiskAnalyzer(
                frame_width=frame_w,
                frame_height=frame_h,
                fps=target_fps,
                suspicious_enter_speed=risk_cfg.get("high_velocity_px_s", 350.0),
                loiter_seconds=risk_cfg.get("loiter_seconds", 30.0),
                loiter_displacement_px=risk_cfg.get("loiter_displacement_px", 30.0),
            )
            self.vehicle_trackers[cam_cfg.camera_id] = VehicleTracker(
                iou_threshold=anpr_cfg.get("vehicle_iou_threshold", 0.25),
                max_age_sec=anpr_cfg.get("vehicle_max_age_sec", 2.0),
            )

        # ── Alerts ────────────────────────────────────────────
        webhook_cfg = integ_cfg.get("webhook", {}) if isinstance(integ_cfg.get("webhook"), dict) else {}
        self.alert_manager = AlertManager(
            events_log_path=log_cfg.get("events_log", "logs/events.jsonl"),
            webhook_url=webhook_cfg.get("url", ""),
            webhook_min_severity=webhook_cfg.get("min_severity", "WARNING"),
            webhook_headers=webhook_cfg.get("headers"),
            site_id=integ_cfg.get("site_id", ""),
        )
        self.alerts_enabled = bool(runtime_cfg.get("alerts_enabled", True))
        self._behaviour_cooldown = float(alert_cfg.get("behaviour_cooldown_sec", 60.0))
        self._face_cooldown = float(alert_cfg.get("face_match_cooldown_sec", 60.0))
        self._night_cooldown = float(alert_cfg.get("night_movement_cooldown_sec", 30.0))
        self._plate_cooldown = float(alert_cfg.get("plate_watchlist_cooldown_sec", 120.0))

        # ── Runtime tuning ────────────────────────────────────
        self._reid_every_n = runtime_cfg.get("reid_every_n_frames", 1)
        self._cam_frame_idx: Dict[str, int] = {}
        self._event_cooldown_sec = runtime_cfg.get("detection_event_cooldown_sec", 3.0)
        self._global_min_track_age = runtime_cfg.get("global_min_track_age", 4)
        self._global_reid_update_every = runtime_cfg.get("global_reid_update_every", 4)
        self._global_embedding_history = runtime_cfg.get("global_embedding_history", 8)
        self._global_identity_ttl_sec = runtime_cfg.get("global_identity_ttl_sec", 180.0)
        self._global_rematch_frames = runtime_cfg.get("global_rematch_frames", 45)
        self._global_rematch_min_age = runtime_cfg.get("global_rematch_min_age", 8)
        self._global_update_min_sim = runtime_cfg.get("global_update_min_sim", 0.72)
        # Cross-camera hand-off is stricter than same-camera re-entry: a wrong merge
        # (two people sharing an ID) is worse than a person getting a fresh ID.
        self._xcam_threshold = gm_cfg.get("cross_camera_match_threshold", 0.78)
        self._xcam_margin = gm_cfg.get("cross_camera_margin", 0.10)
        self._defer_all_matches = bool(gm_cfg.get("defer_matching", False))

        # ── Virtual fences (drawn in the dashboard, persisted) ──
        self._fences_path = self.config.get("zones", {}).get("fences_path", "data/fences.json")
        # Ignore areas: posters, TV screens, mirrors... detections mostly inside are dropped
        self.ignore_zones: Dict[str, dict] = {}   # cam -> {"points": [...], "polys": [Polygon]}
        self._zone_entered: Dict[tuple, float] = {}     # (cam, kind, track) -> entry time
        self._zone_inside: set = set()                  # (cam, vehicle track) inside zone
        self._veh_crossed: set = set()                  # (cam, vehicle track) crossed tripwire
        self._veh_prev_foot: Dict[str, Dict[int, tuple]] = {}
        self._ignore_min_overlap = float(self.config.get("zones", {}).get("ignore_min_overlap", 0.6))
        self.load_fences()

        # ── Performance ───────────────────────────────────────
        self._metrics_window = float(runtime_cfg.get("metrics_window_sec", 3.0))
        self._loop_times = collections.deque()
        self._cam_times: Dict[str, Any] = {}
        self._cam_latency: Dict[str, float] = {}
        self.last_raw_frames: Dict[str, np.ndarray] = {}
        # Pipelined mode: YOLO for batch N+1 runs on its own thread while the
        # CPU-side tracking / Re-ID / risk / drawing for batch N runs here.
        self._pipelined = bool(runtime_cfg.get("pipeline", True))

        # ── Lifecycle ─────────────────────────────────────────
        self._frame_count = 0
        self._start_time = 0.0
        self._face_labels: Dict[str, Any] = {}

        self._frame_log_enabled = bool(log_cfg.get("frame_log_enabled", True))
        self._frame_log_every_n = max(1, int(log_cfg.get("frame_log_every_n", 1)))
        os.makedirs("logs", exist_ok=True)
        self.log_file_path = log_cfg.get("frame_log", "logs/frame_risk_output.jsonl")
        if self._frame_log_enabled:
            open(self.log_file_path, "w").close()  # fresh per run

    # ══════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _probe_resolution(source) -> tuple:
        """Open the video source briefly to read its actual resolution."""
        try:
            if isinstance(source, str) and "://" in source:
                # Network camera (IP cam / phone): short timeout so an offline device
                # can't stall startup; its stream thread keeps retrying in the background.
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            else:
                cap = cv2.VideoCapture(int(source) if str(source).strip().isdigit() else source)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                cap.release()
                return w, h
        except Exception:
            pass
        return 1920, 1080

    def _log_telemetry(self, telemetry: dict):
        if self._frame_log_enabled and self._frame_count % self._frame_log_every_n == 0:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(telemetry, default=str) + "\n")

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]

    def _is_night(self, cam_id: str) -> bool:
        try:
            return self.stream_manager.is_night(cam_id) is True
        except Exception:
            return False

    def _alert(self, alerts: List[dict], key: Optional[str], cooldown: float, *args, **kwargs) -> None:
        if not self.alerts_enabled:
            return
        if key is not None and not self.alert_manager.allow(key, cooldown):
            return
        alerts.append(self.alert_manager.make_alert(*args, **kwargs))

    def stop(self) -> None:
        self._stop_event.set()

    # ══════════════════════════════════════════════════════
    #  Core pipeline
    # ══════════════════════════════════════════════════════

    def process_frames(self, frame_data: Dict[str, Tuple[np.ndarray, float]],
                       detections: Optional[List] = None
                       ) -> Tuple[dict, Dict[str, np.ndarray]]:
        """Run the full analytics pipeline on one batch of frames (one per camera).

        `detections` may be supplied when YOLO already ran on these frames (pipelined mode).
        """
        self._frame_count += 1
        loop_ts = time.time() - self._start_time if self._start_time else time.time()

        telemetry = {"timestamp": time.time(), "frame": self._frame_count, "alerts": [], "streams": {}}
        cam_ids = list(frame_data.keys())
        frames = [frame_data[cid][0] for cid in cam_ids]
        all_detections = detections if detections is not None else self.detector.batch_detect(frames)
        self.last_raw_frames = dict(zip(cam_ids, frames))  # un-annotated, for display overlays

        annotated_frames: Dict[str, np.ndarray] = {}
        active_face_keys: List[str] = []
        for idx, cam_id in enumerate(cam_ids):
            annotated, stream_payload, face_keys = self._process_camera(
                cam_id, frames[idx], all_detections[idx], loop_ts, telemetry["alerts"])
            annotated_frames[cam_id] = annotated
            telemetry["streams"][cam_id] = stream_payload
            active_face_keys.extend(face_keys)

        self.face_auth.prune_tracks(active_face_keys)
        self.alert_manager.publish(telemetry["alerts"])
        self._record_metrics(frame_data, telemetry)
        return telemetry, annotated_frames

    def _zone_events(self, cam_id, kind, objects, loop_ts, alerts) -> None:
        """Entry / exit events for a restricted zone.

        objects: {track_id: (inside, global_id_or_None, bbox)} for this frame.
        Entry -> WARNING (suspicious until they leave); exit -> NORMAL with time spent.
        """
        for tid, (inside, gid, bbox) in objects.items():
            key = (cam_id, kind, tid)
            name = f"G{gid}" if gid else (f"Vehicle V{tid}" if kind == "vehicle" else f"Person T{tid}")
            if inside and key not in self._zone_entered:
                self._zone_entered[key] = loop_ts
                if kind == "vehicle":
                    self._zone_inside.add((cam_id, tid))
                self._alert(alerts, None, 0, "RESTRICTED_ZONE_ENTRY", cam_id, "WARNING",
                            f"Suspicious: {name} entered restricted zone", object_class=kind,
                            track_id=tid, global_id=gid, bbox=[round(float(v), 1) for v in bbox])
            elif not inside and key in self._zone_entered:
                self._zone_exit(cam_id, kind, tid, gid, loop_ts, alerts)

    def _zone_exit(self, cam_id, kind, tid, gid, loop_ts, alerts) -> None:
        entered = self._zone_entered.pop((cam_id, kind, tid), None)
        if kind == "vehicle":
            self._zone_inside.discard((cam_id, tid))
        if entered is None:
            return
        name = f"G{gid}" if gid else (f"Vehicle V{tid}" if kind == "vehicle" else f"Person T{tid}")
        self._alert(alerts, None, 0, "RESTRICTED_ZONE_EXIT", cam_id, "NORMAL",
                    f"{name} left restricted zone after {loop_ts - entered:.0f}s", object_class=kind,
                    track_id=tid, global_id=gid, dwell_sec=round(loop_ts - entered, 1))

    def _vehicle_fence_checks(self, cam_id, seen, ended, loop_ts, alerts) -> None:
        """Tripwire crossings and restricted-zone presence for vehicles (bottom-centre point)."""
        analyzer = self.risk_analyzers[cam_id]
        prev = self._veh_prev_foot.setdefault(cam_id, {})
        for tr in ended:
            prev.pop(tr.track_id, None)
            self._veh_crossed.discard((cam_id, tr.track_id))
            if (cam_id, "vehicle", tr.track_id) in self._zone_entered:
                self._zone_exit(cam_id, "vehicle", tr.track_id, None, loop_ts, alerts)
        zone_state = {}
        for tr in seen:
            if not tr.confirmed:
                continue
            x1, y1, x2, y2 = tr.bbox
            foot = ((x1 + x2) / 2.0, float(y2))
            if analyzer.crossed_tripwire(prev.get(tr.track_id), foot) and (cam_id, tr.track_id) not in self._veh_crossed:
                self._veh_crossed.add((cam_id, tr.track_id))
                self._alert(alerts, None, 0, "VEHICLE_FENCE_CROSSING", cam_id, "CRITICAL",
                            f"Tripwire crossed by {tr.label} V{tr.track_id}" + (f" ({tr.plate})" if tr.plate else ""),
                            object_class=tr.label, track_id=tr.track_id, plate=tr.plate or None,
                            bbox=[round(float(v), 1) for v in tr.bbox])
            prev[tr.track_id] = foot
            zone_state[tr.track_id] = (analyzer.in_zone(foot), None, tr.bbox)
        self._zone_events(cam_id, "vehicle", zone_state, loop_ts, alerts)

    def load_fences(self) -> None:
        """Restore fences saved from the dashboard (data/fences.json)."""
        try:
            with open(self._fences_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, ValueError):
            return
        for cam_id, fence in saved.items():
            analyzer = self.risk_analyzers.get(cam_id)
            if analyzer is not None:
                analyzer.update_fence([tuple(p) for p in fence.get("points", [])],
                                      fence.get("type", "line"))
            if fence.get("ignore"):
                self.set_ignore_zones(cam_id, fence["ignore"])

    def set_ignore_zones(self, cam_id: str, zones) -> int:
        """Replace a camera's ignore areas. Each zone is a list of >= 3 [x, y] points.
        Returns how many valid zones were kept."""
        from shapely.geometry import Polygon
        pts_list, polys = [], []
        for zone in zones or []:
            pts = [(float(x), float(y)) for x, y in zone]
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 1.0:
                continue
            pts_list.append([[x, y] for x, y in pts]); polys.append(poly)
        if polys:
            self.ignore_zones[cam_id] = {"points": pts_list, "polys": polys}
        else:
            self.ignore_zones.pop(cam_id, None)
        return len(polys)

    def _drop_ignored(self, cam_id: str, dets):
        """Drop detections whose box lies mostly (>= ignore_min_overlap) inside an ignore area.
        A real person standing in front of a poster has a much bigger box and is kept."""
        from shapely.geometry import box as sbox
        polys = self.ignore_zones[cam_id]["polys"]
        kept = []
        for d in dets:
            b = sbox(*d.bbox)
            area = b.area or 1.0
            if any(p.intersection(b).area / area >= self._ignore_min_overlap for p in polys):
                continue
            kept.append(d)
        return kept

    def save_fences(self) -> None:
        data = {cid: {**a.fence_dict(), "ignore": self.ignore_zones.get(cid, {}).get("points", [])}
                for cid, a in self.risk_analyzers.items()}
        os.makedirs(os.path.dirname(self._fences_path) or ".", exist_ok=True)
        tmp = self._fences_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self._fences_path)

    def _refresh_global_ids(self, cam_id, tracker, active_tracks, loop_ts, alerts=None) -> None:
        """Keep cross-camera identities current while people are visible.

        * last-seen time/camera updated every frame (transit-time gate stays correct)
        * appearance prototype refreshed every N frames from the track's pooled
          embedding (many views, not just the first 3 frames)
        * a track that got a NEW id at activation is re-matched for a while with its
          better pooled embedding; if it is clearly a known person, it takes that ID
        """
        if alerts is None:
            alerts = []
        every = max(1, int(self._global_reid_update_every))
        window = int(self._global_rematch_frames)
        for t in active_tracks:
            gid = tracker.local_to_global.get(t.track_id)
            if gid is None:
                continue
            self.global_matcher.mark_seen(gid, cam_id, loop_ts)
            if t.pooled_embedding is None or t.age % every:
                continue
            if getattr(t, "gid_provisional", False):
                if t.age < self._global_rematch_min_age:
                    continue  # pooled embedding not yet reliable
                if t.age > window:
                    t.gid_provisional = False
                else:
                    in_use = {g for g in tracker.local_to_global.values()}
                    res = self.global_matcher.rematch(t.pooled_embedding, cam_id, loop_ts, in_use,
                                                      threshold=self._xcam_threshold,
                                                      margin=self._xcam_margin)
                    if res is not None:
                        from_cam = res.previous_camera
                        tracker.set_global_id(t.track_id, res.global_id, res.display_name)
                        self.global_matcher.remove_identity(gid)
                        t.gid_provisional = False
                        self.global_matcher.record_visit(res.global_id, cam_id, loop_ts)
                        if from_cam and from_cam != cam_id:
                            self._alert(alerts, f"handoff:{res.global_id}:{cam_id}", 10.0,
                                        "PERSON_HANDOFF", cam_id, "NORMAL",
                                        f"G{res.global_id} moved {self.camera_names.get(from_cam, from_cam)}"
                                        f" → {self.camera_names.get(cam_id, cam_id)}",
                                        global_id=res.global_id, track_id=t.track_id,
                                        from_camera=from_cam, to_camera=cam_id,
                                        confidence=round(float(res.confidence), 3))
                        continue
            if t.age >= self._global_min_track_age:
                # Only refresh the stored appearance when it clearly is the same person, so an
                # ID switch or a wrong match can't drag the profile towards someone else.
                ident = self.global_matcher.get_identity(gid)
                if ident is not None and ident.prototype is not None and                         float(np.dot(t.pooled_embedding, ident.prototype)) >= self._global_update_min_sim:
                    self.global_matcher.update_gallery(gid, t.pooled_embedding, cam_id, loop_ts)

    # -- Performance metrics (rolling window; what the dashboard shows) --

    def _record_metrics(self, frame_data, telemetry) -> None:
        now = time.time()
        window = self._metrics_window
        self._loop_times.append(now)
        while self._loop_times and now - self._loop_times[0] > window:
            self._loop_times.popleft()
        for cam_id, (_, captured_at) in frame_data.items():
            q = self._cam_times.setdefault(cam_id, collections.deque())
            q.append(now)
            while q and now - q[0] > window:
                q.popleft()
            lat = self._cam_latency.get(cam_id)
            sample = max(0.0, (now - captured_at) * 1000.0)  # capture -> analysis done
            self._cam_latency[cam_id] = sample if lat is None else 0.8 * lat + 0.2 * sample
        perf = self.performance()
        telemetry["processing_fps"] = perf["processing_fps"]
        telemetry["latency_ms"] = perf["latency_ms"]
        for cam_id, payload in telemetry["streams"].items():
            cam_perf = perf["cameras"].get(cam_id, {})
            payload["analysis_fps"] = cam_perf.get("analysis_fps", 0.0)
            payload["latency_ms"] = cam_perf.get("latency_ms")

    @staticmethod
    def _rate(times) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def performance(self) -> dict:
        """Current analysis speed: loops/s, per-camera analysed FPS and latency (ms)."""
        now = time.time()
        cams = {}
        for cam_id, q in list(self._cam_times.items()):
            fresh = bool(q) and now - q[-1] < 2.0   # camera stalled -> 0 FPS
            lat = self._cam_latency.get(cam_id)
            cams[cam_id] = {"analysis_fps": round(self._rate(q), 1) if fresh else 0.0,
                            "latency_ms": round(lat) if lat is not None and fresh else None}
        lats = [c["latency_ms"] for c in cams.values() if c["latency_ms"] is not None]
        loops_fresh = bool(self._loop_times) and now - self._loop_times[-1] < 2.0
        return {"processing_fps": round(self._rate(self._loop_times), 1) if loops_fresh else 0.0,
                "latency_ms": round(sum(lats) / len(lats)) if lats else None,
                "cameras": cams}

    def _process_camera(self, cam_id: str, frame: np.ndarray, detections, loop_ts: float,
                        alerts: List[dict]) -> Tuple[np.ndarray, dict, List[str]]:
        annotated = frame.copy()
        high_dets, low_dets = detections
        if self.ignore_zones.get(cam_id):
            high_dets = self._drop_ignored(cam_id, high_dets)
            low_dets = self._drop_ignored(cam_id, low_dets)
        night = self._is_night(cam_id)

        person_high = [d for d in high_dets if not d.is_vehicle()]
        person_low = [d for d in low_dets if not d.is_vehicle()]
        vehicle_high = [d for d in high_dets if d.is_vehicle()]

        vehicles_payload = self._process_vehicles(cam_id, frame, annotated, vehicle_high, loop_ts, night, alerts)
        risk_result, face_payload, face_keys = self._process_persons(
            cam_id, frame, annotated, person_high, person_low, loop_ts, night, alerts)

        # Virtual fence + HUD
        analyzer = self.risk_analyzers[cam_id]
        for zone_pts in self.ignore_zones.get(cam_id, {}).get("points", []):
            cv2.polylines(annotated, [np.array(zone_pts, np.int32)], True, (128, 128, 128), 1, cv2.LINE_AA)
        if len(analyzer.virtual_fence_coords) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in analyzer.virtual_fence_coords], np.int32)
            cv2.polylines(annotated, [pts], analyzer.fence_type == "zone", (255, 0, 255), 2, cv2.LINE_AA)
        if night:
            cv2.putText(annotated, "NIGHT MODE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        payload = dict(risk_result)
        payload.update({
            "camera_id": cam_id,
            "camera_name": self.camera_names.get(cam_id, cam_id),
            "night_mode": night,
            "vehicles": vehicles_payload,
            "face_matches": face_payload,
            "counts": {"persons": len(risk_result.get("tracks", [])), "vehicles": len(vehicles_payload)},
        })
        return annotated, payload, face_keys

    # ── Vehicles: tracking, classification, ANPR ─────────────

    def _process_vehicles(self, cam_id, frame, annotated, vehicle_dets, loop_ts, night, alerts) -> List[dict]:
        vtracker = self.vehicle_trackers[cam_id]
        seen, ended = vtracker.update(vehicle_dets, loop_ts)
        for tr in ended:
            if self.anpr_engine is not None:
                self.anpr_engine.clear_track(f"{cam_id}:{tr.track_id}")

        # OCR budget: vehicles tried least so far first, then the largest (most readable);
        # give up on a vehicle whose plate stays unreadable after max_attempts tries.
        ocr_queue = sorted([t for t in seen if t.confirmed and not t.plate
                            and getattr(t, "ocr_attempts", 0) < self._ocr_max_attempts],
                           key=lambda t: (getattr(t, "ocr_attempts", 0),
                                          -(t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])))
        if self.anpr_engine is not None and self.anpr_engine.available:
            if self._anpr_async:
                self._apply_async_plates(cam_id, vtracker, alerts)
                submitted = 0
                self._refill_ocr_tokens()
                for tr in ocr_queue:
                    if submitted >= self.anpr_max_vehicles or self._ocr_tokens < 1:
                        break
                    key = f"{cam_id}:{tr.track_id}"
                    with self._anpr_pending_lock:
                        if key in self._anpr_pending:
                            continue
                        self._anpr_pending.add(key)
                    try:
                        self._anpr_jobs.put_nowait((key, self._crop(frame, tr.bbox).copy(), loop_ts,
                                                    cam_id, tr.track_id, tr.label))
                        submitted += 1
                        self._ocr_tokens -= 1
                        tr.ocr_attempts = getattr(tr, "ocr_attempts", 0) + 1
                    except queue.Full:
                        with self._anpr_pending_lock:
                            self._anpr_pending.discard(key)
                        break
            else:
                for tr in ocr_queue[: self.anpr_max_vehicles]:
                    key = f"{cam_id}:{tr.track_id}"
                    confirmed = self.anpr_engine.extract_plate_with_voting(
                        self._crop(frame, tr.bbox), key, now=loop_ts, camera_id=cam_id,
                        track_id=tr.track_id, vehicle_class=tr.label)
                    if confirmed is not None:
                        self._on_plate_confirmed(cam_id, tr, confirmed, alerts)

        self._vehicle_fence_checks(cam_id, seen, ended, loop_ts, alerts)

        payload = []
        for tr in seen:
            if not tr.confirmed:
                continue
            if night and tr.hits == 2:
                self._alert(alerts, f"night:{cam_id}", self._night_cooldown, "NIGHT_MOVEMENT", cam_id,
                            "WARNING", f"Night-time vehicle movement ({tr.label})",
                            object_class=tr.label, track_id=tr.track_id)
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            vkey = (cam_id, tr.track_id)
            color = (0, 0, 255) if (tr.watchlist_tag in CRITICAL_PLATE_TAGS or vkey in self._veh_crossed) else (
                (0, 60, 255) if vkey in self._zone_inside else
                (0, 215, 255) if tr.watchlist_tag else (255, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"V{tr.track_id} {tr.label.upper()}" + (f" | {tr.plate}" if tr.plate else "")
            if tr.watchlist_tag:
                label += f" | {tr.watchlist_tag}"
            if vkey in self._veh_crossed:
                label += " | CROSSED"
            elif vkey in self._zone_inside:
                label += " | IN ZONE"
            cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            payload.append(tr.to_dict())
        return payload

    def _refill_ocr_tokens(self) -> None:
        """Token bucket: at most anpr.max_ocr_per_sec plate reads per second, all cameras.
        EasyOCR shares the GPU with YOLO; unbounded OCR halves detection FPS in traffic."""
        if self._ocr_rate <= 0:
            self._ocr_tokens = float("inf")
            return
        now = time.time()
        self._ocr_tokens = min(self._ocr_rate, self._ocr_tokens + (now - self._ocr_refill_at) * self._ocr_rate)
        self._ocr_refill_at = now

    def _on_plate_confirmed(self, cam_id, tr, confirmed, alerts) -> None:
        tr.plate = confirmed.text
        tr.watchlist_tag = confirmed.watchlist_tag
        self._alert(alerts, None, 0, "VEHICLE_DETECTED", cam_id, "NORMAL",
                    f"Plate {confirmed.text} · {tr.label}", plate=confirmed.text,
                    vehicle_class=tr.label, track_id=tr.track_id,
                    confidence=round(confirmed.confidence, 3), read_count=confirmed.read_count)
        hit = confirmed.watchlist_hit
        if hit:
            sev = "CRITICAL" if hit.tag in CRITICAL_PLATE_TAGS else "WARNING"
            match_note = "exact" if hit.exact else f"fuzzy match {hit.watch_plate} (d={hit.distance:.1f})"
            self._alert(alerts, f"plate:{hit.watch_plate}:{cam_id}", self._plate_cooldown,
                        "ANPR_WATCHLIST_HIT", cam_id, sev,
                        f"{hit.tag}: vehicle {confirmed.text} ({match_note})",
                        plate=confirmed.text, watchlist=hit.to_dict(), vehicle_class=tr.label,
                        track_id=tr.track_id, bbox=[round(float(v), 1) for v in tr.bbox])

    def _anpr_worker(self) -> None:
        """Background thread: runs multi-pass OCR + voting for queued vehicle crops."""
        while not self._stop_event.is_set():
            try:
                key, crop, ts, cam_id, track_id, label = self._anpr_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                confirmed = self.anpr_engine.extract_plate_with_voting(
                    crop, key, now=ts, camera_id=cam_id, track_id=track_id, vehicle_class=label)
                if confirmed is not None:
                    self._anpr_results.put((cam_id, track_id, confirmed))
            except Exception as e:  # never let one bad crop kill the worker
                print(f"  [ANPR] worker error: {e}")
            finally:
                with self._anpr_pending_lock:
                    self._anpr_pending.discard(key)

    def _apply_async_plates(self, cam_id, vtracker, alerts) -> None:
        """Apply confirmed plates finished by the worker for this camera."""
        pending_other = []
        while True:
            try:
                item = self._anpr_results.get_nowait()
            except queue.Empty:
                break
            if item[0] != cam_id:
                pending_other.append(item)
                continue
            _, track_id, confirmed = item
            tr = vtracker.tracks.get(track_id)
            if tr is not None and not tr.plate:
                self._on_plate_confirmed(cam_id, tr, confirmed, alerts)
        for item in pending_other:
            self._anpr_results.put(item)

    # ── Persons: tracking, Re-ID, FRS, behaviour ──────────────

    def _process_persons(self, cam_id, frame, annotated, person_high, person_low, loop_ts, night, alerts):
        crops, crop_idx = [], []
        tracker = self.trackers[cam_id]
        n = self._cam_frame_idx[cam_id] = self._cam_frame_idx.get(cam_id, 0) + 1
        full_reid = self._reid_every_n <= 1 or n % self._reid_every_n == 0
        track_boxes = [] if full_reid else [t.bbox for t in tracker.tracks.values()]
        for i, det in enumerate(person_high):
            if not full_reid:
                # Re-ID is the costliest per-person step. Between full passes, embed only
                # people who NEED appearance: new ones (no track overlaps) every frame,
                # crowded ones (2+ overlapping tracks) every other frame.
                overlaps = sum(1 for b in track_boxes if bbox_iou(b, det.bbox) > 0.3)
                if overlaps == 1 or (overlaps >= 2 and n % 2):
                    continue
            crop = self._crop(frame, det.bbox)
            if crop.size > 0:
                crops.append(crop)
                crop_idx.append(i)

        embeddings = {}
        if crops:
            embs = self.encoder.batch_extract(crops)
            if embs is not None:
                for j, orig in enumerate(crop_idx):
                    embeddings[orig] = embs[j]

        events = tracker.update(person_high, person_low, embeddings, loop_ts)
        for ev in events:
            if ev.type == "TRACK_ACTIVATED":
                if ev.embedding is not None:
                    # Never hand out an ID that another person in this camera already holds
                    in_use = {g for t, g in tracker.local_to_global.items() if t != ev.track_id}
                    # From 3 frames, only same-camera re-entries are reliable; a match to a
                    # person last seen on ANOTHER camera is decided later, by _refresh_global_ids,
                    # once the track has a pooled embedding from many views.
                    if self._defer_all_matches:
                        # Decide every re-identification from the pooled embedding (many views):
                        # in crowds a 3-frame embedding causes ID theft between neighbours.
                        other_cam = {i.global_id for i in self.global_matcher.get_all_identities()
                                     if not i.is_registered}
                    else:
                        other_cam = {i.global_id for i in self.global_matcher.get_all_identities()
                                     if i.last_seen_camera and i.last_seen_camera != cam_id
                                     and not i.is_registered}
                    match = self.global_matcher.match_phase1(ev.embedding, cam_id, loop_ts,
                                                             exclude_gids=in_use | other_cam)
                    tracker.set_global_id(ev.track_id, match.global_id, match.display_name)
                    tr = tracker.tracks.get(ev.track_id)
                    if tr is not None:
                        tr.gid_provisional = match.is_new  # may be corrected with better views
                if night:
                    self._alert(alerts, f"night:{cam_id}", self._night_cooldown, "NIGHT_MOVEMENT", cam_id,
                                "WARNING", "Night-time human movement detected", object_class="person",
                                track_id=ev.track_id)

        active_tracks = tracker.get_active_tracks()
        self._refresh_global_ids(cam_id, tracker, active_tracks, loop_ts, alerts)

        # Facial recognition on each tracked person (engine throttles per track)
        face_payload, face_keys = [], []
        for t in active_tracks:
            key = f"{cam_id}:{t.track_id}"
            face_keys.append(key)
            match, newly = self.face_auth.recognize_track(self._crop(frame, t.bbox), key, loop_ts)
            if match is None:
                continue
            gid = tracker.local_to_global.get(t.track_id)
            entry = {**match.to_dict(), "track_id": t.track_id, "global_id": gid}
            face_payload.append(entry)
            self._face_labels[key] = match
            if newly:
                if match.category in THREAT_CATEGORIES:
                    a_type, sev = "FACE_MATCH_SUSPECT", "CRITICAL"
                    msg = f"CRITICAL_THREAT: FACE_MATCH_SUSPECT — {match.name} ({match.category})"
                elif match.category == "AUTHORIZED":
                    a_type, sev, msg = "FACE_MATCH_AUTHORIZED", "NORMAL", f"Authorised: {match.name}"
                else:
                    a_type, sev, msg = "FACE_MATCH_WATCHLIST", "WARNING", f"{match.category}: {match.name}"
                self._alert(alerts, f"face:{match.identity_id}:{cam_id}", self._face_cooldown, a_type, cam_id,
                            sev, msg, threat=match.category, identity=entry, track_id=t.track_id,
                            global_id=gid, distance=round(match.distance, 4))

        # Behaviour risk
        analyzer = self.risk_analyzers[cam_id]
        risk_inputs = [{"id": t.track_id, "bbox": [float(v) for v in t.bbox]} for t in active_tracks]
        risk_result = analyzer.update(frame_idx=self._frame_count, tracks=risk_inputs, timestamp=loop_ts)

        self._zone_events(cam_id, "person", {
            int(it["id"]): (it.get("in_zone", False), tracker.local_to_global.get(int(it["id"])), it["bbox"])
            for it in risk_result["tracks"]}, loop_ts, alerts)
        for tid in risk_result.get("exits", []):        # left the camera while inside the zone
            if (cam_id, "person", int(tid)) in self._zone_entered:
                self._zone_exit(cam_id, "person", int(tid), tracker.local_to_global.get(int(tid)), loop_ts, alerts)
        for item in risk_result["tracks"]:
            tid = item["id"]
            gid = tracker.local_to_global.get(tid, tid)
            item["global_id"] = gid
            label = item["label"]
            if label in BEHAVIOUR_ALERTS:
                sev, title = BEHAVIOUR_ALERTS[label]
                detail = f" for {item.get('dwell_sec', 0):.0f}s" if label == "LOITERING" else ""
                self._alert(alerts, f"beh:{cam_id}:{tid}:{label}", self._behaviour_cooldown, label, cam_id, sev,
                            f"{title} — G{gid}{detail}", track_id=tid, global_id=gid, bbox=item["bbox"],
                            score=round(item["confidence"] * 100))

            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            risk_score = item["confidence"] * 100
            face = self._face_labels.get(f"{cam_id}:{tid}")
            color = (0, 255, 0)
            if label == "INTRUSION" or (face is not None and face.is_threat):
                color = (0, 0, 255)
            elif label == "RESTRICTED_ZONE":
                color = (0, 60, 255)   # red-orange: inside a restricted zone
            elif label in ("LOITERING", "SUSPICIOUS_MOTION") or risk_score > 70:
                color = (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"G{gid} | R:{risk_score:.0f} | {label}"
            if label == "RESTRICTED_ZONE":
                text = f"G{gid} | IN RESTRICTED ZONE {item.get('zone_dwell_sec', 0):.0f}s"
            cv2.putText(annotated, text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if face is not None:
                cv2.putText(annotated, f"FACE: {face.name} [{face.category}]", (x1, min(annotated.shape[0] - 5, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if face.is_threat else (0, 255, 0), 2)

        live = {f"{cam_id}:{t.track_id}" for t in active_tracks}
        for k in [k for k in self._face_labels if k.startswith(f"{cam_id}:") and k not in live]:
            del self._face_labels[k]
        return risk_result, face_payload, face_keys

    # ══════════════════════════════════════════════════════
    #  Run loop
    # ══════════════════════════════════════════════════════

    def _on_frames(self, annotated_frames: Dict[str, np.ndarray]) -> bool:
        """Hook for subclasses. Return False to stop the loop."""
        if not self.display:
            return True
        frames = list(annotated_frames.values())
        if len(frames) == 1:
            cv2.imshow("SHIELD C&C Monitor", frames[0])
        elif len(frames) > 1:
            cv2.imshow("SHIELD COMMAND CENTER", np.hstack([cv2.resize(f, (640, 360)) for f in frames]))
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), ord('Q'), 27)

    def run(self) -> None:
        self.stream_manager.start_all()
        self._start_time = time.time()
        print("\n🎬 SHIELD Active..." + (" Press Q to quit.\n" if self.display else "\n"))

        try:
            batches = self._start_detection_thread() if self._pipelined else None
            while not self._stop_event.is_set():
                if batches is not None:
                    try:
                        item = batches.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:          # detection thread ended: all streams finished
                        break
                    frame_data, detections = item
                else:
                    frame_data, detections = self.stream_manager.grab_latest(), None
                    if not frame_data:
                        if self.stream_manager.all_stopped():
                            break
                        time.sleep(0.01)
                        continue

                telemetry, annotated_frames = self.process_frames(frame_data, detections)
                self._log_telemetry(telemetry)
                if self.telemetry_callback:
                    self.telemetry_callback(telemetry)
                if not self._on_frames(annotated_frames):
                    break
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        finally:
            self._stop_event.set()
            self.stream_manager.stop_all()
            if self.display:
                cv2.destroyAllWindows()

    def _start_detection_thread(self):
        """YOLO producer: grabs the newest frames and detects while the main thread is
        still processing the previous batch. Queue size 1 keeps latency bounded."""
        batches = queue.Queue(maxsize=1)

        def produce():
            try:
                while not self._stop_event.is_set():
                    frame_data = self.stream_manager.grab_latest()
                    if not frame_data:
                        if self.stream_manager.all_stopped():
                            break
                        self.stream_manager.wait_for_frame(0.05)
                        continue
                    cam_ids = list(frame_data.keys())
                    dets = self.detector.batch_detect([frame_data[c][0] for c in cam_ids])
                    while not self._stop_event.is_set():
                        try:
                            batches.put((frame_data, dets), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                print(f"  [DETECT] detection thread error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    batches.put(None, timeout=1.0)
                except queue.Full:
                    pass

        threading.Thread(target=produce, name="yolo-detect", daemon=True).start()
        return batches
