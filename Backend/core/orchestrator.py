"""
SHIELD Master Orchestrator
--------------------------
Core pipeline integrating Detection, Re-ID, Face Auth, ANPR, and Behavioral Risk.
Streams telemetry via callbacks and writes persistent logs to JSONL for future cloud ingestion.

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
  - Full config.yaml integration for all component parameters
"""
from __future__ import annotations
import os
import time
import json
from typing import Any, Dict, List, Optional, Callable

import cv2
import numpy as np

# Core SHIELD Modules
from core.detector import SecurityDetector
from core.reid_encoder import ReIDEncoder
from core.tracker import SingleCameraTracker
from core.stream_manager import StreamManager, CameraConfig
from core.global_matcher import GlobalMatcher
from core.behavior_risk import RiskAnalyzer
from core.anpr import ANPREngine
from core.face_auth import FaceAuthEngine

class Orchestrator:
    def __init__(self,
                 cameras: List[CameraConfig],
                 device: str = "cuda",
                 display: bool = True,
                 telemetry_callback: Optional[Callable[[Dict], None]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the SHIELD orchestrator.

        Args:
            cameras:             List of camera configurations.
            device:              Compute device ("cuda" or "cpu").
            display:             Whether to show OpenCV windows.
            telemetry_callback:  Optional callback for each telemetry frame.
            config:              Master configuration dict (from config.yaml).
                                 If None, all components use their built-in defaults.
        """
        self.cameras = cameras
        self.display = display
        self.device = device
        self.telemetry_callback = telemetry_callback
        self.config = config or {}

        # ── Extract sub-configs ───────────────────────────────
        det_cfg = self.config.get("detection", {})
        reid_cfg = self.config.get("reid", {})
        tracker_cfg = self.config.get("tracker", {})
        gm_cfg = self.config.get("global_matcher", {})
        risk_cfg = self.config.get("risk", {})
        stream_cfg = self.config.get("stream", {})
        runtime_cfg = self.config.get("runtime", {})
        zone_cfg = self.config.get("zones", {})

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
        self.anpr_engine = ANPREngine(use_gpu=(device == "cuda"))

        # ── Face Auth Engine ──────────────────────────────────
        self.face_auth = FaceAuthEngine()

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

        # ── Stream Manager & Per-Camera Components ────────────
        self.stream_manager = StreamManager()
        self.trackers: Dict[str, SingleCameraTracker] = {}
        self.risk_analyzers: Dict[str, RiskAnalyzer] = {}

        target_fps = stream_cfg.get("target_fps", 30.0)

        for cam_cfg in cameras:
            self.stream_manager.add_camera(cam_cfg)

            # Probe actual video resolution instead of assuming 1920×1080
            frame_w, frame_h = self._probe_resolution(cam_cfg.source)
            print(f"  [INIT] {cam_cfg.camera_id} ({cam_cfg.name}): {frame_w}×{frame_h}")

            # ── Single-Camera Tracker (fully configured) ──────
            self.trackers[cam_cfg.camera_id] = SingleCameraTracker(
                camera_id=cam_cfg.camera_id,
                frame_w=frame_w,
                frame_h=frame_h,
                # Lifecycle
                n_confirm=tracker_cfg.get("n_confirm", 3),
                tentative_max_miss=tracker_cfg.get("tentative_max_miss", 3),
                max_miss=tracker_cfg.get("max_miss", 20),
                lost_buffer_sec=tracker_cfg.get("lost_buffer_sec", 15.0),
                # Cost matrix gating
                gate_dist_px=tracker_cfg.get("gate_dist_px", 200),
                mahal_gate=tracker_cfg.get("mahal_gate", 9.48),
                cost_thr_high=tracker_cfg.get("cost_threshold_high", 0.65),
                cost_thr_low=tracker_cfg.get("cost_threshold_low", 0.70),
                # Adaptive weights
                weights_normal=tuple(tracker_cfg.get("weights_normal", [0.50, 0.40, 0.10])),
                weights_crowd=tuple(tracker_cfg.get("weights_crowd", [0.30, 0.45, 0.25])),
                crowd_overlap_min=tracker_cfg.get("crowd_overlap_min", 0.10),
                crowd_score_thr=tracker_cfg.get("crowd_score_thr", 0.08),
                # Appearance
                sim_thr_live=tracker_cfg.get("sim_threshold_live", 0.45),
                sim_thr_lost=tracker_cfg.get("sim_threshold_lost", 0.35),
                embedding_pool_k=tracker_cfg.get("embedding_pool_k", 5),
                # Duplicate guard
                dup_iou_thr=tracker_cfg.get("dup_iou_thr", 0.50),
                # Quality / export
                min_export_frames=tracker_cfg.get("min_export_frames", 5),
                quality_threshold=tracker_cfg.get("quality_threshold", 0.60),
            )

            # ── Risk Analyzer (fully configured) ──────────────
            self.risk_analyzers[cam_cfg.camera_id] = RiskAnalyzer(
                frame_width=frame_w,
                frame_height=frame_h,
                fps=target_fps,
            )

        # ── Runtime Tuning ────────────────────────────────────
        self._reid_every_n = runtime_cfg.get("reid_every_n_frames", 1)
        self._event_cooldown_sec = runtime_cfg.get("detection_event_cooldown_sec", 3.0)
        self._global_min_track_age = runtime_cfg.get("global_min_track_age", 4)
        self._global_reid_update_every = runtime_cfg.get("global_reid_update_every", 4)
        self._global_embedding_history = runtime_cfg.get("global_embedding_history", 8)
        self._global_identity_ttl_sec = runtime_cfg.get("global_identity_ttl_sec", 180.0)

        # ── Lifecycle ─────────────────────────────────────────
        self._frame_count = 0
        self._start_time = 0.0
        self.plate_memory = set()
        self.alerted_global_ids = set()
        
        # Setup Logging Directory
        os.makedirs("logs", exist_ok=True)
        self.log_file_path = "logs/frame_risk_output.jsonl"
        # Clear previous log on new run
        open(self.log_file_path, 'w').close()

    @staticmethod
    def _probe_resolution(source) -> tuple:
        """Open the video source briefly to read its actual resolution."""
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                cap.release()
                return w, h
        except Exception:
            pass
        return 1920, 1080  # Fallback

    def _log_telemetry(self, telemetry: dict):
        """Appends JSON line to log file for AWS/MongoDB sync pipeline."""
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(telemetry) + "\n")

    def run(self) -> None:
        self.stream_manager.start_all()
        self._start_time = time.time()

        print("\n🎬 SHIELD Active... Press Q to quit.\n")

        try:
            while True:
                frame_data = self.stream_manager.grab_latest()
                if not frame_data:
                    if self.stream_manager.all_stopped():
                        break
                    time.sleep(0.01)
                    continue

                self._frame_count += 1
                loop_ts = time.time() - self._start_time
                
                telemetry = {
                    "timestamp": time.time(),
                    "frame": self._frame_count,
                    "alerts": [],
                    "streams": {}
                }

                cam_ids_ordered = list(frame_data.keys())
                frames_list = [frame_data[cid][0] for cid in cam_ids_ordered]

                all_detections = self.detector.batch_detect(frames_list)
                annotated_frames = {}

                for cam_idx, cam_id in enumerate(cam_ids_ordered):
                    frame = frames_list[cam_idx]
                    fh, fw = frame.shape[:2]
                    annotated = frame.copy()
                    
                    high_dets, low_dets = all_detections[cam_idx]

                    person_high = [d for d in high_dets if not d.is_vehicle()]
                    person_low = [d for d in low_dets if not d.is_vehicle()]
                    vehicle_high = [d for d in high_dets if d.is_vehicle()]

                    # --- ANPR PROCESSING ---
                    for v_det in vehicle_high:
                        x1, y1, x2, y2 = [int(v) for v in v_det.bbox]
                        crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                        
                        plate_text, conf = self.anpr_engine.extract_plate(crop)
                        
                        if plate_text and plate_text not in self.plate_memory:
                            self.plate_memory.add(plate_text)
                            telemetry["alerts"].append({
                                "type": "VEHICLE_DETECTED",
                                "plate": plate_text,
                                "camera": cam_id
                            })
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        display_text = f"PLATE: {plate_text}" if plate_text else "VEHICLE"
                        cv2.putText(annotated, display_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # --- HUMAN TRACKING & RE-ID ---
                    person_crops = []
                    crop_map = []
                    for i, det in enumerate(person_high):
                        x1, y1, x2, y2 = [int(v) for v in det.bbox]
                        crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                        if crop.size > 0:
                            person_crops.append(crop)
                            crop_map.append((i, crop))

                    embeddings = {}
                    if person_crops:
                        embs = self.encoder.batch_extract(person_crops)
                        if embs is not None:
                            for j, (orig_idx, _) in enumerate(crop_map):
                                embeddings[orig_idx] = embs[j]

                    tracker = self.trackers[cam_id]
                    events = tracker.update(person_high, person_low, embeddings, loop_ts)

                    for ev in events:
                        if ev.type == "TRACK_ACTIVATED" and ev.embedding is not None:
                            match = self.global_matcher.match_phase1(ev.embedding, cam_id, loop_ts)
                            tracker.set_global_id(ev.track_id, match.global_id, match.display_name)
                            
                            for _, crop in crop_map:
                                threat = self.face_auth.verify_face(crop, match.global_id)
                                if threat:
                                    telemetry["alerts"].append({"type": "FACE_MATCH", "threat": threat, "camera": cam_id})

                    # --- BEHAVIOR RISK ---
                    risk_analyzer = self.risk_analyzers[cam_id]
                    active_tracks = tracker.get_active_tracks()
                    risk_inputs = [{"id": t.track_id, "bbox": [float(v) for v in t.bbox]} for t in active_tracks]
                    risk_result = risk_analyzer.update(frame_idx=self._frame_count, tracks=risk_inputs)

                    for item in risk_result["tracks"]:
                        tid = item["id"]
                        gid = tracker.local_to_global.get(tid, tid)
                        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
                        risk_score = item["confidence"] * 100
                        
                        color = (0, 255, 0)
                        if item["label"] == "INTRUSION":
                            color = (0, 0, 255) 
                        elif risk_score > 70:
                            color = (0, 165, 255) 

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"G{gid} | R:{risk_score:.0f} | {item['label']}"
                        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    fence_coords = risk_analyzer.virtual_fence_coords
                    cv2.line(annotated, (int(fence_coords[0][0]), int(fence_coords[0][1])), 
                             (int(fence_coords[1][0]), int(fence_coords[1][1])), (255, 0, 255), 2)

                    annotated_frames[cam_id] = annotated
                    telemetry["streams"][cam_id] = risk_result

                # Logging & Broadcast
                self._log_telemetry(telemetry)
                if self.telemetry_callback:
                    self.telemetry_callback(telemetry)

                # Grid Rendering
                if self.display:
                    frames_list = list(annotated_frames.values())
                    if len(frames_list) == 1:
                        cv2.imshow("SHIELD C&C Monitor", frames_list[0])
                    elif len(frames_list) > 1:
                        resized = [cv2.resize(f, (640, 360)) for f in frames_list]
                        grid = np.hstack(resized)
                        cv2.imshow("SHIELD COMMAND CENTER", grid)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break

        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        finally:
            self.stream_manager.stop_all()
            cv2.destroyAllWindows()
