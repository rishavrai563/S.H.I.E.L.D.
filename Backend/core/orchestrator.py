"""
SHIELD Master Orchestrator
--------------------------
Core pipeline integrating Detection, Re-ID, Face Auth, ANPR, and Behavioral Risk.
Streams telemetry via callbacks and writes persistent logs to JSONL for future cloud ingestion.
"""
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

try:
    import pywhatkit
except ImportError:
    pywhatkit = None

class Orchestrator:
    def __init__(self,
                 cameras: List[CameraConfig],
                 device: str = "cuda",
                 display: bool = True,
                 telemetry_callback: Optional[Callable[[Dict], None]] = None):
        
        self.cameras = cameras
        self.display = display
        self.device = device
        self.telemetry_callback = telemetry_callback

        print("\n" + "=" * 60)
        print("  SHIELD: TACTICAL COMMAND & CONTROL")
        print("=" * 60)

        self.detector = SecurityDetector(model_path="models/yolo11m.pt", device=device)
        self.encoder = ReIDEncoder(model_path="models/osnet_x1_0_msmt17.pth", device=device)
        self.anpr_engine = ANPREngine(use_gpu=(device == "cuda"))
        self.face_auth = FaceAuthEngine()

        self.global_matcher = GlobalMatcher()
        self.stream_manager = StreamManager()
        self.trackers: Dict[str, SingleCameraTracker] = {}
        self.risk_analyzers: Dict[str, RiskAnalyzer] = {}

        for cam_cfg in cameras:
            self.stream_manager.add_camera(cam_cfg)
            self.trackers[cam_cfg.camera_id] = SingleCameraTracker(
                camera_id=cam_cfg.camera_id, frame_w=1920, frame_h=1080
            )
            self.risk_analyzers[cam_cfg.camera_id] = RiskAnalyzer(
                frame_width=1920, frame_height=1080
            )

        self._frame_count = 0
        self._start_time = 0.0
        self.plate_memory = set()
        self.alerted_global_ids = set()
        
        # Setup Logging Directory
        os.makedirs("logs", exist_ok=True)
        self.log_file_path = "logs/frame_risk_output.jsonl"
        # Clear previous log on new run
        open(self.log_file_path, 'w').close()

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
                                    telemetry["alerts"].append({"type": "FACE_MATCH", "threat": threat})

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