"""
SHIELD Command & Control API
------------------------------
Production-grade FastAPI server integrating the full ML pipeline:
  - Real-time YOLO + OSNet + ANPR + Behavior Risk processing
  - WebSocket telemetry broadcast to React dashboards
  - REST endpoints for system status, identities, alerts, and camera control
  - MJPEG frame streaming per camera via WebSocket

Architecture:
  The Orchestrator runs synchronously in a background thread. Telemetry is
  pushed via a thread-safe queue.Queue, bridged into asyncio.Queue for
  non-blocking WebSocket broadcast. Annotated frames are stored in a
  thread-safe buffer for snapshot and streaming endpoints.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Core SHIELD Modules
from core.orchestrator import Orchestrator
from core.stream_manager import StreamManager, CameraConfig
from core.global_matcher import GlobalMatcher
from core.reid_encoder import ReIDEncoder


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION LOADER
# ═══════════════════════════════════════════════════════════

def load_config() -> dict:
    """Load the master configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_camera_configs() -> List[CameraConfig]:
    """Load camera configurations from cameras.yaml."""
    cam_path = os.path.join(os.path.dirname(__file__), "config", "cameras.yaml")
    with open(cam_path, "r") as f:
        cam_data = yaml.safe_load(f)

    configs = []
    for cam in cam_data.get("cameras", []):
        configs.append(
            CameraConfig(
                camera_id=cam["camera_id"],
                source=cam["source"],
                name=cam.get("name", "Unknown Node"),
                gps_lat=cam.get("gps_lat", 0.0),
                gps_lng=cam.get("gps_lng", 0.0),
            )
        )
    return configs


# ═══════════════════════════════════════════════════════════
#  SHARED STATE (populated at startup)
# ═══════════════════════════════════════════════════════════

class ShieldState:
    """Thread-safe shared state between the ML thread and the async API."""

    def __init__(self):
        # Pipeline components (set during startup)
        self.orchestrator: Optional[Orchestrator] = None
        self.camera_configs: List[CameraConfig] = []
        self.config: dict = {}

        # Thread-safe telemetry bridge
        self.telemetry_queue: queue.Queue = queue.Queue(maxsize=100)
        self.async_telemetry_queue: Optional[asyncio.Queue] = None

        # Recent alerts buffer (thread-safe via deque's atomic append/pop)
        self.recent_alerts: Deque[dict] = deque(maxlen=1000)

        # Latest telemetry snapshot per camera (for REST endpoints)
        self._latest_telemetry: dict = {}
        self._telemetry_lock = threading.Lock()

        # Annotated frame buffer: camera_id → (jpeg_bytes, timestamp)
        self._frame_buffer: Dict[str, Tuple[bytes, float]] = {}
        self._frame_lock = threading.Lock()

        # Lifecycle
        self.start_time: float = 0.0
        self.frame_count: int = 0
        self.ml_thread: Optional[threading.Thread] = None
        self.running: bool = False

    def push_telemetry(self, telemetry: dict):
        """Called from the ML thread — pushes telemetry into the sync queue."""
        # Store latest snapshot
        with self._telemetry_lock:
            self._latest_telemetry = telemetry
            self.frame_count = telemetry.get("frame", self.frame_count)

        # Extract and store alerts
        for alert in telemetry.get("alerts", []):
            alert["timestamp"] = telemetry.get("timestamp", time.time())
            self.recent_alerts.append(alert)

        # Push to async bridge (drop if full — non-blocking)
        try:
            self.telemetry_queue.put_nowait(telemetry)
        except queue.Full:
            pass  # Drop oldest-unread telemetry to prevent backpressure

    def push_annotated_frame(self, camera_id: str, frame: np.ndarray):
        """Store the latest annotated frame for a camera (called from ML thread)."""
        jpeg_quality = self.config.get("stream", {}).get("jpeg_quality", 70)
        _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        with self._frame_lock:
            self._frame_buffer[camera_id] = (jpeg_buf.tobytes(), time.time())

    def get_latest_frame(self, camera_id: str) -> Optional[Tuple[bytes, float]]:
        """Get the latest JPEG frame for a camera."""
        with self._frame_lock:
            return self._frame_buffer.get(camera_id)

    def get_latest_telemetry(self) -> dict:
        """Get the latest telemetry snapshot."""
        with self._telemetry_lock:
            return self._latest_telemetry.copy()


state = ShieldState()


# ═══════════════════════════════════════════════════════════
#  ENHANCED ORCHESTRATOR (adds frame buffer support)
# ═══════════════════════════════════════════════════════════

class APIOrchestrator(Orchestrator):
    """
    Extends Orchestrator to push annotated frames to the API state buffer.
    Overrides run() to operate headlessly and push frames to the API layer.
    """

    def __init__(self, cameras, device, shared_state: ShieldState):
        super().__init__(
            cameras=cameras,
            device=device,
            display=False,  # Always headless in API mode
            telemetry_callback=shared_state.push_telemetry,
            config=shared_state.config,
        )
        self._shared_state = shared_state
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the orchestrator to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Overridden run loop that pushes annotated frames to the state buffer."""
        self.stream_manager.start_all()
        self._start_time = time.time()

        print("\n🎬 SHIELD API Orchestrator Active...\n")

        try:
            while not self._stop_event.is_set():
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
                    "streams": {},
                }

                cam_ids_ordered = list(frame_data.keys())
                frames_list = [frame_data[cid][0] for cid in cam_ids_ordered]

                all_detections = self.detector.batch_detect(frames_list)

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
                                "camera": cam_id,
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
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    fence_coords = risk_analyzer.virtual_fence_coords
                    cv2.line(annotated,
                             (int(fence_coords[0][0]), int(fence_coords[0][1])),
                             (int(fence_coords[1][0]), int(fence_coords[1][1])),
                             (255, 0, 255), 2)

                    # Push annotated frame to API buffer
                    self._shared_state.push_annotated_frame(cam_id, annotated)

                    telemetry["streams"][cam_id] = risk_result

                # Logging & Broadcast
                self._log_telemetry(telemetry)
                if self.telemetry_callback:
                    self.telemetry_callback(telemetry)

        except Exception as e:
            print(f"\n[CRITICAL] Orchestrator error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stream_manager.stop_all()


# ═══════════════════════════════════════════════════════════
#  ML BACKGROUND THREAD
# ═══════════════════════════════════════════════════════════

def ml_worker(shared_state: ShieldState):
    """Runs the full ML pipeline in a background thread."""
    try:
        config = shared_state.config
        device = config.get("system", {}).get("device", "cpu")

        orchestrator = APIOrchestrator(
            cameras=shared_state.camera_configs,
            device=device,
            shared_state=shared_state,
        )
        shared_state.orchestrator = orchestrator
        shared_state.start_time = time.time()

        print("\n🚀 SHIELD ML Pipeline starting in background thread...")
        orchestrator.run()

    except Exception as e:
        print(f"\n[CRITICAL] ML Worker crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shared_state.running = False
        print("🛑 ML Pipeline stopped.")


# ═══════════════════════════════════════════════════════════
#  ASYNC TELEMETRY BRIDGE
# ═══════════════════════════════════════════════════════════

async def telemetry_bridge(shared_state: ShieldState):
    """
    Transfers telemetry from the thread-safe queue.Queue to the
    asyncio.Queue for non-blocking WebSocket broadcast.
    """
    loop = asyncio.get_event_loop()
    while shared_state.running:
        try:
            # Non-blocking check of the sync queue
            telemetry = await loop.run_in_executor(
                None, lambda: shared_state.telemetry_queue.get(timeout=0.5)
            )
            if shared_state.async_telemetry_queue:
                try:
                    shared_state.async_telemetry_queue.put_nowait(telemetry)
                except asyncio.QueueFull:
                    pass
        except queue.Empty:
            continue
        except Exception:
            break


async def telemetry_broadcaster(shared_state: ShieldState):
    """Reads from the async queue and broadcasts to all WebSocket clients."""
    while shared_state.running:
        try:
            if shared_state.async_telemetry_queue is None:
                await asyncio.sleep(0.1)
                continue
            telemetry = await asyncio.wait_for(
                shared_state.async_telemetry_queue.get(), timeout=1.0
            )
            await ws_manager.broadcast_telemetry(telemetry)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


# ═══════════════════════════════════════════════════════════
#  WEBSOCKET CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════

class WebSocketManager:
    """Manages WebSocket connections for telemetry and MJPEG streaming."""

    def __init__(self):
        self.telemetry_clients: List[WebSocket] = []
        self.stream_clients: Dict[str, List[WebSocket]] = {}  # camera_id → [ws, ...]
        self._lock = asyncio.Lock()

    async def connect_telemetry(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.telemetry_clients.append(websocket)
        print(f"🔗 [WebSocket] Telemetry client connected ({len(self.telemetry_clients)} active)")

    async def disconnect_telemetry(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.telemetry_clients:
                self.telemetry_clients.remove(websocket)
        print(f"❌ [WebSocket] Telemetry client disconnected")

    async def connect_stream(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        async with self._lock:
            if camera_id not in self.stream_clients:
                self.stream_clients[camera_id] = []
            self.stream_clients[camera_id].append(websocket)
        print(f"🔗 [WebSocket] Stream client connected for {camera_id}")

    async def disconnect_stream(self, websocket: WebSocket, camera_id: str):
        async with self._lock:
            if camera_id in self.stream_clients:
                if websocket in self.stream_clients[camera_id]:
                    self.stream_clients[camera_id].remove(websocket)

    async def broadcast_telemetry(self, message: dict):
        async with self._lock:
            dead = []
            for ws in self.telemetry_clients:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.telemetry_clients.remove(ws)


ws_manager = WebSocketManager()


# ═══════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the SHIELD API."""
    # --- Startup ---
    print("\n" + "=" * 60)
    print("  SHIELD API SERVER — INITIALIZING")
    print("=" * 60)

    # Load configuration (use pre-set config from start_api_server if available)
    if not state.config:
        state.config = load_config()
    state.camera_configs = load_camera_configs()
    state.async_telemetry_queue = asyncio.Queue(maxsize=50)
    state.running = True

    # Start ML pipeline in background thread
    state.ml_thread = threading.Thread(target=ml_worker, args=(state,), daemon=True)
    state.ml_thread.start()

    # Start async telemetry bridge tasks
    bridge_task = asyncio.create_task(telemetry_bridge(state))
    broadcast_task = asyncio.create_task(telemetry_broadcaster(state))

    print("✅ SHIELD API Server ready.\n")

    yield

    # --- Shutdown ---
    print("\n🛑 Shutting down SHIELD API...")
    state.running = False
    if state.orchestrator:
        state.orchestrator.stop()
    if state.ml_thread and state.ml_thread.is_alive():
        state.ml_thread.join(timeout=5.0)

    bridge_task.cancel()
    broadcast_task.cancel()
    print("✅ Shutdown complete.")


app = FastAPI(
    title="SHIELD Command & Control API",
    description="AI-powered CCTV surveillance — real-time detection, tracking, and analytics",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════

class FenceConfig(BaseModel):
    """Virtual fence coordinates: two points defining a tripwire line."""
    points: List[List[float]]  # [[x1, y1], [x2, y2]]
    camera_id: Optional[str] = None  # If None, apply to all cameras

class PersonRegistration(BaseModel):
    """Register a known person with their name and face image."""
    name: str
    image: str  # Base64-encoded image (JPEG or PNG)

class CameraControl(BaseModel):
    """Control a camera stream."""
    action: str  # "start" or "stop"


# ═══════════════════════════════════════════════════════════
#  REST ENDPOINTS
# ═══════════════════════════════════════════════════════════

# --- System ---

@app.get("/api/v1/system/status")
async def get_system_status():
    """System health: uptime, FPS, active camera count, total tracks, gallery size."""
    uptime = time.time() - state.start_time if state.start_time > 0 else 0
    fps = state.frame_count / max(1.0, uptime)

    total_tracks = 0
    if state.orchestrator:
        for tracker in state.orchestrator.trackers.values():
            total_tracks += len(tracker.active_ids)

    gallery_size = 0
    if state.orchestrator:
        gallery_size = state.orchestrator.global_matcher.gallery_size

    active_cams = 0
    if state.orchestrator:
        for stream in state.orchestrator.stream_manager.streams.values():
            if stream.is_alive_stream:
                active_cams += 1

    return {
        "status": "operational" if state.running else "stopped",
        "telemetry_mode": "live",
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_seconds": round(uptime, 2),
        "fps": round(fps, 2),
        "frame_count": state.frame_count,
        "active_cameras": active_cams,
        "total_cameras": len(state.camera_configs),
        "active_tracks": total_tracks,
        "gallery_size": gallery_size,
        "alert_buffer_size": len(state.recent_alerts),
    }


# --- Camera Streams ---

@app.get("/api/v1/streams/active")
async def get_active_streams():
    """Returns all camera streams with their status, resolution, and FPS."""
    cameras = []
    for cam_cfg in state.camera_configs:
        cam_info = {
            "id": cam_cfg.camera_id,
            "name": cam_cfg.name,
            "source": str(cam_cfg.source),
            "gps": {"lat": cam_cfg.gps_lat, "lng": cam_cfg.gps_lng},
            "status": "unknown",
            "resolution": None,
            "fps": None,
        }

        if state.orchestrator and cam_cfg.camera_id in state.orchestrator.stream_manager.streams:
            stream = state.orchestrator.stream_manager.streams[cam_cfg.camera_id]
            cam_info["status"] = "active" if stream.is_alive_stream else "stopped"
            w, h = stream.frame_size
            cam_info["resolution"] = {"width": w, "height": h} if w > 0 else None
            cam_info["fps"] = round(stream.fps, 1)

        cameras.append(cam_info)

    return {"status": "success", "cameras": cameras}


@app.get("/api/v1/streams/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str):
    """Returns the latest annotated JPEG frame from a camera."""
    frame_data = state.get_latest_frame(camera_id)
    if frame_data is None:
        raise HTTPException(status_code=404, detail=f"No frame available for camera {camera_id}")

    jpeg_bytes, timestamp = frame_data
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Frame-Timestamp": str(timestamp),
            "Cache-Control": "no-cache, no-store",
        },
    )


@app.post("/api/v1/streams/{camera_id}/control")
async def control_camera_stream(camera_id: str, control: CameraControl):
    """Start or stop an individual camera stream."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    sm = state.orchestrator.stream_manager
    if camera_id not in sm.streams:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    stream = sm.streams[camera_id]

    if control.action == "stop":
        stream.stop()
        return {"status": "success", "message": f"Camera {camera_id} stopped"}
    elif control.action == "start":
        if not stream.is_running:
            # Re-create and start the stream
            config = sm.configs.get(camera_id)
            if config:
                from core.stream_manager import CameraStream
                new_stream = CameraStream(config)
                sm.streams[camera_id] = new_stream
                new_stream.start()
                return {"status": "success", "message": f"Camera {camera_id} restarted"}
        return {"status": "success", "message": f"Camera {camera_id} already running"}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {control.action}. Use 'start' or 'stop'.")


# --- Identities ---

@app.get("/api/v1/identities")
async def get_all_identities():
    """Returns all global identities from the cross-camera matcher gallery."""
    if not state.orchestrator:
        return {"status": "success", "identities": [], "total": 0}

    gm = state.orchestrator.global_matcher
    identities = []
    for ident in gm.get_all_identities():
        identities.append({
            "global_id": ident.global_id,
            "display_name": ident.display_name,
            "is_registered": ident.is_registered,
            "first_seen_time": ident.first_seen_time,
            "last_seen_time": ident.last_seen_time,
            "first_seen_camera": ident.first_seen_camera,
            "last_seen_camera": ident.last_seen_camera,
            "times_seen": ident.times_seen,
            "best_quality": ident.best_quality,
        })

    return {"status": "success", "identities": identities, "total": len(identities)}


@app.get("/api/v1/identities/{global_id}")
async def get_identity(global_id: int):
    """Get details for a single global identity including journey history."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    gm = state.orchestrator.global_matcher
    ident = gm.get_identity(global_id)
    if ident is None:
        raise HTTPException(status_code=404, detail=f"Identity G{global_id} not found")

    journey = gm.get_journey(global_id) or []

    return {
        "status": "success",
        "identity": {
            "global_id": ident.global_id,
            "display_name": ident.display_name,
            "is_registered": ident.is_registered,
            "first_seen_time": ident.first_seen_time,
            "last_seen_time": ident.last_seen_time,
            "first_seen_camera": ident.first_seen_camera,
            "last_seen_camera": ident.last_seen_camera,
            "times_seen": ident.times_seen,
            "best_quality": ident.best_quality,
            "journey": [
                {"camera_id": cam, "entry_time": entry, "exit_time": exit_}
                for cam, entry, exit_ in journey
            ],
        },
    }


@app.get("/api/v1/identities/{global_id}/journey")
async def get_identity_journey(global_id: int):
    """Movement timeline for a person across all cameras."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    gm = state.orchestrator.global_matcher
    journey = gm.get_journey(global_id)
    if journey is None:
        raise HTTPException(status_code=404, detail=f"Identity G{global_id} not found")

    return {
        "status": "success",
        "global_id": global_id,
        "journey": [
            {"camera_id": cam, "entry_time": entry, "exit_time": exit_}
            for cam, entry, exit_ in journey
        ],
    }


@app.post("/api/v1/identities/register")
async def register_person(registration: PersonRegistration):
    """
    Register a known person. Accepts a name and base64-encoded image.
    Extracts a ReID embedding and registers them in the GlobalMatcher gallery.
    """
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(registration.image)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Extract embedding
    embedding = state.orchestrator.encoder.extract(image)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Could not extract features from image. Ensure it contains a visible person.")

    # Register in global matcher
    global_id = state.orchestrator.global_matcher.register_person(
        name=registration.name,
        prototype=embedding,
    )

    return {
        "status": "success",
        "message": f"Person '{registration.name}' registered successfully",
        "global_id": global_id,
    }


# --- Alerts ---

@app.get("/api/v1/alerts")
async def get_recent_alerts(
    limit: int = Query(default=50, ge=1, le=1000, description="Number of recent alerts to return"),
    alert_type: Optional[str] = Query(default=None, description="Filter by alert type"),
):
    """Returns recent alerts from the telemetry buffer."""
    alerts = list(state.recent_alerts)
    alerts.reverse()  # Most recent first

    if alert_type:
        alerts = [a for a in alerts if a.get("type") == alert_type]

    return {
        "status": "success",
        "alerts": alerts[:limit],
        "total_buffered": len(state.recent_alerts),
    }


# --- Risk ---

@app.get("/api/v1/risk/summary")
async def get_risk_summary():
    """Per-camera scene status and density summary from the latest telemetry."""
    telemetry = state.get_latest_telemetry()
    streams = telemetry.get("streams", {})

    summary = {}
    for cam_id, stream_data in streams.items():
        summary[cam_id] = {
            "scene_status": stream_data.get("scene_status", "UNKNOWN"),
            "scene_density": stream_data.get("scene_density", 0.0),
            "track_count": len(stream_data.get("tracks", [])),
            "tracks": [
                {
                    "id": t["id"],
                    "label": t["label"],
                    "confidence": t["confidence"],
                    "speed": t["speed"],
                }
                for t in stream_data.get("tracks", [])
            ],
        }

    return {
        "status": "success",
        "timestamp": telemetry.get("timestamp", 0),
        "cameras": summary,
    }


# --- Virtual Fence ---

@app.post("/api/v1/fence/config")
async def update_virtual_fence(fence: FenceConfig):
    """Update virtual fence coordinates — pushes to RiskAnalyzer instances."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    if len(fence.points) != 2 or any(len(p) != 2 for p in fence.points):
        raise HTTPException(status_code=400, detail="Fence must be exactly 2 points: [[x1,y1], [x2,y2]]")

    new_coords = [(fence.points[0][0], fence.points[0][1]),
                  (fence.points[1][0], fence.points[1][1])]

    if fence.camera_id:
        # Update specific camera
        if fence.camera_id not in state.orchestrator.risk_analyzers:
            raise HTTPException(status_code=404, detail=f"Camera {fence.camera_id} not found")
        state.orchestrator.risk_analyzers[fence.camera_id].update_fence(new_coords)
        return {"status": "success", "message": f"Fence updated for {fence.camera_id}"}
    else:
        # Update all cameras
        for analyzer in state.orchestrator.risk_analyzers.values():
            analyzer.update_fence(new_coords)
        return {"status": "success", "message": "Fence updated for all cameras"}


# ═══════════════════════════════════════════════════════════
#  WEBSOCKET ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    Streams live telemetry (track positions, risk labels, alerts)
    from the running orchestrator to connected dashboards.
    """
    await ws_manager.connect_telemetry(websocket)
    try:
        while True:
            # Keep connection alive — listen for client messages (ping/control)
            data = await websocket.receive_text()
            # Client can send commands via telemetry websocket if needed
    except WebSocketDisconnect:
        await ws_manager.disconnect_telemetry(websocket)
    except Exception:
        await ws_manager.disconnect_telemetry(websocket)


@app.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """
    Streams MJPEG-encoded annotated frames for a specific camera.
    Clients receive binary JPEG frames at the pipeline's processing rate.
    """
    await ws_manager.connect_stream(websocket, camera_id)
    last_ts = 0.0
    try:
        while True:
            frame_data = state.get_latest_frame(camera_id)
            if frame_data is not None:
                jpeg_bytes, ts = frame_data
                if ts > last_ts:
                    await websocket.send_bytes(jpeg_bytes)
                    last_ts = ts
            await asyncio.sleep(0.033)  # ~30 FPS cap
    except WebSocketDisconnect:
        await ws_manager.disconnect_stream(websocket, camera_id)
    except Exception:
        await ws_manager.disconnect_stream(websocket, camera_id)


# Serve Vite's production build when present. Keep this catch-all mount after
# every API and WebSocket route so it cannot shadow them.
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def start_api_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[dict] = None):
    """Start the SHIELD API server."""
    # Store the config (with CLI overrides) so the lifespan can use it
    if config is not None:
        state.config = config

    print(f"\n🌐 Starting SHIELD API on http://{host}:{port}")
    print(f"   📡 Telemetry WebSocket: ws://{host}:{port}/ws/telemetry")
    print(f"   📺 Stream WebSocket:    ws://{host}:{port}/ws/stream/{{camera_id}}")
    print(f"   📖 API Docs:            http://{host}:{port}/docs\n")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    config = load_config()
    api_config = config.get("api", {})
    start_api_server(
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
    )
