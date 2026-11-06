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
    # --cameras / SHIELD_CAMERAS selects another topology file (e.g. config/cameras.live.yaml)
    cam_path = os.environ.get("SHIELD_CAMERAS") or os.path.join(os.path.dirname(__file__), "config", "cameras.yaml")
    if not os.path.isabs(cam_path):
        cam_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cam_path)
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
                night_mode=str(cam.get("night_mode", "")),
                width=int(cam.get("width", 0) or 0),
                height=int(cam.get("height", 0) or 0),
                loop=bool(cam.get("loop", False)),
                rotate=int(cam.get("rotate", 0) or 0),
                realtime=bool(cam.get("realtime", False)),
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
        # Latest analysis overlay per camera: (raw_frame, annotated_frame, timestamp)
        self._overlays: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        self._overlay_lock = threading.Lock()
        self.display_times: Dict[str, Deque[float]] = {}
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

    def set_overlay(self, camera_id: str, raw: np.ndarray, annotated: np.ndarray) -> None:
        """Called from the ML thread after each analysis: cheap, no encoding."""
        with self._overlay_lock:
            self._overlays[camera_id] = (raw, annotated, time.time())

    def get_overlay(self, camera_id: str):
        with self._overlay_lock:
            return self._overlays.get(camera_id)

    def display_fps(self, camera_id: str) -> float:
        q = self.display_times.get(camera_id)
        if not q or len(q) < 2 or time.time() - q[-1] > 2.0:
            return 0.0
        return round((len(q) - 1) / max(1e-6, q[-1] - q[0]), 1)

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
    Headless Orchestrator that publishes annotated frames to the API buffer.
    The analytics pipeline itself lives in Orchestrator.process_frames().
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

    def _on_frames(self, annotated_frames: Dict[str, np.ndarray]) -> bool:
        raw = self.last_raw_frames
        for cam_id, frame in annotated_frames.items():
            if cam_id in raw:
                self._shared_state.set_overlay(cam_id, raw[cam_id], frame)
            else:
                self._shared_state.push_annotated_frame(cam_id, frame)
        return True

    def run(self) -> None:
        try:
            super().run()
        except Exception as e:
            print(f"\n[CRITICAL] Orchestrator error: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════
#  DISPLAY COMPOSITOR (smooth live video, decoupled from analysis)
# ═══════════════════════════════════════════════════════════

def display_worker(shared_state: ShieldState):
    """Publishes dashboard video at the camera's own rate (up to stream.display_fps).

    Each output frame is the camera's NEWEST frame with the most recent analysis
    overlay (boxes, labels, fence) copied on top. The analysis loop therefore never
    limits how smooth the video looks; boxes refresh at the analysis rate.
    """
    stream_cfg = shared_state.config.get("stream", {})
    period = 1.0 / max(1.0, float(stream_cfg.get("display_fps", 15)))
    max_dim = int(stream_cfg.get("display_max_dim", 1280))
    quality = int(stream_cfg.get("jpeg_quality", 70))
    overlay_max_age = float(stream_cfg.get("overlay_max_age_sec", 1.5))
    last_seq: Dict[str, int] = {}
    masks: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}  # cam -> (overlay ts, mask, annotated)

    while shared_state.running:
        tick = time.time()
        orch = shared_state.orchestrator
        if orch is None:
            time.sleep(0.1)
            continue
        for cam_id, stream in list(orch.stream_manager.streams.items()):
            try:
                latest = stream.latest_display()
                if latest is None or latest[2] == last_seq.get(cam_id):
                    continue
                frame, _, seq = latest
                last_seq[cam_id] = seq

                ov = shared_state.get_overlay(cam_id)
                out = frame
                if ov is not None and tick - ov[2] <= overlay_max_age:
                    raw, ann, ov_ts = ov
                    cached = masks.get(cam_id)
                    if cached is None or cached[0] != ov_ts:
                        if raw.shape == ann.shape:
                            mask = np.any(cv2.absdiff(raw, ann), axis=2)
                            masks[cam_id] = cached = (ov_ts, mask, ann)
                        else:
                            cached = None
                    if cached is not None and cached[2].shape == frame.shape:
                        out = frame.copy()
                        out[cached[1]] = cached[2][cached[1]]

                h, w = out.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    now = time.time()
                    with shared_state._frame_lock:
                        shared_state._frame_buffer[cam_id] = (jpeg.tobytes(), now)
                    q = shared_state.display_times.setdefault(cam_id, deque())
                    q.append(now)
                    while q and now - q[0] > 3.0:
                        q.popleft()
            except Exception as e:  # never let one bad frame stop the video
                print(f"  [DISPLAY] {cam_id}: {e}")
        time.sleep(max(0.0, period - (time.time() - tick)))


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
    threading.Thread(target=display_worker, args=(state,), name="display", daemon=True).start()

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
    """Virtual fence drawn in the dashboard.

    type "line": tripwire polyline, >= 2 points (crossing = intrusion)
    type "zone": restricted area polygon, >= 3 points (inside = intrusion)
    An empty points list disables the fence.
    """
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...] in frame pixels
    camera_id: Optional[str] = None  # If None, apply to all cameras
    type: str = "line"

class PersonRegistration(BaseModel):
    """Register a known person with their name and face image."""
    name: str
    image: str  # Base64-encoded image (JPEG or PNG)

class CameraControl(BaseModel):
    """Control a camera stream."""
    action: str  # "start" or "stop"

class FaceEnrollment(BaseModel):
    """Enrol a person into the facial-recognition watchlist."""
    name: str
    image: Optional[str] = None            # base64 JPEG/PNG (data: URLs accepted)
    images: Optional[List[str]] = None     # several reference photos of the same person
    category: str = "SUSPECT"              # SUSPECT | WANTED | BLACKLISTED | MISSING | VIP | AUTHORIZED …
    notes: str = ""
    identity_id: Optional[str] = None      # add references to an existing identity

class ImageQuery(BaseModel):
    image: str                             # base64 JPEG/PNG

class PlateWatchEntry(BaseModel):
    plate: str
    tag: str = "BLACKLISTED"               # BLACKLISTED | STOLEN | WANTED | RESTRICTED | VIP …
    notes: str = ""


def decode_image(data: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) image into a BGR array, or raise HTTP 400."""
    try:
        if "," in data[:100] and data.lstrip().startswith("data:"):
            data = data.split(",", 1)[1]
        np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("not a decodable image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def _require_pipeline():
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")
    return state.orchestrator


# ═══════════════════════════════════════════════════════════
#  REST ENDPOINTS
# ═══════════════════════════════════════════════════════════

# --- System ---

@app.get("/api/v1/system/status")
async def get_system_status():
    """System health: uptime, FPS, active camera count, total tracks, gallery size."""
    uptime = time.time() - state.start_time if state.start_time > 0 else 0
    perf = state.orchestrator.performance() if state.orchestrator else {
        "processing_fps": 0.0, "latency_ms": None, "cameras": {}}
    fps = perf["processing_fps"]

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
        "fps": fps,                               # analysis loops per second (last 3 s)
        "processing_fps": fps,
        "latency_ms": perf["latency_ms"],         # capture -> analysis finished
        "cameras_perf": {
            cid: {**p, "display_fps": state.display_fps(cid)} for cid, p in perf["cameras"].items()
        },
        "frame_count": state.frame_count,
        "active_cameras": active_cams,
        "total_cameras": len(state.camera_configs),
        "active_tracks": total_tracks,
        "gallery_size": gallery_size,
        "alert_buffer_size": len(state.recent_alerts),
        "analytics": _analytics_status(),
    }


def _analytics_status() -> dict:
    orch = state.orchestrator
    if not orch:
        return {}
    out = {"face_recognition": orch.face_auth.get_stats()}
    out["anpr"] = orch.anpr_engine.get_stats() if orch.anpr_engine is not None else {"available": False}
    out["night_mode"] = {cid: orch.stream_manager.is_night(cid) for cid in orch.stream_manager.streams}
    out["alert_counts"] = dict(orch.alert_manager.counts)
    out["webhook"] = {"enabled": bool(orch.alert_manager.webhook_url), **orch.alert_manager.webhook_stats}
    return out


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
            cam_info["fps"] = round(stream.fps, 1)               # camera's own frame rate
            cam_perf = state.orchestrator.performance()["cameras"].get(cam_cfg.camera_id, {})
            cam_info["analysis_fps"] = cam_perf.get("analysis_fps", 0.0)
            cam_info["display_fps"] = state.display_fps(cam_cfg.camera_id)
            cam_info["latency_ms"] = cam_perf.get("latency_ms")

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
        if sm.restart_camera(camera_id):
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

    image = decode_image(registration.image)

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


# --- Facial Recognition (FRS) ---

@app.post("/api/v1/faces/enroll")
async def enroll_face(enrollment: FaceEnrollment):
    """
    Enrol a suspect (or authorised person) from one or more reference photos.
    The largest face in each photo is embedded and stored in the persistent gallery.
    Live tracks are then matched against it; threats raise FACE_MATCH_SUSPECT alerts.
    """
    orch = _require_pipeline()
    raw = list(enrollment.images or []) + ([enrollment.image] if enrollment.image else [])
    if not raw:
        raise HTTPException(status_code=400, detail="Provide 'image' or 'images'")
    images = [decode_image(r) for r in raw]
    try:
        result = await asyncio.to_thread(
            orch.face_auth.enroll_face, enrollment.name, images, enrollment.category,
            enrollment.notes, enrollment.identity_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "success", "message": f"Enrolled '{enrollment.name}'", "identity": result}


@app.get("/api/v1/faces/gallery")
async def get_face_gallery():
    """List every enrolled identity (with thumbnail) in the FRS gallery."""
    orch = _require_pipeline()
    items = orch.face_auth.list_identities()
    return {"status": "success", "identities": items, "total": len(items),
            "engine": orch.face_auth.get_stats()}


@app.delete("/api/v1/faces/{identity_id}")
async def delete_face_identity(identity_id: str):
    orch = _require_pipeline()
    if not orch.face_auth.remove_identity(identity_id):
        raise HTTPException(status_code=404, detail=f"Identity {identity_id} not found")
    return {"status": "success", "message": f"Identity {identity_id} removed"}


@app.post("/api/v1/faces/search")
async def search_faces(query: ImageQuery):
    """Detect all faces in an uploaded image and match each against the gallery."""
    orch = _require_pipeline()
    if not orch.face_auth.available:
        raise HTTPException(status_code=503, detail="Face recognition models not loaded")
    image = decode_image(query.image)
    faces = await asyncio.to_thread(orch.face_auth.identify, image)
    return {"status": "success", "faces": faces, "total": len(faces)}


# --- ANPR ---

def _require_anpr():
    orch = _require_pipeline()
    if orch.anpr_engine is None:
        raise HTTPException(status_code=503, detail="ANPR is disabled in config.yaml")
    return orch.anpr_engine


@app.get("/api/v1/anpr/watchlist")
async def get_plate_watchlist():
    engine = _require_anpr()
    items = engine.list_watchlist()
    return {"status": "success", "watchlist": items, "total": len(items)}


@app.post("/api/v1/anpr/watchlist")
async def add_plate_watchlist(entry: PlateWatchEntry):
    engine = _require_anpr()
    from core.anpr import normalize_plate
    plate = normalize_plate(entry.plate)
    if len(plate) < 4:
        raise HTTPException(status_code=400, detail="Plate must have at least 4 alphanumeric characters")
    engine.update_watchlist({plate: {"tag": entry.tag, "notes": entry.notes}})
    return {"status": "success", "message": f"{plate} added as {entry.tag.upper()}", "plate": plate}


@app.delete("/api/v1/anpr/watchlist/{plate}")
async def remove_plate_watchlist(plate: str):
    engine = _require_anpr()
    if not engine.remove_from_watchlist(plate):
        raise HTTPException(status_code=404, detail=f"{plate} is not on the watchlist")
    return {"status": "success", "message": f"{plate} removed"}


@app.get("/api/v1/anpr/plates")
async def get_recent_plates(limit: int = Query(default=50, ge=1, le=500)):
    """Recently confirmed plates (temporal-voting consensus), newest first."""
    engine = _require_anpr()
    plates = [p.to_dict() for p in reversed(list(engine.recent_plates))][:limit]
    return {"status": "success", "plates": plates, "total": len(engine.recent_plates)}


@app.post("/api/v1/anpr/recognize")
async def recognize_plate(query: ImageQuery):
    """One-shot plate recognition on an uploaded vehicle/plate image."""
    engine = _require_anpr()
    if not engine.available:
        raise HTTPException(status_code=503, detail="OCR engine not available (pip install easyocr)")
    image = decode_image(query.image)
    result = await asyncio.to_thread(engine.recognize_image, image)
    return {"status": "success", "result": result}


# --- Event log (C&C integration) ---

@app.get("/api/v1/events")
async def get_event_log(
    limit: int = Query(default=100, ge=1, le=5000),
    alert_type: Optional[str] = Query(default=None),
):
    """Persistent alert/event log (logs/events.jsonl), newest first — survives restarts."""
    orch = _require_pipeline()
    events = await asyncio.to_thread(orch.alert_manager.read_events, limit, alert_type)
    return {"status": "success", "events": events, "total": len(events)}


# --- Virtual Fence ---

@app.post("/api/v1/fence/config")
async def update_virtual_fence(fence: FenceConfig):
    """Update virtual fence coordinates — pushes to RiskAnalyzer instances."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")

    if fence.type not in ("line", "zone"):
        raise HTTPException(status_code=400, detail="type must be 'line' (tripwire) or 'zone' (area)")
    if any(len(p) != 2 for p in fence.points):
        raise HTTPException(status_code=400, detail="Each point must be [x, y]")
    minimum = 3 if fence.type == "zone" else 2
    if fence.points and len(fence.points) < minimum:
        raise HTTPException(status_code=400,
                            detail=f"A {fence.type} needs at least {minimum} points (or none to disable)")

    new_coords = [(float(p[0]), float(p[1])) for p in fence.points]
    analyzers = state.orchestrator.risk_analyzers
    if fence.camera_id and fence.camera_id not in analyzers:
        raise HTTPException(status_code=404, detail=f"Camera {fence.camera_id} not found")
    targets = [fence.camera_id] if fence.camera_id else list(analyzers)
    for cam_id in targets:
        if analyzers[cam_id].update_fence(new_coords, fence.type) is False:
            raise HTTPException(status_code=400, detail="Shape has no area; spread the points out")
    try:
        state.orchestrator.save_fences()
    except Exception as e:  # saving is best-effort; the live fence is already applied
        print(f"  [FENCE] could not save fences: {e}")
    where = fence.camera_id or "all cameras"
    return {"status": "success", "message": f"Fence updated for {where}"}


class IgnoreZones(BaseModel):
    """Areas whose detections are ignored (posters, screens, mirrors). [] removes all."""
    camera_id: str
    zones: List[List[List[float]]]   # [[[x, y], [x, y], [x, y], ...], ...]


@app.post("/api/v1/fence/ignore")
async def update_ignore_zones(cfg: IgnoreZones):
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")
    if cfg.camera_id not in state.orchestrator.risk_analyzers:
        raise HTTPException(status_code=404, detail=f"Camera {cfg.camera_id} not found")
    if any(len(z) < 3 or any(len(p) != 2 for p in z) for z in cfg.zones):
        raise HTTPException(status_code=400, detail="Each ignore area needs at least 3 [x, y] points")
    kept = state.orchestrator.set_ignore_zones(cfg.camera_id, cfg.zones)
    if kept != len(cfg.zones):
        raise HTTPException(status_code=400, detail="An ignore area has no area; spread its points out")
    try:
        state.orchestrator.save_fences()
    except Exception as e:
        print(f"  [FENCE] could not save ignore areas: {e}")
    return {"status": "success", "message": f"{kept} ignore area(s) set for {cfg.camera_id}"}


@app.get("/api/v1/fence/{camera_id}")
async def get_virtual_fence(camera_id: str):
    """Current fence of a camera (for the dashboard editor)."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="ML pipeline not running")
    analyzer = state.orchestrator.risk_analyzers.get(camera_id)
    if analyzer is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    ignore = getattr(state.orchestrator, "ignore_zones", {}).get(camera_id, {}).get("points", [])
    return {"status": "success", "camera_id": camera_id, **analyzer.fence_dict(),
            "ignore": ignore if isinstance(ignore, list) else []}


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
