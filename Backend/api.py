import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import time

@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(run_ml_orchestrator())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

# Create the FastAPI App
app = FastAPI(
    title="IBVAP Command & Control API",
    description="Intelligent Border Video Analytics Platform",
    version="1.0.0",
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

# WebSocket Connection Manager for Live Telemetry
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("[WebSocket] Frontend C&C Dashboard Connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("[WebSocket] Frontend C&C Dashboard Disconnected")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"WebSocket send error: {e}")

manager = ConnectionManager()

# --- REST Endpoints ---

@app.get("/api/v1/streams/active")
async def get_active_streams():
    """Returns the list of active border camera streams."""
    streams = await get_streams()
    return {
        "status": streams["status"],
        "cameras": [
            {"id": camera["camera_id"], "name": camera["camera_name"], "status": camera["status"]}
            for camera in streams["cameras"]
        ],
    }

@app.get("/api/v1/streams")
async def get_streams():
    """Frontend-compatible stream list (static until StreamManager is shared with this process)."""
    return {
        "status": "success",
        "cameras": [
            {"camera_id": "cam1", "camera_name": "Highway Checkpost", "status": "configured"},
            {"camera_id": "cam2", "camera_name": "Border Patrol (Crowd)", "status": "configured"},
            {"camera_id": "cam3", "camera_name": "Night Vision Sector", "status": "configured"},
        ],
    }

@app.get("/api/v1/status")
async def get_status():
    """Basic API health; processing metrics are populated when a live pipeline is connected."""
    return {
        "status": "online",
        "telemetry_mode": "simulation",
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

@app.post("/api/v1/fence/config")
@app.post("/api/v1/config/fence")
async def update_virtual_fence(payload: Dict[str, Any]):
    """Rejects changes until this API shares a live RiskAnalyzer with the pipeline."""
    raise HTTPException(
        status_code=501,
        detail="Fence updates are not connected to a running risk engine yet; no changes were applied.",
    )

# --- WebSocket Endpoint ---

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Streams live bounding boxes, IDs, and alerts to the frontend."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for any client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Background ML Task Simulation ---

async def run_ml_orchestrator():
    """
    This background task will eventually run our actual YOLO + OSNet + ANPR pipeline.
    For now, it sends dummy telemetry to prove the WebSocket connection works.
    """
    print("Starting Background ML Orchestrator...")
    frame_number = 0
    while True:
        await asyncio.sleep(2.0) # Simulate 2 seconds of frame processing
        if manager.active_connections:
            frame_number += 1
            dummy_telemetry = {
                "timestamp": time.time(),
                "frame": frame_number,
                "camera_id": "cam1",
                "tracks": [
                    {"id": "G1", "class": "person", "bbox": [100, 100, 200, 300], "risk": 45},
                    {"id": "G2", "class": "truck", "bbox": [400, 200, 600, 450], "plate": "DL-1C-AA-1111"}
                ]
            }
            await manager.broadcast(dummy_telemetry)

# Serve the Vite build from FastAPI at the same origin. Keep this mount last so
# API and WebSocket routes remain reachable before the catch-all frontend.
FRONTEND_DIST = Path(__file__).resolve().parent / "static"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

if __name__ == "__main__":
    print("Starting IBVAP API Server on port 8080...")
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
