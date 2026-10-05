import asyncio
import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import time

# Create the FastAPI App
app = FastAPI(
    title="IBVAP Command & Control API",
    description="Intelligent Border Video Analytics Platform",
    version="1.0.0"
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
        print(f"🔗 [WebSocket] Frontend C&C Dashboard Connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"❌ [WebSocket] Frontend C&C Dashboard Disconnected")

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
    # TODO: Connect this to the new StreamManager
    return {
        "status": "success",
        "cameras": [
            {"id": "cam1", "name": "Border Checkpoint Alpha", "status": "active"},
            {"id": "cam2", "name": "Night-Vision Outpost Bravo", "status": "active"}
        ]
    }

@app.post("/api/v1/fence/config")
async def update_virtual_fence(payload: Dict[str, Any]):
    """Receives updated virtual fence coordinates from the frontend."""
    # TODO: Push these new coordinates into behavior_risk.py math logic
    print(f"🚧 [API] Virtual Fence Updated: {payload}")
    return {"status": "success", "message": "Fence updated successfully"}

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
    print("🚀 Starting Background ML Orchestrator...")
    while True:
        await asyncio.sleep(2.0) # Simulate 2 seconds of frame processing
        if manager.active_connections:
            dummy_telemetry = {
                "timestamp": time.time(),
                "camera_id": "cam1",
                "tracks": [
                    {"id": "G1", "class": "person", "bbox": [100, 100, 200, 300], "risk": 45},
                    {"id": "G2", "class": "truck", "bbox": [400, 200, 600, 450], "plate": "DL-1C-AA-1111"}
                ]
            }
            await manager.broadcast(dummy_telemetry)

@app.on_event("startup")
async def startup_event():
    # Spin up the ML pipeline in the background when the server starts
    asyncio.create_task(run_ml_orchestrator())

if __name__ == "__main__":
    print("Starting IBVAP API Server on port 8080...")
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)