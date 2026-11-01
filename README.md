# S.H.I.E.L.D.

**Secure Hostile Intelligence & Enforcement for Live Defense**

AI-powered multi-camera CCTV video analytics platform for real-time person detection, cross-camera tracking, re-identification, vehicle plate recognition, and behavioral risk analysis.

---

## Features

| Module | Description |
|---|---|
| **Person Detection** | YOLOv11m multi-class detector (persons + vehicles) with dual-confidence split |
| **Single-Camera Tracking** | 8-step Kalman + Hungarian pipeline with lost-buffer re-linking and occlusion awareness |
| **Cross-Camera Re-ID** | OSNet_x1_0 embeddings with 2-phase global matching (early + refined) |
| **ANPR** | Automatic Number Plate Recognition via EasyOCR |
| **Face Auth** | Haar cascade face detection with configurable threat watchlist |
| **Behavioral Risk** | Temporal risk classifier with virtual fence (tripwire) using Shapely geometry |
| **Night Vision** | CLAHE-based auto-enhancement during nighttime hours |
| **REST API** | FastAPI server with 11 REST endpoints + 2 WebSocket streams |
| **Telemetry Logging** | JSONL frame-level logs for cloud ingestion pipelines |

---

## Architecture

```
StreamManager (threaded capture per camera)
    |
    v
SecurityDetector (YOLOv11m -- batched GPU inference)
    |
    +-- Person detections --> ReIDEncoder (OSNet_x1_0 -- batched)
    |                              |
    |                              v
    |                    SingleCameraTracker (per camera)
    |                              |
    |                              v
    |                       GlobalMatcher (cross-camera identity)
    |                              |
    |                              v
    |                        RiskAnalyzer (behavioral analysis)
    |
    +-- Vehicle detections --> ANPREngine (EasyOCR plate extraction)

Orchestrator ties all modules together in a single loop.
API server wraps the orchestrator with FastAPI + WebSocket telemetry.
```

---

## Project Structure

```
SHIELD/
|-- Backend/
|   |-- core/
|   |   |-- __init__.py          # Package exports
|   |   |-- orchestrator.py      # Master pipeline -- wires all modules
|   |   |-- detector.py          # YOLOv11m multi-class detector
|   |   |-- reid_encoder.py      # OSNet_x1_0 person re-ID encoder
|   |   |-- tracker.py           # 8-step single-camera tracker
|   |   |-- kalman.py            # 8-state Kalman filter
|   |   |-- global_matcher.py    # Cross-camera identity matching
|   |   |-- behavior_risk.py     # Temporal risk classifier + virtual fence
|   |   |-- stream_manager.py    # Threaded camera capture + CLAHE
|   |   |-- anpr.py              # Automatic Number Plate Recognition
|   |   +-- face_auth.py         # Face detection + threat watchlist
|   |-- config/
|   |   |-- config.yaml          # Master configuration
|   |   +-- cameras.yaml         # Camera topology
|   |-- models/
|   |   +-- yolo11m.pt           # YOLOv11m weights
|   |-- videos/                  # Test video files
|   |-- logs/                    # Runtime telemetry logs (auto-created)
|   |-- tests/                   # Pytest test suite (13 files, ~100 tests)
|   |-- api.py                   # FastAPI server with WebSocket streaming
|   |-- run_shield.py            # CLI + API launcher
|   +-- requirements.txt         # Python dependencies
|-- .gitignore
|-- LICENSE
+-- README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- pip or conda

### 2. Install Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

> **Note:** On first run, if `models/osnet_x1_0_msmt17.pth` is not present, the ReID encoder will auto-download pretrained MSMT17 weights via torchreid.

### 3. Configure Cameras

Edit `Backend/config/cameras.yaml` to define your camera sources:

```yaml
cameras:
  - camera_id: "cam1"
    name: "Main Gate"
    source: "rtsp://user:pass@192.168.1.100/stream"
    gps_lat: 28.6139
    gps_lng: 77.2090

  - camera_id: "cam2"
    name: "Parking Lot"
    source: 0  # Local webcam
```

Supported sources:
- **RTSP streams:** `rtsp://...`
- **Video files:** `videos/recording.mp4`
- **Webcams:** `0`, `1`, etc.

### 4. Run

**CLI mode** (with OpenCV display windows):

```bash
cd Backend
python run_shield.py
```

**API server mode** (headless, for dashboard integration):

```bash
cd Backend
python run_shield.py --api
```

**With options:**

```bash
python run_shield.py --api --port 9000 --host 0.0.0.0 --device cuda
```

### 5. Run Tests

```bash
cd Backend
python -m pytest tests/ -v
```

---

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/system/status` | System health -- uptime, FPS, active cameras, tracks |
| `GET` | `/api/v1/streams/active` | All camera streams with status and resolution |
| `GET` | `/api/v1/streams/{id}/snapshot` | Latest annotated JPEG frame from a camera |
| `POST` | `/api/v1/streams/{id}/control` | Start/stop an individual camera |
| `GET` | `/api/v1/identities` | All tracked global identities |
| `GET` | `/api/v1/identities/{id}` | Single identity details + journey history |
| `GET` | `/api/v1/identities/{id}/journey` | Movement timeline across cameras |
| `POST` | `/api/v1/identities/register` | Register a known person (name + face image) |
| `GET` | `/api/v1/alerts` | Recent alerts with optional type filter |
| `GET` | `/api/v1/risk/summary` | Per-camera risk analysis summary |
| `POST` | `/api/v1/fence/config` | Update virtual fence coordinates |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://host:port/ws/telemetry` | Live JSON telemetry stream (tracks, risk, alerts) |
| `ws://host:port/ws/stream/{camera_id}` | Live MJPEG frame stream for a specific camera |

### Interactive Docs

When running in API mode, visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## Configuration

All tunable parameters live in `Backend/config/config.yaml`. The orchestrator reads this file and passes values to each component -- **no hardcoded values in the pipeline**.

Key sections:

| Section | Controls |
|---------|----------|
| `system` | Device (cuda/cpu), number of cameras |
| `detection` | YOLO confidence thresholds, NMS IOU, image size |
| `reid` | OSNet model, embedding dimensions |
| `tracker` | Lifecycle thresholds, cost matrix weights, appearance similarity |
| `global_matcher` | Cross-camera match/margin thresholds, EMA update rate |
| `risk` | Alert threshold, loiter/velocity/crowd parameters |
| `zones` | Polygon coordinates for entry/restricted areas |
| `stream` | JPEG quality, target FPS |
| `runtime` | ReID frequency, global ID settings, alert cooldowns |
| `api` | Host, port |

---

## How It Works

### Detection Pipeline

1. **Frame Capture** -- `StreamManager` runs a daemon thread per camera, storing only the latest frame (no backlog)
2. **Batched Detection** -- All camera frames are batched into a single YOLO inference call on GPU
3. **Dual-Confidence Split** -- Detections above `high_conf` (0.40) get full ReID processing; detections between `low_conf` (0.07) and `high_conf` are used only for keeping existing tracks alive during occlusion

### Tracking Pipeline

4. **Kalman Predict** -- 8-state constant-velocity Kalman filter predicts each track's next position
5. **Hungarian Assignment** -- Gated cost matrix (motion + appearance + IOU) with cascade matching (active tracks get priority over tentative)
6. **Lost Buffer** -- Tracks missing for a few frames are held in a buffer for re-linking via appearance matching
7. **Birth Guard** -- Multi-layer duplicate prevention (IOU, appearance, aspect ratio) before creating new tracks

### Cross-Camera Re-ID

8. **Phase 1 (Early)** -- When a track is confirmed, its best embedding is matched against the global gallery for immediate GlobalID assignment
9. **Phase 2 (Refined)** -- When a track ends, its pooled embedding (median of top-K) is used for more accurate cross-camera association and prototype update

### Risk Analysis

10. **Behavioral Classification** -- Temporal analysis of speed, proximity, zone dwell time, and interaction patterns
11. **Virtual Fence** -- Mathematical tripwire using Shapely `LineString.intersects()` on track movement vectors
12. **Scene Status** -- Aggregated scene-level classification (CALM > ACTIVE > HIGH_ACTIVITY > BREACH_DETECTED)

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
