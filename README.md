# S.H.I.E.L.D.

SHIELD (IBVAP — Intelligent Border Video Analytics Platform) turns ordinary IP/RTSP CCTV cameras into an intelligent surveillance network in software, with no FRS, ANPR or smart-camera hardware. FastAPI starts the configured camera streams and the analytics Orchestrator in a worker thread, then serves telemetry, annotated video, alerts and REST APIs to the React command dashboard.

## Capabilities

| Capability | How it works | Module |
|---|---|---|
| Human detection & tracking | YOLOv11m (person) → Kalman + cascade Hungarian tracker → OSNet Re-ID across cameras | `detector.py`, `tracker.py`, `global_matcher.py` |
| Vehicle detection & classification | YOLOv11m car / motorcycle / bus / truck, IoU vehicle tracker with majority-vote class | `detector.py`, `vehicle_tracker.py` |
| Face detection & **recognition (FRS)** | YuNet detector → 5-point alignment → SFace (ArcFace-family) embeddings → cosine-distance match (≤ 0.6) against an enrolled, persistent gallery; 2 consecutive hits required before alerting | `face_auth.py` |
| **ANPR** | Plate localisation (Sobel-X/Otsu/morphology + plate-background cue, aspect 2.5–5:1) → CLAHE/denoise/deskew → multi-pass EasyOCR (bumper → localised → full) → Indian-format correction → 2-read temporal voting → OCR-confusion-tolerant watchlist | `anpr.py` |
| Virtual fence intrusion | Foot-point movement vector vs tripwire (Shapely), editable from the dashboard | `behavior_risk.py` |
| Suspicious activity | Loitering (dwell ≥ `loiter_seconds` within `loiter_displacement_px`), sustained high-speed motion, interaction, isolation, crowd density | `behavior_risk.py` |
| Night-time movement | Night mode when the clock is in the night window **or** the scene is dark (mean luma, hysteresis); CLAHE enhancement + `NIGHT_MOVEMENT` alerts | `stream_manager.py` |
| Real-time alerts & logging | Uniform alert schema with cooldowns → WebSocket, `logs/events.jsonl` (persistent, rotated), optional C&C webhook | `alerts.py`, `orchestrator.py` |

### Alert types

| Type | Severity | Trigger |
|---|---|---|
| `FACE_MATCH_SUSPECT` | CRITICAL | Enrolled SUSPECT / WANTED / BLACKLISTED face recognised (`CRITICAL_THREAT: FACE_MATCH_SUSPECT`) |
| `FACE_MATCH_WATCHLIST` / `FACE_MATCH_AUTHORIZED` | WARNING / NORMAL | MISSING or VIP / AUTHORIZED face recognised |
| `ANPR_WATCHLIST_HIT` | CRITICAL (WARNING for VIP) | Confirmed plate matches the plate watchlist (exact or fuzzy) |
| `VEHICLE_DETECTED` | NORMAL | Plate confirmed by temporal voting |
| `INTRUSION` | CRITICAL | Virtual fence crossed |
| `LOITERING`, `SUSPICIOUS_MOTION` | WARNING | Behavioural anomaly |
| `NIGHT_MOVEMENT` | WARNING | Person or vehicle appears while the camera is in night mode |

## Requirements

- Windows PowerShell
- Python 3.10 or newer
- Node.js and npm
- A CUDA-capable GPU is recommended. CPU mode is available with `--device cpu` but will process more slowly.
- The configured camera videos or reachable RTSP sources

The first Python setup may take time because PyTorch, EasyOCR, YOLO, and Re-ID dependencies are large. If the OSNet checkpoint is absent, `torchreid` downloads pretrained weights on first startup.

## Build and start the dashboard

Run these commands from the directory containing the `SHIELD` folder. Build the frontend before launching the API so FastAPI has static dashboard files to serve.

```powershell
cd SHIELD\Backend
python -m venv ..\venv
..\venv\Scripts\python.exe -m pip install --upgrade pip
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe scripts\download_models.py
cd ..\Frontend
npm.cmd install
npm.cmd run build
cd ..\Backend
```

Choose one of these commands from `SHIELD\Backend` to start the API and live pipeline.

**CPU mode** (works without a CUDA-capable GPU, but processes more slowly):

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cpu
```

**CUDA setup:** the general requirements can install a CPU-only PyTorch build. From `SHIELD\Backend`, install the CUDA build and verify it detects the GPU:

```powershell
..\venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
..\venv\Scripts\python.exe scripts\check_gpu.py
```

`check_gpu.py` should print your GPU name and `OK - ready`. The cu124 wheel (verified on an RTX 3050 6 GB, driver 581.95) works with any NVIDIA driver that reports CUDA ≥ 12.4 in `nvidia-smi`. If you request `--device cuda` without a working CUDA build, SHIELD prints a warning and falls back to CPU instead of crashing.

**Live cameras (phones + webcam):** edit the phone IPs in `Backend/config/cameras.live.yaml`, then start with `--cameras config/cameras.live.yaml`. Supported sources: webcam index (`0`, `1`), `http(s)://` MJPEG (phone apps such as IP Webcam: `http://<phone-ip>:8080/video`), `rtsp://` IP cameras/NVRs, and video files. Live sources reconnect automatically and never block the other cameras.

**Download locations:** all model weights downloaded at runtime (OSNet, EasyOCR, Ultralytics assets) are stored in `Backend/.cache/` (git-ignored), never in your home folder; override with the `SHIELD_CACHE_DIR` environment variable. On the reference machine pip and npm caches are redirected to `F:\HS202\.cache` via `pip config --user set global.cache-dir` and `npm config set cache`.

On GPU, YOLO runs in FP16 and ANPR OCR runs on a background thread (`anpr.async_ocr`). Throughput on an RTX 3050 with 3 cameras: `detection.imgsz` 1280 ≈ 4.0 FPS/camera, 960 (default) ≈ 5.4, 640 ≈ 7.9.

**CUDA mode** (for a supported NVIDIA GPU with CUDA-enabled PyTorch):

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cuda
```

Open [http://localhost:8000](http://localhost:8000). The API starts the live pipeline from `Backend/config/cameras.yaml`; the dashboard bundle is served from `Backend/static`. Change camera sources in that YAML file before starting the API. If CUDA initialization fails, use the CPU command.

## Frontend development mode

Start the API in a first PowerShell terminal from `SHIELD\Backend`. Choose either CPU or CUDA mode:

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cpu
```

For CUDA, first install and verify the CUDA-enabled PyTorch build using the command above, then use this instead:

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cuda
```

Start Vite in a second terminal from `SHIELD\Frontend`:

```powershell
npm.cmd run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the terminal, normally [http://localhost:5173](http://localhost:5173). Vite proxies `/api` and `/ws` traffic to FastAPI on port `8000`. Re-run `npm.cmd run build` after frontend changes to update the production dashboard served by FastAPI.

## Standalone computer-vision mode

To run the pipeline with OpenCV display windows and without the web dashboard, run from `SHIELD\Backend`:

```powershell
..\venv\Scripts\python.exe run_shield.py --device cpu
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/system/status` | Runtime health, processing FPS, cameras, tracks, and uptime |
| `GET` | `/api/v1/streams/active` | Camera status, resolution, and configured source |
| `GET` | `/api/v1/streams/{camera_id}/snapshot` | Latest annotated JPEG frame |
| `POST` | `/api/v1/streams/{camera_id}/control` | Start or stop a camera stream |
| `GET` | `/api/v1/identities` | Tracked global identities |
| `GET` | `/api/v1/identities/{global_id}` | Identity details and journey |
| `GET` | `/api/v1/identities/{global_id}/journey` | Camera movement timeline |
| `POST` | `/api/v1/identities/register` | Register an identity from a base64 image |
| `GET` | `/api/v1/alerts` | Recent detected events |
| `GET` | `/api/v1/risk/summary` | Per-camera risk summary |
| `POST` | `/api/v1/fence/config` | Set a two-point tripwire for one camera or all cameras |
| `POST` | `/api/v1/faces/enroll` | Enrol a face: `{name, image \| images[], category, notes}` (base64 or data-URL) |
| `GET` | `/api/v1/faces/gallery` | Enrolled identities with thumbnails and engine stats |
| `DELETE` | `/api/v1/faces/{identity_id}` | Remove an enrolled identity |
| `POST` | `/api/v1/faces/search` | Detect and identify every face in an uploaded image |
| `GET` / `POST` | `/api/v1/anpr/watchlist` | List / add plate watchlist entries `{plate, tag, notes}` |
| `DELETE` | `/api/v1/anpr/watchlist/{plate}` | Remove a watchlisted plate |
| `GET` | `/api/v1/anpr/plates` | Recently confirmed plates |
| `POST` | `/api/v1/anpr/recognize` | One-shot plate read from an uploaded image |
| `GET` | `/api/v1/events` | Persistent event log (survives restarts), filter by `alert_type` |

| WebSocket | Purpose |
|---|---|
| `/ws/telemetry` | Live frame telemetry, tracks, scene risk, and alerts |
| `/ws/stream/{camera_id}` | Binary JPEG frames with annotation overlays for a camera |

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration and integration notes

- `Backend/config/config.yaml` controls the model, processing, tracking, stream, risk, and API settings, plus the `face_recognition`, `anpr`, `night_vision`, `alerts`, `logging` and `integrations` sections.
- Watchlists are managed from the dashboard (**Watchlists** button) or the API. They persist in `Backend/data/face_gallery.json` and `Backend/data/plate_watchlist.json`; `Backend/data/` is git-ignored because it holds biometric data.
- **Command & Control integration:** set `integrations.webhook.url` to POST every alert at or above `min_severity` as JSON (`{"source": "SHIELD-IBVAP", "event": {...}}`) to an existing C&C system. `integrations.site_id` is stamped on every alert. Systems can also poll `/api/v1/events` or subscribe to `/ws/telemetry`.
- The face models (YuNet + SFace ONNX, ~39 MB) are downloaded and SHA-256 verified by `scripts/download_models.py`. Without them, FRS falls back to Haar face detection only and recognition is disabled. Set `face_recognition.backend: facenet` and `pip install facenet-pytorch` for 512-d FaceNet embeddings (re-enrol faces after switching backends).
- Recognition thresholds: a face matches when cosine distance ≤ `match_distance_threshold` (0.60) and it beats the next identity by `margin`. Lower the threshold to reduce false alarms; enrol several photos per person to improve recall.
- `Backend/config/cameras.yaml` defines the camera IDs, names, and video or RTSP sources.
- The fence editor uses two endpoints because the current risk analyzer implements a tripwire, not a polygon.
- The REST API and dashboard share the same origin in production. During development, Vite proxies the API and both WebSocket paths.

## Tests

```powershell
cd SHIELD\Backend
..\venv\Scripts\python.exe -m pytest tests -q
```

Heavy dependencies (YOLO, torch, torchreid, EasyOCR) are stubbed so the suite runs anywhere; the face-recognition integration tests use the real ONNX models when they are present.

## What the commands do

1. `cd SHIELD\Backend` enters the backend folder; the pipeline expects model and video paths relative to it.
2. `python -m venv ..\venv` creates a Python virtual environment beside the project folder.
3. `..\venv\Scripts\python.exe -m pip install --upgrade pip` updates pip in that environment.
4. `..\venv\Scripts\python.exe -m pip install -r requirements.txt` installs the backend, ML, API, and test dependencies.
4b. `..\venv\Scripts\python.exe scripts\download_models.py` downloads and verifies the face detection/recognition models.
5. `cd ..\Frontend` enters the React/Vite app.
6. `npm.cmd install` installs the frontend packages recorded in `package-lock.json`.
7. `npm.cmd run build` type-checks and builds the dashboard into `Backend/static`.
8. `cd ..\Backend` returns to the backend folder for startup.
9. `..\venv\Scripts\python.exe run_shield.py --api --device cpu` starts FastAPI and the live camera analytics pipeline in headless CPU mode.
10. `..\venv\Scripts\python.exe run_shield.py --api --device cuda` starts the same API and pipeline using CUDA acceleration.
11. `..\venv\Scripts\python.exe run_shield.py --device cpu` runs the standalone pipeline with OpenCV display windows.
12. `npm.cmd run dev -- --host 127.0.0.1` runs Vite locally and proxies backend requests to port `8000`.
