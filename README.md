# S.H.I.E.L.D.

SHIELD is a multi-camera video analytics dashboard backed by a Python computer-vision pipeline. FastAPI starts the configured camera streams and Orchestrator in a worker thread, then serves telemetry, camera frames, and REST APIs to the React dashboard.

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
..\venv\Scripts\python.exe -m pip install --force-reinstall torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
..\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

The verification command should print `True`. This CUDA wheel is large; see the [official PyTorch install commands](https://pytorch.org/get-started/previous-versions/) for other supported versions.

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

| WebSocket | Purpose |
|---|---|
| `/ws/telemetry` | Live frame telemetry, tracks, scene risk, and alerts |
| `/ws/stream/{camera_id}` | Binary JPEG frames with annotation overlays for a camera |

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration and integration notes

- `Backend/config/config.yaml` controls the model, processing, tracking, stream, risk, and API settings.
- `Backend/config/cameras.yaml` defines the camera IDs, names, and video or RTSP sources.
- The fence editor uses two endpoints because the current risk analyzer implements a tripwire, not a polygon.
- The REST API and dashboard share the same origin in production. During development, Vite proxies the API and both WebSocket paths.

## What the commands do

1. `cd SHIELD\Backend` enters the backend folder; the pipeline expects model and video paths relative to it.
2. `python -m venv ..\venv` creates a Python virtual environment beside the project folder.
3. `..\venv\Scripts\python.exe -m pip install --upgrade pip` updates pip in that environment.
4. `..\venv\Scripts\python.exe -m pip install -r requirements.txt` installs the backend, ML, API, and test dependencies.
5. `cd ..\Frontend` enters the React/Vite app.
6. `npm.cmd install` installs the frontend packages recorded in `package-lock.json`.
7. `npm.cmd run build` type-checks and builds the dashboard into `Backend/static`.
8. `cd ..\Backend` returns to the backend folder for startup.
9. `..\venv\Scripts\python.exe run_shield.py --api --device cpu` starts FastAPI and the live camera analytics pipeline in headless CPU mode.
10. `..\venv\Scripts\python.exe run_shield.py --api --device cuda` starts the same API and pipeline using CUDA acceleration.
11. `..\venv\Scripts\python.exe run_shield.py --device cpu` runs the standalone pipeline with OpenCV display windows.
12. `npm.cmd run dev -- --host 127.0.0.1` runs Vite locally and proxies backend requests to port `8000`.
