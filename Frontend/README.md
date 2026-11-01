# SHIELD Frontend

React, TypeScript, and Vite dashboard for the FastAPI video analytics backend.

## Development

Start FastAPI from `SHIELD\Backend` in one PowerShell terminal. Use CPU mode:

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cpu
```

For a supported NVIDIA GPU, first install CUDA-enabled PyTorch as described in the root README, then start it with:

```powershell
..\venv\Scripts\python.exe run_shield.py --api --device cuda
```

Then run Vite from this directory in a second terminal:

```powershell
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

Open the local URL Vite prints, normally `http://localhost:5173`. The Vite configuration proxies API requests and telemetry/video WebSockets to FastAPI at `localhost:8000`.

## Production build

Run from this directory:

```powershell
npm.cmd run build
```

Vite writes the bundle to `Backend/static`. Start the API from `SHIELD\Backend` with `..\venv\Scripts\python.exe run_shield.py --api --device cpu` and open `http://localhost:8000`.

The dashboard consumes `/api/v1/system/status`, `/api/v1/streams/active`, `/api/v1/fence/config`, `/ws/telemetry`, and `/ws/stream/{camera_id}`. Fence configuration is a two-point tripwire matching the current `RiskAnalyzer`. See the root README for setup, the full endpoint list, and command explanations.
