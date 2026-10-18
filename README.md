# S.H.I.E.L.D.

AI-powered CCTV analytics project with a React command center and a Python computer-vision backend.

## Current integration status

The React dashboard is served by FastAPI at `/`. It connects to `/api/v1/streams`, `/api/v1/status`, `/api/v1/config/fence`, and `/ws/telemetry` using same-origin requests.

The current API is a **demo integration**: its WebSocket sends simulated telemetry for one camera, and its stream list reports the three configured cameras. It does not yet share a process with the real `Orchestrator`, provide live video, or apply fence changes. The fence endpoint returns `501` so the UI cannot mistakenly report a configuration as applied. The dashboard can normalize both the requested `TelemetryFrame` format and the orchestrator's current telemetry format when that telemetry is connected.

## Prerequisites

- Windows PowerShell
- Python 3.10 (the commands below use the `py` launcher)
- Node.js with npm

The Python requirements include the machine-learning and computer-vision dependencies and can take time and disk space to install. The frontend dependencies are installed separately with npm.

## Build and start the dashboard

Run these commands from the directory containing the `SHIELD` folder. The backend command runs in the foreground; keep that terminal open while using the dashboard.

```powershell
cd SHIELD
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
cd Frontend
npm install
npm run build
cd ..\Backend
..\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
```

Open [http://localhost:8080](http://localhost:8080). The frontend build is written to `Backend/static`, which FastAPI serves from `/`.

## Frontend development mode

First start FastAPI in one terminal using the final two commands from the previous section. Then open a second PowerShell terminal and run:

```powershell
cd SHIELD\Frontend
npm run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the terminal, normally [http://localhost:5173](http://localhost:5173). Vite proxies `/api` and `/ws` requests to FastAPI on port `8080`. After changing frontend code, run `npm run build` to refresh the production bundle served by FastAPI.

## Run the standalone computer-vision pipeline

The pipeline launcher is separate from the API/dashboard. It opens the configured camera sources and displays OpenCV output; it does not send its telemetry to the dashboard.

```powershell
cd SHIELD\Backend
..\venv\Scripts\python.exe run_shield.py
```

The current checkout includes `models/yolo11m.pt`, but the standalone pipeline also imports `torchreid` (not listed in `requirements.txt`) and expects OSNet weights at `models/osnet_x1_0_msmt17.pth` (not included in the checkout). Install that dependency and provide the model file before starting the standalone pipeline. Connecting this pipeline to the web dashboard requires wiring its telemetry callback into the FastAPI WebSocket and sharing runtime stream and fence state.

## What the commands do

1. `cd SHIELD` moves to the project root so `requirements.txt`, `Frontend`, and `Backend` paths resolve.
2. `py -3.10 -m venv venv` creates the Python virtual environment used by the backend.
3. `.\venv\Scripts\python.exe -m pip install --upgrade pip` updates pip inside that environment.
4. `.\venv\Scripts\python.exe -m pip install -r requirements.txt` installs the Python API, computer-vision, and ML dependencies.
5. `cd Frontend` moves to the React/Vite project.
6. `npm install` installs the frontend packages from `package-lock.json`.
7. `npm run build` type-checks and bundles the frontend into `Backend/static` for FastAPI to serve.
8. `cd ..\Backend` moves to the directory where the API module is located.
9. `..\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080` starts the API and dashboard on local port `8080`.
10. `cd SHIELD\Frontend` moves to the frontend project in the development terminal.
11. `npm run dev -- --host 127.0.0.1` starts Vite's development server, with API and WebSocket traffic proxied to FastAPI.
12. `cd SHIELD\Backend` moves to the backend directory required by the pipeline's relative config and model paths.
13. `..\venv\Scripts\python.exe run_shield.py` starts the standalone OpenCV/ML pipeline; it is not currently the dashboard's telemetry source.
