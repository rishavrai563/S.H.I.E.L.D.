# SHIELD Frontend

React + TypeScript + Vite dashboard. Vite writes its production build into `Backend/static`, which FastAPI serves at `/`.

## Development

Start the API in one PowerShell terminal from `SHIELD/Backend`:

```powershell
..\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
```

Then run these commands from this directory in a second terminal:

```powershell
npm install
npm run dev -- --host 127.0.0.1
```

Open the Vite URL printed in the terminal, normally `http://localhost:5173`. The Vite config proxies API and WebSocket requests to FastAPI on port `8080`.

## Production bundle

```powershell
npm run build
```

Then start FastAPI from `SHIELD/Backend` with the Uvicorn command above and open `http://localhost:8080`.

The dashboard understands both the requested `TelemetryFrame` shape and the legacy `streams`/`alerts` shape currently written by `Orchestrator`. The API currently labels its WebSocket feed as simulated, and live camera video is not exposed, so camera cards show telemetry overlays on a placeholder. The fence endpoint returns `501` until it can apply changes to a running risk engine; that engine currently supports a two-point tripwire rather than a polygon. See the root README for the current integration status and complete setup commands.
