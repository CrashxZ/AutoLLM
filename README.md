# AutoLLM – CARLA AI-in-the-Loop Dashboard

AutoLLM couples a FastAPI backend that orchestrates a CARLA simulation with a React/Tailwind dashboard for monitoring, control, and LLM-assisted intent generation. The project delivers live telemetry, multiple video transport modes (MJPEG + WebRTC), dataset capture, and configurable simulation resets from the browser.

## Repository Layout
- `client/web/` – Vite + React + Tailwind web client (`package.json`, `src/`, `vite.config.js`)
- `server/` – FastAPI backend, CARLA integration, WebRTC helpers, and WebSocket routes
- `data/` – Session logs (`telemetry.jsonl`, captured frames) generated at runtime
- `scripts/`, `old/` – Ancillary utilities or legacy code

## Client (React Dashboard)
- Entry: `client/web/src/main.jsx` mounts the modern `Dashboard` view while keeping legacy layouts (`App`, `WebRTCDashboard`) around for comparison.
- Core layout: `client/web/src/components/Dashboard.jsx` renders telemetry, control panel, drawered configuration, and intent tooling around an MJPEG `<img>` stream.
- Hooks:
  - `client/web/src/hooks/useWebSocket.js` maintains a resilient WS connection, normalises outgoing commands, and exposes `telemetry`, `vehicles`, and `sendCommand`.
  - `client/web/src/hooks/useFrameCapture.js` pulls `/frame/...` JPEG snapshots, pairs them with telemetry, and exports ZIP archives (JSONL + CSV + images).
  - `client/web/src/hooks/useWebRTC.js` negotiates WebRTC offers with the backend for low-latency video (`useRef`-driven `<video>` element via `WebRTCVideo.jsx`).
- Control components:
  - `ControlPanel.jsx` (MJPEG view) exposes manual vehicle commands, mode toggles, and view switching.
  - `ControlPanelWebRTC.jsx` adapts the panel to WebRTC renegotiation flow.
- Intent tooling:
  - `IntentPanel.jsx` lets operators run a rule policy or call OpenAI (if a key is provided) via `agents/llmAgent.js:proposeAction`.
  - `IntentGeneratorPanel.jsx` periodically captures top-down imagery and submits multi-vehicle prompts to OpenAI, persisting results per vehicle.
- Styling: Tailwind is configured in `tailwind.config.js`; shared classes live in `src/index.css`.

## Server (FastAPI + CARLA)
- Entry: `server/server.py` boots FastAPI, connects to CARLA, manages synchronous ticking (`tick_loop`), spawns vehicles/sensors on `/config`, and tears down on `/reset`.
- Telemetry: `telemetry_snapshot()` (every ~100 ms) gathers pose, lane metadata, speed, and FSM lane-change state for each vehicle, streaming results through WebSocket broadcasts and logging to `data/logs/<timestamp>/telemetry.jsonl`.
- Video:
  - MJPEG endpoints `/video/{veh_id}`, `/video_top/{veh_id}`, `/video_global` stream the latest JPEG buffers maintained in `Session.buffers`.
  - WebRTC endpoints `/webrtc/offer` and `/webrtc/close` leverage `server/webrtc.py:CarlaVideoTrack` to reuse the same JPEG buffers over H.264/VP8.
- Commands: `handle_command()` converts high-level WS payloads (speed, lane, intent) into CARLA Traffic Manager calls or direct vehicle controls. The FSM in `server/lane_change.py:LaneChangeController` handles safe TM lane changes.
- WebSocket routes: `server/ws_routes.py` registers `/ws_ui` and legacy `/ws`, funnels telemetry, optional JPEG frames, and normalises inbound commands before enqueuing them to `SESSION.cmd_queue`.
- Session management: `/archive` produces a ZIP of the current `data/logs` session; `/telemetry.jsonl` exposes the latest log slice.

## Data & Control Flow
- WebSocket (`/ws_ui`): The client subscribes via `useWebSocket`, receiving `{type:"telemetry"}` payloads about 10 Hz. Outgoing actions (manual commands, intent approvals, mode switches) are queued locally until the socket is open and then forwarded to the server command queue.
- Video:
  - MJPEG `<img>` tags hit `/video*` endpoints and rely on continuous JPEG buffer updates from CARLA camera callbacks.
  - WebRTC clients post an SDP offer to `/webrtc/offer`; the server answers with an SDP that wraps `CarlaVideoTrack`, pulling the same JPEG buffers and respecting optional bitrate/fps parameters.
- Frame capture (`useFrameCapture`): On demand, the client polls `/frame/top/{veh_id}.jpg` (or other views) and combines imagery with last-seen telemetry for offline datasets.
- LLM intent loop: `IntentPanel` or `IntentGeneratorPanel` call OpenAI’s Chat Completions API in-browser (if an API key is provided). Approved intents are relayed back to the server as `{cmd:"intent", intent:{...}}`, which the backend maps to TM speed/lane adjustments.

## Getting Started

### Prerequisites
- CARLA 0.9.15 simulator accessible from the server host (default `127.0.0.1:2000`)
- Python 3.10+ for the backend
- Node.js 18+ and npm for the frontend
- Optional: OpenAI API key for LLM-assisted intent features

### Backend Setup
```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Set any CARLA environment overrides as needed (`CARLA_DIR`, `CARLA_HOST`, `CARLA_PORT`). Start the server (after launching CARLA):
```bash
uvicorn server.server:app --host 0.0.0.0 --port 8000
```
The server will automatically keep CARLA in synchronous mode, manage vehicle sensors, and serve both REST and WebSocket endpoints.

### Frontend Setup
```bash
cd client/web
npm install
npm run dev
```
Vite defaults to `http://localhost:5173`, proxying API/WebSocket calls to the backend. Configure `.env` variables (e.g., `VITE_API_BASE`, `VITE_WS_URL`) if the backend runs on a different host/port.

## Key Endpoints & Commands
- REST:
  - `POST /config` – spawn vehicles with desired speeds and spawn indices
  - `POST /reset` – destroy actors and clear buffers/logs
  - `GET /spawns` – list available spawn points (with coordinates)
  - `GET /archive` – download the current session log archive
- WebSocket `/ws_ui` message types:
  - Telemetry: `{type:"telemetry", payload:{<veh_id>:{...}}}`
  - Command aliases accepted: `speed_up`, `slow_down`, `lane_left`, `lane_right`, `mode`, `select`, `intent`, `brake`, `release`
  - Acknowledgements: `{type:"ack"}`, `{type:"nack"}`
- Video: `/video/{veh_id}`, `/video_top/{veh_id}`, `/video_global`, `/frame/front/{veh_id}.jpg`, `/frame/top/{veh_id}.jpg`, `/frame/global.jpg`, `/webrtc/offer`

## Additional Notes
- `client/web/src/App.jsx` contains legacy scaffold code and duplicate hook declarations; use `Dashboard.jsx` or `WebRTCDashboard.jsx` as the authoritative client entry points.
- Two FastAPI entry modules (`server/server.py`, `server/server2.py`) exist; `server/server.py` is the current default and should be edited preferentially.
- Session artifacts (frames + telemetry) accumulate under `data/logs/<timestamp>/`; ensure sufficient disk space for long captures.

---

With both services running, point a browser at the Vite dev server (or the built `client/web/dist` served directly by FastAPI) to monitor CARLA, issue manual or LLM-assisted commands, and export curated driving datasets.
