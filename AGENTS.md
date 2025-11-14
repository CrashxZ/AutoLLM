# Repository Guidelines

## Project Structure & Module Organization
- `client/web/` hosts the Vite + React dashboard; entry point `src/main.jsx` mounts the modern `Dashboard` layout, while legacy agents live under `src/agents/`.
- `server/` contains the FastAPI backend; `server.py` is the default ASGI app, with telemetry helpers in `lane_change.py`, `webrtc.py`, and `ws_routes.py`.
- `data/` is created at runtime for session archives (telemetry logs, captured frames); keep it out of commits.
- `scripts/` and `old/` store one-off utilities and deprecated prototypes—scan before reusing any snippet.

## Build, Test, and Development Commands
- `cd server && uvicorn server.server:app --reload` launches the API against a running CARLA instance.
- `cd client/web && npm run dev` serves the dashboard with Vite’s proxy towards the backend.
- `npm run build` produces the production bundle in `client/web/dist`; serve it with the FastAPI static mounts when packaging.
- `npm run lint` and `cd server && python -m pytest` should be clean before opening a pull request; add `pytest.ini` markers if you introduce slow CARLA fixtures.

## Coding Style & Naming Conventions
- Frontend code uses 2-space indentation, functional React components, and camelCase filenames (`useWebSocket.js`, `IntentPanel.jsx`). Favor small hooks under `src/hooks/` and colocate UI state in zustand stores.
- Tailwind utility classes live in JSX; share global tokens through `src/index.css`.
- Python modules follow snake_case names; format with `black` (line length 88) and import-sort with `isort`. Prefer explicit typing on public functions to ease FastAPI response models.

## Testing Guidelines
- Backend tests belong in `server/tests/` with filenames like `test_ws_routes.py`; mock CARLA clients to keep runs under 30 seconds.
- Frontend tests can live in `client/web/src/__tests__/` using Vitest + React Testing Library (add via `npm install -D vitest @testing-library/react` when you create the first spec).
- Record coverage deltas in the PR description if you touch critical telemetry or command routing.

## Commit & Pull Request Guidelines
- Write imperative, single-sentence commit subjects mirroring the existing history (“Add CARLA server setup…”). Include focused commits per feature or fix.
- Reference linked issues in the commit body or PR description and attach dashboards or terminal captures that demonstrate telemetry, video, and intent flows.
- PRs should describe CARLA setup assumptions, new `.env` keys (`VITE_API_BASE`, `CARLA_HOST`), and any migration steps for `data/` artifacts.
