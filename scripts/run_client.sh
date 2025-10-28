# scripts/run_client.sh
#!/usr/bin/env bash
#
# CARLA AI-in-the-Loop: Web Client Launcher (Vite + React + Tailwind)
# -------------------------------------------------------------------
# Purpose:
#   - Installs NPM dependencies (if needed)
#   - Starts the Vite dev server for the dashboard UI
#
# Usage:
#   chmod +x scripts/run_client.sh
#   ./scripts/run_client.sh
#
# Optional environment variables:
#   PORT=5173                      # Vite dev server port
#   VITE_WS_URL=ws://localhost:8000/ws_ui
#   VITE_API_BASE=http://localhost:8000
#   OPEN_BROWSER=true              # open default browser automatically
#   NODE_ENV=development
#
# Notes:
#   - Run from the repo root (carla-ai-loop/)
#   - Make sure the Python server is running (scripts/run_server.sh)
#   - For Vercel, the app is built via `npm run build` in client/web/

set -euo pipefail

# ---- Resolve paths ----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
WEB_DIR="${REPO_ROOT}/client/web"

cd "${REPO_ROOT}"

# ---- Defaults ----
export PORT="${PORT:-5173}"
export VITE_WS_URL="${VITE_WS_URL:-ws://localhost:8000/ws_ui}"
export VITE_API_BASE="${VITE_API_BASE:-http://localhost:8000}"
export NODE_ENV="${NODE_ENV:-development}"
OPEN_BROWSER="${OPEN_BROWSER:-false}"

# ---- Dep checks ----
need() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing dependency: $1"; exit 1; }; }
need node
need npm

# ---- Install deps if needed ----
cd "${WEB_DIR}"
if [[ ! -d "node_modules" ]]; then
  echo "[INFO] Installing web client dependencies ..."
  npm install
fi

# ---- Show effective config ----
echo "[INFO] Launching web client:"
echo "       DIR           : ${WEB_DIR}"
echo "       PORT          : ${PORT}"
echo "       VITE_WS_URL   : ${VITE_WS_URL}"
echo "       VITE_API_BASE : ${VITE_API_BASE}"
echo "       OPEN_BROWSER  : ${OPEN_BROWSER}"

# ---- Optionally open browser after server starts ----
open_browser_after_start() {
  local url="http://localhost:${PORT}"
  # Wait a bit for Vite to boot
  for i in {1..30}; do
    if curl -sSf "${url}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" || true
  elif command -v open >/dev/null 2>&1; then
    open "${url}" || true
  else
    echo "[INFO] Please open ${url} in your browser."
  fi
}

if [[ "${OPEN_BROWSER}" == "true" ]]; then
  open_browser_after_start &
fi

# ---- Start Vite dev server ----
# Expose env so Vite can pick up VITE_* vars
export VITE_WS_URL VITE_API_BASE
npm run dev -- --port "${PORT}"
