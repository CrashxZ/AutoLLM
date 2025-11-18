# scripts/run_server.sh
#!/usr/bin/env bash
#
# CARLA AI-in-the-Loop: FastAPI Server Launcher
# ---------------------------------------------
# Purpose:
#   - Exports CARLA env vars
#   - Waits for CARLA (localhost:2000)
#   - Starts uvicorn serving server/server.py
#
# Usage:
#   chmod +x scripts/run_server.sh
#   ./scripts/run_server.sh
#
# Optional environment variables:
#   CARLA_HOST=127.0.0.1
#   CARLA_PORT=2000
#   CARLA_TIMEOUT=10.0
#   CARLA_DIR=/opt/CARLA_0.9.15          # path to CARLA root (adds PythonAPI egg to PYTHONPATH)
#   API_PORT=8000                        # FastAPI server port
#   UVICORN_RELOAD=false                 # true/false (dev hot-reload)

set -euo pipefail

# ---------- Resolve repo root ----------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

# ---------- Defaults ----------
export CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
export CARLA_PORT="${CARLA_PORT:-2000}"
export CARLA_TIMEOUT="${CARLA_TIMEOUT:-10.0}"
API_PORT="${API_PORT:-8000}"
UVICORN_RELOAD="${UVICORN_RELOAD:-false}"

# ---------- Optional: add CARLA egg to PYTHONPATH ----------
if [[ -n "${CARLA_DIR:-}" ]]; then
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="win-amd64"
  else
    PLATFORM="linux-x86_64"
  fi
  EGG_PATH=$(ls -1 "${CARLA_DIR}/PythonAPI/carla/dist"/carla-*${PY_VER}-${PLATFORM}.egg 2>/dev/null | head -n 1 || true)
  if [[ -n "$EGG_PATH" ]]; then
    export PYTHONPATH="${EGG_PATH}:${CARLA_DIR}/PythonAPI/carla:${PYTHONPATH:-}"
    echo "[INFO] Added CARLA egg to PYTHONPATH:"
    echo "       ${EGG_PATH}"
  else
    echo "[WARN] Could not find CARLA egg in ${CARLA_DIR}/PythonAPI/carla/dist for Python ${PY_VER} (${PLATFORM})"
  fi
fi

# ---------- Check deps quickly ----------
need() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing dependency: $1"; exit 1; }; }
need python3
need uvicorn

# ---------- Wait for CARLA server ----------
echo "[INFO] Waiting for CARLA at ${CARLA_HOST}:${CARLA_PORT} ..."
ATTEMPTS=60
SLEEP=1
i=0
while ! (exec 3<>/dev/tcp/${CARLA_HOST}/${CARLA_PORT}) 2>/dev/null; do
  ((i++)) || true
  if [[ "$i" -ge "$ATTEMPTS" ]]; then
    echo "[ERROR] CARLA did not respond on ${CARLA_HOST}:${CARLA_PORT} after $((ATTEMPTS*SLEEP))s."
    echo "        Start CARLA first, e.g.:"
    echo "        ${CARLA_DIR:-/opt/CARLA}/CarlaUE4.sh -quality-level=Low -RenderOffScreen"
    exit 1
  fi
  sleep "$SLEEP"
done
exec 3>&-

echo "[INFO] CARLA reachable. Launching FastAPI (port ${API_PORT}) ..."
# ---------- Launch uvicorn ----------
if [[ -f "server/server.py" ]]; then
  APP_PATH="server.server:app"
elif [[ -f "server.py" ]]; then
  APP_PATH="server:app"
else
  echo "[ERROR] Cannot locate server/server2.py. Run from repo root (carla-ai-loop/)."
  exit 1
fi

if [[ "${UVICORN_RELOAD}" == "true" ]]; then
  uvicorn "${APP_PATH}" --host 0.0.0.0 --port "${API_PORT}" --reload
else
  uvicorn "${APP_PATH}" --host 0.0.0.0 --port "${API_PORT}"
fi
