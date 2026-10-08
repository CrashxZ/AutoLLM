# server/ws_routes.py
"""
WebSocket routes for the React dashboard UI

Endpoints:
  /ws_ui (primary)
  /ws     (legacy alias)

Features:
  - Periodic telemetry (~10 Hz): {type:"telemetry", payload:{...}}
  - Periodic base64 JPEG frames for ego & global:
      {type:"frame", camera:"ego"|"global", base64:"..."}
  - Receives commands from UI in either shape:
      A) { "cmd": "...", ... }                      # plain (preferred by client)
      B) { "type": "...", ... }                     # legacy typed
    → Normalizes and enqueues into SESSION.cmd_queue

Also registers each WS with session.ws_clients and assigns a per-WS
asyncio.Queue at ws._send_queue so server.broadcast_ws_json() can deliver:
  - {type:"veh_list", veh_ids:[...]}
  - {type:"reset", ts:...}
  - {type:"approval", ...}
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Callable, Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Rates (Hz)
TELEMETRY_HZ = 10
EGO_FRAME_HZ = 6
GLOBAL_FRAME_HZ = 3


def register_ws_routes(
    app,
    telemetry_fn: Callable[[], Dict[str, Any]],
    session: Any,
    enqueue_fn: Callable[[Dict[str, Any]], None],
):
    """Register WebSocket endpoints onto an existing FastAPI app."""

    async def _sender(ws: WebSocket, q: "asyncio.Queue[str]"):
        try:
            while True:
                payload = await q.get()
                if ws.application_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(payload)
        except Exception:
            pass

    async def _heartbeat(ws: WebSocket):
        try:
            while True:
                await asyncio.sleep(10.0)
                if ws.application_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(json.dumps({"type": "ping", "t": time.time()}))
        except Exception:
            pass

    async def _handle_messages(ws: WebSocket):
        """Receive messages from client, accept either {cmd:...} or {type:...}."""
        while True:
            msg = await ws.receive_text()
            try:
                obj = json.loads(msg)
            except Exception:
                continue

            # ---- Plain command path: {cmd: ...} ----
            if "cmd" in obj or "intent" in obj:
                norm = _normalize_plain_command(obj)
                if norm:
                    try:
                        enqueue_fn(norm)
                        await ws.send_text(json.dumps({"type": "ack", "t": time.time()}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "nack", "error": str(e)}))
                continue

            # ---- Typed path: {type: ...} ----
            mtype = (obj.get("type") or "").lower()

            if mtype == "mode":
                mode = (obj.get("control_mode") or "USER").upper()
                enqueue_fn({"cmd": "mode", "control_mode": mode})

            elif mtype == "set_goal":
                goal = str(obj.get("goal", "")).strip()
                if goal:
                    enqueue_fn({"cmd": "set_goal", "goal": goal})

            elif mtype == "select_vehicle":
                try:
                    vid = int(obj.get("veh_id"))
                    enqueue_fn({"cmd": "select", "veh_id": vid})
                except Exception:
                    pass

            elif mtype == "control":
                # {"type":"control","veh_id": id, "action":{"speed_kmh":..,"lane_cmd":"left|right|none","brake":bool}}
                try:
                    vid = int(obj.get("veh_id"))
                    action = obj.get("action") or {}
                    brake = bool(action.get("brake", False))
                    if brake:
                        enqueue_fn({"cmd": "brake", "veh_id": vid})
                    else:
                        if "speed_kmh" in action:
                            enqueue_fn({"cmd": "speed", "veh_id": vid, "kmh": float(action["speed_kmh"])})
                        lane = (action.get("lane_cmd") or "none").lower()
                        if lane in ("left", "right"):
                            enqueue_fn({"cmd": "lane", "veh_id": vid, "dir": lane})
                except Exception:
                    pass

            elif mtype == "approve_action":
                try:
                    vid = int(obj.get("veh_id"))
                    act = obj.get("action") or {}
                    brake = bool(act.get("brake", False))
                    if brake:
                        enqueue_fn({"cmd": "brake", "veh_id": vid})
                    else:
                        if "speed_kmh" in act:
                            enqueue_fn({"cmd": "speed", "veh_id": vid, "kmh": float(act["speed_kmh"])})
                        lane = (act.get("lane_cmd") or "none").lower()
                        if lane in ("left", "right"):
                            enqueue_fn({"cmd": "lane", "veh_id": vid, "dir": lane})
                    await ws.send_text(json.dumps({"type": "approved", "veh_id": vid, "ts": time.time(), "action": act}))
                except Exception:
                    pass

            elif mtype == "deny_action":
                try:
                    vid = int(obj.get("veh_id"))
                    await ws.send_text(json.dumps({"type": "denied", "veh_id": vid, "ts": time.time()}))
                except Exception:
                    pass

            elif mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": time.time()}))
            else:
                # unknown message; ignore
                pass

    def _normalize_plain_command(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map friendly client tokens to server-native schema used by handle_command()."""
        cmd = (obj.get("cmd") or "").lower()

        # direct passthroughs
        if cmd in ("mode", "select", "speed", "lane", "brake", "release", "intent"):
            return obj

        # aliases
        if cmd in ("lane_left", "lane_right"):
            return {
                "cmd": "lane",
                "veh_id": obj.get("veh_id"),
                "dir": "left" if "left" in cmd else "right",
            }

        if cmd in ("speed_up", "slow_down"):
            return {
                "cmd": "intent",
                "intent": {
                    "ego_veh_id": obj.get("veh_id"),
                    "ego_action": "speed up" if cmd == "speed_up" else "slow down",
                },
            }

        # unknown -> None
        return None

    async def _telemetry_tx(ws: WebSocket):
        period = 1.0 / max(1, int(TELEMETRY_HZ))
        try:
            while True:
                t0 = time.time()
                tele = telemetry_fn()
                await ws.send_text(json.dumps({"type": "telemetry", "payload": tele}, separators=(",", ":")))
                dt = time.time() - t0
                await asyncio.sleep(max(0.0, period - dt))
        except asyncio.CancelledError:
            pass

    async def _ego_frame_tx(ws: WebSocket):
        period = 1.0 / max(1, int(EGO_FRAME_HZ))
        try:
            while True:
                t0 = time.time()
                veh_id = getattr(session, "selected_vehicle_id", None)
                jpg = None
                if veh_id is not None:
                    with session.buffers.lock:
                        jpg = session.buffers.front_last_jpeg.get(veh_id)
                if jpg:
                    b64 = base64.b64encode(jpg).decode("ascii")
                    await ws.send_text(json.dumps({"type": "frame", "camera": "ego", "base64": b64}))
                dt = time.time() - t0
                await asyncio.sleep(max(0.0, period - dt))
        except asyncio.CancelledError:
            pass

    async def _global_frame_tx(ws: WebSocket):
        period = 1.0 / max(1, int(GLOBAL_FRAME_HZ))
        try:
            while True:
                t0 = time.time()
                with session.buffers.lock:
                    jpg = session.buffers.global_last_jpeg
                if jpg:
                    b64 = base64.b64encode(jpg).decode("ascii")
                    await ws.send_text(json.dumps({"type": "frame", "camera": "global", "base64": b64}))
                dt = time.time() - t0
                await asyncio.sleep(max(0.0, period - dt))
        except asyncio.CancelledError:
            pass

    async def _handle_connection(ws: WebSocket):
        await ws.accept()

        # attach per-WS queue so server.broadcast_ws_json() can deliver veh_list/reset/etc.
        q: "asyncio.Queue[str]" = asyncio.Queue()
        ws._send_queue = q  # type: ignore
        session.ws_clients.append(ws)

        # hello
        try:
            await ws.send_text(json.dumps({
                "type": "hello",
                "veh_ids": list(getattr(session, "vehicles", {}).keys()),
                "selected_veh_id": getattr(session, "selected_vehicle_id", None),
                "control_mode": getattr(session, "control_mode", "USER"),
                "t": time.time(),
            }))
        except Exception:
            await _safe_close(ws)
            return

        send_task  = asyncio.create_task(_sender(ws, q))
        hb_task    = asyncio.create_task(_heartbeat(ws))
        tele_task  = asyncio.create_task(_telemetry_tx(ws))
        ego_task   = asyncio.create_task(_ego_frame_tx(ws))
        glob_task  = asyncio.create_task(_global_frame_tx(ws))
        recv_task  = asyncio.create_task(_handle_messages(ws))

        try:
            await asyncio.gather(recv_task)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            for t in (send_task, hb_task, tele_task, ego_task, glob_task, recv_task):
                try:
                    t.cancel()
                except Exception:
                    pass
            try:
                session.ws_clients.remove(ws)
            except Exception:
                pass
            await _safe_close(ws)

    @app.websocket("/ws_ui")
    async def ws_ui(ws: WebSocket):
        await _handle_connection(ws)

    @app.websocket("/ws")
    async def ws_legacy(ws: WebSocket):
        await _handle_connection(ws)


async def _safe_close(ws: WebSocket):
    try:
        await ws.close()
    except Exception:
        pass