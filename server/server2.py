# server/server.py
"""
CARLA AI-in-the-Loop Server (FastAPI)
-------------------------------------
Purpose:
  - Orchestrates a CARLA simulation with synchronous ticking.
  - Exposes a WebSocket for commands + telemetry (/ws via ws_routes).
  - Streams:
      * MJPEG:
          /video/{veh_id}        : front-view (per-vehicle)
          /video_top/{veh_id}    : per-vehicle top-down
          /video_global          : global overhead camera
      * Single frames:
          /frame/front/{veh_id}.jpg
          /frame/top/{veh_id}.jpg
          /frame/global.jpg
          (legacy) /frame/{veh_id}.jpg -> redirects to /frame/top/{veh_id}.jpg
      * WebRTC (H.264/VP8):
          POST /webrtc/offer?view=front|top|global&veh_id=&max_bitrate=&fps=
          POST /webrtc/close

  - Supports session config (/config) and reset (/reset).
  - Logs telemetry JSONL + all frames to data/logs/<SESSION>/.

Assumptions:
  - CARLA is running (default host/port: 127.0.0.1:2000).
  - Python 3.10+.

Run (dev):
  uvicorn server.server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import glob
import io
import cv2
import gc
import json
import time
import math
import queue
import shutil
import base64
import random
import string
import zipfile
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request, HTTPException, Body, Query
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

# --- WebRTC (no circular imports) ---
from aiortc import RTCPeerConnection, RTCSessionDescription
from .webrtc import CarlaVideoTrack, PCS, set_bitrate_cap, close_all_pcs

# ---- CARLA import / setup ----------------------------------------------------

CARLA_HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
CARLA_TIMEOUT = float(os.environ.get("CARLA_TIMEOUT", "10.0"))
CARLA_DIR = os.environ.get("CARLA_DIR", "/home/labsdr/carla_simulator")  # update to your install

def _try_import_carla():
    if CARLA_DIR:
        try:
            sys.path.append(glob.glob(
                f'{CARLA_DIR}/PythonAPI/carla/dist/carla-*{sys.version_info.major}.{sys.version_info.minor}-{"win-amd64" if os.name=="nt" else "linux-x86_64"}.egg'
            )[0])
            sys.path.append(f'{CARLA_DIR}/PythonAPI/carla')
        except Exception:
            pass
    try:
        import carla  # type: ignore
        return carla
    except Exception as e:
        print("[FATAL] Could not import CARLA. Set CARLA_DIR or install egg. Error:", e)
        raise

carla = _try_import_carla()

# lane change controller (TM-based) from sibling file
from .lane_change import LaneChangeController, LaneChangeState

# ---- FastAPI app -------------------------------------------------------------

app = FastAPI(title="CARLA AI-in-the-Loop Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally serve built web UI if present
DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "client", "web", "dist")
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")

# ---- Config constants --------------------------------------------------------

FIXED_DELTA = 0.05           # 20 Hz
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "logs")

FRONT_RES = (1280, 720)
TOP_RES = (1024, 1024)
GLOBAL_RES = (1920, 1080)

JPEG_QUALITY = 80

TOP_DOWN_HEIGHT = 55.0       # meters above car
GLOBAL_CAM_HEIGHT = 120.0
GLOBAL_CAM_FOV = 90.0

WS_HEARTBEAT_SEC = 10.0

# ---- Data models -------------------------------------------------------------

class ConfigRequest(BaseModel):
    num_cars: int = Field(2, ge=1, le=12)
    spawn_indices: List[int] = Field(default_factory=lambda: [110, 112])
    initial_speeds: List[float] = Field(default_factory=lambda: [50.0, 40.0])

class IntentMessage(BaseModel):
    ego_veh_id: int
    ego_action: str
    reason: Optional[str] = None
    target_lane: Optional[str] = None
    confidence: Optional[float] = None
    backend: Optional[str] = None
    tier: Optional[str] = None
    goal: Optional[str] = None

# ---- Session state -----------------------------------------------------------

class CameraBuffers:
    """In-memory buffers for last frames + counters for logging."""
    def __init__(self):
        self.front_last_jpeg: Dict[int, bytes] = {}
        self.top_last_jpeg: Dict[int, bytes] = {}
        self.global_last_jpeg: Optional[bytes] = None
        self.front_counter: Dict[int, int] = {}
        self.top_counter: Dict[int, int] = {}
        self.global_counter: int = 0
        self.lock = threading.Lock()

class VehicleCtx:
    """Holds references for a single vehicle and its sensors/controllers."""
    def __init__(self, veh: carla.Vehicle, tm, world_map):
        self.veh = veh
        self.tm = tm
        self.lane_ctl = LaneChangeController(veh, tm, world_map)
        self.cam_front: Optional[carla.Sensor] = None
        self.cam_top: Optional[carla.Sensor] = None

class Session:
    """Holds the entire CARLA + server runtime for a single run."""
    def __init__(self):
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.tm: Optional[carla.TrafficManager] = None
        self.bp: Optional[carla.BlueprintLibrary] = None
        self.spectator: Optional[carla.Actor] = None

        self.global_cam: Optional[carla.Sensor] = None

        self.vehicles: Dict[int, VehicleCtx] = {}
        self.spawn_points: List[carla.Transform] = []
        self.session_dir: Optional[str] = None
        self.telemetry_jsonl: Optional[io.TextIOWrapper] = None

        self.running: bool = False
        self.tick_thread: Optional[threading.Thread] = None
        self.ws_clients: List[WebSocket] = []
        self.ws_lock = threading.Lock()

        self.buffers = CameraBuffers()
        self.control_mode: str = "USER"  # USER | LLM

        self.selected_vehicle_id: Optional[int] = None

        # routes / path planning placeholders
        self.route_waypoints: List[carla.Location] = []

        # internal queues
        self.cmd_queue: "queue.Queue[dict]" = queue.Queue()

    # --- util paths ---
    def _make_session_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(os.path.join(LOG_ROOT, ts))
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "frames_front"), exist_ok=True)
        os.makedirs(os.path.join(root, "frames_top"), exist_ok=True)
        os.makedirs(os.path.join(root, "frames_global"), exist_ok=True)
        return root

    def open_log(self):
        if not self.session_dir:
            self.session_dir = self._make_session_dir()
        path = os.path.join(self.session_dir, "telemetry.jsonl")
        self.telemetry_jsonl = open(path, "a", buffering=1, encoding="utf-8")

    def close_log(self):
        if self.telemetry_jsonl:
            try:
                self.telemetry_jsonl.close()
            except Exception:
                pass
        self.telemetry_jsonl = None

SESSION = Session()

# ---- CARLA lifecycle helpers -------------------------------------------------

def carla_connect():
    if SESSION.client:
        return
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    world = client.load_world('Town04_Opt', map_layers=carla.MapLayer.Ground)
    # enforce synchronous + fixed delta
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8005)
    tm.set_synchronous_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_random_device_seed(42)
    tm.set_global_distance_to_leading_vehicle(3.0)

    bp = world.get_blueprint_library()
    spectator = world.get_spectator()
    world_map = world.get_map()

    SESSION.client = client
    SESSION.world = world
    SESSION.tm = tm
    SESSION.bp = bp
    SESSION.spectator = spectator
    SESSION.map = world_map
    SESSION.spawn_points = world_map.get_spawn_points()

def carla_reset_settings():
    if not SESSION.world or not SESSION.tm:
        return
    try:
        settings = SESSION.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        SESSION.world.apply_settings(settings)
        SESSION.tm.set_synchronous_mode(False)
    except Exception as e:
        print("[WARN] Failed to reset CARLA settings:", e)

def safe_destroy(actors: List[carla.Actor]):
    if not actors:
        return
    # sensors first
    sensors = [a for a in actors if "sensor" in a.type_id]
    others = [a for a in actors if "sensor" not in a.type_id]

    for s in sensors:
        try:
            if hasattr(s, "is_listening") and s.is_listening:
                s.stop()
        except Exception:
            pass
        time.sleep(0.02)
    all_actors = sensors + others
    try:
        SESSION.client.apply_batch([
            carla.command.DestroyActor(a) for a in all_actors
        ])
    except Exception:
        # fallback one-by-one
        for a in all_actors:
            try:
                a.destroy()
            except Exception:
                pass
            time.sleep(0.01)

# ---- Telemetry ---------------------------------------------------------------

def veh_speed_kmh(v: carla.Vehicle) -> float:
    vel = v.get_velocity()
    return float(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6)

def waypoint_info(map_obj: carla.Map, loc: carla.Location):
    try:
        w = map_obj.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not w:
            return None
        return {
            "lane_id": int(w.lane_id),
            "lane_type": str(w.lane_type).split(".")[-1].lower(),
            "is_junction": bool(w.is_junction),
            "center": (float(w.transform.location.x), float(w.transform.location.y)),
        }
    except Exception:
        return None

def telemetry_snapshot() -> dict:
    """Builds the telemetry for all vehicles (dict keyed by string veh_id)."""
    out: Dict[str, dict] = {}
    now = time.time()
    for vid, vctx in list(SESSION.vehicles.items()):
        veh = vctx.veh
        try:
            tr = veh.get_transform()
            rot = tr.rotation
            loc = tr.location
            vel = veh.get_velocity()

            wpi = waypoint_info(SESSION.map, loc) if SESSION.map else None
            d2c = 0.0
            if wpi and "center" in wpi:
                cx, cy = wpi["center"]
                d2c = math.hypot(loc.x - cx, loc.y - cy)

            # lane change state (FSM)
            lc = vctx.lane_ctl.get_public_state()

            out[str(vid)] = {
                "ts": now,
                "veh_id": vid,
                "pose": {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z), "yaw": float(rot.yaw)},
                "vel": {"x": float(vel.x), "y": float(vel.y), "z": float(vel.z)},
                "speed_kmh": veh_speed_kmh(veh),
                "lane_id": (wpi["lane_id"] if wpi else None),
                "lane_type": (wpi["lane_type"] if wpi else None),
                "is_junction": (wpi["is_junction"] if wpi else False),
                "distance_to_center": d2c,
                "goal_distance": None,
                "lane_change": lc,  # {"state":"IDLE/ARMING/EXECUTING/SETTLING/DONE/ABORT","direction":"left/right/none"}
            }
        except Exception:
            continue
    return out

def log_telemetry(json_obj: dict):
    """Append telemetry JSON object to session JSONL."""
    if not SESSION.telemetry_jsonl:
        return
    try:
        SESSION.telemetry_jsonl.write(json.dumps(json_obj, separators=(",", ":")) + "\n")
    except Exception as e:
        print("[WARN] telemetry write failed:", e)

# ---- Camera helpers ----------------------------------------------------------

def encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return enc.tobytes()

def draw_overlay_front(bgr: np.ndarray, veh_data: dict | None):
    """Front camera overlay: FPS-like tick info, lane id/type, LC state, yaw arrow."""
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    y = 24
    def put(txt, color=(0,255,0)):
        nonlocal y
        cv2.putText(bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22
    put("CARLA Front View")
    if veh_data:
        try:
            put(f"Veh {veh_data.get('veh_id','?')}  Speed {veh_data.get('speed_kmh',0.0):.1f} km/h", (0,255,255))
        except Exception:
            put("Speed -.- km/h", (0,255,255))
        put(f"Lane {veh_data.get('lane_id','-')} ({veh_data.get('lane_type','-')})  Jct {veh_data.get('is_junction',False)}", (255,255,0))
        lc = veh_data.get("lane_change", {})
        put(f"LANE CHANGE: {lc.get('state','IDLE')} {str(lc.get('direction','') ).upper()}", (0,200,255))
    # compass
    yaw = veh_data.get("pose", {}).get("yaw", 0.0) if veh_data else 0.0
    cx, cy, r = w - 60, 60, 28
    cv2.circle(bgr, (cx, cy), r, (200,200,200), 2)
    ang = math.radians(yaw)
    ex = int(cx + r * math.cos(ang))
    ey = int(cy + r * math.sin(ang))
    cv2.line(bgr, (cx, cy), (ex, ey), (0,0,255), 2)

def draw_overlay_top(bgr: np.ndarray, veh_data: dict | None):
    if bgr is None:
        return
    cv2.putText(bgr, "Top-Down (Vehicle)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def draw_overlay_global(bgr: np.ndarray):
    if bgr is None:
        return
    cv2.putText(bgr, "Global Overhead", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)

def carla_img_to_bgr(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # ensures writeable
    return bgr

# ---- Spawn / Sensor setup ----------------------------------------------------

def spawn_vehicle(bp: carla.BlueprintLibrary, world: carla.World, sp: carla.Transform) -> carla.Vehicle:
    car_bp = random.choice([bp.find('vehicle.audi.tt'), bp.find('vehicle.audi.etron')])
    if car_bp.has_attribute("color"):
        car_bp.set_attribute("color", random.choice(car_bp.get_attribute("color").recommended_values))
    veh = world.try_spawn_actor(car_bp, sp)
    if veh is None:
        raise RuntimeError("Failed to spawn vehicle at given spawn point.")
    return veh

def attach_front_cam(bp: carla.BlueprintLibrary, veh: carla.Vehicle) -> carla.Sensor:
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(FRONT_RES[0]))
    cam_bp.set_attribute("image_size_y", str(FRONT_RES[1]))
    cam_bp.set_attribute("fov", "90.0")
    rel = carla.Transform(carla.Location(x=0.8, z=1.4), carla.Rotation(pitch=0.0))
    cam = SESSION.world.spawn_actor(cam_bp, rel, attach_to=veh)
    return cam

def attach_top_cam(bp: carla.BlueprintLibrary, veh: carla.Vehicle) -> carla.Sensor:
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(TOP_RES[0]))
    cam_bp.set_attribute("image_size_y", str(TOP_RES[1]))
    cam_bp.set_attribute("fov", "60.0")
    rel = carla.Transform(carla.Location(z=TOP_DOWN_HEIGHT), carla.Rotation(pitch=-90.0))
    cam = SESSION.world.spawn_actor(cam_bp, rel, attach_to=veh)
    return cam

def spawn_global_cam(bp: carla.BlueprintLibrary, world: carla.World, center: Optional[carla.Location] = None) -> carla.Sensor:
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(GLOBAL_RES[0]))
    cam_bp.set_attribute("image_size_y", str(GLOBAL_RES[1]))
    cam_bp.set_attribute("fov", str(GLOBAL_CAM_FOV))
    if center is None:
        sp0 = SESSION.spawn_points[0] if SESSION.spawn_points else world.get_map().get_spawn_points()[0]
        center = sp0.location
    tr = carla.Transform(carla.Location(x=center.x, y=center.y, z=center.z + GLOBAL_CAM_HEIGHT), carla.Rotation(pitch=-90.0))
    cam = world.spawn_actor(cam_bp, tr)
    return cam

# ---- Camera listeners --------------------------------------------------------

def make_front_callback(veh_id: int):
    def _cb(image):
        try:
            bgr = carla_img_to_bgr(image)
            try:
                tele = None   # keep callbacks light
                draw_overlay_front(bgr, tele)
            except Exception as e:
                print(f"[FRONT] overlay fail veh={veh_id}: {e}")

            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return
            jpg = enc.tobytes()

            with SESSION.buffers.lock:
                SESSION.buffers.front_last_jpeg[veh_id] = jpg
                cnt = SESSION.buffers.front_counter.get(veh_id, 0) + 1
                SESSION.buffers.front_counter[veh_id] = cnt

            if SESSION.session_dir:
                try:
                    out_dir = os.path.join(SESSION.session_dir, "frames_front")
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    out = os.path.join(out_dir, f"veh_{veh_id:03d}_{cnt:06d}.jpg")
                    with open(out, "wb") as f:
                        f.write(jpg)
                except Exception as e:
                    print(f"[FRONT] persist fail veh={veh_id}: {e}")

        except Exception as e:
            print(f"[FRONT] callback error veh={veh_id}: {e}")
    return _cb

def make_top_callback(veh_id: int):
    def _cb(image):
        try:
            bgr = carla_img_to_bgr(image)
            try:
                tele = None
                draw_overlay_top(bgr, tele)
            except Exception as e:
                print(f"[TOP] overlay fail veh={veh_id}: {e}")

            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return
            jpg = enc.tobytes()

            with SESSION.buffers.lock:
                SESSION.buffers.top_last_jpeg[veh_id] = jpg
                cnt = SESSION.buffers.top_counter.get(veh_id, 0) + 1
                SESSION.buffers.top_counter[veh_id] = cnt

            if SESSION.session_dir:
                try:
                    out = os.path.join(SESSION.session_dir, "frames_top", f"veh_{veh_id:03d}_top_{cnt:06d}.jpg")
                    with open(out, "wb") as f:
                        f.write(jpg)
                except Exception as e:
                    print(f"[TOP] persist fail veh={veh_id}: {e}")

        except Exception as e:
            print(f"[TOP] callback error veh={veh_id}: {e}")
    return _cb

def global_cam_callback(image):
    try:
        bgr = carla_img_to_bgr(image)
        try:
            draw_overlay_global(bgr)
        except Exception as e:
            print(f"[GLOBAL] overlay fail: {e}")

        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        jpg = enc.tobytes()

        with SESSION.buffers.lock:
            SESSION.buffers.global_last_jpeg = jpg
            SESSION.buffers.global_counter += 1

        if SESSION.session_dir:
            try:
                out = os.path.join(SESSION.session_dir, "frames_global", f"global_{SESSION.buffers.global_counter:06d}.jpg")
                with open(out, "wb") as f:
                    f.write(jpg)
            except Exception as e:
                print(f"[GLOBAL] persist fail: {e}")

    except Exception as e:
        print(f"[GLOBAL] callback error: {e}")

# ---- Ticking & Broadcasting --------------------------------------------------

def broadcast_ws_json(obj: dict):
    payload = json.dumps(obj, separators=(",", ":"))
    with SESSION.ws_lock:
        dead = []
        for ws in SESSION.ws_clients:
            try:
                if hasattr(ws, "_send_queue"):
                    ws._send_queue.put_nowait(payload)  # type: ignore[attr-defined]
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                SESSION.ws_clients.remove(ws)
            except Exception:
                pass

def drain_commands_nonblocking():
    while True:
        try:
            cmd = SESSION.cmd_queue.get_nowait()
        except queue.Empty:
            break
        try:
            handle_command(cmd)
        except Exception as e:
            print("[CMD] error handling command:", e)

def tick_loop():
    """Drives the CARLA world and periodically broadcasts telemetry to WS clients."""
    world = SESSION.world
    if world is None:
        return
    last_broadcast = 0.0
    while SESSION.running:
        try:
            world.tick()
            now = time.time()
            # process queued commands (non-blocking)
            drain_commands_nonblocking()
            # telemetry broadcast at ~10 Hz
            if now - last_broadcast >= 0.1:
                tele = telemetry_snapshot()
                msg = {
                    "type": "telemetry",
                    "t": now,
                    "vehicles": tele,
                    "payload": tele,
                    "veh_ids": list(tele.keys()),
                    "control_mode": SESSION.control_mode,
                    "selected_veh_id": SESSION.selected_vehicle_id,
                }
                broadcast_ws_json(msg)
                # also log JSONL per tick
                log_telemetry({"t": now, "vehicles": tele})
                last_broadcast = now
        except Exception as e:
            print("[TICK] error:", e)
            time.sleep(0.05)

def start_tick_thread():
    if SESSION.tick_thread and SESSION.tick_thread.is_alive():
        return
    SESSION.running = True
    th = threading.Thread(target=tick_loop, daemon=True)
    SESSION.tick_thread = th
    th.start()

def stop_tick_thread():
    SESSION.running = False
    if SESSION.tick_thread and SESSION.tick_thread.is_alive():
        SESSION.tick_thread.join(timeout=2.0)
    SESSION.tick_thread = None

# ---- Command handling --------------------------------------------------------

def handle_command(cmd: dict):
    c = (cmd.get("cmd") or "").lower()
    if c == "mode":
        m = (cmd.get("control_mode") or "USER").upper()
        SESSION.control_mode = "USER" if m == "USER" else "LLM"
        return

    if c == "select":
        SESSION.selected_vehicle_id = int(cmd.get("veh_id"))
        return

    vid = int(cmd.get("veh_id") or 0)
    vctx = SESSION.vehicles.get(vid)
    if not vctx:
        return

    veh = vctx.veh
    tm = SESSION.tm

    if c == "speed":
        kmh = float(cmd.get("kmh") or 0.0)
        tm.set_desired_speed(veh, kmh)

    elif c == "lane":
        d = (cmd.get("dir") or "").lower()
        if d in ("left", "l"):
            vctx.lane_ctl.request_change("left")
        elif d in ("right", "r"):
            vctx.lane_ctl.request_change("right")

    elif c == "lane_left":
        vctx.lane_ctl.request_change("left")

    elif c == "lane_right":
        vctx.lane_ctl.request_change("right")
        print(f"[CMD] veh={vid} lane change RIGHT")

    elif c == "brake":
        veh.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        tm.set_desired_speed(veh, 0)

    elif c == "release":
        veh.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(veh, 30)

    elif c == "intent":
        # Global coordinator: approve by default; map intent -> action
        intent = cmd.get("intent") or {}
        act = (intent.get("ego_action") or intent.get("action") or "keep").lower()
        if "left" in act:
            vctx.lane_ctl.request_change("left")
            print(f"[INTENT] veh={vid} lane change LEFT")
        elif "right" in act:
            vctx.lane_ctl.request_change("right")
            print(f"[INTENT] veh={vid} lane change RIGHT") 
        elif "speed up" in act or "faster" in act:
            cur = veh_speed_kmh(veh)
            tm.set_desired_speed(veh, cur + 10.0)
            print(f"[INTENT] veh={vid} speed up from {cur:.1f} km/h")
            #SESSION.tm.set_desired_speed(veh, min(cur + 10.0, 120.0))
        elif "slow" in act or "slower" in act:
            cur = veh_speed_kmh(veh)
            tm.set_desired_speed(veh, cur - 10.0)
            print(f"[INTENT] veh={vid} slow down from {cur:.1f} km/h")
            #SESSION.tm.set_desired_speed(veh, max(cur - 10.0, 0.0))
        broadcast_ws_json({
            "type": "approval",
            "ts": time.time(),
            "intent": intent,
            "approved": True,
            "plan": {"action_applied": act}
        })

# ---- Routes: health, spawns, previews, config/reset --------------------------

@app.get("/debug/session")
def debug_session():
    return {
        "veh_ids": list(SESSION.vehicles.keys()),
        "selected": SESSION.selected_vehicle_id,
        "tick_running": SESSION.running,
    }

@app.get("/vehicles")
def list_vehicles():
    return {"veh_ids": list(SESSION.vehicles.keys())}

@app.get("/health")
def health():
    return {"ok": True, "running": SESSION.running, "vehicles": list(SESSION.vehicles.keys())}

@app.get("/spawns")
def list_spawns():
    carla_connect()
    out = []
    for i, sp in enumerate(SESSION.spawn_points):
        loc = sp.location
        rot = sp.rotation
        out.append({
            "index": i,
            "x": float(loc.x), "y": float(loc.y), "z": float(loc.z),
            "yaw": float(rot.yaw)
        })
    return {"spawns": out}

@app.get("/preview_spawn/{idx}.jpg")
def preview_spawn(idx: int):
    carla_connect()
    if idx < 0 or idx >= len(SESSION.spawn_points):
        raise HTTPException(404, "spawn index out of range")
    sp = SESSION.spawn_points[idx]
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    txt = f"Spawn #{idx}\nX={sp.location.x:.1f}\nY={sp.location.y:.1f}\nYaw={sp.rotation.yaw:.1f}"
    y = 30
    for line in txt.splitlines():
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2, cv2.LINE_AA)
        y += 30
    jpg = encode_jpeg(img, 85)
    return Response(content=jpg, media_type="image/jpeg")

@app.post("/config")
def configure(cfg: ConfigRequest):
    carla_connect()
    reset_internal(destroy_only=True)

    SESSION.open_log()

    world = SESSION.world
    tm = SESSION.tm
    bp = SESSION.bp
    world_map = SESSION.map
    if not (world and tm and bp and world_map):
        raise HTTPException(500, "CARLA not ready")

    veh_ids = []
    for i in range(cfg.num_cars):
        sp_idx = cfg.spawn_indices[i] if i < len(cfg.spawn_indices) else cfg.spawn_indices[-1]
        if sp_idx < 0 or sp_idx >= len(SESSION.spawn_points):
            raise HTTPException(400, f"spawn index {sp_idx} invalid")
        sp = SESSION.spawn_points[sp_idx]
        veh = spawn_vehicle(bp, world, sp)
        tm.set_desired_speed(veh, float(cfg.initial_speeds[i] if i < len(cfg.initial_speeds) else cfg.initial_speeds[-1]))
        veh.set_autopilot(True, tm.get_port())

        vctx = VehicleCtx(veh, tm, world_map)
        vctx.cam_front = attach_front_cam(bp, veh)
        vctx.cam_top = attach_top_cam(bp, veh)

        vctx.cam_front.listen(make_front_callback(veh.id))
        vctx.cam_top.listen(make_top_callback(veh.id))

        SESSION.vehicles[veh.id] = vctx
        veh_ids.append(veh.id)

    if SESSION.global_cam:
        safe_destroy([SESSION.global_cam])
        SESSION.global_cam = None
    gcam = spawn_global_cam(bp, world, None)
    gcam.listen(global_cam_callback)
    SESSION.global_cam = gcam

    SESSION.selected_vehicle_id = veh_ids[0] if veh_ids else None
    if SESSION.vehicles and SESSION.selected_vehicle_id is None:
        SESSION.selected_vehicle_id = next(iter(SESSION.vehicles.keys()))

    start_tick_thread()

    try:
        broadcast_ws_json({"type": "veh_list", "veh_ids": veh_ids, "selected_veh_id": SESSION.selected_vehicle_id})
    except Exception:
        pass
    return {"ok": True, "veh_ids": veh_ids}

@app.post("/reset")
def reset():
    reset_internal(destroy_only=False)
    return {"ok": True}

def reset_internal(destroy_only: bool):
    stop_tick_thread()
    actors = []
    for vctx in list(SESSION.vehicles.values()):
        if vctx.cam_front: actors.append(vctx.cam_front)
        if vctx.cam_top: actors.append(vctx.cam_top)
        actors.append(vctx.veh)
    if SESSION.global_cam:
        actors.append(SESSION.global_cam)
    safe_destroy(actors)

    SESSION.vehicles.clear()
    SESSION.global_cam = None
    SESSION.selected_vehicle_id = None

    with SESSION.buffers.lock:
        SESSION.buffers.front_last_jpeg.clear()
        SESSION.buffers.top_last_jpeg.clear()
        SESSION.buffers.global_last_jpeg = None
        SESSION.buffers.front_counter.clear()
        SESSION.buffers.top_counter.clear()
        SESSION.buffers.global_counter = 0

    if not destroy_only:
        SESSION.close_log()
        SESSION.session_dir = None
        try:
            broadcast_ws_json({"type": "veh_list", "veh_ids": []})
            broadcast_ws_json({"type": "reset", "ts": time.time()})
        except Exception:
            pass
    gc.collect()

    # Close any active WebRTC peer connections
    try:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if loop and loop.is_running():
            asyncio.create_task(close_all_pcs())
        else:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(close_all_pcs())
            finally:
                loop.close()
    except Exception:
        pass

# ---- WS helpers (used by ws_routes) -----------------------------------------

async def ws_sender(ws: WebSocket, q: "asyncio.Queue[str]"):
    try:
        while True:
            payload = await q.get()
            if ws.application_state != WebSocketState.CONNECTED:
                break
            await ws.send_text(payload)
    except Exception:
        pass

async def ws_heartbeat(ws: WebSocket, q: "asyncio.Queue[str]"):
    try:
        while True:
            await asyncio.sleep(WS_HEARTBEAT_SEC)
            if ws.application_state != WebSocketState.CONNECTED:
                break
            try:
                await ws.send_text(json.dumps({"type": "ping", "t": time.time()}))
            except Exception:
                break
    except asyncio.CancelledError:
        pass

def normalize_incoming(obj: dict) -> dict:
    cmd = (obj.get("cmd") or "").lower()
    if cmd in ("lane_left", "lane_right"):
        return {"cmd": "lane", "veh_id": obj.get("veh_id"), "dir": "left" if "left" in cmd else "right"}
    if cmd in ("speed_up", "slow_down"):
        return {"cmd": "intent", "intent": {"ego_veh_id": obj.get("veh_id"),
                                            "ego_action": "speed up" if cmd == "speed_up" else "slow down"}}
    return obj

# ---------------- MJPEG Streaming helpers ----------------

def blank_jpeg(w=320, h=180):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "waiting...", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    return encode_jpeg(img, 80)

def mjpeg_generator(get_jpeg_callable, boundary=b"--frame", fps_limit=20):
    period = 1.0 / max(1, int(fps_limit))
    last = 0.0
    while True:
        now = time.time()
        slp = period - (now - last)
        if slp > 0:
            time.sleep(slp)
        last = time.time()
        try:
            jpg = get_jpeg_callable()
            if not jpg:
                blk = blank_jpeg()
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(blk)).encode() + b"\r\n\r\n" + blk + b"\r\n"
            else:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        except GeneratorExit:
            break
        except Exception:
            continue

# ---------------- Video (MJPEG) Endpoints ----------------

@app.get("/video/{veh_id}")
def video_front(veh_id: int):
    def getter():
        with SESSION.buffers.lock:
            return SESSION.buffers.front_last_jpeg.get(veh_id, None)
    return StreamingResponse(
        mjpeg_generator(getter, boundary=b"--frame", fps_limit=20),
        media_type="multipart/x-mixed-replace; boundary=--frame",
    )

@app.get("/video_top/{veh_id}")
def video_top(veh_id: int):
    def getter():
        with SESSION.buffers.lock:
            return SESSION.buffers.top_last_jpeg.get(veh_id, None)
    return StreamingResponse(
        mjpeg_generator(getter, boundary=b"--frame", fps_limit=12),
        media_type="multipart/x-mixed-replace; boundary=--frame",
    )

@app.get("/video_global")
def video_global():
    def getter():
        with SESSION.buffers.lock:
            return SESSION.buffers.global_last_jpeg
    return StreamingResponse(
        mjpeg_generator(getter, boundary=b"--frame", fps_limit=10),
        media_type="multipart/x-mixed-replace; boundary=--frame",
    )

# ---------------- WebRTC Endpoints ----------------

def _frame_provider(view: str, veh_id: int | None) -> bytes | None:
    with SESSION.buffers.lock:
        if view == "global":
            return SESSION.buffers.global_last_jpeg
        if view == "front" and veh_id is not None:
            return SESSION.buffers.front_last_jpeg.get(veh_id)
        if view == "top" and veh_id is not None:
            return SESSION.buffers.top_last_jpeg.get(veh_id)
    return None

@app.post("/webrtc/offer")
async def webrtc_offer(
    payload: dict = Body(...),
    view: str = Query("front", pattern="^(front|top|global)$"),
    veh_id: int | None = Query(None),
    max_bitrate: int | None = Query(None, description="bps, e.g., 1200000"),
    fps: float = Query(20.0),
):
    offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
    pc = RTCPeerConnection()
    PCS.add(pc)

    track = CarlaVideoTrack(
        frame_provider=_frame_provider,
        view=view,
        veh_id=veh_id,
        target_fps=fps,
        scale_to=(960, 540),
        max_bitrate=max_bitrate,
    )
    pc.addTrack(track)

    @pc.on("iceconnectionstatechange")
    def on_state_change():
        st = pc.iceConnectionState
        if st in ("failed", "closed", "disconnected"):
            try:
                PCS.remove(pc)
            except KeyError:
                pass

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    if max_bitrate:
        await set_bitrate_cap(pc, int(max_bitrate))

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.post("/webrtc/close")
async def webrtc_close():
    await close_all_pcs()
    return {"ok": True}

# ---------------- Frame (single JPEG) Endpoints ----------------

@app.get("/frame/{veh_id}.jpg")
def legacy_frame_redirect(veh_id: int):
    return RedirectResponse(url=f"/frame/top/{veh_id}.jpg", status_code=307)

@app.get("/frame/front/{veh_id}.jpg")
def frame_front(veh_id: int):
    with SESSION.buffers.lock:
        jpg = SESSION.buffers.front_last_jpeg.get(veh_id)
    if not jpg:
        print(f"[FRAME] miss veh={veh_id}. keys(front)={list(SESSION.buffers.front_last_jpeg.keys())}")
        raise HTTPException(404, "no frame yet")
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/frame/top/{veh_id}.jpg")
def frame_top(veh_id: int):
    with SESSION.buffers.lock:
        jpg = SESSION.buffers.top_last_jpeg.get(veh_id)
    if not jpg:
        raise HTTPException(404, "no frame yet")
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/frame/global.jpg")
def frame_global():
    with SESSION.buffers.lock:
        jpg = SESSION.buffers.global_last_jpeg
    if not jpg:
        raise HTTPException(404, "no frame yet")
    return Response(content=jpg, media_type="image/jpeg")

# ---------------- Archive current session ----------------

@app.get("/archive")
def archive_session():
    if not SESSION.session_dir or not os.path.isdir(SESSION.session_dir):
        raise HTTPException(404, "no active session directory")
    zpath = os.path.join(SESSION.session_dir, "session_archive.zip")
    try:
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(SESSION.session_dir):
                for f in files:
                    if f.endswith(".zip"):
                        continue
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, SESSION.session_dir)
                    zf.write(full, arcname=rel)
        with open(zpath, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{os.path.basename(zpath)}"'},
        )
    except Exception as e:
        raise HTTPException(500, f"archive failed: {e}")

# ---------------- Utility: graceful shutdown from client ----------------

@app.post("/shutdown")
def shutdown_endpoint():
    def _shutdown():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return {"ok": True, "msg": "Server exiting in 0.5s"}

# ---------------- Optional REST command injection ----------------

@app.post("/command")
async def post_command(req: Request):
    try:
        obj = await req.json()
    except Exception:
        raise HTTPException(400, "invalid JSON")
    try:
        SESSION.cmd_queue.put_nowait(obj)
    except queue.Full:
        raise HTTPException(429, "command queue full")
    return {"ok": True}

@app.get("/telemetry.jsonl")
def get_current_telemetry():
    if not SESSION.session_dir:
        raise HTTPException(404, "no session")
    path = os.path.join(SESSION.session_dir, "telemetry.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "no telemetry file yet")
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-100:]
        return JSONResponse([json.loads(x) for x in lines])
    except Exception as e:
        raise HTTPException(500, f"read failed: {e}")

# ---- Register UI WebSocket endpoint (base64 frames + telemetry) ----
from .ws_routes import register_ws_routes
register_ws_routes(
    app,
    telemetry_fn=telemetry_snapshot,
    session=SESSION,
    enqueue_fn=lambda cmd: SESSION.cmd_queue.put_nowait(cmd),
)

# ---------------- App lifecycle (startup/shutdown) ----------------

@app.on_event("startup")
def on_startup():
    carla_connect()
    print("[APP] Startup complete: connected to CARLA.")

@app.on_event("shutdown")
def on_shutdown():
    try:
        reset_internal(destroy_only=False)
    except Exception as e:
        print("[APP] reset on shutdown failed:", e)
    try:
        carla_reset_settings()
    except Exception as e:
        print("[APP] reset CARLA settings failed:", e)
    print("[APP] Shutdown done.")

# ---------------- Entry point for standalone run ----------------

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting CARLA AI-in-the-Loop FastAPI server on port 8000 ...")
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)