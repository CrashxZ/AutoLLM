# server/server.py
"""
CARLA AI-in-the-Loop Server (FastAPI)
-------------------------------------
Purpose:
  - Orchestrates a CARLA simulation with synchronous ticking.
  - Exposes a WebSocket for commands + telemetry (/ws).
  - Streams MJPEG video for:
      * /video/{veh_id}        : front-view (per-vehicle)
      * /video_top/{veh_id}    : per-vehicle top-down
      * /video_global          : global overhead camera
    And single-frame snapshots:
      * /frame/{veh_id}.jpg, /frame_top/{veh_id}.jpg, /frame_global.jpg
  - Supports session config (/config) and reset (/reset).
  - Logs telemetry JSONL + all frames to data/logs/<SESSION>/.

Assumptions:
  - CARLA is running (default host/port: 127.0.0.1:2000).
  - Python 3.10+.
  - lane_change.py (TM-based FSM) is present in the same folder.

Usage (dev):
  uvicorn server.server:app --reload --port 8000

Key constants:
  - FIXED_DELTA = 0.05 (20Hz)
  - TOP_DOWN_HEIGHT = 55.0 per-vehicle, pitch -90
  - GLOBAL_CAM_HEIGHT = 120.0, pitch -90

Notes:
  - We keep dependencies minimal: fastapi, uvicorn, pydantic, opencv-python, numpy.
  - We do NOT depend on FastAPI background tasks for CARLA ticking; we run our own thread.

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
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse , RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Body, Query
from aiortc import RTCPeerConnection, RTCSessionDescription
from .webrtc import CarlaVideoTrack, PCS, set_bitrate_cap, close_all_pcs

from pydantic import BaseModel, Field

import numpy as np

# ---- Environment loader -----------------------------------------------------

def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                val = value.strip().strip('"').strip("'")
                os.environ[key] = val
    except FileNotFoundError:
        pass

load_env_file()

# ---- CARLA import / setup ----------------------------------------------------

CARLA_HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
CARLA_TIMEOUT = float(os.environ.get("CARLA_TIMEOUT", "10.0"))
CARLA_DIR = os.environ.get("CARLA_DIR", "/home/labsdr/carla_simulator")  # optional: override if needed

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
from .mec import MECController

# ---- FastAPI app -------------------------------------------------------------

app = FastAPI(title="CARLA AI-in-the-Loop Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for local dev; tighten in prod
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

class IntentLogEntry(BaseModel):
    ts: float
    veh_id: int
    source: str
    intent: Dict[str, Any]
    telemetry: Optional[Dict[str, Any]] = None

class MecReviewPayload(BaseModel):
    veh_id: int
    plan: Dict[str, Any] = Field(default_factory=dict)
    intent: Optional[Dict[str, Any]] = None
    request: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    top_frame_b64: Optional[str] = None
    goal: Optional[str] = None

class MecConfig(BaseModel):
    safety_posture: Literal["strict", "balanced", "relaxed"]

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
        self.mec_log: Optional[io.TextIOWrapper] = None
        self.vehicle_logs: Dict[int, io.TextIOWrapper] = {}
        self.last_intents: Dict[int, dict] = {}

        self.running: bool = False
        self.tick_thread: Optional[threading.Thread] = None
        self.ws_clients: List[WebSocket] = []
        self.ws_lock = threading.Lock()

        self.tick_task: asyncio.Task | None = None

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
        os.makedirs(os.path.join(root, "vehicular"), exist_ok=True)
        return root

    def open_log(self):
        if not self.session_dir:
            self.session_dir = self._make_session_dir()
        path = os.path.join(self.session_dir, "telemetry.jsonl")
        self.telemetry_jsonl = open(path, "a", buffering=1, encoding="utf-8")
        mec_path = os.path.join(self.session_dir, "mec_reviews.jsonl")
        self.mec_log = open(mec_path, "a", buffering=1, encoding="utf-8")

    def close_log(self):
        if self.telemetry_jsonl:
            try:
                self.telemetry_jsonl.close()
            except Exception:
                pass
        self.telemetry_jsonl = None
        if self.mec_log:
            try:
                self.mec_log.close()
            except Exception:
                pass
        self.mec_log = None
        for fp in list(self.vehicle_logs.values()):
            try:
                fp.close()
            except Exception:
                pass
        self.vehicle_logs = {}

SESSION = Session()
MEC = MECController()

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

            last_intent = SESSION.last_intents.get(vid)
            frame_idx = SESSION.buffers.top_counter.get(vid)

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
                "goal_distance": None,  # optional: compute from final route waypoint
                "lane_change": lc,      # {"state":"IDLE/ARMING/EXECUTING/SETTLING/DONE/ABORT","direction": "left/right/none"}
                "intent": last_intent,
                "frame_top_file": f"veh_{vid}_top_{frame_idx:06d}.jpg" if frame_idx is not None else None,
            }
        except Exception:
            continue
    return out

# def telemetry_snapshot() -> dict[str, dict]:
#     out = {}
#     now = time.time()
#     for vid, vehicle in SESSION.vehicles.items():
#         try:
#             tr = vehicle.get_transform()
#             vel = vehicle.get_velocity()
#             spd_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
#             wp = WORLD.get_map().get_waypoint(tr.location)

#             out[str(vid)] = {
#                 "timestamp": now,
#                 "vehicle_id": vid,
#                 "speed_kmh": float(spd_kmh),
#                 "lane_id": int(wp.lane_id) if wp else None,
#                 "lane_type": str(wp.lane_type).split(".")[-1].lower() if wp else "unknown",
#                 "is_junction": bool(wp.is_junction) if wp else False,
#                 "distance_to_center": 0.0 if not wp else float(
#                     wp.transform.location.distance(tr.location)
#                 ),
#                 "pose": {
#                     "x": float(tr.location.x),
#                     "y": float(tr.location.y),
#                     "z": float(tr.location.z),
#                     "yaw": float(tr.rotation.yaw),
#                 },
#             }
#         except Exception:
#             # swallow vehicle that fails this tick
#             continue
#     return out


def log_telemetry(json_obj: dict):
    """Append telemetry JSON object to session JSONL."""
    if not SESSION.telemetry_jsonl:
        return
    try:
        SESSION.telemetry_jsonl.write(json.dumps(json_obj, separators=(",", ":")) + "\n")
    except Exception as e:
        print("[WARN] telemetry write failed:", e)

def log_intent(entry: Dict[str, Any]):
    """Append intent/plan JSON object to session JSONL."""
    if not SESSION.telemetry_jsonl:
        return
    try:
        SESSION.telemetry_jsonl.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        print("[WARN] intent write failed:", e)


def log_mec(entry: Dict[str, Any]):
    if not SESSION.mec_log:
        return
    try:
        SESSION.mec_log.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        print("[WARN] mec write failed:", e)


def log_vehicle(entry: Dict[str, Any]):
    vid = entry.get("veh_id")
    if vid is None:
        return
    if vid not in SESSION.vehicle_logs:
        try:
            path = os.path.join(SESSION.session_dir or "", "vehicular", f"{vid}.jsonl")
            SESSION.vehicle_logs[vid] = open(path, "a", buffering=1, encoding="utf-8")
        except Exception as e:
            print("[WARN] vehicle log open failed:", e)
            return
    try:
        SESSION.vehicle_logs[vid].write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        print("[WARN] vehicle log write failed:", e)


def sanitize_intent_request(intent: dict | None, veh_id: int) -> dict | None:
    if not intent:
        return intent
    req = intent.get("request") or {}
    to_list = req.get("to")
    if isinstance(to_list, list):
        filtered = [str(x) for x in to_list if str(x) != str(veh_id)]
    else:
        filtered = []
    if filtered:
        req["to"] = filtered
    else:
        req = {"to": [], "ask": "none"}
    intent = {**intent, "request": req}
    return intent

# ---- Camera callbacks & JPEG helpers ----------------------------------------

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
    # simple HUD text
    y = 24
    def put(txt, color=(0,255,0)):
        nonlocal y
        cv2.putText(bgr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22
    put("CARLA Front View")
    if veh_data:
        put(f"Veh {veh_data.get('veh_id','?')}  Speed {veh_data.get('speed_kmh','-'):.1f} km/h", (0,255,255))
        put(f"Lane {veh_data.get('lane_id','-')} ({veh_data.get('lane_type','-')})  Jct {veh_data.get('is_junction',False)}", (255,255,0))
        lc = veh_data.get("lane_change", {})
        put(f"LANE CHANGE: {lc.get('state','IDLE')} {str(lc.get('direction','') ).upper()}", (0,200,255))
    # compass dot (very simple)
    yaw = veh_data.get("pose", {}).get("yaw", 0.0) if veh_data else 0.0
    cx, cy, r = w - 60, 60, 28
    cv2.circle(bgr, (cx, cy), r, (200,200,200), 2)
    # heading line
    ang = math.radians(yaw)
    ex = int(cx + r * math.cos(ang))
    ey = int(cy + r * math.sin(ang))
    cv2.line(bgr, (cx, cy), (ex, ey), (0,0,255), 2)

def draw_overlay_top(bgr: np.ndarray, veh_data: dict | None):
    """Top camera overlay: label + simple breadcrumb optional (TODO)."""
    if bgr is None:
        return
    cv2.putText(bgr, "Top-Down (Vehicle)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

def draw_overlay_global(bgr: np.ndarray):
    if bgr is None:
        return
    cv2.putText(bgr, "Global Overhead", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)

def carla_img_to_bgr(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
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
    # mount near windshield
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

    # Derive center/height from spawn points to cover more of the map
    spawns = SESSION.spawn_points or world.get_map().get_spawn_points()
    if spawns:
        xs = [sp.location.x for sp in spawns]
        ys = [sp.location.y for sp in spawns]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        span_x = max_x - min_x
        span_y = max_y - min_y
        fov_rad = math.radians(GLOBAL_CAM_FOV)
        # height so that visible width ~= max(span_x, span_y)
        height = max(span_x, span_y) / (2.0 * math.tan(max(fov_rad, 0.1) / 2.0)) + 20.0
        height = max(height, 120.0)
        if center is None:
            center = carla.Location(x=cx, y=cy, z=0.0)
    if center is None:
        sp0 = spawns[0] if spawns else world.get_map().get_spawn_points()[0]
        center = sp0.location
        height = GLOBAL_CAM_HEIGHT
    tr = carla.Transform(
        carla.Location(x=center.x, y=center.y, z=center.z + height),
        carla.Rotation(pitch=-90.0)
    )
    cam = world.spawn_actor(cam_bp, tr)
    return cam

# ---- Camera listeners --------------------------------------------------------


#     try:
#         bgr = carla_img_to_bgr(image)
#         draw_overlay_global(bgr)
#         jpg = encode_jpeg(bgr, JPEG_QUALITY)
#         with SESSION.buffers.lock:
#             SESSION.buffers.global_last_jpeg = jpg
#             SESSION.buffers.global_counter += 1
#         if SESSION.session_dir:
#             out = os.path.join(SESSION.session_dir, "frames_global", f"global_{SESSION.buffers.global_counter:06d}.jpg")
#             try:
#                 with open(out, "wb") as f:
#                     f.write(jpg)
#             except Exception:
#                 pass
#     except Exception:
#         pass


# def make_front_callback(veh_id: int):
#     def _cb(image):
#         try:
#             # Convert BGRA -> BGR
#             arr = np.frombuffer(image.raw_data, dtype=np.uint8)
#             arr = arr.reshape((image.height, image.width, 4))
#             bgr = arr[:, :, :3]

#             # Best-effort overlay (should not prevent buffer write)
#             try:
#                 tele = None  # avoid calling telemetry_snapshot() here (heavy)
#                 draw_overlay_front(bgr, tele)  # make this handle tele=None
#             except Exception as e:
#                 print(f"[FRONT] overlay fail veh={veh_id}: {e}")

#             ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
#             if not ok:
#                 print(f"[FRONT] encode fail veh={veh_id}")
#                 return
#             jpg = enc.tobytes()

#             with SESSION.buffers.lock:
#                 SESSION.buffers.front_last_jpeg[veh_id] = jpg
#                 cnt = SESSION.buffers.front_counter.get(veh_id, 0) + 1
#                 SESSION.buffers.front_counter[veh_id] = cnt

#             if SESSION.session_dir:
#                 try:
#                     out = os.path.join(SESSION.session_dir, "frames_front", f"veh_{veh_id:03d}_{cnt:06d}.jpg")
#                     with open(out, "wb") as f:
#                         f.write(jpg)
#                 except Exception as e:
#                     print(f"[FRONT] persist fail veh={veh_id}: {e}")

#         except Exception as e:
#             print(f"[FRONT] callback error veh={veh_id}: {e}")
#     return _cb


# def make_top_callback(veh_id: int):
#     def _cb(image):
#         try:
#             arr = np.frombuffer(image.raw_data, dtype=np.uint8)
#             arr = arr.reshape((image.height, image.width, 4))
#             bgr = arr[:, :, :3]

#             try:
#                 tele = None
#                 draw_overlay_top(bgr, tele)
#             except Exception as e:
#                 print(f"[TOP] overlay fail veh={veh_id}: {e}")

#             ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
#             if not ok:
#                 print(f"[TOP] encode fail veh={veh_id}")
#                 return
#             jpg = enc.tobytes()

#             with SESSION.buffers.lock:
#                 SESSION.buffers.top_last_jpeg[veh_id] = jpg
#                 cnt = SESSION.buffers.top_counter.get(veh_id, 0) + 1
#                 SESSION.buffers.top_counter[veh_id] = cnt

#             if SESSION.session_dir:
#                 try:
#                     out = os.path.join(SESSION.session_dir, "frames_top", f"veh_{veh_id:03d}_top_{cnt:06d}.jpg")
#                     with open(out, "wb") as f:
#                         f.write(jpg)
#                 except Exception as e:
#                     print(f"[TOP] persist fail veh={veh_id}: {e}")

#         except Exception as e:
#             print(f"[TOP] callback error veh={veh_id}: {e}")
#     return _cb


# def global_cam_callback(image):
#     try:
#         arr = np.frombuffer(image.raw_data, dtype=np.uint8)
#         arr = arr.reshape((image.height, image.width, 4))
#         bgr = arr[:, :, :3]

#         try:
#             draw_overlay_global(bgr)
#         except Exception as e:
#             print(f"[GLOBAL] overlay fail: {e}")

#         ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
#         if not ok:
#             print("[GLOBAL] encode fail")
#             return
#         jpg = enc.tobytes()

#         with SESSION.buffers.lock:
#             SESSION.buffers.global_last_jpeg = jpg
#             SESSION.buffers.global_counter += 1

#         if SESSION.session_dir:
#             try:
#                 out = os.path.join(SESSION.session_dir, "frames_global", f"global_{SESSION.buffers.global_counter:06d}.jpg")
#                 with open(out, "wb") as f:
#                     f.write(jpg)
#             except Exception as e:
#                 print(f"[GLOBAL] persist fail: {e}")

#     except Exception as e:
#         print(f"[GLOBAL] callback error: {e}")


def make_front_callback(veh_id: int):
    """
    CARLA RGB front camera callback factory.
    - Converts BGRA -> BGR into a writeable, contiguous array
    - Best-effort overlay (won't block frame publishing)
    - Stores latest JPEG in SESSION.buffers.front_last_jpeg[veh_id]
    - Increments per-vehicle frame counter
    - Optionally persists to session_dir/frames_front/
    """
    def _cb(image):
        try:
            # Convert raw BGRA to writeable BGR
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # ensures contiguous, writeable

            # Optional overlay (must not prevent buffer write)
            try:
                tele = None  # avoid heavy calls inside sensor thread
                # If you have a cached last-telemetry dict, you can read from it instead:
                # tele = LAST_TELEMETRY.get(veh_id)
                draw_overlay_front(bgr, tele)  # make this tolerant to tele=None
            except Exception as e:
                print(f"[FRONT] overlay fail veh={veh_id}: {e}")

            # JPEG encode
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                print(f"[FRONT] encode fail veh={veh_id}")
                return
            jpg = enc.tobytes()

            # Publish to in-memory buffer
            with SESSION.buffers.lock:
                SESSION.buffers.front_last_jpeg[veh_id] = jpg
                cnt = SESSION.buffers.front_counter.get(veh_id, 0) + 1
                SESSION.buffers.front_counter[veh_id] = cnt

            # Optional persistence (best-effort)
            if SESSION.session_dir:
                try:
                    out_dir = os.path.join(SESSION.session_dir, "frames_front")
                    # ensure directory exists (only when needed)
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
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # writeable

            try:
                tele = None  # avoid heavy work here
                draw_overlay_top(bgr, tele)  # must draw in-place; tolerate tele=None
            except Exception as e:
                print(f"[TOP] overlay fail veh={veh_id}: {e}")

            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                print(f"[TOP] encode fail veh={veh_id}")
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
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # writeable

        try:
            draw_overlay_global(bgr)
        except Exception as e:
            print(f"[GLOBAL] overlay fail: {e}")

        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            print("[GLOBAL] encode fail")
            return
        jpg = enc.tobytes()

        with SESSION.buffers.lock:
            SESSION.buffers.global_last_jpeg = jpg
            SESSION.buffers.global_counter += 1

    except Exception as e:
        print(f"[GLOBAL] callback error: {e}")

# ---- Ticking & Broadcasting --------------------------------------------------

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
                    "payload": tele,                   # client-friendly alias
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

# ---- WebSocket management ----------------------------------------------------

def broadcast_ws_json(obj: dict):
    payload = json.dumps(obj, separators=(",", ":"))
    with SESSION.ws_lock:
        dead = []
        for ws in SESSION.ws_clients:
            try:
                # send text (non-await; we are outside asyncio loop) -> schedule in thread-safe way
                # We'll push via a background thread to avoid blocking; for simplicity
                # rely on FastAPI's WS send_text in async context only. Here we stash for on-loop?
                # Simpler: put into a shared queue and the WS connection task pulls. We'll do both:
                # For now, try to schedule via ws.send_text using starlette's 'send' is async -> we cannot here.
                # So we keep per-WS queues. We'll implement them in the WS connection handler.
                if hasattr(ws, "_send_queue"):
                    ws._send_queue.put_nowait(payload)  # type: ignore
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

    # commands that require a vehicle
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

    elif c == "brake":
        veh.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

    elif c == "release":
        veh.set_autopilot(True, tm.get_port())

    elif c == "intent":
        # For now, global always approves. We turn intent into an action.
        intent = cmd.get("intent") or {}
        intent = sanitize_intent_request(intent, vid)
        try:
            SESSION.last_intents[vid] = intent
            entry = {
                "ts": time.time(),
                "veh_id": vid,
                "source": "vehicle_intent",
                "intent": intent,
                "telemetry": telemetry_snapshot().get(str(vid)),
            }
            log_intent(entry)
            log_vehicle(entry)
        except Exception:
            pass
        # simple mapping
        act = (intent.get("ego_action") or intent.get("action") or "keep").lower()
        if "left" in act:
            vctx.lane_ctl.request_change("left")
        elif "right" in act:
            vctx.lane_ctl.request_change("right")
        elif "speed up" in act or "faster" in act:
            cur = veh_speed_kmh(veh)
            SESSION.tm.set_desired_speed(veh, min(cur + 10.0, 120.0))
        elif "slow" in act or "slower" in act:
            cur = veh_speed_kmh(veh)
            SESSION.tm.set_desired_speed(veh, max(cur - 10.0, 0.0))
        # broadcast approval message
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

@app.post("/mec/review")
async def mec_review(payload: MecReviewPayload):
    telemetry = telemetry_snapshot()
    result = await MEC.review(payload, telemetry)
    try:
        entry = {
            "ts": time.time(),
            "veh_id": payload.veh_id,
            "source": "mec",
            "intent": result,
            "telemetry": telemetry.get(str(payload.veh_id)),
            "goal": payload.goal or payload.context.get("goal"),
        }
        log_intent(entry)
        log_mec(entry)
        log_vehicle(entry)
    except Exception:
        pass
    broadcast_ws_json({"type": "mec_decision", **result})
    return result

@app.get("/mec/history")
def mec_history():
    return {"decisions": MEC.get_history()}

@app.get("/mec/config")
def mec_config():
    return {"safety_posture": MEC.get_safety_posture()}

@app.post("/mec/config")
def update_mec_config(cfg: MecConfig):
    try:
        MEC.set_safety_posture(cfg.safety_posture)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    updated = MEC.get_safety_posture()
    return {"safety_posture": updated}

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
    """
    Simplified spawn preview: we render a placeholder image with spawn coordinates.
    (A full 3D render would require switching spectator + grabbing a camera; here we keep it light.)
    """
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
    """
    Spawns vehicles/sensors per request and starts ticking + logging.
    Destroys any existing session actors first.
    """
    carla_connect()
    # clean old
    reset_internal(destroy_only=True)

    # session dir + log
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
        # attach cams
        cam_front = attach_front_cam(bp, veh)
        cam_top = attach_top_cam(bp, veh)
        vctx.cam_front = cam_front
        vctx.cam_top = cam_top

        # listeners
        cam_front.listen(make_front_callback(veh.id))
        cam_top.listen(make_top_callback(veh.id))

        SESSION.vehicles[veh.id] = vctx
        veh_ids.append(veh.id)

    # global cam
    if SESSION.global_cam:
        safe_destroy([SESSION.global_cam])
        SESSION.global_cam = None
    gcam = spawn_global_cam(bp, world, None)
    gcam.listen(global_cam_callback)
    SESSION.global_cam = gcam

    SESSION.selected_vehicle_id = veh_ids[0] if veh_ids else None
    # Auto-select first vehicle so ego camera starts sending frames
    if SESSION.vehicles and SESSION.selected_vehicle_id is None:
        SESSION.selected_vehicle_id = next(iter(SESSION.vehicles.keys()))

    # start ticking
    start_tick_thread()
    try:
        broadcast_ws_json({
            "type": "veh_list",
            "veh_ids": veh_ids,
            "selected_veh_id": SESSION.selected_vehicle_id
        })
    except Exception:
        pass
    return {"ok": True, "veh_ids": veh_ids}

@app.post("/reset")
def reset():
    # after clearing vehicles/buffers
    reset_internal(destroy_only=False)
    return {"ok": True}

def reset_internal(destroy_only: bool):
    """Destroy all actors and optionally close logs; keep CARLA sync setting alive."""
    stop_tick_thread()
    # stop sensors + destroy
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
    # clear buffers
    with SESSION.buffers.lock:
        SESSION.buffers.front_last_jpeg.clear()
        SESSION.buffers.top_last_jpeg.clear()
        SESSION.buffers.global_last_jpeg = None
        SESSION.buffers.front_counter.clear()
        SESSION.buffers.top_counter.clear()
        SESSION.buffers.global_counter = 0

    if not destroy_only:
        SESSION.close_log()
        # keep session_dir for archives; create a fresh one on next /config
        SESSION.session_dir = None
        try:
            broadcast_ws_json({"type": "veh_list", "veh_ids": []})
            broadcast_ws_json({"type": "reset", "ts": time.time()})
        except Exception:
            pass
    gc.collect()



import asyncio
from starlette.websockets import WebSocketState


WS_HEARTBEAT_SEC = 10.0

async def ws_sender(ws: WebSocket, q: "asyncio.Queue[str]"):
    """Dedicated async sender: drains queue and send_text to the client."""
    try:
        while True:
            payload = await q.get()
            if ws.application_state != WebSocketState.CONNECTED:
                break
            await ws.send_text(payload)
    except Exception:
        # exit silently; connection likely closed
        pass

async def ws_heartbeat(ws: WebSocket, q: "asyncio.Queue[str]"):
    """Send periodic ping (as JSON) to keep connection alive."""
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
    """Maps friendly client tokens to server-native command schema."""
    cmd = (obj.get("cmd") or "").lower()
    if cmd in ("lane_left", "lane_right"):
        return {
            "cmd": "lane",
            "veh_id": obj.get("veh_id"),
            "dir": "left" if "left" in cmd else "right",
        }
    if cmd in ("speed_up", "slow_down"):
        # interpret as intent so planner can adjust
        return {
            "cmd": "intent",
            "intent": {
                "ego_veh_id": obj.get("veh_id"),
                "ego_action": "speed up" if cmd == "speed_up" else "slow down",
            },
        }
    # passthrough defaults
    return obj

# ---------------- MJPEG Streaming helpers ----------------

def mjpeg_generator(get_jpeg_callable, boundary=b"--frame", fps_limit=20):
    """
    Generic MJPEG generator. get_jpeg_callable() should return bytes or None.
    """
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
                # send a tiny black frame placeholder to keep stream alive
                blk = blank_jpeg()
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(blk)).encode() + b"\r\n\r\n" + blk + b"\r\n"
            else:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        except GeneratorExit:
            break
        except Exception:
            # swallow and keep streaming
            continue

def blank_jpeg(w=320, h=180):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "waiting...", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    return encode_jpeg(img, 80)

# ---------------- Video Endpoints ----------------

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
# Keep PCs to allow closing later (optional)
PCS = set()

def _frame_provider(view: str, veh_id: int | None) -> bytes | None:
    """
    Safely read the latest JPEG from SESSION buffers.
    """
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
    """
    WebRTC signaling:
      - POST SDP offer, receive SDP answer.
      - Query selects which view and veh_id (for front/top) to stream.
    """
    offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
    pc = RTCPeerConnection()
    PCS.add(pc)

    track = CarlaVideoTrack(
        frame_provider=_frame_provider,
        view=view,
        veh_id=veh_id,
        target_fps=fps,
        scale_to=(960, 540),  # adjust if you want
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
    # Optionally close all peer connections (useful on sim reset)
    for pc in list(PCS):
        await pc.close()
        PCS.discard(pc)
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
        SESSION.buffers.top_counter[veh_id] = SESSION.buffers.top_counter.get(veh_id, 0) + 1
    if not jpg:
        raise HTTPException(404, "no frame yet")
    fname = f"veh_{veh_id}_top_{SESSION.buffers.top_counter.get(veh_id,0):06d}.jpg"
    try:
        if SESSION.session_dir:
            path = os.path.join(SESSION.session_dir, "frames_top", fname)
            with open(path, "wb") as fh:
                fh.write(jpg)
    except Exception:
        pass
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/frame/global.jpg")
def frame_global():
    with SESSION.buffers.lock:
        jpg = SESSION.buffers.global_last_jpeg
    if not jpg:
        raise HTTPException(404, "no frame yet")
    return Response(content=jpg, media_type="image/jpeg")

# ---------------- App lifecycle (startup/shutdown) ----------------

@app.on_event("startup")
def on_startup():
    # connect to CARLA and stay in sync mode; no actors yet
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
# (continuation) server/server.py

import zipfile

# ---------------- Dynamic global camera positioning ----------------

def update_global_camera_transform():
    """
    Repositions the global overhead camera to keep all vehicles in view.
    Runs in its own lightweight thread, every few seconds.
    """
    world = SESSION.world
    if not (SESSION.global_cam and world):
        return
    while SESSION.running:
        try:
            if not SESSION.vehicles:
                time.sleep(2.0)
                continue
            xs, ys = [], []
            for vctx in list(SESSION.vehicles.values()):
                tr = vctx.veh.get_transform()
                xs.append(tr.location.x)
                ys.append(tr.location.y)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            # keep Z constant
            tr = SESSION.global_cam.get_transform()
            new_loc = carla.Location(x=cx, y=cy, z=tr.location.z)
            new_rot = carla.Rotation(pitch=-90.0)
            SESSION.global_cam.set_transform(carla.Transform(new_loc, new_rot))
        except Exception:
            pass
        time.sleep(1.5)

def start_global_reposition_thread():
    th = threading.Thread(target=update_global_camera_transform, daemon=True)
    th.start()

# ---------------- Archive current session ----------------

@app.get("/archive")
def archive_session():
    """
    Creates a ZIP of the current session log directory (telemetry + frames)
    and returns it for download.
    """
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
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(zpath)}"'
            },
        )
    except Exception as e:
        raise HTTPException(500, f"archive failed: {e}")

# ---------------- Utility: graceful shutdown from client ----------------

@app.post("/shutdown")
def shutdown_endpoint():
    """
    Allows client to request a clean shutdown of the FastAPI server.
    """
    def _shutdown():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_shutdown, daemon=True).start()
    return {"ok": True, "msg": "Server exiting in 0.5s"}

# ---------------- Optional helpers for external control ----------------

@app.post("/command")
async def post_command(req: Request):
    """
    Accepts a JSON command (identical to WebSocket protocol).
    """
    try:
        obj = await req.json()
    except Exception:
        raise HTTPException(400, "invalid JSON")
    try:
        SESSION.cmd_queue.put_nowait(obj)
    except queue.Full:
        raise HTTPException(429, "command queue full")
    return {"ok": True}

# ---------------- Optional manual telemetry dump ----------------

@app.get("/telemetry.jsonl")
def get_current_telemetry():
    """Return a snapshot of last 100 lines of telemetry.jsonl."""
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

# ---------------- Entry point for standalone run ----------------

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting CARLA AI-in-the-Loop FastAPI server on port 8000 ...")
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
