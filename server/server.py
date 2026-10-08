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
CARLA_DIR = os.environ.get("CARLA_DIR", "")  # optional: override if needed
TM_SAFE_DISTANCE = float(os.environ.get("TM_SAFE_DISTANCE", "3.0"))

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
from .road_geometry import ordered_driving_lane_ids
from .coordination import (
    CoordinationOrchestrator,
    DeterministicYieldProposer,
    OpenAIJointPlanProposer,
    RankedCandidateProposer,
)
from .coordination.validator import DeterministicPlanValidator, ValidationConfig
from .coordination.legacy_adapter import (
    decision_to_legacy,
    payload_to_proposal,
    telemetry_to_states,
)
from marl.numpy_policy import NumpyActorPolicy, telemetry_observation
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

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

# Coordination baseline constants
FCFS_DISTANCE_M = 50.0
FCFS_TTC_S = 4.0
FCFS_WINDOW_S = 5.0
FLOW_TARGET_KMH = 50.0

MARL_MODEL_PATH = os.environ.get(
    "MARL_POLICY_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "models", "mappo_policy.npz"),
)
MARL_MODEL = None

# ---- Data models -------------------------------------------------------------

class ConfigRequest(BaseModel):
    num_cars: int = Field(2, ge=1, le=12)
    spawn_indices: List[int] = Field(default_factory=lambda: [110, 112])
    spawn_longitudinal_offsets_m: List[float] = Field(default_factory=list)
    initial_speeds: List[float] = Field(default_factory=lambda: [50.0, 40.0])
    coordination_mode: Literal["IA", "FCFS", "MARL", "MIND_CAVS"] = "MIND_CAVS"
    scenario_id: Optional[str] = None
    seed: Optional[int] = None
    lane_goals: Optional[Dict[str, Any]] = None
    persist_frames: bool = True
    enable_cameras: bool = True
    tm_auto_lane_change: bool = True
    tm_route: List[Literal["Left", "Right", "Straight"]] = Field(
        default_factory=list
    )
    start_paused: bool = False


class SimulationWarmupRequest(BaseModel):
    vehicle_ids: List[int] = Field(default_factory=list)
    minimum_speed_kmh: float = Field(30.0, ge=0.0, le=200.0)
    require_non_junction: bool = True
    max_ticks: int = Field(400, ge=1, le=2000)


class SimulationStepRequest(BaseModel):
    vehicle_ids: List[int] = Field(default_factory=list)
    ticks: int = Field(10, ge=1, le=100)

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
    transaction_id: Optional[str] = None
    created_at_s: Optional[float] = None
    expires_at_s: Optional[float] = None
    observation_ts_s: Optional[float] = None
    source: Optional[str] = None
    plan: Dict[str, Any] = Field(default_factory=dict)
    intent: Optional[Dict[str, Any]] = None
    request: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    top_frame_b64: Optional[str] = None
    goal: Optional[str] = None

class MecConfig(BaseModel):
    safety_posture: Literal["strict", "balanced", "relaxed"]


class MecV2RankerConfig(BaseModel):
    variant: Literal["deterministic", "learned"]


class UnsafeToggle(BaseModel):
    unsafe: bool


class MecOutcomePayload(BaseModel):
    transaction_id: str
    outcome: Literal["executing", "completed", "failed"]
    details: Dict[str, Any] = Field(default_factory=dict)

class CoordinationMode(str):
    IA = "IA"
    FCFS = "FCFS"
    MARL = "MARL"
    MIND_CAVS = "MIND_CAVS"

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
        self.lane_ctl = LaneChangeController(
            veh,
            tm,
            world_map,
            get_desired_speed_kmh=lambda: self.last_desired_speed_kmh,
            get_sim_time_s=lambda: SESSION.sim_time_s,
            get_auto_lane_change_enabled=lambda: SESSION.tm_auto_lane_change_enabled,
        )
        self.cam_front: Optional[carla.Sensor] = None
        self.cam_top: Optional[carla.Sensor] = None
        self.collision_sensor: Optional[carla.Sensor] = None
        self.last_desired_speed_kmh: Optional[float] = None

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
        self.collision_events: List[Dict[str, Any]] = []
        self.collision_counts: Dict[int, int] = {}
        self.collision_lock = threading.Lock()
        self.sim_frame: Optional[int] = None
        self.sim_time_s: Optional[float] = None
        self.sim_delta_s: Optional[float] = None
        self.persist_frames: bool = True

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

        # traffic manager safety mode
        self.tm_unsafe_mode: bool = False
        self.tm_auto_lane_change_enabled: bool = True

        # coordination mode + run metadata
        self.coordination_mode: str = CoordinationMode.MIND_CAVS
        self.run_config: Dict[str, Any] = {}

        # FCFS state
        self.fcfs_queue: List[Dict[str, Any]] = []
        self.pending_intents: Dict[int, Dict[str, Any]] = {}

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


def build_mind_v2_orchestrator() -> CoordinationOrchestrator:
    validator = build_execution_aware_validator()
    proposer_mode = os.getenv("MIND_CAV_V2_PROPOSER", "ranked").strip().lower()
    if proposer_mode == "deterministic":
        proposer = DeterministicYieldProposer()
    elif proposer_mode in {"ranked", "local", "learned"}:
        proposer = RankedCandidateProposer(
            validator=validator,
            model_path=(os.getenv("MIND_CAV_RANKER_MODEL") or None)
            if os.getenv("MIND_CAV_RANKER_VARIANT", "deterministic") == "learned" else None,
            enable_global_conflict_recovery=(
                os.getenv("MIND_CAV_V2_GLOBAL_CONFLICT_RECOVERY", "0") == "1"
            ),
            enable_liveness_preparation=(
                os.getenv("MIND_CAV_V2_LIVENESS_PREPARATION", "0") == "1"
            ),
        )
    else:
        proposer = OpenAIJointPlanProposer()
    return CoordinationOrchestrator(
        proposer=proposer,
        validator=validator,
        proposer_timeout_s=float(os.getenv("MIND_CAV_V2_TIMEOUT_S", "30")),
        semantic_call_cap=int(os.getenv("MIND_CAV_V2_CALL_CAP", "500")),
    )


def build_execution_aware_validator() -> DeterministicPlanValidator:
    enabled = os.getenv("MIND_CAV_V2_EXECUTION_AWARE_VALIDATOR", "0") == "1"
    return DeterministicPlanValidator(
        ValidationConfig(
            model_lane_change_execution_speed=enabled,
            lane_change_execution_target_speed_kmh=float(
                os.getenv("LANE_CHANGE_EXECUTION_TARGET_SPEED_KMH", "30.0")
            ),
            lane_change_execution_speed_factor=float(
                os.getenv("LANE_CHANGE_SPEED_FACTOR", "1.0")
            ),
        )
    )


MIND_V2_ENABLED = os.getenv("MIND_CAV_V2_ENABLED", "1") == "1"
MIND_V2 = build_mind_v2_orchestrator()


def build_ranked_candidate_proposer(
    variant: str,
    validator: Optional[DeterministicPlanValidator] = None,
) -> RankedCandidateProposer:
    """Build one of the two validator-constrained ranking variants."""
    liveness_preparation = (
        os.getenv("MIND_CAV_V2_LIVENESS_PREPARATION", "0") == "1"
    )
    model_path = None
    if variant == "learned":
        model_path = os.getenv("MIND_CAV_RANKER_MODEL") or None
        if not model_path:
            raise ValueError("MIND_CAV_RANKER_MODEL is required for learned ranking")
    proposer = RankedCandidateProposer(
        validator=validator or build_execution_aware_validator(),
        model_path=model_path,
        enable_liveness_preparation=liveness_preparation,
        enable_global_conflict_recovery=(
            os.getenv("MIND_CAV_V2_GLOBAL_CONFLICT_RECOVERY", "0") == "1"
        ),
    )
    if variant == "learned" and proposer.model is None:
        raise ValueError(
            "learned ranker could not be loaded "
            f"({proposer.model_error or 'unknown error'})"
        )
    return proposer

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
    tm.set_global_distance_to_leading_vehicle(TM_SAFE_DISTANCE)

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
        center = w.transform.location
        yaw_rad = math.radians(float(w.transform.rotation.yaw))
        # Signed road-relative lateral offset. Positive values lie to the
        # waypoint heading's left; CARLA's ``s`` is the longitudinal coordinate
        # used by the finite-horizon validator.
        dx = float(loc.x - center.x)
        dy = float(loc.y - center.y)
        signed_d = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        driving_lane_ids = ordered_driving_lane_ids(w)
        return {
            "road_id": int(w.road_id),
            "section_id": int(w.section_id),
            "s_m": float(w.s),
            "d_m": float(signed_d),
            "lane_id": int(w.lane_id),
            "driving_lane_ids": list(driving_lane_ids),
            "driving_lane_index": driving_lane_ids.index(int(w.lane_id)),
            "lane_type": str(w.lane_type).split(".")[-1].lower(),
            "is_junction": bool(w.is_junction),
            "center": (float(center.x), float(center.y)),
        }
    except Exception:
        return None


def forward_non_junction_m(
    waypoint: Optional[carla.Waypoint],
    *,
    step_m: float = 5.0,
    maximum_m: float = 150.0,
) -> float:
    """Estimate usable forward lane distance for calibration case selection."""
    if waypoint is None or waypoint.is_junction:
        return 0.0
    current = waypoint
    distance_m = 0.0
    while distance_m + step_m <= maximum_m:
        candidates = [
            candidate
            for candidate in current.next(step_m)
            if candidate.lane_type == carla.LaneType.Driving
            and int(candidate.lane_id) == int(current.lane_id)
            and not candidate.is_junction
        ]
        if not candidates:
            break
        current = candidates[0]
        distance_m += step_m
    return distance_m


def shifted_spawn_transform(
    world_map: carla.Map,
    spawn_transform: carla.Transform,
    longitudinal_offset_m: float,
) -> carla.Transform:
    """Move an experiment spawn along its lane topology without changing lanes."""
    offset_m = float(longitudinal_offset_m)
    if abs(offset_m) < 1e-6:
        return spawn_transform
    waypoint = world_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise ValueError("base spawn has no driving-lane waypoint")
    candidates = (
        waypoint.next(offset_m)
        if offset_m > 0.0
        else waypoint.previous(abs(offset_m))
    )
    candidates = [
        candidate
        for candidate in candidates
        if candidate.lane_type == carla.LaneType.Driving
        and not candidate.is_junction
    ]
    if not candidates:
        raise ValueError(f"no non-junction lane at offset {offset_m:g} m")

    def heading_delta(candidate: carla.Waypoint) -> float:
        delta = (
            float(candidate.transform.rotation.yaw)
            - float(waypoint.transform.rotation.yaw)
            + 180.0
        ) % 360.0 - 180.0
        return abs(delta)

    shifted = min(candidates, key=heading_delta).transform
    clearance_z = max(
        0.5,
        float(spawn_transform.location.z - waypoint.transform.location.z),
    )
    return carla.Transform(
        carla.Location(
            x=float(shifted.location.x),
            y=float(shifted.location.y),
            z=float(shifted.location.z + clearance_z),
        ),
        carla.Rotation(
            pitch=float(shifted.rotation.pitch),
            yaw=float(shifted.rotation.yaw),
            roll=float(shifted.rotation.roll),
        ),
    )

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
            if wpi and wpi.get("d_m") is not None:
                d2c = abs(float(wpi["d_m"]))

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
                "desired_speed_kmh": vctx.last_desired_speed_kmh,
                "sim_frame": SESSION.sim_frame,
                "sim_time_s": SESSION.sim_time_s,
                "sim_delta_s": SESSION.sim_delta_s,
                "road_id": (wpi["road_id"] if wpi else None),
                "section_id": (wpi["section_id"] if wpi else None),
                "s_m": (wpi["s_m"] if wpi else None),
                "d_m": (wpi["d_m"] if wpi else None),
                "lane_id": (wpi["lane_id"] if wpi else None),
                "driving_lane_ids": (wpi["driving_lane_ids"] if wpi else None),
                "driving_lane_index": (
                    wpi["driving_lane_index"] if wpi else None
                ),
                "lane_type": (wpi["lane_type"] if wpi else None),
                "is_junction": (wpi["is_junction"] if wpi else False),
                "distance_to_center": d2c,
                "goal_distance": None,  # optional: compute from final route waypoint
                "lane_change": lc,      # {"state":"IDLE/ARMING/EXECUTING/SETTLING/DONE/ABORT","direction": "left/right/none"}
                "occupied_lane_ids": lc.get("occupied_lane_ids") or None,
                "lane_change_remaining_s": float(
                    lc.get("lane_change_remaining_s") or 0.0
                ),
                "collision_count": SESSION.collision_counts.get(vid, 0),
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

def write_session_meta(meta: Dict[str, Any]):
    if not SESSION.session_dir:
        return
    path = os.path.join(SESSION.session_dir, "run_metadata.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception as e:
        print("[WARN] metadata write failed:", e)

def apply_tm_collision_mode(unsafe: bool):
    """Apply Traffic Manager collision/avoidance settings across all vehicles."""
    tm = SESSION.tm
    if not tm:
        return
    vehicles = [vctx.veh for vctx in SESSION.vehicles.values()]
    ignore_pct = 100.0 if unsafe else 0.0
    try:
        tm.set_global_distance_to_leading_vehicle(0.0 if unsafe else TM_SAFE_DISTANCE)
    except Exception:
        pass
    for veh in vehicles:
        try:
            tm.set_percentage_ignore_vehicles(veh, ignore_pct)
            tm.set_percentage_ignore_walkers(veh, ignore_pct)
            tm.set_distance_to_leading_vehicle(veh, 0.0 if unsafe else TM_SAFE_DISTANCE)
        except Exception:
            pass
    # Pairwise collision detection toggle (small fleet -> OK to do O(n^2))
    for i, va in enumerate(vehicles):
        for vb in vehicles[i + 1:]:
            try:
                tm.set_collision_detection(va, vb, not unsafe)
                tm.set_collision_detection(vb, va, not unsafe)
            except Exception:
                pass

# ---- Coordination helpers ---------------------------------------------------

def _get_target_lane_from_plan(plan: Dict[str, Any], intent: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if plan:
        steps = plan.get("steps") or []
        for step in steps:
            if not isinstance(step, dict):
                continue
            lane = step.get("target_lane_id")
            if lane is not None:
                try:
                    return int(lane)
                except Exception:
                    return None
    if intent:
        lane = intent.get("target_lane_id") or intent.get("target_lane")
        if lane is not None:
            try:
                return int(lane)
            except Exception:
                return None
    return None

def _is_emergency(intent: Optional[Dict[str, Any]], ctx: Optional[Dict[str, Any]]) -> bool:
    val = None
    if intent:
        val = intent.get("vehicle_class") or intent.get("class")
    if not val and ctx:
        val = ctx.get("vehicle_class")
    return str(val).lower() == "emergency"

def _veh_snapshot(telemetry_snapshot: Dict[str, Any], veh_id: int) -> Optional[Dict[str, Any]]:
    return telemetry_snapshot.get(str(veh_id)) if telemetry_snapshot else None

def _distance_m(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    try:
        ax, ay = a["pose"]["x"], a["pose"]["y"]
        bx, by = b["pose"]["x"], b["pose"]["y"]
        return math.hypot(ax - bx, ay - by)
    except Exception:
        return None

def _relative_speed_mps(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    try:
        va = float(a.get("speed_kmh") or 0.0) / 3.6
        vb = float(b.get("speed_kmh") or 0.0) / 3.6
        return abs(va - vb)
    except Exception:
        return None

def _compute_ttc_s(dist_m: Optional[float], rel_speed_mps: Optional[float]) -> Optional[float]:
    if dist_m is None or rel_speed_mps is None or rel_speed_mps <= 0.01:
        return None
    return dist_m / rel_speed_mps

def _load_marl_model():
    global MARL_MODEL
    if MARL_MODEL is not None:
        return MARL_MODEL
    if not os.path.exists(MARL_MODEL_PATH):
        return None
    try:
        if MARL_MODEL_PATH.endswith(".npz"):
            MARL_MODEL = NumpyActorPolicy.load(MARL_MODEL_PATH)
        elif joblib is not None:
            MARL_MODEL = joblib.load(MARL_MODEL_PATH)
    except Exception:
        MARL_MODEL = None
    return MARL_MODEL

def _marl_features(ego: Dict[str, Any], other: Optional[Dict[str, Any]], target_lane: Optional[int]) -> List[float]:
    return telemetry_observation(ego, other, target_lane, flow_speed_kmh=FLOW_TARGET_KMH).tolist()

def _marl_policy_action(ego: Dict[str, Any], other: Optional[Dict[str, Any]], target_lane: Optional[int]) -> str:
    model = _load_marl_model()
    if model is not None:
        feats = _marl_features(ego, other, target_lane)
        try:
            if isinstance(model, NumpyActorPolicy):
                return model.action_name(np.asarray(feats, dtype=np.float32))
            pred = model.predict([feats])[0]
            return str(pred)
        except Exception:
            pass
    # heuristic fallback
    if target_lane is not None and (ego.get("lane_id") != target_lane):
        if other:
            dist = _distance_m(ego, other)
            if dist is not None and dist <= FCFS_DISTANCE_M:
                return "hold"
        return "lane_left" if target_lane > ego.get("lane_id") else "lane_right"
    speed = float(ego.get("speed_kmh") or 0.0)
    if speed < FLOW_TARGET_KMH - 2.0:
        return "speed_up"
    if speed > FLOW_TARGET_KMH + 5.0:
        return "speed_down"
    return "hold"

def _plan_first_action(plan: Dict[str, Any]) -> Optional[str]:
    steps = plan.get("steps") or []
    if not steps:
        return None
    first = steps[0]
    if isinstance(first, dict):
        return first.get("action")
    return None

def fcfs_review(payload: MecReviewPayload, telemetry_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    # prune old entries
    SESSION.fcfs_queue = [q for q in SESSION.fcfs_queue if now - q["ts"] <= FCFS_WINDOW_S]
    SESSION.pending_intents = {
        k: v for k, v in SESSION.pending_intents.items()
        if now - v.get("ts", 0) <= FCFS_WINDOW_S
    }

    veh_id = payload.veh_id
    target_lane = _get_target_lane_from_plan(payload.plan, payload.intent)
    SESSION.pending_intents[veh_id] = {"target_lane_id": target_lane, "ts": now}

    if not any(q["veh_id"] == veh_id for q in SESSION.fcfs_queue):
        SESSION.fcfs_queue.append({
            "veh_id": veh_id,
            "ts": now,
            "priority": 0 if _is_emergency(payload.intent, payload.context) else 1,
        })

    ego = _veh_snapshot(telemetry_snapshot, veh_id)
    conflicts: List[int] = []
    if target_lane is not None and ego:
        for other_id, other in SESSION.pending_intents.items():
            if other_id == veh_id:
                continue
            if other.get("target_lane_id") != target_lane:
                continue
            other_snap = _veh_snapshot(telemetry_snapshot, other_id)
            if not other_snap:
                continue
            dist = _distance_m(ego, other_snap)
            rel_speed = _relative_speed_mps(ego, other_snap)
            ttc = _compute_ttc_s(dist, rel_speed)
            if dist is not None and dist <= FCFS_DISTANCE_M:
                conflicts.append(other_id)
            elif ttc is not None and ttc <= FCFS_TTC_S:
                conflicts.append(other_id)

    if not conflicts:
        return {
            "veh_id": veh_id,
            "decision": "allow",
            "reason": "FCFS: no conflict",
            "plan": None,
            "model": "fcfs",
            "ts": now,
            "decision_id": f"fcfs-{int(now * 1000)}-{veh_id}",
        }

    # Determine priority ordering (emergency first, then arrival time)
    ordered = sorted(SESSION.fcfs_queue, key=lambda x: (x["priority"], x["ts"]))
    top = ordered[0]["veh_id"] if ordered else veh_id
    if top == veh_id:
        return {
            "veh_id": veh_id,
            "decision": "allow",
            "reason": f"FCFS: priority over {conflicts[0]}",
            "plan": None,
            "model": "fcfs",
            "ts": now,
            "decision_id": f"fcfs-{int(now * 1000)}-{veh_id}",
        }
    return {
        "veh_id": veh_id,
        "decision": "reject",
        "reason": f"FCFS: conflict with veh {conflicts[0]} (first-come-first-serve)",
        "plan": None,
        "model": "fcfs",
        "ts": now,
        "decision_id": f"fcfs-{int(now * 1000)}-{veh_id}",
    }

def marl_review(payload: MecReviewPayload, telemetry_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    veh_id = payload.veh_id
    target_lane = _get_target_lane_from_plan(payload.plan, payload.intent)
    ego = _veh_snapshot(telemetry_snapshot, veh_id)
    if not ego:
        return {
            "veh_id": veh_id,
            "decision": "allow",
            "reason": "policy",
            "plan": None,
            "model": "marl_stub",
            "ts": now,
            "decision_id": f"marl-{int(now * 1000)}-{veh_id}",
        }

    # pick closest other vehicle for policy input
    closest = None
    closest_dist = None
    for key, other in telemetry_snapshot.items():
        oid = other.get("veh_id") or int(key)
        if oid == veh_id:
            continue
        dist = _distance_m(ego, other)
        if dist is None:
            continue
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest = other

    action = _marl_policy_action(ego, closest, target_lane)
    requested_action = _plan_first_action(payload.plan) or ""

    # Map action into plan override if needed
    if action in ("lane_left", "lane_right"):
        decision = "allow" if action == requested_action else "override"
        plan = None if decision == "allow" else {
            "summary": "Policy override: lane change",
            "steps": [
                {
                    "id": "marl-step-1",
                    "description": "Change lane per policy",
                    "action": action,
                    "target_lane_id": target_lane,
                    "target_speed_kmh": None,
                }
            ],
        }
    elif action in ("accelerate", "yield", "speed_up", "speed_down"):
        decision = "override"
        target_speed = FLOW_TARGET_KMH + (5.0 if action in ("accelerate", "speed_up") else -10.0)
        plan = {
            "summary": "Policy override: speed adjust",
            "steps": [
                {
                    "id": "marl-step-1",
                    "description": "Adjust speed to maintain flow",
                    "action": "set_speed",
                    "target_lane_id": ego.get("lane_id"),
                    "target_speed_kmh": target_speed,
                }
            ],
        }
    else:
        decision = "override"
        plan = {
            "summary": "Policy override: hold lane",
            "steps": [
                {
                    "id": "marl-step-1",
                    "description": "Hold lane and maintain flow speed",
                    "action": "set_speed",
                    "target_lane_id": ego.get("lane_id"),
                    "target_speed_kmh": FLOW_TARGET_KMH,
                }
            ],
        }

    return {
        "veh_id": veh_id,
        "decision": decision,
        "reason": "policy",
        "plan": plan,
        "model": (
            "mappo"
            if isinstance(_load_marl_model(), NumpyActorPolicy)
            else "marl_mlp" if _load_marl_model() is not None else "marl_stub"
        ),
        "ts": now,
        "decision_id": f"marl-{int(now * 1000)}-{veh_id}",
    }

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


def attach_collision_sensor(
    bp: carla.BlueprintLibrary, veh: carla.Vehicle
) -> carla.Sensor:
    collision_bp = bp.find("sensor.other.collision")
    return SESSION.world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=veh,
    )


def make_collision_callback(veh_id: int):
    def _callback(event):
        impulse = event.normal_impulse
        row = {
            "ts": time.time(),
            "frame": int(event.frame),
            "veh_id": int(veh_id),
            "other_actor_id": int(event.other_actor.id),
            "other_actor_type": str(event.other_actor.type_id),
            "normal_impulse": {
                "x": float(impulse.x),
                "y": float(impulse.y),
                "z": float(impulse.z),
            },
            "impulse_magnitude": float(
                math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            ),
        }
        with SESSION.collision_lock:
            SESSION.collision_events.append(row)
            SESSION.collision_counts[veh_id] = (
                SESSION.collision_counts.get(veh_id, 0) + 1
            )
            if SESSION.session_dir:
                path = os.path.join(SESSION.session_dir, "collision_events.jsonl")
                try:
                    with open(path, "a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(row, separators=(",", ":")) + "\n"
                        )
                except Exception as exc:
                    print(f"[COLLISION] persist fail veh={veh_id}: {exc}")

    return _callback

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
            if SESSION.session_dir and SESSION.persist_frames:
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

            if SESSION.session_dir and SESSION.persist_frames:
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
            snapshot = world.get_snapshot()
            SESSION.sim_frame = int(snapshot.frame)
            SESSION.sim_time_s = float(snapshot.timestamp.elapsed_seconds)
            SESSION.sim_delta_s = float(snapshot.timestamp.delta_seconds)
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
                    "coordination_mode": SESSION.coordination_mode,
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
        try:
            vctx.last_desired_speed_kmh = kmh
        except Exception:
            pass
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
        # Stop TM control so manual brake persists
        try:
            veh.set_autopilot(False)
        except Exception:
            pass
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
    mode = SESSION.coordination_mode
    if mode == CoordinationMode.IA:
        result = {
            "veh_id": payload.veh_id,
            "decision": "allow",
            "reason": "IA bypass",
            "plan": None,
            "model": "ia",
            "ts": time.time(),
            "decision_id": f"ia-{int(time.time() * 1000)}-{payload.veh_id}",
        }
    elif mode == CoordinationMode.FCFS:
        result = fcfs_review(payload, telemetry)
    elif mode == CoordinationMode.MARL:
        result = marl_review(payload, telemetry)
    else:
        if MIND_V2_ENABLED:
            try:
                states = telemetry_to_states(telemetry)
                proposal = payload_to_proposal(payload, states)
                proposal = proposal.model_copy(
                    update={
                        key: value
                        for key, value in {
                            "transaction_id": payload.transaction_id,
                            "created_at_s": payload.created_at_s,
                            "expires_at_s": payload.expires_at_s,
                            "observation_ts_s": payload.observation_ts_s,
                            "source": payload.source,
                        }.items()
                        if value is not None
                    }
                )
                decision = await MIND_V2.review(proposal, states)
                result = decision_to_legacy(decision, payload.veh_id)
            except Exception as exc:
                # Fail closed even when legacy input cannot be converted into a
                # typed transaction. No API/parser error may become approval.
                now = time.time()
                result = {
                    "veh_id": payload.veh_id,
                    "decision": "reject",
                    "reason": f"MIND-CAV v2 input failure ({type(exc).__name__}); no maneuver authorized.",
                    "reason_code": "V2_INPUT_ERROR",
                    "plan": None,
                    "joint_plan": None,
                    "fallback_plan": None,
                    "validation": {"safe": False, "errors": [type(exc).__name__]},
                    "model": "mind-cav-v2",
                    "ts": now,
                    "decision_id": f"v2-error-{int(now * 1000)}-{payload.veh_id}",
                    "latency_ms": 0.0,
                }
        else:
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
        if mode != CoordinationMode.IA:
            log_mec(entry)
            log_vehicle(entry)
    except Exception:
        pass
    broadcast_ws_json({"type": "mec_decision", **result})
    return result

@app.get("/mec/history")
def mec_history():
    return {"decisions": MEC.get_history()}


def mec_v2_status_payload() -> Dict[str, Any]:
    status = {
        "enabled": MIND_V2_ENABLED,
        "proposer": MIND_V2.proposer.name,
        "semantic_calls": MIND_V2.semantic_calls,
        "semantic_call_cap": MIND_V2.semantic_call_cap,
        "active_transactions": MIND_V2.store.active_transaction_ids(),
        "execution_speed_prediction_v5": bool(
            MIND_V2.validator.config.model_lane_change_execution_speed
        ),
    }
    if isinstance(MIND_V2.proposer, RankedCandidateProposer):
        status.update(
            {
                "ranker_model_path": MIND_V2.proposer.model_path,
                "ranker_model_loaded": MIND_V2.proposer.model is not None,
                "ranker_model_error": MIND_V2.proposer.model_error,
                "liveness_preparation": bool(
                    MIND_V2.proposer.enable_liveness_preparation
                ),
                "global_conflict_recovery": bool(
                    MIND_V2.proposer.generator.enable_global_conflict_recovery
                ),
            }
        )
    return status


@app.get("/mec/v2/status")
def mec_v2_status():
    return mec_v2_status_payload()


@app.post("/mec/v2/ranker")
def update_mec_v2_ranker(cfg: MecV2RankerConfig):
    """Select a frozen local ranker between otherwise identical pilot runs."""
    if MIND_V2.store.active_transaction_ids():
        raise HTTPException(409, "cannot switch ranker with active transactions")
    try:
        proposer = build_ranked_candidate_proposer(cfg.variant, MIND_V2.validator)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    MIND_V2.proposer = proposer
    return mec_v2_status_payload()


@app.post("/mec/v2/outcome")
def mec_v2_outcome(payload: MecOutcomePayload):
    """Record executor progress for an approved MIND-CAV transaction."""
    try:
        if payload.outcome == "executing":
            MIND_V2.store.mark_executing(payload.transaction_id)
        else:
            MIND_V2.store.mark_outcome(
                payload.transaction_id,
                success=payload.outcome == "completed",
                payload=payload.details,
            )
        state = MIND_V2.store.get(payload.transaction_id).state.value
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    entry = {
        "ts": time.time(),
        "source": "mec_execution_outcome",
        "transaction_id": payload.transaction_id,
        "outcome": payload.outcome,
        "state": state,
        "details": payload.details,
    }
    if SESSION.coordination_mode != CoordinationMode.IA:
        log_mec(entry)
    return {"ok": True, "transaction_id": payload.transaction_id, "state": state}

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

@app.get("/tm/unsafe")
def get_tm_unsafe():
    return {"unsafe_mode": SESSION.tm_unsafe_mode}

@app.post("/tm/unsafe")
def set_tm_unsafe(cfg: UnsafeToggle):
    SESSION.tm_unsafe_mode = bool(cfg.unsafe)
    apply_tm_collision_mode(SESSION.tm_unsafe_mode)
    return {"unsafe_mode": SESSION.tm_unsafe_mode}

@app.get("/spawns")
def list_spawns():
    carla_connect()
    out = []
    for i, sp in enumerate(SESSION.spawn_points):
        loc = sp.location
        rot = sp.rotation
        waypoint = SESSION.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        left = waypoint.get_left_lane() if waypoint else None
        right = waypoint.get_right_lane() if waypoint else None
        row = {
            "index": i,
            "x": float(loc.x), "y": float(loc.y), "z": float(loc.z),
            "yaw": float(rot.yaw),
            "road_id": int(waypoint.road_id) if waypoint else None,
            "section_id": int(waypoint.section_id) if waypoint else None,
            "s_m": float(waypoint.s) if waypoint else None,
            "lane_id": int(waypoint.lane_id) if waypoint else None,
            "is_junction": bool(waypoint.is_junction) if waypoint else None,
            "left_lane_id": (
                int(left.lane_id)
                if left and left.lane_type == carla.LaneType.Driving
                else None
            ),
            "right_lane_id": (
                int(right.lane_id)
                if right and right.lane_type == carla.LaneType.Driving
                else None
            ),
            "forward_non_junction_m": forward_non_junction_m(waypoint),
        }
        out.append(row)
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
    SESSION.coordination_mode = cfg.coordination_mode
    SESSION.tm_auto_lane_change_enabled = bool(cfg.tm_auto_lane_change)
    SESSION.run_config = {
        "coordination_mode": cfg.coordination_mode,
        "scenario_id": cfg.scenario_id,
        "seed": cfg.seed,
        "lane_goals": cfg.lane_goals,
        "spawn_indices": cfg.spawn_indices,
        "spawn_longitudinal_offsets_m": cfg.spawn_longitudinal_offsets_m,
        "initial_speeds": cfg.initial_speeds,
        "num_cars": cfg.num_cars,
        "persist_frames": cfg.persist_frames,
        "enable_cameras": cfg.enable_cameras,
        "tm_auto_lane_change": cfg.tm_auto_lane_change,
        "tm_route": list(cfg.tm_route),
        "start_paused": cfg.start_paused,
        "created_at": time.time(),
    }
    SESSION.persist_frames = bool(cfg.persist_frames)
    write_session_meta(SESSION.run_config)

    world = SESSION.world
    tm = SESSION.tm
    bp = SESSION.bp
    world_map = SESSION.map
    if not (world and tm and bp and world_map):
        raise HTTPException(500, "CARLA not ready")

    if cfg.seed is not None:
        try:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            tm.set_random_device_seed(int(cfg.seed))
        except Exception:
            pass

    veh_ids = []
    for i in range(cfg.num_cars):
        sp_idx = cfg.spawn_indices[i] if i < len(cfg.spawn_indices) else cfg.spawn_indices[-1]
        if sp_idx < 0 or sp_idx >= len(SESSION.spawn_points):
            raise HTTPException(400, f"spawn index {sp_idx} invalid")
        sp = SESSION.spawn_points[sp_idx]
        offset_m = float(
            cfg.spawn_longitudinal_offsets_m[i]
            if i < len(cfg.spawn_longitudinal_offsets_m)
            else 0.0
        )
        try:
            sp = shifted_spawn_transform(world_map, sp, offset_m)
        except ValueError as exc:
            raise HTTPException(
                400,
                f"spawn index {sp_idx} offset {offset_m:g} m invalid: {exc}",
            ) from exc
        veh = spawn_vehicle(bp, world, sp)
        initial_speed = float(cfg.initial_speeds[i] if i < len(cfg.initial_speeds) else cfg.initial_speeds[-1])
        tm.set_desired_speed(veh, initial_speed)
        veh.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(veh, SESSION.tm_auto_lane_change_enabled)
        if cfg.tm_route:
            try:
                tm.set_route(veh, list(cfg.tm_route))
            except Exception as exc:
                raise HTTPException(
                    500,
                    f"failed to set Traffic Manager route: {exc}",
                ) from exc

        vctx = VehicleCtx(veh, tm, world_map)
        vctx.last_desired_speed_kmh = initial_speed
        # Camera sensors are optional for headless non-visual experiments.
        # Collision sensing and all kinematic telemetry remain enabled.
        collision_sensor = attach_collision_sensor(bp, veh)
        vctx.collision_sensor = collision_sensor

        if cfg.enable_cameras:
            cam_front = attach_front_cam(bp, veh)
            cam_top = attach_top_cam(bp, veh)
            vctx.cam_front = cam_front
            vctx.cam_top = cam_top
            cam_front.listen(make_front_callback(veh.id))
            cam_top.listen(make_top_callback(veh.id))
        collision_sensor.listen(make_collision_callback(veh.id))

        SESSION.vehicles[veh.id] = vctx
        veh_ids.append(veh.id)

    # Apply current TM safety mode to all vehicles (safe vs. unsafe)
    apply_tm_collision_mode(SESSION.tm_unsafe_mode)

    # global cam
    if SESSION.global_cam:
        safe_destroy([SESSION.global_cam])
        SESSION.global_cam = None
    if cfg.enable_cameras:
        gcam = spawn_global_cam(bp, world, None)
        gcam.listen(global_cam_callback)
        SESSION.global_cam = gcam

    SESSION.selected_vehicle_id = veh_ids[0] if veh_ids else None
    # Auto-select first vehicle so ego camera starts sending frames
    if SESSION.vehicles and SESSION.selected_vehicle_id is None:
        SESSION.selected_vehicle_id = next(iter(SESSION.vehicles.keys()))

    # Dashboard runs start immediately. Experiment runners can opt into a
    # deterministic warm-up barrier and explicitly resume after capturing the
    # first qualifying simulation tick.
    if not cfg.start_paused:
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


@app.post("/simulation/warmup")
def simulation_warmup(request: SimulationWarmupRequest):
    """Advance a paused synchronous world to an exact readiness tick.

    This endpoint is opt-in and intended for paired headless experiments. It
    avoids wall-clock polling jitter while preserving the existing dashboard
    startup behavior.
    """
    world = SESSION.world
    if world is None:
        raise HTTPException(503, "CARLA world is unavailable")
    if SESSION.tick_thread and SESSION.tick_thread.is_alive():
        raise HTTPException(409, "simulation must be paused before warm-up")

    vehicle_ids = request.vehicle_ids or list(SESSION.vehicles)
    missing = sorted(set(vehicle_ids) - set(SESSION.vehicles))
    if missing:
        raise HTTPException(404, f"unknown vehicle ids: {missing}")
    if not vehicle_ids:
        raise HTTPException(409, "no vehicles are configured")

    expected = {str(veh_id) for veh_id in vehicle_ids}
    last: Dict[str, Any] = {}
    for tick_count in range(1, request.max_ticks + 1):
        world.tick()
        snapshot = world.get_snapshot()
        SESSION.sim_frame = int(snapshot.frame)
        SESSION.sim_time_s = float(snapshot.timestamp.elapsed_seconds)
        SESSION.sim_delta_s = float(snapshot.timestamp.delta_seconds)
        last = telemetry_snapshot()
        ready = expected.issubset(last) and all(
            float(last[veh_id].get("speed_kmh") or 0.0)
            >= request.minimum_speed_kmh
            and last[veh_id].get("lane_id") is not None
            and last[veh_id].get("sim_time_s") is not None
            and (
                not request.require_non_junction
                or not bool(last[veh_id].get("is_junction"))
            )
            for veh_id in expected
        )
        if ready:
            return {
                "ok": True,
                "ticks": tick_count,
                "sim_frame": SESSION.sim_frame,
                "sim_time_s": SESSION.sim_time_s,
                "vehicles": {veh_id: last[veh_id] for veh_id in expected},
            }

    state = {
        veh_id: {
            "speed_kmh": (last.get(veh_id) or {}).get("speed_kmh"),
            "lane_id": (last.get(veh_id) or {}).get("lane_id"),
            "is_junction": (last.get(veh_id) or {}).get("is_junction"),
        }
        for veh_id in expected
    }
    raise HTTPException(
        408,
        f"fleet did not become ready within {request.max_ticks} ticks; state={state}",
    )


@app.post("/simulation/resume")
def simulation_resume():
    """Resume the normal background tick loop after deterministic warm-up."""
    if not SESSION.vehicles:
        raise HTTPException(409, "no vehicles are configured")
    start_tick_thread()
    return {
        "ok": True,
        "running": SESSION.running,
        "sim_frame": SESSION.sim_frame,
        "sim_time_s": SESSION.sim_time_s,
    }


@app.post("/simulation/step")
def simulation_step(request: SimulationStepRequest):
    """Advance a paused synchronous world by an exact number of ticks.

    This endpoint is for reproducible headless experiments. Dashboard runs
    continue to use the existing background tick loop.
    """
    world = SESSION.world
    if world is None:
        raise HTTPException(503, "CARLA world is unavailable")
    if SESSION.tick_thread and SESSION.tick_thread.is_alive():
        raise HTTPException(409, "simulation must be paused for fixed stepping")

    vehicle_ids = request.vehicle_ids or list(SESSION.vehicles)
    missing = sorted(set(vehicle_ids) - set(SESSION.vehicles))
    if missing:
        raise HTTPException(404, f"unknown vehicle ids: {missing}")
    if not vehicle_ids:
        raise HTTPException(409, "no vehicles are configured")

    expected = {str(veh_id) for veh_id in vehicle_ids}
    telemetry: Dict[str, Any] = {}
    for _ in range(request.ticks):
        # Commands submitted since the preceding step affect the next exact
        # simulation tick rather than a wall-clock-dependent future frame.
        drain_commands_nonblocking()
        world.tick()
        snapshot = world.get_snapshot()
        SESSION.sim_frame = int(snapshot.frame)
        SESSION.sim_time_s = float(snapshot.timestamp.elapsed_seconds)
        SESSION.sim_delta_s = float(snapshot.timestamp.delta_seconds)
        telemetry = telemetry_snapshot()

    if not expected.issubset(telemetry):
        raise HTTPException(503, "telemetry missing after fixed simulation step")
    selected = {veh_id: telemetry[veh_id] for veh_id in expected}
    log_telemetry({"t": time.time(), "vehicles": selected})
    return {
        "ok": True,
        "ticks": request.ticks,
        "sim_frame": SESSION.sim_frame,
        "sim_time_s": SESSION.sim_time_s,
        "vehicles": selected,
    }

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
        if vctx.collision_sensor: actors.append(vctx.collision_sensor)
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
    with SESSION.collision_lock:
        SESSION.collision_events.clear()
        SESSION.collision_counts.clear()

    # restore TM safety defaults
    SESSION.tm_unsafe_mode = False
    SESSION.tm_auto_lane_change_enabled = True
    apply_tm_collision_mode(False)
    SESSION.coordination_mode = CoordinationMode.MIND_CAVS
    SESSION.run_config = {}
    SESSION.fcfs_queue = []
    SESSION.pending_intents = {}
    SESSION.sim_frame = None
    SESSION.sim_time_s = None
    SESSION.sim_delta_s = None
    SESSION.persist_frames = True
    MIND_V2.reset()

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
            lines = f.readlines()
        if lines and not lines[-1].endswith("\n"):
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                # The logger may still be appending the final JSONL record.
                lines.pop()
        lines = lines[-100:]
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

# Register static fallback last so built assets cannot shadow API/WS routes.
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")

# ---------------- Entry point for standalone run ----------------

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting CARLA AI-in-the-Loop FastAPI server on port 8000 ...")
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
