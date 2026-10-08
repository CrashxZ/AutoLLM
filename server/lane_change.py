# server/lane_change.py
"""
Lane Change Controller (Traffic Manager-based FSM)
--------------------------------------------------
Purpose:
  Robust, one-shot lane change using CARLA Traffic Manager (TM) only.
  No manual steering or repeated commands required.

FSM:
  IDLE -> ARMING -> EXECUTING -> SETTLING -> DONE
                         \-> ABORT (on failure/unsafe)
  After DONE/ABORT -> COOLDOWN -> IDLE

Key behaviors:
  - Pre-checks before arming:
      * vehicle speed >= MIN_SPEED_KMH
      * adjacent lane exists and is driving lane
      * not at a junction (waypoint.is_junction)
  - During EXECUTING:
      * TM is instructed: auto_lane_change(False) + force_lane_change(dir)
      * Temporarily clamp desired speed to CLAMP_FACTOR × current speed
  - SETTLING:
      * Wait until vehicle sits in target lane (lane_id match) and is centered (|d_center| < CENTER_THRESH)
  - Completion:
      * Restore prior desired speed (approx), re-enable auto_lane_change
      * Enter COOLDOWN (ignore new requests for COOLDOWN_SEC)
  - ABORT:
      * If junction/no target lane/not moving
      * Restore settings + cooldown

Integration:
  - Construct per vehicle: LaneChangeController(vehicle, traffic_manager, world_map)
  - Initiate: controller.request_change("left"|"right")
  - Query state (and advance FSM): controller.get_public_state()
    (This advances the FSM opportunistically; server calls it in telemetry loop.)

Assumptions:
  - Server runs CARLA in synchronous mode.
  - TM is already configured and vehicle autopilot is ON.
  - Server may change desired speeds independently; we approximate "restore"
    by snapshotting current speed at ARMING.

Public API:
  - request_change(direction: Literal["left","right"]) -> bool
  - get_public_state() -> dict: {"state": ..., "direction": ...}

"""

from __future__ import annotations

import time
import math
import os
from typing import Optional, Literal

try:
    import carla  # type: ignore
except Exception as e:
    # The server module loads CARLA earlier; this file is imported after that.
    raise

# ---------------- Tunables ----------------

MIN_SPEED_KMH: float = float(os.getenv("LANE_CHANGE_MIN_SPEED_KMH", "10.0"))
ALLOW_JUNCTION: bool = os.getenv("LANE_CHANGE_ALLOW_JUNCTION", "0") == "1"
CLAMP_FACTOR: float = float(os.getenv("LANE_CHANGE_SPEED_FACTOR", "1.0"))
CENTER_THRESH: float = 0.50        # meters to lane center to consider "centered"
SETTLING_TIME: float = 0.75        # seconds to remain centered before DONE
COOLDOWN_SEC: float = 0          # seconds after DONE/ABORT before next request
ARMING_TIMEOUT: float = float(
    os.getenv("LANE_CHANGE_ARMING_TIMEOUT_S", "4.0")
)                                   # simulation seconds allowed to reach force speed
EXEC_TIMEOUT: float = 8.0          # seconds allowed in EXECUTING
PROGRESS_TIMEOUT: float = float(
    os.getenv("LANE_CHANGE_PROGRESS_TIMEOUT_S", "1.5")
)                                   # seconds before physical progress is required
MIN_LATERAL_PROGRESS_M: float = float(
    os.getenv("LANE_CHANGE_MIN_PROGRESS_M", "0.25")
)
EXECUTION_TARGET_SPEED_KMH: float = float(
    os.getenv("LANE_CHANGE_EXECUTION_TARGET_SPEED_KMH", "30.0")
)
MAX_FORCE_SPEED_KMH: float = float(
    os.getenv("LANE_CHANGE_MAX_FORCE_SPEED_KMH", "32.0")
)
SETTLING_TIMEOUT: float = float(
    os.getenv("LANE_CHANGE_SETTLING_TIMEOUT_S", "4.0")
)                                   # simulation seconds allowed in SETTLING
OCCUPANCY_HORIZON_S: float = float(
    os.getenv("LANE_CHANGE_OCCUPANCY_HORIZON_S", "3.0")
)

# ---------------- States ------------------

class LaneChangeState:
    IDLE = "IDLE"
    ARMING = "ARMING"
    EXECUTING = "EXECUTING"
    SETTLING = "SETTLING"
    DONE = "DONE"
    ABORT = "ABORT"
    COOLDOWN = "COOLDOWN"   # internal (maps to IDLE externally with cooldown info)

# ---------------- Controller --------------

class LaneChangeController:
    """
    TM-based lane change controller with an internal FSM.

    Note:
      The FSM advances inside get_public_state() to avoid requiring an explicit update()
      call each tick. The server calls get_public_state() while building telemetry
      (every ~0.1s), which is sufficient to progress the FSM.
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        tm: "carla.TrafficManager",
        world_map: "carla.Map",
        get_desired_speed_kmh: Optional[callable] = None,
        get_sim_time_s: Optional[callable] = None,
        get_auto_lane_change_enabled: Optional[callable] = None,
    ):
        self.veh = vehicle
        self.tm = tm
        self.map = world_map
        self._get_desired_speed_kmh = get_desired_speed_kmh
        self._get_sim_time_s = get_sim_time_s
        self._get_auto_lane_change_enabled = get_auto_lane_change_enabled

        # Bind each maneuver to one clock domain. In synchronous CARLA runs,
        # wall time can be several times slower than simulation time because of
        # rendering and telemetry work. Maneuver safety and completion timers
        # therefore use simulation time when the server exposes it, while wall
        # time remains available for latency observability.
        self._fsm_clock_source: str = "wall"
        self._fsm_last_time_s: Optional[float] = None

        # FSM
        self.state: str = LaneChangeState.IDLE
        self.direction: Optional[Literal["left","right"]] = None
        self.t_state_enter: float = 0.0

        # Targets / refs
        self.target_lane_id: Optional[int] = None
        self.source_lane_id: Optional[int] = None
        self._request_started_at: Optional[float] = None
        self._request_started_sim_s: Optional[float] = None
        self._force_issued_at: Optional[float] = None
        self._force_issued_sim_s: Optional[float] = None
        self._settle_entry_time: Optional[float] = None
        self._source_center_distance_m: Optional[float] = None
        self._max_lateral_progress_m: float = 0.0
        self._progress_acknowledged: bool = False

        # Terminal observability. These fields intentionally do not alter the
        # FSM; they expose whether a physical completion followed an internal
        # timeout during calibration.
        self.request_sequence: int = 0
        self.last_terminal_state: Optional[str] = None
        self.last_terminal_reason: Optional[str] = None
        self.last_terminal_at: Optional[float] = None
        self.last_terminal_sim_s: Optional[float] = None
        self.last_duration_s: Optional[float] = None
        self.last_duration_sim_s: Optional[float] = None

        # Speed management
        self._pre_desired_kmh: Optional[float] = None   # approx "pre" speed to restore
        self._clamped: bool = False

        # Cooldown
        self._cooldown_until: float = 0.0

        # Internal guard: if TM auto lane change was previously on; we restore this
        self._prev_auto_lane: Optional[bool] = None

    # ------------- Public API -------------

    def request_change(self, direction: Literal["left","right"]) -> bool:
        """
        One-shot lane change request. Returns True if the FSM accepted the request.
        """
        now = self._fsm_now_s()
        # respect cooldown
        if now < self._cooldown_until:
            return False
        if self.state not in (LaneChangeState.IDLE, LaneChangeState.DONE, LaneChangeState.ABORT, LaneChangeState.COOLDOWN):
            # already busy
            return False

        # Pre-checks
        speed_kmh = self._current_speed_kmh()
        if speed_kmh < MIN_SPEED_KMH:
            self._enter_abort("below_min_speed")
            return False

        wp = self._waypoint(self.veh.get_location())
        if not wp or ((not ALLOW_JUNCTION) and wp.is_junction):
            self._enter_abort("at_junction_or_no_waypoint")
            return False

        adj = wp.get_left_lane() if direction == "left" else wp.get_right_lane()
        if not adj or adj.lane_type != carla.LaneType.Driving:
            self._enter_abort("no_adjacent_driving_lane")
            return False

        # Passed checks: ARMING
        self.direction = direction
        self.source_lane_id = int(wp.lane_id)
        self.target_lane_id = int(adj.lane_id)
        request_wall_s = time.time()
        request_sim_s = self._sim_time_s()
        self._bind_fsm_clock(request_wall_s, request_sim_s)
        self._request_started_at = request_wall_s
        self._request_started_sim_s = request_sim_s
        self._force_issued_at = None
        self._force_issued_sim_s = None
        self._source_center_distance_m = self._lateral_distance_to_center(
            wp,
            self.veh.get_location(),
        )
        self._max_lateral_progress_m = 0.0
        self._progress_acknowledged = False
        self.request_sequence += 1

        # Snapshot desired speed (prefer externally tracked target if available)
        target_kmh = None
        if self._get_desired_speed_kmh is not None:
            try:
                target_kmh = float(self._get_desired_speed_kmh() or 0.0)
            except Exception:
                target_kmh = None
        if target_kmh is None or target_kmh <= 0.0:
            target_kmh = max(speed_kmh, 0.0)
        self._pre_desired_kmh = target_kmh

        # Disable TM auto lane changing during our one-shot
        try:
            if self._prev_auto_lane is None:
                # TM has no getter. The server supplies the configured
                # steady-state value so explicit-control experiments do not
                # silently re-enable unsolicited lane changes after completion.
                self._prev_auto_lane = self._configured_auto_lane_change()
            self.tm.auto_lane_change(self.veh, False)
        except Exception:
            pass

        self._switch_state(LaneChangeState.ARMING)
        return True

    def get_public_state(self) -> dict:
        """
        Advances the FSM opportunistically and returns a small public struct.
        """
        self._maybe_advance()
        active = self.state in {
            LaneChangeState.ARMING,
            LaneChangeState.EXECUTING,
            LaneChangeState.SETTLING,
        }
        occupied_lane_ids = []
        if active:
            occupied_lane_ids = sorted(
                {
                    lane_id
                    for lane_id in (self.source_lane_id, self.target_lane_id)
                    if lane_id is not None
                }
            )
        remaining_s = 0.0
        if active and self._request_started_at is not None:
            elapsed_s = self._request_elapsed_s()
            # Keep a short non-zero horizon while CARLA still reports the
            # vehicle as transitioning, even if TM takes longer than nominal.
            remaining_s = max(0.2, OCCUPANCY_HORIZON_S - elapsed_s)
        return {
            "state": self._public_state_label(),
            "direction": self.direction or "none",
            "target_lane_id": self.target_lane_id,
            "occupied_lane_ids": occupied_lane_ids,
            "lane_change_remaining_s": remaining_s,
            "request_sequence": self.request_sequence,
            "request_started_at": self._request_started_at,
            "request_started_sim_s": self._request_started_sim_s,
            "force_issued_at": self._force_issued_at,
            "force_issued_sim_s": self._force_issued_sim_s,
            "execution_target_speed_kmh": EXECUTION_TARGET_SPEED_KMH,
            "maximum_force_speed_kmh": MAX_FORCE_SPEED_KMH,
            "fsm_clock_source": self._fsm_clock_source,
            "last_terminal_state": self.last_terminal_state,
            "last_terminal_reason": self.last_terminal_reason,
            "last_terminal_at": self.last_terminal_at,
            "last_terminal_sim_s": self.last_terminal_sim_s,
            "last_duration_s": self.last_duration_s,
            "last_duration_sim_s": self.last_duration_sim_s,
            "lateral_progress_m": self._max_lateral_progress_m,
            "progress_acknowledged": self._progress_acknowledged,
        }

    # ------------- FSM mechanics -------------

    def _maybe_advance(self):
        now = self._fsm_now_s()

        # Cooldown handled as separate state; expose as IDLE publicly
        if self.state == LaneChangeState.COOLDOWN:
            if now >= self._cooldown_until:
                self._reset_to_idle()
            return

        if self.state == LaneChangeState.IDLE:
            return

        if self.state == LaneChangeState.ARMING:
            # Town04 calibration shows that Traffic Manager can silently
            # consume forced lane changes at cruise speed without initiating
            # lateral motion. Enter a bounded, shared execution envelope before
            # issuing the one-shot command, then restore the prior target speed
            # when the maneuver terminates.
            if now - self.t_state_enter > ARMING_TIMEOUT:
                self._enter_abort("force_speed_timeout")
                return

            self._apply_speed_clamp()
            if self._current_speed_kmh() > MAX_FORCE_SPEED_KMH:
                return

            self._issue_tm_force_once()
            self._switch_state(LaneChangeState.EXECUTING)
            return

        if self.state == LaneChangeState.EXECUTING:
            # Monitor progress: reached target lane id?
            if now - self.t_state_enter > EXEC_TIMEOUT:
                self._enter_abort("exec_timeout")
                return

            location = self.veh.get_location()
            current_wp = self._waypoint(location)
            if not current_wp:
                if now - self.t_state_enter > PROGRESS_TIMEOUT:
                    self._enter_abort("no_lateral_progress")
                return
            if int(current_wp.lane_id) == int(self.target_lane_id or 0):
                self._progress_acknowledged = True
                # Now ensure centering
                self._settle_entry_time = None
                self._switch_state(LaneChangeState.SETTLING)
                return
            if int(current_wp.lane_id) == int(self.source_lane_id or 0):
                center_distance = self._lateral_distance_to_center(
                    current_wp,
                    location,
                )
                baseline = self._source_center_distance_m or 0.0
                self._max_lateral_progress_m = max(
                    self._max_lateral_progress_m,
                    max(0.0, center_distance - baseline),
                )
                if self._max_lateral_progress_m >= MIN_LATERAL_PROGRESS_M:
                    self._progress_acknowledged = True
            if (
                not self._progress_acknowledged
                and now - self.t_state_enter > PROGRESS_TIMEOUT
            ):
                self._enter_abort("no_lateral_progress")
            return

        if self.state == LaneChangeState.SETTLING:
            # compute distance to lane center
            wp = self._waypoint(self.veh.get_location())
            if not wp:
                if now - self.t_state_enter > SETTLING_TIMEOUT:
                    self._enter_abort("settling_timeout")
                return
            d_center = self._lateral_distance_to_center(
                wp, self.veh.get_location()
            )
            if d_center <= CENTER_THRESH:
                if self._settle_entry_time is None:
                    self._settle_entry_time = now
                if now - self._settle_entry_time >= SETTLING_TIME:
                    self._complete_done()
                    return
            else:
                # reset settling timer if deviated again
                self._settle_entry_time = None
                if now - self.t_state_enter > SETTLING_TIMEOUT:
                    self._enter_abort("settling_timeout")
            return

        if self.state in (LaneChangeState.DONE, LaneChangeState.ABORT):
            # Switch to cooldown immediately (managed by _complete_done/_enter_abort)
            return

    def _issue_tm_force_once(self):
        """Send the current TM one-shot lane-change command."""
        try:
            # TM API: force_lane_change(actor, to_right: bool)
            to_right = (self.direction == "right")
            self.tm.force_lane_change(self.veh, to_right)
            self._force_issued_at = time.time()
            self._force_issued_sim_s = self._sim_time_s()
        except Exception:
            # If this failed, abort early
            self._enter_abort("tm_force_failed")

    def _apply_speed_clamp(self):
        """Optionally limit desired speed during a maneuver.

        The default factor is 1.0, which preserves the requested speed. Earlier
        versions used 0.85 and introduced braking as a side effect of every lane
        change, confounding the coordination comparison.
        """
        if self._clamped:
            return
        try:
            cur_kmh = max(self._current_speed_kmh(), 0.0)
            clamp_kmh = max(
                min(
                    CLAMP_FACTOR * cur_kmh,
                    EXECUTION_TARGET_SPEED_KMH,
                ),
                0.0,
            )
            self.tm.set_desired_speed(self.veh, clamp_kmh)
            self._clamped = True
        except Exception:
            # If TM rejects, carry on without clamping
            self._clamped = False

    def _restore_speed_and_auto(self):
        # Preserve a command that superseded the pre-maneuver target. Restoring
        # the captured value can silently undo a newer MEC yield instruction.
        try:
            restore_kmh = self._pre_desired_kmh
            if self._get_desired_speed_kmh is not None:
                latest_kmh = self._get_desired_speed_kmh()
                if latest_kmh is not None and math.isfinite(float(latest_kmh)):
                    restore_kmh = float(latest_kmh)
            if restore_kmh is not None:
                self.tm.set_desired_speed(self.veh, max(restore_kmh, 0.0))
        except Exception:
            pass
        try:
            # Rejected pre-checks have no captured prior setting. Preserve the
            # server's configuration instead of enabling unsolicited maneuvers.
            auto_lane = (
                self._configured_auto_lane_change()
                if self._prev_auto_lane is None
                else self._prev_auto_lane
            )
            self.tm.auto_lane_change(self.veh, auto_lane)
        except Exception:
            pass
        self._clamped = False
        self._pre_desired_kmh = None
        self._prev_auto_lane = None

    def _configured_auto_lane_change(self) -> bool:
        if self._get_auto_lane_change_enabled is None:
            return True
        try:
            return bool(self._get_auto_lane_change_enabled())
        except Exception:
            return True

    def _complete_done(self):
        self._record_terminal(LaneChangeState.DONE, "target_lane_settled")
        self._restore_speed_and_auto()
        self._switch_state(LaneChangeState.DONE)
        self._enter_cooldown()

    def _enter_abort(self, reason: str):
        self._record_terminal(LaneChangeState.ABORT, reason)
        # If we aborted after disabling auto LC, restore it
        self._restore_speed_and_auto()
        self._switch_state(LaneChangeState.ABORT)
        self._enter_cooldown()

    def _enter_cooldown(self):
        now = self._fsm_now_s()
        self._cooldown_until = now + COOLDOWN_SEC
        self.state = LaneChangeState.COOLDOWN
        self.t_state_enter = now
        # keep direction/target to expose in telemetry during cooldown if needed

    def _reset_to_idle(self):
        self.state = LaneChangeState.IDLE
        self.direction = None
        self.target_lane_id = None
        self.source_lane_id = None
        self._request_started_at = None
        self._request_started_sim_s = None
        self._force_issued_at = None
        self._force_issued_sim_s = None
        self._settle_entry_time = None
        self._source_center_distance_m = None
        self._max_lateral_progress_m = 0.0
        self._progress_acknowledged = False

    def _switch_state(self, new_state: str):
        self.state = new_state
        self.t_state_enter = self._fsm_now_s()

    def _record_terminal(self, state: str, reason: str) -> None:
        now = time.time()
        now_sim_s = self._sim_time_s()
        self.last_terminal_state = state
        self.last_terminal_reason = reason
        self.last_terminal_at = now
        self.last_terminal_sim_s = now_sim_s
        self.last_duration_s = (
            None
            if self._request_started_at is None
            else max(0.0, now - self._request_started_at)
        )
        self.last_duration_sim_s = (
            None
            if self._request_started_sim_s is None or now_sim_s is None
            else max(0.0, now_sim_s - self._request_started_sim_s)
        )

    # ------------- Helpers -----------------

    def _current_speed_kmh(self) -> float:
        vel = self.veh.get_velocity()
        return float(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6)

    def _sim_time_s(self) -> Optional[float]:
        if self._get_sim_time_s is None:
            return None
        try:
            value = self._get_sim_time_s()
            return None if value is None else float(value)
        except Exception:
            return None

    def _bind_fsm_clock(
        self, request_wall_s: float, request_sim_s: Optional[float]
    ) -> None:
        if request_sim_s is None:
            self._fsm_clock_source = "wall"
            self._fsm_last_time_s = request_wall_s
        else:
            self._fsm_clock_source = "simulation"
            self._fsm_last_time_s = request_sim_s

    def _fsm_now_s(self) -> float:
        if self._fsm_clock_source == "simulation":
            sim_time_s = self._sim_time_s()
            if sim_time_s is not None:
                self._fsm_last_time_s = sim_time_s
            if self._fsm_last_time_s is not None:
                return self._fsm_last_time_s
        return time.time()

    def _request_elapsed_s(self) -> float:
        if (
            self._fsm_clock_source == "simulation"
            and self._request_started_sim_s is not None
        ):
            return max(0.0, self._fsm_now_s() - self._request_started_sim_s)
        if self._request_started_at is None:
            return 0.0
        return max(0.0, time.time() - self._request_started_at)

    def _waypoint(self, loc: "carla.Location") -> Optional["carla.Waypoint"]:
        try:
            return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            return None

    @staticmethod
    def _distance_to(a: "carla.Location", b: "carla.Location") -> float:
        return float(math.hypot(a.x - b.x, a.y - b.y))

    @staticmethod
    def _lateral_distance_to_center(
        waypoint: "carla.Waypoint", location: "carla.Location"
    ) -> float:
        """Return lateral offset without including along-road displacement."""
        center = waypoint.transform.location
        yaw_rad = math.radians(float(waypoint.transform.rotation.yaw))
        dx = float(location.x - center.x)
        dy = float(location.y - center.y)
        return abs(-math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy)

    def _public_state_label(self) -> str:
        # Expose COOLDOWN as IDLE but keep direction info
        if self.state == LaneChangeState.COOLDOWN:
            return LaneChangeState.IDLE
        return self.state
