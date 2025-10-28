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
from typing import Optional, Literal

try:
    import carla  # type: ignore
except Exception as e:
    # The server module loads CARLA earlier; this file is imported after that.
    raise

# ---------------- Tunables ----------------

MIN_SPEED_KMH: float = 10.0        # min speed to allow a lane change
CLAMP_FACTOR: float = 0.85         # temp desired speed factor during maneuver
CENTER_THRESH: float = 0.50        # meters to lane center to consider "centered"
SETTLING_TIME: float = 0.75        # seconds to remain centered before DONE
COOLDOWN_SEC: float = 0          # seconds after DONE/ABORT before next request
ARMING_TIMEOUT: float = 2.0        # seconds allowed in ARMING
EXEC_TIMEOUT: float = 8.0          # seconds allowed in EXECUTING
SETTLING_TIMEOUT: float = 3.0      # seconds allowed in SETTLING

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

    def __init__(self, vehicle: "carla.Vehicle", tm: "carla.TrafficManager", world_map: "carla.Map"):
        self.veh = vehicle
        self.tm = tm
        self.map = world_map

        # FSM
        self.state: str = LaneChangeState.IDLE
        self.direction: Optional[Literal["left","right"]] = None
        self.t_state_enter: float = 0.0

        # Targets / refs
        self.target_lane_id: Optional[int] = None
        self._settle_entry_time: Optional[float] = None

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
        now = time.time()
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
        if not wp or wp.is_junction:
            self._enter_abort("at_junction_or_no_waypoint")
            return False

        adj = wp.get_left_lane() if direction == "left" else wp.get_right_lane()
        if not adj or adj.lane_type != carla.LaneType.Driving:
            self._enter_abort("no_adjacent_driving_lane")
            return False

        # Passed checks: ARMING
        self.direction = direction
        self.target_lane_id = int(adj.lane_id)

        # Snapshot current effective speed as "to restore" later
        self._pre_desired_kmh = max(speed_kmh, 0.0)

        # Disable TM auto lane changing during our one-shot
        try:
            # store prior
            if self._prev_auto_lane is None:
                self._prev_auto_lane = True  # default assume ON
            # TM has no getter; we assume ON and enforce OFF for maneuver.
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
        return {"state": self._public_state_label(), "direction": self.direction or "none"}

    # ------------- FSM mechanics -------------

    def _maybe_advance(self):
        now = time.time()

        # Cooldown handled as separate state; expose as IDLE publicly
        if self.state == LaneChangeState.COOLDOWN:
            if now >= self._cooldown_until:
                self._reset_to_idle()
            return

        if self.state == LaneChangeState.IDLE:
            return

        if self.state == LaneChangeState.ARMING:
            # Arm immediately by issuing TM force lane change
            if now - self.t_state_enter > ARMING_TIMEOUT:
                self._enter_abort("arming_timeout")
                return

            # Issue TM command once on ARMING entry
            self._issue_tm_force_once()
            # Apply speed clamp
            self._apply_speed_clamp()

            self._switch_state(LaneChangeState.EXECUTING)
            return

        if self.state == LaneChangeState.EXECUTING:
            # Monitor progress: reached target lane id?
            if now - self.t_state_enter > EXEC_TIMEOUT:
                self._enter_abort("exec_timeout")
                return

            current_wp = self._waypoint(self.veh.get_location())
            if not current_wp:
                return
            if int(current_wp.lane_id) == int(self.target_lane_id or 0):
                # Now ensure centering
                self._settle_entry_time = None
                self._switch_state(LaneChangeState.SETTLING)
            return

        if self.state == LaneChangeState.SETTLING:
            if now - self.t_state_enter > SETTLING_TIMEOUT:
                self._enter_abort("settling_timeout")
                return
            # compute distance to lane center
            wp = self._waypoint(self.veh.get_location())
            if not wp:
                return
            d_center = self._distance_to(wp.transform.location, self.veh.get_location())
            if d_center <= CENTER_THRESH:
                if self._settle_entry_time is None:
                    self._settle_entry_time = now
                if now - self._settle_entry_time >= SETTLING_TIME:
                    self._complete_done()
            else:
                # reset settling timer if deviated again
                self._settle_entry_time = None
            return

        if self.state in (LaneChangeState.DONE, LaneChangeState.ABORT):
            # Switch to cooldown immediately (managed by _complete_done/_enter_abort)
            return

    def _issue_tm_force_once(self):
        """Send the TM one-shot lane change command."""
        try:
            # TM API: force_lane_change(actor, to_right: bool)
            to_right = (self.direction == "right")
            self.tm.force_lane_change(self.veh, to_right)
        except Exception:
            # If this failed, abort early
            self._enter_abort("tm_force_failed")

    def _apply_speed_clamp(self):
        """Clamp TM desired speed during maneuver, relative to current speed."""
        if self._clamped:
            return
        try:
            cur_kmh = max(self._current_speed_kmh(), 0.0)
            clamp_kmh = max(CLAMP_FACTOR * cur_kmh, 0.0)
            self.tm.set_desired_speed(self.veh, clamp_kmh)
            self._clamped = True
        except Exception:
            # If TM rejects, carry on without clamping
            self._clamped = False

    def _restore_speed_and_auto(self):
        # Restore desired speed (approx) and re-enable auto lane change as before
        try:
            if self._pre_desired_kmh is not None:
                self.tm.set_desired_speed(self.veh, max(self._pre_desired_kmh, 0.0))
        except Exception:
            pass
        try:
            # Re-enable auto lane change (assume ON pre-maneuver if unknown)
            self.tm.auto_lane_change(self.veh, True if self._prev_auto_lane is None else self._prev_auto_lane)
        except Exception:
            pass
        self._clamped = False
        self._pre_desired_kmh = None
        self._prev_auto_lane = None

    def _complete_done(self):
        self._restore_speed_and_auto()
        self._switch_state(LaneChangeState.DONE)
        self._enter_cooldown()

    def _enter_abort(self, reason: str):
        # If we aborted after disabling auto LC, restore it
        self._restore_speed_and_auto()
        self._switch_state(LaneChangeState.ABORT)
        self._enter_cooldown()

    def _enter_cooldown(self):
        self._cooldown_until = time.time() + COOLDOWN_SEC
        self.state = LaneChangeState.COOLDOWN
        self.t_state_enter = time.time()
        # keep direction/target to expose in telemetry during cooldown if needed

    def _reset_to_idle(self):
        self.state = LaneChangeState.IDLE
        self.direction = None
        self.target_lane_id = None
        self._settle_entry_time = None

    def _switch_state(self, new_state: str):
        self.state = new_state
        self.t_state_enter = time.time()

    # ------------- Helpers -----------------

    def _current_speed_kmh(self) -> float:
        vel = self.veh.get_velocity()
        return float(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6)

    def _waypoint(self, loc: "carla.Location") -> Optional["carla.Waypoint"]:
        try:
            return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            return None

    @staticmethod
    def _distance_to(a: "carla.Location", b: "carla.Location") -> float:
        return float(math.hypot(a.x - b.x, a.y - b.y))

    def _public_state_label(self) -> str:
        # Expose COOLDOWN as IDLE but keep direction info
        if self.state == LaneChangeState.COOLDOWN:
            return LaneChangeState.IDLE
        return self.state
