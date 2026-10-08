"""Explicit baseline policies and validator adapters for benchmark v2.

This module keeps mechanism labels honest.  In particular, queue-only FCFS
does not inspect target-lane gaps after granting a request, while the existing
reactive controller remains available under its own name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from server.coordination.models import (
    Action,
    JointPlan,
    PlanStep,
    ValidationResult,
    VehiclePlan,
    VehicleState,
)
from server.coordination.validator import DeterministicPlanValidator, ValidationConfig

from .fcfs_policy import ManeuverRequest
from .fleet_fcfs_policy import FleetFCFSPolicy
from .fleet_highway_env import FleetHighwayEnv
from .highway_env import (
    ACCELERATE,
    ACTION_NAMES,
    KEEP_LANE,
    LANE_LEFT,
    LANE_RIGHT,
    YIELD,
)


class QueueOnlyFleetFCFS(FleetFCFSPolicy):
    """FCFS serialization with no post-grant gap or TTC controller."""

    def _granted_action(
        self,
        env: FleetHighwayEnv,
        request: ManeuverRequest,
    ) -> int:
        agent = request.agent
        changing = abs(
            float(env.lane_destination[agent] - env.lane_pos[agent])
        ) > 1e-4
        if changing:
            return KEEP_LANE
        lane_error = request.target_lane - float(env.lane_pos[agent])
        if abs(lane_error) < 0.08:
            return KEEP_LANE
        return LANE_RIGHT if lane_error > 0.0 else LANE_LEFT


class GapControlledFleetFCFS(FleetFCFSPolicy):
    """Named form of the existing FCFS plus reactive gap/TTC controller."""


@dataclass(frozen=True)
class ActionValidationDecision:
    """Result of deterministic admission applied to one joint action vector."""

    requested_actions: np.ndarray
    authorized_actions: np.ndarray
    intervention_mask: np.ndarray
    final_validation: ValidationResult
    attempted_vehicle_ids: tuple[int, ...]
    rejected_vehicle_ids: tuple[int, ...]

    @property
    def intervention_count(self) -> int:
        return int(np.count_nonzero(self.intervention_mask))


class FleetActionValidator:
    """Translate fleet actions to typed plans and admit a safe subset.

    Non-KEEP requests are considered in a fixed priority order.  A request is
    retained only when the complete candidate joint plan passes the same
    deterministic validator used by MIND-CAV.  Rejected requests become
    KEEP_LANE; no alternative maneuver is synthesized here.
    """

    def __init__(
        self,
        env: FleetHighwayEnv,
        *,
        priority_order: Optional[Iterable[int]] = None,
        validator: Optional[DeterministicPlanValidator] = None,
        allow_cooperative_support: bool = False,
        goal_directed_lane_actions: bool = False,
    ) -> None:
        count = env.cfg.num_vehicles
        priority = tuple(range(count) if priority_order is None else priority_order)
        if sorted(int(index) for index in priority) != list(range(count)):
            raise ValueError("priority_order must contain every vehicle index once")
        self.priority_order = tuple(int(index) for index in priority)
        self.allow_cooperative_support = bool(allow_cooperative_support)
        self.goal_directed_lane_actions = bool(goal_directed_lane_actions)
        self.validator = validator or DeterministicPlanValidator(
            ValidationConfig(
                horizon_s=8.0,
                dt_s=env.cfg.dt_s,
                lane_change_duration_s=env.cfg.lane_change_duration_s,
                minimum_clearance_m=env.cfg.safe_gap_m,
                model_error_margin_m=env.cfg.model_error_margin_m,
                ttc_threshold_s=env.cfg.ttc_threshold_s,
                max_accel_mps2=env.cfg.accel_mps2,
                max_decel_mps2=env.cfg.yield_decel_mps2,
                max_speed_mps=env.cfg.max_speed_kmh / 3.6,
                allowed_lane_ids=set(range(env.cfg.num_lanes)),
            )
        )
        self._transaction_counter = 0

    def authorize(
        self,
        env: FleetHighwayEnv,
        actions: np.ndarray,
    ) -> ActionValidationDecision:
        """Authorize actions without modifying the environment."""
        requested = np.asarray(actions, dtype=np.int64).copy()
        expected_shape = (env.cfg.num_vehicles,)
        if requested.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}")
        if np.any((requested < KEEP_LANE) | (requested > YIELD)):
            raise ValueError("actions contain an unsupported action ID")

        eligible = np.asarray(env.goal_active & ~env.completed, dtype=bool)
        if self.allow_cooperative_support:
            support = np.isin(requested, (ACCELERATE, YIELD))
            cooperative = np.asarray(
                getattr(
                    env,
                    "cooperative",
                    np.ones(env.cfg.num_vehicles, dtype=bool),
                ),
                dtype=bool,
            )
            eligible |= support & cooperative
        normalized = requested.copy()
        normalized[~eligible] = KEEP_LANE
        changing = np.abs(env.lane_destination - env.lane_pos) > 1e-4
        normalized[changing] = KEEP_LANE
        if self.goal_directed_lane_actions:
            for index in range(env.cfg.num_vehicles):
                current_lane = env.current_lane(index)
                goal_lane = int(env.goal_lane[index])
                action = int(normalized[index])
                if action == LANE_LEFT and goal_lane >= current_lane:
                    normalized[index] = KEEP_LANE
                elif action == LANE_RIGHT and goal_lane <= current_lane:
                    normalized[index] = KEEP_LANE

        authorized = np.full(env.cfg.num_vehicles, KEEP_LANE, dtype=np.int64)
        states = fleet_states_from_env(env)
        attempted = []
        rejected = []
        for index in self.priority_order:
            action = int(normalized[index])
            if action == KEEP_LANE:
                continue
            attempted.append(int(env.vehicle_ids[index]))
            candidate = authorized.copy()
            candidate[index] = action
            validation = self._validate(env, states, candidate)
            if validation.safe:
                authorized = candidate
            else:
                rejected.append(int(env.vehicle_ids[index]))

        final_validation = self._validate(env, states, authorized)
        intervention_mask = requested != authorized
        return ActionValidationDecision(
            requested_actions=requested,
            authorized_actions=authorized,
            intervention_mask=intervention_mask,
            final_validation=final_validation,
            attempted_vehicle_ids=tuple(attempted),
            rejected_vehicle_ids=tuple(rejected),
        )

    def admissible_action_mask(
        self,
        env: FleetHighwayEnv,
        authorized_prefix: np.ndarray,
        vehicle_index: int,
    ) -> np.ndarray:
        """Return actions that preserve safety after prior admissions.

        ``authorized_prefix`` contains actions already selected for
        higher-priority vehicles and KEEP_LANE elsewhere.  This makes the
        resulting masks conditional on the same deterministic admission order
        used by :meth:`authorize`.
        """
        prefix = np.asarray(authorized_prefix, dtype=np.int64)
        expected_shape = (env.cfg.num_vehicles,)
        if prefix.shape != expected_shape:
            raise ValueError(f"authorized_prefix must have shape {expected_shape}")
        index = int(vehicle_index)
        if index < 0 or index >= env.cfg.num_vehicles:
            raise IndexError("vehicle_index is outside the fleet")

        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        # KEEP_LANE leaves the already admitted prefix unchanged.  It is also
        # the deterministic fail-safe when the starting snapshot itself is
        # outside a conservative prediction bound.
        mask[KEEP_LANE] = True
        eligible = bool(env.goal_active[index] and not env.completed[index])
        support_eligible = bool(
            self.allow_cooperative_support
            and getattr(env, "cooperative", np.ones(env.cfg.num_vehicles, dtype=bool))[
                index
            ]
        )
        changing = abs(
            float(env.lane_destination[index] - env.lane_pos[index])
        ) > 1e-4
        if changing:
            return mask

        current_lane = env.current_lane(index)
        candidates = [ACCELERATE, YIELD] if eligible or support_eligible else []
        if eligible and self.goal_directed_lane_actions:
            goal_lane = int(env.goal_lane[index])
            if goal_lane < current_lane and current_lane > 0:
                candidates.append(LANE_LEFT)
            elif goal_lane > current_lane and current_lane < env.cfg.num_lanes - 1:
                candidates.append(LANE_RIGHT)
        elif eligible:
            if current_lane > 0:
                candidates.append(LANE_LEFT)
            if current_lane < env.cfg.num_lanes - 1:
                candidates.append(LANE_RIGHT)
        states = fleet_states_from_env(env)
        for action in candidates:
            candidate = prefix.copy()
            candidate[index] = int(action)
            mask[int(action)] = self._validate(env, states, candidate).safe
        return mask

    def _validate(
        self,
        env: FleetHighwayEnv,
        states: dict[int, VehicleState],
        actions: np.ndarray,
    ) -> ValidationResult:
        self._transaction_counter += 1
        plan = joint_plan_from_actions(
            env,
            actions,
            transaction_id=f"action-filter-{self._transaction_counter}",
        )
        now_s = env.steps * env.cfg.dt_s
        return self.validator.validate(plan, states, now_s=now_s)


def fleet_states_from_env(env: FleetHighwayEnv) -> dict[int, VehicleState]:
    """Create a typed, timestamp-consistent global state snapshot."""
    observed_at_s = env.steps * env.cfg.dt_s
    states = {}
    for index, raw_vehicle_id in enumerate(env.vehicle_ids):
        vehicle_id = int(raw_vehicle_id)
        states[vehicle_id] = VehicleState(
            veh_id=vehicle_id,
            observed_at_s=observed_at_s,
            x_m=float(env.x[index]),
            y_m=float(env.lane_pos[index] * env.cfg.lane_width_m),
            yaw_deg=0.0,
            speed_mps=float(env.speed[index]),
            lane_id=env.current_lane(index),
            s_m=float(env.x[index]),
            d_m=float(env.lane_pos[index] * env.cfg.lane_width_m),
            length_m=env.cfg.vehicle_length_m,
            width_m=env.cfg.vehicle_width_m,
            localization_error_m=env.cfg.localization_error_m,
        )
    return states


def validator_masked_argmax(
    env: FleetHighwayEnv,
    validator: FleetActionValidator,
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the best admissible joint action in validator priority order."""
    scores = np.asarray(logits, dtype=np.float64)
    expected_shape = (env.cfg.num_vehicles, len(ACTION_NAMES))
    if scores.shape != expected_shape:
        raise ValueError(f"logits must have shape {expected_shape}")

    actions = np.full(env.cfg.num_vehicles, KEEP_LANE, dtype=np.int64)
    masks = np.zeros(expected_shape, dtype=bool)
    for index in validator.priority_order:
        mask = validator.admissible_action_mask(env, actions, index)
        masks[index] = mask
        masked_scores = np.where(mask, scores[index], -np.inf)
        actions[index] = int(np.argmax(masked_scores))
    return actions, masks


def joint_plan_from_actions(
    env: FleetHighwayEnv,
    actions: np.ndarray,
    *,
    transaction_id: str,
) -> JointPlan:
    """Encode every environment action with an explicit target lane."""
    actions = np.asarray(actions, dtype=np.int64)
    if actions.shape != (env.cfg.num_vehicles,):
        raise ValueError("actions must have shape (num_vehicles,)")
    plans = {}
    for index, raw_vehicle_id in enumerate(env.vehicle_ids):
        vehicle_id = int(raw_vehicle_id)
        action_id = int(actions[index])
        current_lane = env.current_lane(index)
        changing = abs(
            float(env.lane_destination[index] - env.lane_pos[index])
        ) > 1e-4
        if changing:
            target_lane = int(round(float(env.lane_destination[index])))
        elif action_id == LANE_LEFT:
            target_lane = max(0, current_lane - 1)
        elif action_id == LANE_RIGHT:
            target_lane = min(env.cfg.num_lanes - 1, current_lane + 1)
        else:
            target_lane = current_lane

        speed_kmh = float(env.speed[index] * 3.6)
        if action_id == ACCELERATE:
            target_speed = min(env.cfg.max_speed_kmh, speed_kmh + 10.0)
        elif action_id == YIELD:
            target_speed = max(0.0, speed_kmh - 15.0)
        else:
            target_speed = env.cfg.flow_speed_kmh

        plans[vehicle_id] = VehiclePlan(
            veh_id=vehicle_id,
            summary="benchmark action",
            horizon_s=8.0,
            steps=[
                PlanStep(
                    action=_protocol_action(action_id),
                    target_lane_id=target_lane,
                    target_speed_kmh=target_speed,
                    duration_s=(
                        env.cfg.lane_change_duration_s
                        if action_id in {LANE_LEFT, LANE_RIGHT} or changing
                        else env.cfg.dt_s
                    ),
                )
            ],
        )
    return JointPlan(
        transaction_id=transaction_id,
        plans=plans,
        summary="benchmark joint action",
        proposer="benchmark-v2-adapter",
    )


def _protocol_action(action_id: int) -> Action:
    mapping = {
        KEEP_LANE: Action.KEEP_LANE,
        LANE_LEFT: Action.LANE_LEFT,
        LANE_RIGHT: Action.LANE_RIGHT,
        ACCELERATE: Action.ACCELERATE,
        YIELD: Action.YIELD,
    }
    try:
        return mapping[action_id]
    except KeyError as exc:
        raise ValueError(f"unsupported action ID: {action_id}") from exc
