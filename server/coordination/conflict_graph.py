"""Sparse, finite-horizon conflict graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

from .models import Action, Conflict, JointPlan, VehiclePlan, VehicleState


@dataclass(frozen=True)
class PredictionConfig:
    horizon_s: float = 8.0
    dt_s: float = 0.2
    lane_change_duration_s: float = 3.0
    minimum_clearance_m: float = 5.0
    model_error_margin_m: float = 0.75
    ttc_threshold_s: float = 4.0
    candidate_range_m: float = 120.0
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 4.5
    max_speed_mps: float = 160.0 / 3.6
    # Opt-in for a new evaluation version. Historical campaigns used the
    # submitted lane-change speed throughout the prediction horizon.
    model_lane_change_execution_speed: bool = False
    lane_change_execution_target_speed_kmh: float = 30.0
    lane_change_execution_speed_factor: float = 1.0
    cross_road_range_m: float = 30.0
    cross_road_lateral_m: float = 16.0
    cross_road_vertical_m: float = 2.5
    cross_road_heading_tolerance_deg: float = 30.0


@dataclass
class PredictedPoint:
    t_s: float
    s_m: float
    speed_mps: float
    occupied_lanes: Set[int]


@dataclass
class ConflictGraph:
    vehicle_ids: Set[int] = field(default_factory=set)
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)
    conflicts: List[Conflict] = field(default_factory=list)
    total_pair_count: int = 0
    candidate_pair_count: int = 0
    sample_count: int = 0

    def add_conflict(self, conflict: Conflict) -> None:
        self.vehicle_ids.update((conflict.veh_a, conflict.veh_b))
        self.adjacency.setdefault(conflict.veh_a, set()).add(conflict.veh_b)
        self.adjacency.setdefault(conflict.veh_b, set()).add(conflict.veh_a)
        self.conflicts.append(conflict)

    def conflicts_for(self, veh_id: int) -> List[Conflict]:
        return [c for c in self.conflicts if veh_id in (c.veh_a, c.veh_b)]


def target_lane_for(plan: VehiclePlan, current_lane: int) -> int:
    lane = current_lane
    for step in plan.steps:
        if step.target_lane_id is not None:
            lane = step.target_lane_id
        elif step.action == Action.LANE_LEFT:
            lane += 1
        elif step.action == Action.LANE_RIGHT:
            lane -= 1
    return lane


def target_speed_for(plan: VehiclePlan, current_speed_mps: float) -> float:
    target = current_speed_mps
    for step in plan.steps:
        if step.target_speed_kmh is not None:
            target = step.target_speed_kmh / 3.6
        elif step.action == Action.ACCELERATE:
            target += 10.0 / 3.6
        elif step.action == Action.YIELD:
            target = max(0.0, target - 15.0 / 3.6)
        elif step.action == Action.BRAKE:
            target = 0.0
    return target


def predict_plan(
    state: VehicleState,
    plan: VehiclePlan,
    config: PredictionConfig,
) -> List[PredictedPoint]:
    horizon = min(plan.horizon_s, config.horizon_s)
    steps = max(1, int(round(horizon / config.dt_s)))
    start_lane = state.lane_id
    start_occupied_lanes = state.occupied_lanes()
    target_lane = target_lane_for(plan, start_lane)
    lane_change = target_lane != start_lane
    desired_speed = min(config.max_speed_mps, target_speed_for(plan, state.speed_mps))
    if lane_change and config.model_lane_change_execution_speed:
        # The CARLA executor applies min(factor * current speed, target)
        # during ARMING and keeps that target until DONE/ABORT. The finite
        # prediction horizon covers this temporary maneuver target; it does
        # not model Traffic Manager's variable force-command delay.
        desired_speed = min(
            config.lane_change_execution_speed_factor * state.speed_mps,
            config.lane_change_execution_target_speed_kmh / 3.6,
        )
    speed = state.speed_mps
    s_m = state.longitudinal_position_m()
    output: List[PredictedPoint] = []

    for index in range(steps + 1):
        t_s = index * config.dt_s
        continuing_transition = (
            len(start_occupied_lanes) > 1
            and t_s < state.lane_change_remaining_s
        )
        if lane_change and t_s < config.lane_change_duration_s:
            occupied = start_occupied_lanes | {target_lane}
        elif continuing_transition:
            occupied = set(start_occupied_lanes)
        else:
            occupied = {target_lane if lane_change else start_lane}
        output.append(PredictedPoint(t_s=t_s, s_m=s_m, speed_mps=speed, occupied_lanes=occupied))
        delta = desired_speed - speed
        accel = max(-config.max_decel_mps2, min(config.max_accel_mps2, delta / config.dt_s))
        speed = max(0.0, min(config.max_speed_mps, speed + accel * config.dt_s))
        s_m += speed * config.dt_s
    return output


def robust_center_separation_m(
    state_a: VehicleState,
    state_b: VehicleState,
    config: PredictionConfig,
) -> float:
    return (
        0.5 * (state_a.length_m + state_b.length_m)
        + config.minimum_clearance_m
        + state_a.localization_error_m
        + state_b.localization_error_m
        + config.model_error_margin_m
    )


def _pair_is_candidate(
    state_a: VehicleState,
    plan_a: VehiclePlan,
    state_b: VehicleState,
    plan_b: VehiclePlan,
    config: PredictionConfig,
) -> bool:
    # CARLA may split one continuous carriageway into consecutive road IDs and
    # reset ``s`` at that boundary. Preserve strict road scoping unless the
    # pair is close, heading-aligned, and inside one local road-width corridor.
    if (
        state_a.road_id is not None
        and state_b.road_id is not None
        and state_a.road_id != state_b.road_id
        and not _cross_road_local_corridor(state_a, state_b, config)
    ):
        return False
    lanes_a = state_a.occupied_lanes() | {
        target_lane_for(plan_a, state_a.lane_id)
    }
    lanes_b = state_b.occupied_lanes() | {
        target_lane_for(plan_b, state_b.lane_id)
    }
    if not lanes_a.intersection(lanes_b):
        return False
    if (
        state_a.road_id is not None
        and state_b.road_id is not None
        and state_a.road_id != state_b.road_id
    ):
        longitudinal_gap = abs(_ego_local_delta(state_a, state_b)[0])
    else:
        longitudinal_gap = abs(
            state_a.longitudinal_position_m()
            - state_b.longitudinal_position_m()
        )
    reach = config.candidate_range_m + config.horizon_s * abs(
        state_a.speed_mps - state_b.speed_mps
    )
    return longitudinal_gap <= reach


def _ego_local_delta(
    state_a: VehicleState, state_b: VehicleState
) -> tuple[float, float]:
    yaw = math.radians(state_a.yaw_deg)
    dx = state_b.x_m - state_a.x_m
    dy = state_b.y_m - state_a.y_m
    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return longitudinal, lateral


def _heading_difference_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _cross_road_local_corridor(
    state_a: VehicleState,
    state_b: VehicleState,
    config: PredictionConfig,
) -> bool:
    longitudinal, lateral = _ego_local_delta(state_a, state_b)
    return bool(
        math.hypot(longitudinal, lateral) <= config.cross_road_range_m
        and abs(lateral) <= config.cross_road_lateral_m
        and abs(state_a.z_m - state_b.z_m) <= config.cross_road_vertical_m
        and _heading_difference_deg(state_a.yaw_deg, state_b.yaw_deg)
        <= config.cross_road_heading_tolerance_deg
    )


def _closing_speed(a: PredictedPoint, b: PredictedPoint, b_s_m: float) -> float:
    if a.s_m <= b_s_m:
        return a.speed_mps - b.speed_mps
    return b.speed_mps - a.speed_mps


def build_conflict_graph(
    joint_plan: JointPlan,
    states: Dict[int, VehicleState],
    config: PredictionConfig,
) -> ConflictGraph:
    graph = ConflictGraph(vehicle_ids=set(joint_plan.plans))
    predictions: Dict[int, List[PredictedPoint]] = {}
    for veh_id, plan in joint_plan.plans.items():
        state = states.get(veh_id)
        if state is not None:
            predictions[veh_id] = predict_plan(state, plan, config)

    graph.total_pair_count = len(predictions) * (len(predictions) - 1) // 2
    for veh_a, veh_b in combinations(sorted(predictions), 2):
        state_a, state_b = states[veh_a], states[veh_b]
        plan_a, plan_b = joint_plan.plans[veh_a], joint_plan.plans[veh_b]
        if not _pair_is_candidate(state_a, plan_a, state_b, plan_b, config):
            continue
        graph.candidate_pair_count += 1
        threshold = robust_center_separation_m(state_a, state_b, config)
        road_seam_offset_m = 0.0
        if (
            state_a.road_id is not None
            and state_b.road_id is not None
            and state_a.road_id != state_b.road_id
        ):
            longitudinal, _ = _ego_local_delta(state_a, state_b)
            road_seam_offset_m = (
                predictions[veh_a][0].s_m
                + longitudinal
                - predictions[veh_b][0].s_m
            )
        recorded: Set[str] = set()
        for point_a, point_b in zip(predictions[veh_a], predictions[veh_b]):
            graph.sample_count += 1
            if not point_a.occupied_lanes.intersection(point_b.occupied_lanes):
                continue
            point_b_s_m = point_b.s_m + road_seam_offset_m
            separation = abs(point_a.s_m - point_b_s_m)
            if separation < threshold and "separation" not in recorded:
                graph.add_conflict(
                    Conflict(
                        veh_a=veh_a,
                        veh_b=veh_b,
                        kind="predicted_separation",
                        predicted_at_s=point_a.t_s,
                        separation_m=separation,
                        threshold_m=threshold,
                        details="overlapping lane occupancy below robust centre separation",
                    )
                )
                recorded.add("separation")
            closing = _closing_speed(point_a, point_b, point_b_s_m)
            bumper_gap = max(0.0, separation - 0.5 * (state_a.length_m + state_b.length_m))
            if closing > 0.05:
                ttc = bumper_gap / closing
                if ttc < config.ttc_threshold_s and "ttc" not in recorded:
                    graph.add_conflict(
                        Conflict(
                            veh_a=veh_a,
                            veh_b=veh_b,
                            kind="ttc",
                            predicted_at_s=point_a.t_s,
                            separation_m=separation,
                            threshold_m=threshold,
                            ttc_s=ttc,
                            details="closing time below configured TTC threshold",
                        )
                    )
                    recorded.add("ttc")
            if len(recorded) == 2:
                break
    return graph


def connected_components(graph: ConflictGraph) -> Iterable[Set[int]]:
    remaining = set(graph.vehicle_ids)
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor in graph.adjacency.get(node, set()):
                if neighbor not in component:
                    component.add(neighbor)
                    remaining.discard(neighbor)
                    stack.append(neighbor)
        yield component
