"""Shared schema helpers for the compact multimodal intent-policy pilot."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from server.coordination.models import VehicleState


INTENT_ACTIONS: tuple[str, ...] = (
    "lane_left",
    "lane_right",
    "hold",
)
REQUEST_ACTIONS: tuple[str, ...] = (
    "none",
    "yield",
    "accelerate",
)
MAX_NEIGHBOURS = 8
EGO_FEATURE_NAMES: tuple[str, ...] = (
    "speed_fraction",
    "current_lane_fraction",
    "goal_lane_fraction",
    "lane_delta_fraction",
    "route_exit",
    "deadline_fraction",
    "fleet_fraction",
)
NEIGHBOUR_FEATURE_NAMES: tuple[str, ...] = (
    "present",
    "relative_longitudinal_fraction",
    "relative_lane_fraction",
    "relative_speed_fraction",
    "range_fraction",
    "ttc_fraction",
)


def protocol_lane_id(lane_index: int) -> int:
    """Map zero-based left-to-right lanes to CARLA-style negative lane IDs."""
    return -(int(lane_index) + 1)


def lane_index(protocol_id: int) -> int:
    return abs(int(protocol_id)) - 1


def ordered_neighbour_ids(
    states: Mapping[int, VehicleState],
    ego_veh_id: int,
    *,
    limit: int = MAX_NEIGHBOURS,
) -> list[int]:
    """Return stable perception slots ordered by range and then identity."""
    ego = states[ego_veh_id]
    ego_s = ego.longitudinal_position_m()
    ordered = sorted(
        (
            (abs(state.longitudinal_position_m() - ego_s), int(veh_id))
            for veh_id, state in states.items()
            if int(veh_id) != int(ego_veh_id)
        ),
        key=lambda item: (item[0], item[1]),
    )
    return [veh_id for _distance, veh_id in ordered[:limit]]


def ego_feature_vector(
    *,
    speed_mps: float,
    current_lane_index: int,
    goal_lane_index: int,
    route_exit: bool,
    deadline_s: float | None,
    fleet_count: int,
) -> list[float]:
    """Build the non-visual input without leaking neighbour kinematics."""
    values = [
        float(np.clip(speed_mps / 25.0, 0.0, 1.5)),
        float(np.clip(current_lane_index / 3.0, 0.0, 1.0)),
        float(np.clip(goal_lane_index / 3.0, 0.0, 1.0)),
        float(np.clip((goal_lane_index - current_lane_index) / 3.0, -1.0, 1.0)),
        float(bool(route_exit)),
        float(np.clip((deadline_s or 0.0) / 60.0, 0.0, 2.0)),
        float(np.clip(fleet_count / 8.0, 0.0, 1.0)),
    ]
    if len(values) != len(EGO_FEATURE_NAMES):  # pragma: no cover - schema guard
        raise AssertionError("ego feature schema mismatch")
    return values


def neighbour_feature_matrix(
    states: Mapping[int, VehicleState],
    ego_veh_id: int,
    neighbour_ids: Sequence[int],
    *,
    limit: int = MAX_NEIGHBOURS,
) -> list[list[float]]:
    """Build padded local-neighbour telemetry in the frozen slot order."""
    ego = states[ego_veh_id]
    ego_s = ego.longitudinal_position_m()
    ego_lane = lane_index(ego.lane_id)
    output: list[list[float]] = []
    for veh_id in list(neighbour_ids)[:limit]:
        other = states[int(veh_id)]
        relative_s = other.longitudinal_position_m() - ego_s
        relative_lane = lane_index(other.lane_id) - ego_lane
        relative_speed = other.speed_mps - ego.speed_mps
        ttc_s = relative_ttc_s(ego, other)
        output.append(
            [
                1.0,
                float(np.clip(relative_s / 120.0, -1.5, 1.5)),
                float(np.clip(relative_lane / 3.0, -1.0, 1.0)),
                float(np.clip(relative_speed / 15.0, -1.5, 1.5)),
                float(np.clip(abs(relative_s) / 120.0, 0.0, 1.5)),
                float(np.clip(ttc_s / 10.0, 0.0, 1.0)) if math.isfinite(ttc_s) else 1.0,
            ]
        )
    padding = [0.0] * len(NEIGHBOUR_FEATURE_NAMES)
    output.extend([list(padding) for _ in range(limit - len(output))])
    return output


def goal_sentence(
    *,
    goal_lane_index: int,
    route_exit: bool,
    deadline_s: float | None,
) -> str:
    if route_exit:
        if deadline_s is not None:
            return f"Take the right exit through lane {goal_lane_index + 1} within {deadline_s:.0f} seconds."
        return f"Take the right exit through lane {goal_lane_index + 1}."
    return f"Reach target lane {goal_lane_index + 1} while maintaining traffic flow."


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_ttc_s(ego: VehicleState, other: VehicleState) -> float:
    """Return bumper-clearance TTC for a same-corridor pair."""
    ego_s = ego.longitudinal_position_m()
    other_s = other.longitudinal_position_m()
    if ego_s <= other_s:
        rear, front = ego, other
    else:
        rear, front = other, ego
    closing = rear.speed_mps - front.speed_mps
    if closing <= 0.05:
        return math.inf
    centre_distance = abs(ego_s - other_s)
    clearance = max(0.0, centre_distance - 0.5 * (ego.length_m + other.length_m))
    return clearance / closing


def class_index(values: Sequence[str], value: str) -> int:
    try:
        return tuple(values).index(value)
    except ValueError as exc:
        raise ValueError(f"unsupported class {value!r}") from exc
