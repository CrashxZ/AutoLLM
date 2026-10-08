"""Portable NumPy inference for the MAPPO shared actor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .highway_env import ACTION_NAMES


def lane_id_to_position(lane_id: Any, num_lanes: int = 4) -> float:
    """Map Town04 negative driving-lane ids to left-to-right positions."""
    try:
        lane = abs(int(lane_id)) - 1
    except (TypeError, ValueError):
        lane = 0
    return float(np.clip(lane, 0, num_lanes - 1))


def telemetry_observation(
    ego: Dict[str, Any],
    other: Optional[Dict[str, Any]],
    target_lane_id: Optional[int],
    *,
    flow_speed_kmh: float = 50.0,
    num_lanes: int = 4,
    goal_is_exit: bool = False,
) -> np.ndarray:
    """Build the same local observation used by the kinematic trainer."""
    max_lane = max(num_lanes - 1, 1)
    ego_lane = lane_id_to_position(ego.get("lane_id"), num_lanes)
    target_lane = lane_id_to_position(
        target_lane_id if target_lane_id is not None else ego.get("lane_id"), num_lanes
    )
    ego_speed_mps = float(ego.get("speed_kmh") or 0.0) / 3.6
    flow_mps = flow_speed_kmh / 3.6

    rel_x = 100.0
    rel_lane = 0.0
    rel_speed = 0.0
    other_target = ego_lane
    ttc_s = 20.0
    if other:
        other_lane = lane_id_to_position(other.get("lane_id"), num_lanes)
        rel_lane = other_lane - ego_lane
        other_speed_mps = float(other.get("speed_kmh") or 0.0) / 3.6
        rel_speed = other_speed_mps - ego_speed_mps
        other_target = other_lane
        try:
            dx = float(other["pose"]["x"]) - float(ego["pose"]["x"])
            dy = float(other["pose"]["y"]) - float(ego["pose"]["y"])
            yaw_rad = math.radians(float(ego["pose"].get("yaw") or 0.0))
            rel_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
        except (KeyError, TypeError, ValueError):
            rel_x = 100.0
        closing = ego_speed_mps - other_speed_mps
        if rel_x > 0.0 and abs(rel_lane) < 0.75 and closing > 0.05:
            ttc_s = rel_x / closing

    return np.asarray(
        [
            ego_lane / max_lane,
            ego_speed_mps / max(flow_mps, 1e-6),
            target_lane / max_lane,
            (target_lane - ego_lane) / max_lane,
            np.clip(rel_x, -100.0, 100.0) / 100.0,
            rel_lane / max_lane,
            np.clip(rel_speed, -20.0, 20.0) / 20.0,
            other_target / max_lane,
            np.clip(ttc_s, 0.0, 20.0) / 20.0,
            float(goal_is_exit),
            0.0,
        ],
        dtype=np.float32,
    )


class NumpyActorPolicy:
    def __init__(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        if len(weights) != len(biases) or not weights:
            raise ValueError("actor requires matching non-empty weights and biases")
        self.weights = [np.asarray(value, dtype=np.float32) for value in weights]
        self.biases = [np.asarray(value, dtype=np.float32) for value in biases]

    @classmethod
    def load(cls, path: str) -> "NumpyActorPolicy":
        archive = np.load(path, allow_pickle=False)
        layer_count = int(archive["layer_count"])
        weights = [archive[f"weight_{index}"] for index in range(layer_count)]
        biases = [archive[f"bias_{index}"] for index in range(layer_count)]
        return cls(weights, biases)

    def logits(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight.T + bias
            if index < len(self.weights) - 1:
                value = np.tanh(value)
        return value

    def action(self, observation: np.ndarray) -> int:
        logits = self.logits(observation)
        return int(np.argmax(logits, axis=-1))

    def action_name(self, observation: np.ndarray) -> str:
        return ACTION_NAMES[self.action(observation)]


def write_policy_metadata(path: str, training_metadata: Dict[str, Any]) -> None:
    metadata_path = Path(path).with_suffix(".json")
    metadata_path.write_text(json.dumps(training_metadata, indent=2) + "\n", encoding="utf-8")
