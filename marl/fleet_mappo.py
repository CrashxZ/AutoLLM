"""Variable-fleet observation and reward adapter for MAPPO.

The adapter deliberately leaves :class:`StressFleetHighwayEnv` unchanged.  It
provides a fixed-size decentralized actor observation, a permutation-invariant
centralized critic context, and the predeclared cooperative training reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from .highway_env import YIELD
from .stress_highway_env import StressFleetHighwayEnv


MAX_NEIGHBORS = 4
EGO_FEATURE_NAMES = (
    "lane_position",
    "speed_to_flow",
    "goal_lane",
    "goal_lane_delta",
    "lane_destination",
    "lane_changing",
    "goal_active",
    "goal_is_exit",
    "completed",
    "cooperative",
)
PRIORITY_FEATURE_NAMES = ("priority_rank",)
NEIGHBOR_FEATURE_NAMES = (
    "relative_x",
    "relative_lane",
    "relative_speed",
    "relative_destination",
    "neighbor_goal_delta",
    "neighbor_goal_active",
    "neighbor_cooperative",
    "present",
)
GLOBAL_FEATURE_NAMES = (
    "lane_position",
    "speed_to_flow",
    "goal_lane",
    "goal_lane_delta",
    "lane_destination",
    "goal_active",
    "goal_is_exit",
    "completed",
    "cooperative",
)

OBSERVATION_DIM = len(EGO_FEATURE_NAMES) + MAX_NEIGHBORS * len(
    NEIGHBOR_FEATURE_NAMES
)
PRIORITY_OBSERVATION_DIM = OBSERVATION_DIM + len(PRIORITY_FEATURE_NAMES)
CONTEXT_DIM = 2 * len(GLOBAL_FEATURE_NAMES) + 1


@dataclass(frozen=True)
class FleetRewardConfig:
    lane_progress: float = 4.0
    exit_progress: float = 0.05
    first_completion: float = 25.0
    team_completion: float = 30.0
    flow_error: float = 0.04
    step_cost: float = 0.01
    unnecessary_yield: float = 0.4
    gap_violation: float = 2.0
    ttc_violation: float = 0.5
    collision: float = 100.0
    shield_intervention: float = 0.5
    conflict_distance_m: float = 50.0

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _normalizers(env: StressFleetHighwayEnv) -> Tuple[float, float]:
    return float(max(env.cfg.num_lanes - 1, 1)), float(
        max(env.cfg.flow_speed_kmh / 3.6, 1e-6)
    )


def _ego_features(env: StressFleetHighwayEnv, index: int) -> np.ndarray:
    max_lane, flow_mps = _normalizers(env)
    lane = float(env.lane_pos[index])
    destination = float(env.lane_destination[index])
    goal = float(env.goal_lane[index])
    return np.asarray(
        [
            lane / max_lane,
            float(env.speed[index]) / flow_mps,
            goal / max_lane,
            (goal - lane) / max_lane,
            destination / max_lane,
            float(abs(destination - lane) > 1e-4),
            float(env.goal_active[index]),
            float(env.goal_is_exit[index]),
            float(env.completed[index]),
            float(env.cooperative[index]),
        ],
        dtype=np.float32,
    )


def actor_observations(
    env: StressFleetHighwayEnv,
    *,
    max_neighbors: int = MAX_NEIGHBORS,
    priority_order: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Build one fixed-width local observation per vehicle.

    Neighbors are selected by current absolute longitudinal distance.  Vehicle
    identifier is the deterministic tie breaker; no array-index identity is
    encoded in the features.
    """
    if max_neighbors != MAX_NEIGHBORS:
        raise ValueError(f"max_neighbors is frozen at {MAX_NEIGHBORS}")
    count = env.cfg.num_vehicles
    max_lane, _ = _normalizers(env)
    include_priority = priority_order is not None
    if include_priority:
        if sorted(int(value) for value in priority_order) != list(range(count)):
            raise ValueError("priority_order must contain every vehicle index once")
        priority_rank = {
            int(vehicle_index): rank / max(count - 1, 1)
            for rank, vehicle_index in enumerate(priority_order)
        }
    else:
        priority_rank = {}
    observation_dim = (
        PRIORITY_OBSERVATION_DIM if include_priority else OBSERVATION_DIM
    )
    neighbor_base = len(EGO_FEATURE_NAMES) + int(include_priority)
    observations = np.zeros((count, observation_dim), dtype=np.float32)
    for ego in range(count):
        observations[ego, : len(EGO_FEATURE_NAMES)] = _ego_features(env, ego)
        if include_priority:
            observations[ego, len(EGO_FEATURE_NAMES)] = priority_rank[ego]
        candidates = np.asarray([index for index in range(count) if index != ego])
        distances = np.abs(env.x[candidates] - env.x[ego])
        order = np.lexsort((env.vehicle_ids[candidates], distances))
        for slot, other in enumerate(candidates[order[:MAX_NEIGHBORS]]):
            offset = neighbor_base + slot * len(NEIGHBOR_FEATURE_NAMES)
            features = np.asarray(
                [
                    np.clip(float(env.x[other] - env.x[ego]), -150.0, 150.0)
                    / 150.0,
                    float(env.lane_pos[other] - env.lane_pos[ego]) / max_lane,
                    np.clip(float(env.speed[other] - env.speed[ego]), -20.0, 20.0)
                    / 20.0,
                    float(
                        env.lane_destination[other]
                        - env.lane_destination[ego]
                    )
                    / max_lane,
                    float(env.goal_lane[other] - env.lane_pos[other]) / max_lane,
                    float(env.goal_active[other]),
                    float(env.cooperative[other]),
                    1.0,
                ],
                dtype=np.float32,
            )
            observations[ego, offset : offset + len(features)] = features
    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("non-finite actor observation")
    return observations


def centralized_context(env: StressFleetHighwayEnv) -> np.ndarray:
    """Return a permutation-invariant summary of the complete current fleet."""
    max_lane, flow_mps = _normalizers(env)
    features = np.stack(
        [
            env.lane_pos / max_lane,
            env.speed / flow_mps,
            env.goal_lane / max_lane,
            (env.goal_lane - env.lane_pos) / max_lane,
            env.lane_destination / max_lane,
            env.goal_active.astype(np.float32),
            env.goal_is_exit.astype(np.float32),
            env.completed.astype(np.float32),
            env.cooperative.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    context = np.concatenate(
        [
            np.mean(features, axis=0),
            np.max(features, axis=0),
            np.asarray([env.cfg.num_vehicles / 8.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    if context.shape != (CONTEXT_DIM,) or not np.all(np.isfinite(context)):
        raise FloatingPointError("invalid centralized critic context")
    return context


class FleetMAPPOAdapter:
    """Add MAPPO observations and rewards to one stress environment."""

    def __init__(
        self,
        env: StressFleetHighwayEnv,
        reward_config: FleetRewardConfig | None = None,
    ) -> None:
        self.env = env
        self.reward_config = reward_config or FleetRewardConfig()

    def reset(self, scenario_name: str) -> Tuple[np.ndarray, np.ndarray]:
        self.env.reset_stress(scenario_name)
        return actor_observations(self.env), centralized_context(self.env)

    def observations(self) -> Tuple[np.ndarray, np.ndarray]:
        return actor_observations(self.env), centralized_context(self.env)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, object]]:
        actions = np.asarray(actions, dtype=np.int64)
        previous_lane_error = np.abs(self.env.goal_lane - self.env.lane_pos)
        previous_completed = self.env.completed.copy()
        previous_success = bool(np.all(previous_completed[self.env.goal_active]))
        previous_x = self.env.x.copy()

        done, info = self.env.step(actions)
        cfg = self.reward_config
        lane_progress = previous_lane_error - np.abs(
            self.env.goal_lane - self.env.lane_pos
        )
        lane_progress *= self.env.goal_active.astype(np.float32)
        exit_progress = np.maximum(self.env.x - previous_x, 0.0)
        exit_progress *= (
            self.env.goal_active & self.env.goal_is_exit & ~previous_completed
        ).astype(np.float32)
        just_completed = self.env.completed & ~previous_completed
        flow_mps = max(self.env.cfg.flow_speed_kmh / 3.6, 1e-6)
        flow_error = np.abs(self.env.speed - flow_mps) / flow_mps

        pair_distances = np.asarray(info["pair_center_distance_m"], dtype=np.float32)
        nearest_distance = np.full(self.env.cfg.num_vehicles, np.inf, dtype=np.float32)
        pairs = np.asarray(info["pairs"], dtype=np.int64)
        id_to_index = {
            int(vehicle_id): index
            for index, vehicle_id in enumerate(self.env.vehicle_ids)
        }
        for pair, distance in zip(pairs, pair_distances):
            first = id_to_index[int(pair[0])]
            second = id_to_index[int(pair[1])]
            nearest_distance[first] = min(nearest_distance[first], float(distance))
            nearest_distance[second] = min(nearest_distance[second], float(distance))

        requested = np.asarray(info["requested_actions"], dtype=np.int64)
        unnecessary_yield = (requested == YIELD) & (
            nearest_distance > cfg.conflict_distance_m
        )
        reward = (
            cfg.lane_progress * lane_progress
            + cfg.exit_progress * exit_progress
            + cfg.first_completion * just_completed.astype(np.float32)
            - cfg.flow_error * flow_error
            - cfg.step_cost
            - cfg.unnecessary_yield * unnecessary_yield.astype(np.float32)
            - cfg.shield_intervention
            * np.asarray(info["safety_interventions"], dtype=np.float32)
        )

        team_penalty = 0.0
        team_penalty += cfg.gap_violation * float(
            np.any(np.asarray(info["gap_violation_pairs"], dtype=bool))
        )
        team_penalty += cfg.ttc_violation * float(
            np.any(np.asarray(info["ttc_violation_pairs"], dtype=bool))
        )
        team_penalty += cfg.collision * float(bool(info["collision"]))
        reward -= team_penalty

        success = bool(info["success"])
        if success and not previous_success:
            reward += cfg.team_completion
        if not np.all(np.isfinite(reward)):
            raise FloatingPointError("non-finite MAPPO reward")
        observation, context = self.observations()
        return observation, context, reward.astype(np.float32), done, info
