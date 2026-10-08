"""Vectorized kinematic highway environment for coordination experiments.

The environment intentionally models only the decision layer used in the paper:
longitudinal speed control and finite-duration lane changes. It is fast enough for
MARL training while retaining simultaneous actions, collision geometry, TTC, and
scenario-level goals. CARLA remains the higher-fidelity validation environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


KEEP_LANE = 0
LANE_LEFT = 1
LANE_RIGHT = 2
ACCELERATE = 3
YIELD = 4
ACTION_NAMES = ("keep_lane", "lane_left", "lane_right", "accelerate", "yield")


@dataclass(frozen=True)
class HighwayConfig:
    num_agents: int = 2
    num_lanes: int = 4
    dt_s: float = 0.2
    lane_width_m: float = 3.5
    lane_change_duration_s: float = 3.0
    flow_speed_kmh: float = 50.0
    max_speed_kmh: float = 90.0
    accel_mps2: float = 2.0
    yield_decel_mps2: float = 3.0
    vehicle_length_m: float = 4.7
    vehicle_width_m: float = 1.9
    safe_gap_m: float = 5.0
    ttc_threshold_s: float = 4.0
    conflict_distance_m: float = 50.0
    route_length_m: float = 650.0
    exit_position_m: float = 500.0
    max_steps: int = 300


class VectorHighwayEnv:
    """A batched, cooperative highway environment with a shared team reward."""

    observation_dim = 11

    def __init__(
        self,
        num_envs: int = 1,
        config: Optional[HighwayConfig] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = config or HighwayConfig()
        if self.cfg.num_agents != 2:
            raise ValueError("The current policy baseline supports exactly two agents")
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.scenario_ids = np.full(self.num_envs, 1, dtype=np.int64)
        self.x = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.speed = np.zeros_like(self.x)
        self.lane_pos = np.zeros_like(self.x)
        self.lane_destination = np.zeros_like(self.x)
        self.goal_lane = np.zeros((self.num_envs, 2), dtype=np.int64)
        self.goal_is_exit = np.zeros((self.num_envs, 2), dtype=bool)
        self.completed = np.zeros((self.num_envs, 2), dtype=bool)
        self.ever_completed = np.zeros((self.num_envs, 2), dtype=bool)
        self.collision = np.zeros(self.num_envs, dtype=bool)
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_actions = np.zeros((self.num_envs, 2), dtype=np.int64)
        self.reset()

    @property
    def state_dim(self) -> int:
        return 10

    def reset(
        self,
        scenario_ids: Optional[np.ndarray] = None,
        seeds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=np.int64)
            if seeds.shape != (self.num_envs,):
                raise ValueError("seeds must have shape (num_envs,)")
            # One deterministic stream for the complete vectorized reset.
            mixed_seed = int(np.bitwise_xor.reduce(seeds)) if len(seeds) else 0
            self.rng = np.random.default_rng(mixed_seed)

        if scenario_ids is None:
            self.scenario_ids = self.rng.integers(1, 4, size=self.num_envs)
        else:
            scenario_ids = np.asarray(scenario_ids, dtype=np.int64)
            if scenario_ids.shape == ():
                scenario_ids = np.full(self.num_envs, int(scenario_ids), dtype=np.int64)
            if scenario_ids.shape != (self.num_envs,):
                raise ValueError("scenario_ids must have shape (num_envs,)")
            if np.any((scenario_ids < 1) | (scenario_ids > 3)):
                raise ValueError("scenario ids must be 1, 2, or 3")
            self.scenario_ids = scenario_ids.copy()

        self._reset_indices(np.arange(self.num_envs, dtype=np.int64))
        return self.observations(), self.global_state()

    def reset_done(self, done: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reset completed vector slots while preserving active environments."""
        done = np.asarray(done, dtype=bool)
        if done.shape != (self.num_envs,):
            raise ValueError("done must have shape (num_envs,)")
        indices = np.flatnonzero(done)
        if len(indices):
            self.scenario_ids[indices] = self.rng.integers(1, 4, size=len(indices))
            self._reset_indices(indices)
        return self.observations(), self.global_state()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs, 2):
            raise ValueError("actions must have shape (num_envs, 2)")
        if np.any((actions < 0) | (actions >= len(ACTION_NAMES))):
            raise ValueError("invalid action")

        previous_lane_error = np.abs(self.goal_lane - self.lane_pos)
        previous_completed = self.completed.copy()
        previous_ever_completed = self.ever_completed.copy()
        previous_speed = self.speed.copy()

        changing = np.abs(self.lane_destination - self.lane_pos) > 1e-4
        start_left = (actions == LANE_LEFT) & ~changing
        start_right = (actions == LANE_RIGHT) & ~changing
        self.lane_destination = np.where(
            start_left,
            np.maximum(0.0, np.rint(self.lane_pos) - 1.0),
            self.lane_destination,
        )
        self.lane_destination = np.where(
            start_right,
            np.minimum(float(self.cfg.num_lanes - 1), np.rint(self.lane_pos) + 1.0),
            self.lane_destination,
        )

        acceleration = np.zeros_like(self.speed)
        acceleration[actions == ACCELERATE] = self.cfg.accel_mps2
        acceleration[actions == YIELD] = -self.cfg.yield_decel_mps2
        flow_mps = self.cfg.flow_speed_kmh / 3.6
        cruising = (actions == KEEP_LANE) | (actions == LANE_LEFT) | (actions == LANE_RIGHT)
        acceleration[cruising] += np.clip(flow_mps - self.speed[cruising], -1.0, 1.0)
        self.speed += acceleration * self.cfg.dt_s
        self.speed[:] = np.clip(self.speed, 0.0, self.cfg.max_speed_kmh / 3.6)
        self.x += self.speed * self.cfg.dt_s

        lane_step = self.cfg.dt_s / self.cfg.lane_change_duration_s
        lane_delta = self.lane_destination - self.lane_pos
        self.lane_pos += np.clip(lane_delta, -lane_step, lane_step)

        self.steps += 1
        self.last_actions[:] = actions
        self._update_completion()
        just_completed = self.completed & ~previous_ever_completed
        self.ever_completed |= self.completed

        dx = np.abs(self.x[:, 0] - self.x[:, 1])
        dy = np.abs(self.lane_pos[:, 0] - self.lane_pos[:, 1]) * self.cfg.lane_width_m
        self.collision |= (dx < self.cfg.vehicle_length_m) & (dy < self.cfg.vehicle_width_m)
        euclidean_gap = np.sqrt(dx * dx + dy * dy)
        gap_violation = euclidean_gap < self.cfg.safe_gap_m
        ttc = self._pair_ttc()
        ttc_violation = ttc < self.cfg.ttc_threshold_s

        lane_progress = previous_lane_error - np.abs(self.goal_lane - self.lane_pos)
        flow_error = np.abs(self.speed - flow_mps) / max(flow_mps, 1e-6)
        individual = 4.0 * lane_progress - 0.04 * flow_error - 0.01
        individual += just_completed.astype(np.float32) * 25.0
        unnecessary_yield = (actions == YIELD) & (dx[:, None] > self.cfg.conflict_distance_m)
        individual -= unnecessary_yield.astype(np.float32) * 0.4

        cooperative_penalty = gap_violation.astype(np.float32) * 2.0
        cooperative_penalty += ttc_violation.astype(np.float32) * 0.5
        cooperative_penalty += self.collision.astype(np.float32) * 100.0
        agent_reward = individual - cooperative_penalty[:, None]

        success = np.all(self.completed, axis=1)
        previous_success = np.all(previous_completed, axis=1)
        agent_reward += (success & ~previous_success).astype(np.float32)[:, None] * 30.0
        timeout = self.steps >= self.cfg.max_steps
        done = success | timeout | self.collision
        info = {
            "success": success,
            "timeout": timeout,
            "collision": self.collision.copy(),
            "gap_m": euclidean_gap.astype(np.float32),
            "gap_violation": gap_violation,
            "ttc_s": ttc.astype(np.float32),
            "ttc_violation": ttc_violation,
            "speed_delta_mps": (self.speed - previous_speed).astype(np.float32),
            "completed": self.completed.copy(),
        }
        return self.observations(), self.global_state(), agent_reward.astype(np.float32), done, info

    def observations(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, 2, self.observation_dim), dtype=np.float32)
        max_lane = max(self.cfg.num_lanes - 1, 1)
        flow_mps = self.cfg.flow_speed_kmh / 3.6
        for agent in range(2):
            other = 1 - agent
            rel_x = np.clip(self.x[:, other] - self.x[:, agent], -100.0, 100.0)
            rel_speed = np.clip(self.speed[:, other] - self.speed[:, agent], -20.0, 20.0)
            ttc = np.clip(self._agent_ttc(agent, other), 0.0, 20.0)
            obs[:, agent, 0] = self.lane_pos[:, agent] / max_lane
            obs[:, agent, 1] = self.speed[:, agent] / max(flow_mps, 1e-6)
            obs[:, agent, 2] = self.goal_lane[:, agent] / max_lane
            obs[:, agent, 3] = (self.goal_lane[:, agent] - self.lane_pos[:, agent]) / max_lane
            obs[:, agent, 4] = rel_x / 100.0
            obs[:, agent, 5] = (self.lane_pos[:, other] - self.lane_pos[:, agent]) / max_lane
            obs[:, agent, 6] = rel_speed / 20.0
            obs[:, agent, 7] = self.goal_lane[:, other] / max_lane
            obs[:, agent, 8] = ttc / 20.0
            obs[:, agent, 9] = self.goal_is_exit[:, agent].astype(np.float32)
            remaining = np.maximum(self.cfg.exit_position_m - self.x[:, agent], 0.0)
            obs[:, agent, 10] = np.where(
                self.goal_is_exit[:, agent], remaining / self.cfg.exit_position_m, 0.0
            )
        return obs

    def global_state(self) -> np.ndarray:
        state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        max_lane = max(self.cfg.num_lanes - 1, 1)
        state[:, 0:2] = self.x / self.cfg.route_length_m
        state[:, 2:4] = self.lane_pos / max_lane
        state[:, 4:6] = self.speed / (self.cfg.max_speed_kmh / 3.6)
        state[:, 6:8] = self.goal_lane / max_lane
        state[:, 8:10] = self.completed.astype(np.float32)
        return state

    def _update_completion(self) -> None:
        lane_reached = np.abs(self.lane_pos - self.goal_lane) < 0.08
        exit_reached = self.x >= self.cfg.exit_position_m
        exit_goal = lane_reached & exit_reached & self.goal_is_exit
        self.completed = np.where(
            self.goal_is_exit,
            self.completed | exit_goal,
            lane_reached,
        )

    def _reset_indices(self, indices: np.ndarray) -> None:
        count = len(indices)
        if count == 0:
            return
        # Lane 0 is leftmost and lane 3 is rightmost.
        self.lane_pos[indices, 0] = 0.0
        self.lane_pos[indices, 1] = 1.0
        self.lane_destination[indices] = self.lane_pos[indices]
        self.goal_lane[indices, 0] = self.cfg.num_lanes - 1
        self.goal_lane[indices, 1] = 1
        s2_indices = indices[self.scenario_ids[indices] == 2]
        self.goal_lane[s2_indices, 1] = 0
        self.goal_is_exit[indices] = False
        exit_indices = indices[self.scenario_ids[indices] == 3]
        self.goal_is_exit[exit_indices, 0] = True

        self.x[indices, 0] = self.rng.uniform(0.0, 16.0, size=count)
        self.x[indices, 1] = self.x[indices, 0] + self.rng.uniform(-12.0, 12.0, size=count)
        base_speed = self.rng.uniform(35.0, 45.0, size=(count, 2)) / 3.6
        self.speed[indices] = base_speed.astype(np.float32)
        self.completed[indices] = False
        self.ever_completed[indices] = False
        self.collision[indices] = False
        self.steps[indices] = 0
        self.last_actions[indices] = KEEP_LANE
        self._update_completion()
        self.ever_completed[indices] = self.completed[indices]

    def _pair_ttc(self) -> np.ndarray:
        ahead_is_one = self.x[:, 1] >= self.x[:, 0]
        rear_speed = np.where(ahead_is_one, self.speed[:, 0], self.speed[:, 1])
        front_speed = np.where(ahead_is_one, self.speed[:, 1], self.speed[:, 0])
        closing = rear_speed - front_speed
        distance = np.abs(self.x[:, 1] - self.x[:, 0])
        same_corridor = np.abs(self.lane_pos[:, 1] - self.lane_pos[:, 0]) < 0.75
        result = np.full(self.num_envs, np.inf, dtype=np.float32)
        valid = same_corridor & (closing > 0.05)
        np.divide(distance, closing, out=result, where=valid)
        return result

    def _agent_ttc(self, agent: int, other: int) -> np.ndarray:
        other_ahead = self.x[:, other] > self.x[:, agent]
        closing = self.speed[:, agent] - self.speed[:, other]
        distance = np.maximum(self.x[:, other] - self.x[:, agent], 0.0)
        same_corridor = np.abs(self.lane_pos[:, other] - self.lane_pos[:, agent]) < 0.75
        result = np.full(self.num_envs, np.inf, dtype=np.float32)
        valid = other_ahead & same_corridor & (closing > 0.05)
        np.divide(distance, closing, out=result, where=valid)
        return result
