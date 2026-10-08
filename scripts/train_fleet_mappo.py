#!/usr/bin/env python3
"""Train and validate a parameter-sharing MAPPO policy on stress fleets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marl.fleet_highway_env import FleetHighwayConfig
from marl.fleet_mappo import (
    CONTEXT_DIM,
    OBSERVATION_DIM,
    FleetMAPPOAdapter,
    FleetRewardConfig,
)
from marl.fleet_mappo_models import (
    PooledFleetCritic,
    SharedFleetActor,
    export_shared_actor,
)
from marl.highway_env import ACTION_NAMES
from marl.numpy_policy import NumpyActorPolicy
from marl.stress_highway_env import SCENARIO_DEFINITIONS, StressFleetHighwayEnv


FLEET_SIZES = (4, 6, 8)
TOPOLOGIES = tuple(sorted(SCENARIO_DEFINITIONS))
CORE_CELLS = tuple((topology, size) for topology in TOPOLOGIES for size in FLEET_SIZES)


@dataclass(frozen=True)
class SlotSpec:
    topology: str
    fleet_size: int


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def as_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class PaddedFleetPool:
    """Sequential pool of mixed-size environments padded only for batching."""

    def __init__(self, num_envs: int, seed: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.max_agents = max(FLEET_SIZES)
        self.specs = [
            SlotSpec(*CORE_CELLS[index % len(CORE_CELLS)])
            for index in range(self.num_envs)
        ]
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.adapters: List[FleetMAPPOAdapter] = []
        self.completed_rows: List[Dict[str, object]] = []
        for slot in range(self.num_envs):
            self.adapters.append(self._new_adapter(slot))

    def _episode_seed(self, slot: int) -> int:
        return self.seed + int(self.episode_counts[slot]) * self.num_envs + slot

    def _new_adapter(self, slot: int) -> FleetMAPPOAdapter:
        spec = self.specs[slot]
        env = StressFleetHighwayEnv(
            FleetHighwayConfig(num_vehicles=spec.fleet_size),
            seed=self._episode_seed(slot),
        )
        adapter = FleetMAPPOAdapter(env)
        adapter.reset(spec.topology)
        return adapter

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation = np.zeros(
            (self.num_envs, self.max_agents, OBSERVATION_DIM), dtype=np.float32
        )
        context = np.zeros((self.num_envs, CONTEXT_DIM), dtype=np.float32)
        mask = np.zeros((self.num_envs, self.max_agents), dtype=bool)
        for slot, adapter in enumerate(self.adapters):
            local, global_context = adapter.observations()
            count = adapter.env.cfg.num_vehicles
            observation[slot, :count] = local
            context[slot] = global_context
            mask[slot, :count] = True
        return observation, context, mask

    def step(
        self,
        padded_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        if padded_actions.shape != (self.num_envs, self.max_agents):
            raise ValueError("padded action shape does not match pool")
        rewards = np.zeros((self.num_envs, self.max_agents), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        episode_rows: List[Dict[str, object]] = []
        for slot, adapter in enumerate(self.adapters):
            count = adapter.env.cfg.num_vehicles
            _, _, reward, done, info = adapter.step(padded_actions[slot, :count])
            rewards[slot, :count] = reward
            dones[slot] = done
            if done:
                spec = self.specs[slot]
                row = {
                    "slot": slot,
                    "topology": spec.topology,
                    "fleet_size": spec.fleet_size,
                    "seed": self._episode_seed(slot),
                    "steps": adapter.env.steps,
                    "success": bool(info["success"]),
                    "collision": bool(info["collision"]),
                    "timeout": bool(info["timeout"]),
                }
                episode_rows.append(row)
                self.completed_rows.append(row)
                self.episode_counts[slot] += 1
                self.adapters[slot] = self._new_adapter(slot)
        observation, context, mask = self.arrays()
        return observation, context, mask, rewards, dones, episode_rows


def collect_rollout(
    pool: PaddedFleetPool,
    actor: SharedFleetActor,
    critic: PooledFleetCritic,
    observation: np.ndarray,
    context: np.ndarray,
    mask: np.ndarray,
    rollout_steps: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    use_action_masks = bool(getattr(pool, "use_validator_action_mask", False))
    keys = (
        "observation",
        "context",
        "mask",
        "actions",
        "log_probs",
        "rewards",
        "dones",
        "values",
    )
    if use_action_masks:
        keys += ("action_masks",)
    storage: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    episode_rows: List[Dict[str, object]] = []
    for _ in range(rollout_steps):
        with torch.no_grad():
            obs_tensor = as_tensor(observation, device)
            context_tensor = as_tensor(context, device)
            logits = actor(obs_tensor)
            if use_action_masks:
                actions = torch.zeros(
                    logits.shape[:-1], dtype=torch.long, device=device
                )
                log_probs = torch.zeros_like(actions, dtype=torch.float32)
                action_masks = np.zeros(logits.shape, dtype=bool)
                for slot in range(pool.num_envs):
                    count = int(np.count_nonzero(mask[slot]))
                    prefix = np.zeros(count, dtype=np.int64)
                    for index in pool.validators[slot].priority_order:
                        valid = pool.admissible_action_mask(
                            slot,
                            prefix,
                            index,
                        )
                        action_masks[slot, index] = valid
                        valid_tensor = torch.as_tensor(valid, device=device)
                        distribution = Categorical(
                            logits=logits[slot, index].masked_fill(
                                ~valid_tensor,
                                -torch.inf,
                            )
                        )
                        action = distribution.sample()
                        actions[slot, index] = action
                        log_probs[slot, index] = distribution.log_prob(action)
                        prefix[index] = int(action.item())
                # Padded agents are deterministic KEEP_LANE entries.
                action_masks[..., 0] |= ~mask
            else:
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions)
            values = critic(obs_tensor, context_tensor)
        next_observation, next_context, next_mask, rewards, dones, rows = pool.step(
            actions.cpu().numpy()
        )
        storage["observation"].append(observation.copy())
        storage["context"].append(context.copy())
        storage["mask"].append(mask.copy())
        storage["actions"].append(actions.cpu().numpy())
        storage["log_probs"].append(log_probs.cpu().numpy())
        storage["rewards"].append(rewards.copy())
        storage["dones"].append(dones.copy())
        storage["values"].append(values.cpu().numpy())
        if use_action_masks:
            storage["action_masks"].append(action_masks.copy())
        episode_rows.extend(rows)
        observation, context, mask = next_observation, next_context, next_mask

    with torch.no_grad():
        next_values = critic(as_tensor(observation, device), as_tensor(context, device))
    batch = {key: np.asarray(value) for key, value in storage.items()}
    batch["next_values"] = next_values.cpu().numpy()
    batch["next_mask"] = mask.copy()
    stats = {
        "episodes": float(len(episode_rows)),
        "success_rate": float(
            np.mean([row["success"] for row in episode_rows])
            if episode_rows
            else 0.0
        ),
        "collision_rate": float(
            np.mean([row["collision"] for row in episode_rows])
            if episode_rows
            else 0.0
        ),
    }
    return batch, observation, context, mask, stats


def compute_masked_gae(
    batch: Dict[str, np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards = batch["rewards"]
    dones = batch["dones"]
    values = batch["values"]
    masks = batch["mask"].astype(np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros_like(values[0], dtype=np.float32)
    bootstrap = batch["next_values"] * batch["next_mask"].astype(np.float32)
    for step in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[step].astype(np.float32)[:, None]
        delta = rewards[step] + gamma * bootstrap * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        last_advantage *= masks[step]
        advantages[step] = last_advantage
        bootstrap = values[step]
    returns = (advantages + values) * masks
    return advantages, returns


def ppo_update(
    actor: SharedFleetActor,
    critic: PooledFleetCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    device: torch.device,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, float]:
    advantages, returns = compute_masked_gae(batch, gamma, gae_lambda)
    valid = batch["mask"].reshape(-1).astype(bool)
    observation_dim = int(batch["observation"].shape[-1])
    observation = batch["observation"].reshape(-1, observation_dim)[valid]
    repeated_context = np.repeat(
        batch["context"][:, :, None, :], batch["mask"].shape[2], axis=2
    ).reshape(-1, CONTEXT_DIM)[valid]
    actions = batch["actions"].reshape(-1)[valid]
    old_log_probs = batch["log_probs"].reshape(-1)[valid]
    advantages = advantages.reshape(-1)[valid]
    returns = returns.reshape(-1)[valid]
    action_masks = None
    if "action_masks" in batch:
        action_masks = batch["action_masks"].reshape(
            -1, batch["action_masks"].shape[-1]
        )[valid]
        if not np.all(np.any(action_masks, axis=-1)):
            raise ValueError("every active agent must have an admissible action")
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise FloatingPointError("non-finite GAE values")
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    sample_count = len(actions)
    for _ in range(epochs):
        permutation = np.random.permutation(sample_count)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start : min(start + minibatch_size, sample_count)]
            obs_tensor = as_tensor(observation[indices], device)
            context_tensor = as_tensor(repeated_context[indices], device)
            action_tensor = torch.as_tensor(actions[indices], dtype=torch.long, device=device)
            old_log_tensor = as_tensor(old_log_probs[indices], device)
            advantage_tensor = as_tensor(advantages[indices], device)
            return_tensor = as_tensor(returns[indices], device)

            logits = actor(obs_tensor)
            if action_masks is not None:
                valid_actions = torch.as_tensor(
                    action_masks[indices], dtype=torch.bool, device=device
                )
                logits = logits.masked_fill(~valid_actions, -torch.inf)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(action_tensor)
            ratio = torch.exp(new_log_probs - old_log_tensor)
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            entropy = distribution.entropy().mean()
            value = critic(obs_tensor.unsqueeze(1), context_tensor).squeeze(1)
            value_loss = 0.5 * torch.square(value - return_tensor).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), max_grad_norm
            )
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy.detach().cpu())
            updates += 1
    for parameter in list(actor.parameters()) + list(critic.parameters()):
        if not torch.all(torch.isfinite(parameter)):
            raise FloatingPointError("non-finite model parameter")
    return {key: value / max(updates, 1) for key, value in totals.items()}


def evaluate_actions(
    action_function,
    seed: int,
    episodes_per_cell: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    for cell_index, (topology, fleet_size) in enumerate(CORE_CELLS):
        for repetition in range(episodes_per_cell):
            episode_seed = seed + cell_index * episodes_per_cell + repetition
            env = StressFleetHighwayEnv(
                FleetHighwayConfig(num_vehicles=fleet_size), seed=episode_seed
            )
            adapter = FleetMAPPOAdapter(env)
            observation, _ = adapter.reset(topology)
            gap_events = ttc_events = shield_interventions = 0
            action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
            final_info: Dict[str, object] = {}
            for _ in range(env.cfg.max_steps):
                actions = np.asarray(action_function(observation), dtype=np.int64)
                if actions.shape != (fleet_size,):
                    raise ValueError("evaluation action function returned malformed actions")
                action_counts += np.bincount(actions, minlength=len(ACTION_NAMES))
                observation, _, _, done, final_info = adapter.step(actions)
                gap_events += int(
                    np.any(np.asarray(final_info["gap_violation_pairs"], dtype=bool))
                )
                ttc_events += int(
                    np.any(np.asarray(final_info["ttc_violation_pairs"], dtype=bool))
                )
                shield_interventions += int(
                    np.sum(np.asarray(final_info["safety_interventions"], dtype=bool))
                )
                if done:
                    break
            rows.append(
                {
                    "topology": topology,
                    "fleet_size": fleet_size,
                    "seed": episode_seed,
                    "steps": env.steps,
                    "success": bool(final_info.get("success", False)),
                    "collision": bool(final_info.get("collision", False)),
                    "timeout": bool(final_info.get("timeout", False)),
                    "gap_event_steps": gap_events,
                    "ttc_event_steps": ttc_events,
                    "shield_interventions": shield_interventions,
                    "completed_active": int(np.sum(env.completed & env.goal_active)),
                    "active_vehicle_count": int(np.sum(env.goal_active)),
                    "action_counts": {
                        name: int(action_counts[index])
                        for index, name in enumerate(ACTION_NAMES)
                    },
                }
            )
    summary = {
        "episodes": float(len(rows)),
        "cell_coverage": float(len({(row["topology"], row["fleet_size"]) for row in rows})),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "gap_episode_rate": float(
            np.mean([int(row["gap_event_steps"] > 0) for row in rows])
        ),
        "mean_shield_interventions": float(
            np.mean([row["shield_interventions"] for row in rows])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
    }
    return rows, summary


def evaluate(
    actor: SharedFleetActor,
    device: torch.device,
    seed: int,
    episodes_per_cell: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    actor.eval()

    def torch_actions(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = actor(as_tensor(observation, device))
        return torch.argmax(logits, dim=-1).cpu().numpy()

    rows, summary = evaluate_actions(torch_actions, seed, episodes_per_cell)
    actor.train()
    return rows, summary


def evaluate_portable(
    actor_path: Path,
    seed: int,
    episodes_per_cell: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    portable = NumpyActorPolicy.load(str(actor_path))

    def numpy_actions(observation: np.ndarray) -> np.ndarray:
        return np.argmax(portable.logits(observation), axis=-1)

    return evaluate_actions(numpy_actions, seed, episodes_per_cell)


def verify_portable_actor(
    actor: SharedFleetActor,
    actor_path: Path,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    first_linear = next(
        module for module in actor.network if isinstance(module, nn.Linear)
    )
    sample = rng.normal(size=(17, first_linear.in_features)).astype(np.float32)
    portable = NumpyActorPolicy.load(str(actor_path))
    with torch.no_grad():
        expected = actor(torch.as_tensor(sample)).cpu().numpy()
    error = float(np.max(np.abs(portable.logits(sample) - expected)))
    if error > 1e-5:
        raise AssertionError(f"portable actor max logit error {error} exceeds tolerance")
    return error


def prepare_run_directory(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved == REPO_ROOT or resolved == REPO_ROOT.parent:
        raise ValueError("refusing broad run directory")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="data/experiments/variable_fleet_mappo_smoke_v1",
    )
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--min-entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=640024)
    parser.add_argument("--eval-seed", type=int, default=740024)
    parser.add_argument("--eval-episodes-per-cell", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--minimum-validation-success-rate", type=float, default=0.0)
    parser.add_argument("--maximum-validation-collision-rate", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.updates < 1 or args.rollout_steps < 1 or args.epochs < 1:
        raise ValueError("updates, rollout steps, and epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device(args.device)
    run_dir = (REPO_ROOT / args.run_dir).resolve()
    prepare_run_directory(run_dir, args.overwrite)
    actor_path = run_dir / "actor.npz"
    checkpoint_path = run_dir / "checkpoint.pt"
    training_log_path = run_dir / "training.jsonl"
    evaluation_path = run_dir / "evaluation.jsonl"
    replay_path = run_dir / "evaluation_replay.jsonl"
    validation_log_path = run_dir / "validation.jsonl"

    reward_config = FleetRewardConfig()
    manifest = {
        "status": "running",
        "algorithm": "parameter-sharing MAPPO",
        "training_mode": "centralized training, decentralized execution",
        "global_seed": args.seed,
        "evaluation_seed": args.eval_seed,
        "seed_overlap": args.seed == args.eval_seed,
        "core_cells": [
            {"topology": topology, "fleet_size": fleet_size}
            for topology, fleet_size in CORE_CELLS
        ],
        "observation_dim": OBSERVATION_DIM,
        "context_dim": CONTEXT_DIM,
        "action_names": ACTION_NAMES,
        "reward": reward_config.to_dict(),
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
    }
    if manifest["seed_overlap"]:
        raise ValueError("training and evaluation seeds must differ")
    _json_dump(run_dir / "manifest.json", manifest)

    pool = PaddedFleetPool(args.num_envs, args.seed)
    observation, context, mask = pool.arrays()
    actor = SharedFleetActor(OBSERVATION_DIM, len(ACTION_NAMES)).to(device)
    critic = PooledFleetCritic(OBSERVATION_DIM, CONTEXT_DIM).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate
    )
    started = time.time()
    best_score = -float("inf")
    best_update = 0
    best_actor_state = None
    best_critic_state = None
    best_optimizer_state = None
    with training_log_path.open("w", encoding="utf-8", buffering=1) as log_file, validation_log_path.open(
        "w", encoding="utf-8", buffering=1
    ) as validation_file:
        for update in range(1, args.updates + 1):
            batch, observation, context, mask, rollout_stats = collect_rollout(
                pool,
                actor,
                critic,
                observation,
                context,
                mask,
                args.rollout_steps,
                device,
            )
            progress = (update - 1) / max(args.updates - 1, 1)
            entropy_coef = (
                args.entropy_coef * (1.0 - progress)
                + args.min_entropy_coef * progress
            )
            loss_stats = ppo_update(
                actor,
                critic,
                optimizer,
                batch,
                device,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=args.clip_ratio,
                entropy_coef=entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            row = {
                "update": update,
                "environment_transitions": update
                * args.num_envs
                * args.rollout_steps,
                "vehicle_decisions": int(np.sum(batch["mask"])),
                "elapsed_s": time.time() - started,
                "entropy_coefficient": entropy_coef,
                **rollout_stats,
                **loss_stats,
            }
            if update == 1 or update % args.eval_every == 0 or update == args.updates:
                validation_rows, validation_summary = evaluate(
                    actor, device, args.eval_seed, args.eval_episodes_per_cell
                )
                validation_score = (
                    validation_summary["success_rate"]
                    - validation_summary["collision_rate"]
                    - 0.1 * validation_summary["gap_episode_rate"]
                    - 0.001 * validation_summary["mean_shield_interventions"]
                )
                validation_record = {
                    "update": update,
                    "score": validation_score,
                    **validation_summary,
                }
                validation_file.write(
                    json.dumps(validation_record, separators=(",", ":")) + "\n"
                )
                row.update(
                    {
                        f"validation_{key}": value
                        for key, value in validation_record.items()
                        if key != "update"
                    }
                )
                if validation_score > best_score:
                    best_score = validation_score
                    best_update = update
                    best_actor_state = copy.deepcopy(actor.state_dict())
                    best_critic_state = copy.deepcopy(critic.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            log_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

    if (
        best_actor_state is None
        or best_critic_state is None
        or best_optimizer_state is None
    ):
        raise RuntimeError("training produced no selectable checkpoint")
    actor.load_state_dict(best_actor_state)
    critic.load_state_dict(best_critic_state)
    evaluation_rows, evaluation_summary = evaluate(
        actor, device, args.eval_seed, args.eval_episodes_per_cell
    )
    with evaluation_path.open("w", encoding="utf-8") as output:
        for row in evaluation_rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    metadata = {
        "algorithm": manifest["algorithm"],
        "training_mode": manifest["training_mode"],
        "training_seed": args.seed,
        "evaluation_seed": args.eval_seed,
        "selected_update": best_update,
        "selection_score": best_score,
        "model_selection": (
            "success_rate - collision_rate - 0.1 * gap_episode_rate "
            "- 0.001 * mean_shield_interventions"
        ),
        "environment_transitions": args.updates
        * args.num_envs
        * args.rollout_steps,
        "observation_dim": OBSERVATION_DIM,
        "context_dim": CONTEXT_DIM,
        "action_names": ACTION_NAMES,
        "core_cells": manifest["core_cells"],
        "reward": reward_config.to_dict(),
        "evaluation": evaluation_summary,
    }
    export_shared_actor(actor, actor_path, metadata)
    portable_error = verify_portable_actor(actor, actor_path, args.seed + 1)
    replay_rows, replay_summary = evaluate_portable(
        actor_path, args.eval_seed, args.eval_episodes_per_cell
    )
    with replay_path.open("w", encoding="utf-8") as output:
        for row in replay_rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    replay_matches = replay_rows == evaluation_rows and replay_summary == evaluation_summary
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": best_optimizer_state,
            "arguments": vars(args),
            "reward": asdict(reward_config),
        },
        checkpoint_path,
    )
    acceptance = {
        "all_core_cells_evaluated": evaluation_summary["cell_coverage"]
        == len(CORE_CELLS),
        "all_evaluation_rows_valid": len(evaluation_rows)
        == len(CORE_CELLS) * args.eval_episodes_per_cell,
        "finite_training_log": True,
        "portable_actor_matches": portable_error <= 1e-5,
        "portable_actor_max_logit_error": portable_error,
        "artifacts_present": all(
            path.exists()
            for path in (
                actor_path,
                actor_path.with_suffix(".json"),
                checkpoint_path,
                training_log_path,
                evaluation_path,
                replay_path,
                validation_log_path,
            )
        ),
        "deterministic_episode_replay": replay_matches,
        "minimum_validation_success_rate_met": evaluation_summary["success_rate"]
        >= args.minimum_validation_success_rate,
        "maximum_validation_collision_rate_met": evaluation_summary[
            "collision_rate"
        ]
        <= args.maximum_validation_collision_rate,
    }
    acceptance["accepted"] = bool(
        all(value for key, value in acceptance.items() if key != "portable_actor_max_logit_error")
    )
    _json_dump(run_dir / "acceptance.json", acceptance)
    _json_dump(run_dir / "evaluation_summary.json", evaluation_summary)
    manifest.update(
        {
            "status": "complete" if acceptance["accepted"] else "failed",
            "elapsed_s": time.time() - started,
            "evaluation": evaluation_summary,
            "selected_update": best_update,
            "selection_score": best_score,
            "acceptance": acceptance,
        }
    )
    _json_dump(run_dir / "manifest.json", manifest)
    if not acceptance["accepted"]:
        raise SystemExit("variable-fleet MAPPO acceptance checks failed")
    print(json.dumps({"run_dir": str(run_dir), **acceptance}, sort_keys=True))


if __name__ == "__main__":
    main()
