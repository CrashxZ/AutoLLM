#!/usr/bin/env python3
"""Train MAPPO on randomized connected scenarios with leakage-safe splits."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import resource
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from marl.adapted_mappo_scenarios import (  # noqa: E402
    CORE_CELLS,
    FLEET_SIZES,
    adapted_scenario_spec,
    split_seed_ranges_disjoint,
)
from marl.carla_aligned_mappo import (  # noqa: E402
    ALIGNED_MINIMUM_CLEARANCE_M,
    CORE_CELLS as CARLA_ALIGNED_CORE_CELLS,
    FLEET_SIZES as CARLA_ALIGNED_FLEET_SIZES,
    CarlaAlignedFleetEnv,
    CarlaAlignedMAPPOAdapter,
    CarlaAlignedSplit,
    balanced_priority_order,
    scenario_spec as carla_aligned_scenario_spec,
    split_seed_ranges_disjoint as carla_aligned_splits_disjoint,
)
from marl.benchmark_v2_policies import (  # noqa: E402
    FleetActionValidator,
    validator_masked_argmax,
)
from marl.fleet_highway_env import FleetHighwayConfig  # noqa: E402
from marl.fleet_mappo import (  # noqa: E402
    CONTEXT_DIM,
    OBSERVATION_DIM,
    PRIORITY_OBSERVATION_DIM,
)
from marl.fleet_mappo_models import (  # noqa: E402
    PooledFleetCritic,
    SharedFleetActor,
    export_shared_actor,
)
from marl.highway_env import ACTION_NAMES  # noqa: E402
from marl.numpy_policy import NumpyActorPolicy  # noqa: E402
from marl.procedural_mappo import ProceduralMAPPOAdapter  # noqa: E402
from marl.procedural_scenarios import (  # noqa: E402
    ProceduralFleetHighwayEnv,
    ScenarioSplit,
)
from scripts.train_fleet_mappo import (  # noqa: E402
    _git_revision,
    _json_dump,
    as_tensor,
    collect_rollout,
    ppo_update,
    prepare_run_directory,
    verify_portable_actor,
)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def ensure_within_budget(started: float, max_wall_s: float, max_rss_mb: float) -> None:
    if max_wall_s > 0.0 and time.time() - started > max_wall_s:
        raise TimeoutError(f"adapted MAPPO wall-time budget exceeded: {max_wall_s}s")
    observed_rss = peak_rss_mb()
    if max_rss_mb > 0.0 and observed_rss > max_rss_mb:
        raise MemoryError(
            f"adapted MAPPO RSS budget exceeded: {observed_rss:.1f} > {max_rss_mb} MiB"
        )


def family_name(value) -> str:
    return str(getattr(value, "value", value))


class RandomizedConnectedPool:
    """Mixed-family pool drawing independent scenarios from the training split."""

    use_validator_action_mask = True

    def __init__(
        self,
        num_envs: int,
        scenario_profile: str = "randomized_connected",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.scenario_profile = str(scenario_profile)
        if self.scenario_profile == "carla_aligned":
            self.core_cells = CARLA_ALIGNED_CORE_CELLS
            self.max_agents = max(CARLA_ALIGNED_FLEET_SIZES)
            self.observation_dim = PRIORITY_OBSERVATION_DIM
        else:
            self.core_cells = CORE_CELLS
            self.max_agents = max(FLEET_SIZES)
            self.observation_dim = OBSERVATION_DIM
        self.cells = [
            self.core_cells[index % len(self.core_cells)]
            for index in range(num_envs)
        ]
        self.episode_counts = np.zeros(num_envs, dtype=np.int64)
        self.adapters: List[object] = []
        self.validators: List[FleetActionValidator] = []
        self.consumed_scenarios: List[Dict[str, object]] = []
        for slot in range(num_envs):
            adapter, validator = self._new_episode(slot)
            self.adapters.append(adapter)
            self.validators.append(validator)

    def _ordinal(self, slot: int) -> int:
        return int(self.episode_counts[slot]) * self.num_envs + slot

    def _new_episode(
        self, slot: int
    ) -> tuple[object, FleetActionValidator]:
        family, fleet_size = self.cells[slot]
        ordinal = self._ordinal(slot)
        if self.scenario_profile == "carla_aligned":
            spec = carla_aligned_scenario_spec(
                CarlaAlignedSplit.TRAIN,
                family_name(family),
                fleet_size,
                ordinal,
            )
            env = CarlaAlignedFleetEnv(
                FleetHighwayConfig(
                    num_vehicles=fleet_size,
                    dt_s=0.5,
                    max_steps=60,
                    safe_gap_m=ALIGNED_MINIMUM_CLEARANCE_M,
                ),
                seed=spec.seed,
            )
            priority = list(balanced_priority_order(fleet_size, ordinal))
            adapter = CarlaAlignedMAPPOAdapter(env, tuple(priority))
            adapter.reset_aligned(spec)
            validator = FleetActionValidator(
                env,
                priority_order=priority,
                allow_cooperative_support=True,
                goal_directed_lane_actions=True,
            )
            self.consumed_scenarios.append(
                {
                    "split": CarlaAlignedSplit.TRAIN.value,
                    "slot": slot,
                    "ordinal": ordinal,
                    "seed": spec.seed,
                    "family": family_name(family),
                    "fleet_size": fleet_size,
                    "priority_order": priority,
                    "initial_state_sha256": env.initial_state_sha256,
                }
            )
            return adapter, validator
        spec = adapted_scenario_spec(
            ScenarioSplit.TRAIN,
            family,
            fleet_size,
            ordinal,
        )
        env = ProceduralFleetHighwayEnv(
            FleetHighwayConfig(num_vehicles=fleet_size),
            seed=spec.seed,
        )
        adapter = ProceduralMAPPOAdapter(env)
        adapter.reset_procedural(spec)
        assert env.generated_scenario is not None
        priority = list(range(fleet_size))
        random.Random(f"priority:{spec.seed}").shuffle(priority)
        validator = FleetActionValidator(env, priority_order=priority)
        self.consumed_scenarios.append(
            {
                "split": ScenarioSplit.TRAIN.value,
                "slot": slot,
                "ordinal": ordinal,
                "seed": spec.seed,
                "family": family_name(family),
                "fleet_size": fleet_size,
                "initial_state_sha256": env.generated_scenario.initial_state_sha256(),
            }
        )
        return adapter, validator

    def arrays(self):
        observation = np.zeros(
            (self.num_envs, self.max_agents, self.observation_dim),
            dtype=np.float32,
        )
        context = np.zeros((self.num_envs, CONTEXT_DIM), dtype=np.float32)
        mask = np.zeros((self.num_envs, self.max_agents), dtype=bool)
        for slot, adapter in enumerate(self.adapters):
            prepare_step = getattr(adapter, "prepare_step", None)
            if prepare_step is not None:
                prepare_step()
            local, global_context = adapter.observations()
            count = adapter.env.cfg.num_vehicles
            observation[slot, :count] = local
            context[slot] = global_context
            mask[slot, :count] = True
        return observation, context, mask

    def admissible_action_mask(
        self,
        slot: int,
        authorized_prefix: np.ndarray,
        vehicle_index: int,
    ) -> np.ndarray:
        return self.validators[slot].admissible_action_mask(
            self.adapters[slot].env,
            authorized_prefix,
            vehicle_index,
        )

    def step(self, padded_actions: np.ndarray):
        expected = (self.num_envs, self.max_agents)
        if padded_actions.shape != expected:
            raise ValueError(f"padded actions must have shape {expected}")
        rewards = np.zeros(expected, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        completed_rows = []
        for slot, adapter in enumerate(self.adapters):
            count = adapter.env.cfg.num_vehicles
            _, _, reward, done, info = adapter.step(padded_actions[slot, :count])
            rewards[slot, :count] = reward
            dones[slot] = done
            if not done:
                continue
            family, fleet_size = self.cells[slot]
            completed_rows.append(
                {
                    "slot": slot,
                    "family": family_name(family),
                    "fleet_size": fleet_size,
                    "steps": adapter.env.steps,
                    "success": bool(info["success"]),
                    "collision": bool(info["collision"]),
                    "timeout": bool(info["timeout"]),
                }
            )
            self.episode_counts[slot] += 1
            self.adapters[slot], self.validators[slot] = self._new_episode(slot)
        observation, context, mask = self.arrays()
        return observation, context, mask, rewards, dones, completed_rows


def evaluate_policy(
    policy,
    *,
    split: ScenarioSplit,
    episodes_per_cell: int,
    scenario_profile: str,
) -> tuple[List[Dict[str, object]], Dict[str, float]]:
    rows = []
    core_cells = (
        CARLA_ALIGNED_CORE_CELLS
        if scenario_profile == "carla_aligned"
        else CORE_CELLS
    )
    for cell_index, (family, fleet_size) in enumerate(core_cells):
        for repetition in range(episodes_per_cell):
            ordinal = cell_index * episodes_per_cell + repetition
            if scenario_profile == "carla_aligned":
                aligned_split = CarlaAlignedSplit(split.value)
                spec = carla_aligned_scenario_spec(
                    aligned_split,
                    family_name(family),
                    fleet_size,
                    ordinal,
                )
                env = CarlaAlignedFleetEnv(
                    FleetHighwayConfig(
                        num_vehicles=fleet_size,
                        dt_s=0.5,
                        max_steps=60,
                        safe_gap_m=ALIGNED_MINIMUM_CLEARANCE_M,
                    ),
                    seed=spec.seed,
                )
                priority = list(balanced_priority_order(fleet_size, ordinal))
                adapter = CarlaAlignedMAPPOAdapter(env, tuple(priority))
                observation, _ = adapter.reset_aligned(spec)
                initial_hash = env.initial_state_sha256
                validator = FleetActionValidator(
                    env,
                    priority_order=priority,
                    allow_cooperative_support=True,
                    goal_directed_lane_actions=True,
                )
            else:
                spec = adapted_scenario_spec(split, family, fleet_size, ordinal)
                env = ProceduralFleetHighwayEnv(
                    FleetHighwayConfig(num_vehicles=fleet_size),
                    seed=spec.seed,
                )
                adapter = ProceduralMAPPOAdapter(env)
                observation, _ = adapter.reset_procedural(spec)
                assert env.generated_scenario is not None
                initial_hash = env.generated_scenario.initial_state_sha256()
                priority = list(range(fleet_size))
                random.Random(f"priority:{spec.seed}").shuffle(priority)
                validator = FleetActionValidator(env, priority_order=priority)
            gap_events = ttc_events = validator_interventions = 0
            final_info: Dict[str, object] = {}
            for _step in range(env.cfg.max_steps):
                prepare_step = getattr(adapter, "prepare_step", None)
                if prepare_step is not None:
                    prepare_step()
                observation, _ = adapter.observations()
                logits = np.asarray(policy(observation), dtype=np.float64)
                requested = np.argmax(logits, axis=-1)
                actions, _masks = validator_masked_argmax(env, validator, logits)
                validator_interventions += int(np.count_nonzero(requested != actions))
                observation, _, _, done, final_info = adapter.step(actions)
                gap_events += int(
                    np.any(np.asarray(final_info["gap_violation_pairs"], dtype=bool))
                )
                ttc_events += int(
                    np.any(np.asarray(final_info["ttc_violation_pairs"], dtype=bool))
                )
                if done:
                    break
            rows.append(
                {
                    "split": split.value,
                    "family": family_name(family),
                    "fleet_size": fleet_size,
                    "repetition": repetition + 1,
                    "ordinal": ordinal,
                    "seed": spec.seed,
                    "initial_state_sha256": initial_hash,
                    "steps": env.steps,
                    "success": bool(final_info.get("success", False)),
                    "collision": bool(final_info.get("collision", False)),
                    "timeout": bool(final_info.get("timeout", False)),
                    "gap_event_steps": gap_events,
                    "ttc_event_steps": ttc_events,
                    "validator_interventions": validator_interventions,
                }
            )
    summary = {
        "episodes": len(rows),
        "cell_coverage": len({(row["family"], row["fleet_size"]) for row in rows}),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "gap_episode_rate": float(
            np.mean([int(row["gap_event_steps"] > 0) for row in rows])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_validator_interventions": float(
            np.mean([row["validator_interventions"] for row in rows])
        ),
    }
    return rows, summary


def hashes(rows: List[Dict[str, object]]) -> set[str]:
    return {str(row["initial_state_sha256"]) for row in rows}


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--min-entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--model-seed", type=int, default=2_026_082_601)
    parser.add_argument("--validation-episodes-per-cell", type=int, default=2)
    parser.add_argument("--test-episodes-per-cell", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-wall-s", type=float, default=900.0)
    parser.add_argument("--max-rss-mb", type=float, default=4096.0)
    parser.add_argument(
        "--scenario-profile",
        choices=("randomized_connected", "carla_aligned"),
        default="randomized_connected",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.updates < 1 or args.num_envs < 1 or args.rollout_steps < 1:
        raise ValueError("updates, num-envs, and rollout-steps must be positive")
    splits_disjoint = (
        carla_aligned_splits_disjoint()
        if args.scenario_profile == "carla_aligned"
        else split_seed_ranges_disjoint()
    )
    if not splits_disjoint:
        raise RuntimeError("adapted MAPPO split namespaces overlap")
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device(args.device)
    run_dir = (REPO_ROOT / args.run_dir).resolve()
    core_cells = (
        CARLA_ALIGNED_CORE_CELLS
        if args.scenario_profile == "carla_aligned"
        else CORE_CELLS
    )
    observation_dim = (
        PRIORITY_OBSERVATION_DIM
        if args.scenario_profile == "carla_aligned"
        else OBSERVATION_DIM
    )
    prepare_run_directory(run_dir, args.overwrite)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "running",
        "claim_status": "training feasibility" if args.skip_test else "exploratory",
        "algorithm": "parameter-sharing validator-masked MAPPO",
        "comparison_label": "MAPPO_ADAPTED",
        "scenario_profile": args.scenario_profile,
        "cooperative_support_actions": args.scenario_profile == "carla_aligned",
        "priority_observation": args.scenario_profile == "carla_aligned",
        "model_seed": args.model_seed,
        "split_seed_namespaces_disjoint": True,
        "core_cells": [
            {"family": family_name(family), "fleet_size": fleet_size}
            for family, fleet_size in core_cells
        ],
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
    }
    _json_dump(manifest_path, manifest)
    started = time.time()
    try:
        pool = RandomizedConnectedPool(args.num_envs, args.scenario_profile)
        observation, context, mask = pool.arrays()
        actor = SharedFleetActor(observation_dim, len(ACTION_NAMES)).to(device)
        critic = PooledFleetCritic(observation_dim, CONTEXT_DIM).to(device)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=args.learning_rate,
        )
        best_score = -float("inf")
        best_update = 0
        best_actor_state = best_critic_state = best_optimizer_state = None
        training_rows = []
        validation_rows_all = []
        for update in range(1, args.updates + 1):
            ensure_within_budget(started, args.max_wall_s, args.max_rss_mb)
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
            record = {
                "update": update,
                "environment_transitions": update
                * args.num_envs
                * args.rollout_steps,
                "elapsed_s": time.time() - started,
                "peak_rss_mb": peak_rss_mb(),
                **rollout_stats,
                **loss_stats,
            }
            if update == 1 or update % args.eval_every == 0 or update == args.updates:
                actor.eval()

                def actor_policy(value):
                    with torch.no_grad():
                        return actor(as_tensor(value, device)).cpu().numpy()

                validation_rows, validation_summary = evaluate_policy(
                    actor_policy,
                    split=ScenarioSplit.VALIDATION,
                    episodes_per_cell=args.validation_episodes_per_cell,
                    scenario_profile=args.scenario_profile,
                )
                actor.train()
                for row in validation_rows:
                    row["checkpoint_update"] = update
                validation_rows_all.extend(validation_rows)
                score = (
                    validation_summary["success_rate"]
                    - 3.0 * validation_summary["collision_rate"]
                    - 0.1 * validation_summary["gap_episode_rate"]
                    - 0.0005
                    * validation_summary["mean_validator_interventions"]
                )
                record.update(
                    {f"validation_{key}": value for key, value in validation_summary.items()}
                )
                record["validation_score"] = score
                if score > best_score:
                    best_score = score
                    best_update = update
                    best_actor_state = copy.deepcopy(actor.state_dict())
                    best_critic_state = copy.deepcopy(critic.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            training_rows.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if best_actor_state is None:
            raise RuntimeError("training produced no selectable checkpoint")
        actor.load_state_dict(best_actor_state)
        critic.load_state_dict(best_critic_state)
        actor_path = run_dir / "actor.npz"
        metadata = {
            "algorithm": manifest["algorithm"],
            "comparison_label": "MAPPO_ADAPTED",
            "model_seed": args.model_seed,
            "selected_update": best_update,
            "selection_score": best_score,
            "training_transitions": args.updates
            * args.num_envs
            * args.rollout_steps,
            "core_cells": manifest["core_cells"],
        }
        export_shared_actor(actor, actor_path, metadata)
        portable_error = verify_portable_actor(actor, actor_path, args.model_seed + 1)
        portable = NumpyActorPolicy.load(str(actor_path))
        replay_rows, replay_summary = evaluate_policy(
            portable.logits,
            split=ScenarioSplit.VALIDATION,
            episodes_per_cell=args.validation_episodes_per_cell,
            scenario_profile=args.scenario_profile,
        )
        selected_validation_rows, selected_validation_summary = evaluate_policy(
            portable.logits,
            split=ScenarioSplit.VALIDATION,
            episodes_per_cell=args.validation_episodes_per_cell,
            scenario_profile=args.scenario_profile,
        )
        replay_matches = (
            replay_rows == selected_validation_rows
            and replay_summary == selected_validation_summary
        )
        test_rows = []
        test_summary = None
        if not args.skip_test:
            test_rows, test_summary = evaluate_policy(
                portable.logits,
                split=ScenarioSplit.TEST,
                episodes_per_cell=args.test_episodes_per_cell,
                scenario_profile=args.scenario_profile,
            )

        write_jsonl(run_dir / "training.jsonl", training_rows)
        write_jsonl(run_dir / "validation.jsonl", validation_rows_all)
        write_jsonl(run_dir / "selected_validation.jsonl", selected_validation_rows)
        write_jsonl(run_dir / "test.jsonl", test_rows)
        write_jsonl(run_dir / "consumed_training_scenarios.jsonl", pool.consumed_scenarios)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": best_optimizer_state,
                "arguments": vars(args),
            },
            run_dir / "checkpoint.pt",
        )
        train_hashes = hashes(pool.consumed_scenarios)
        validation_hashes = hashes(selected_validation_rows)
        test_hashes = hashes(test_rows)
        split_hashes_disjoint = (
            train_hashes.isdisjoint(validation_hashes)
            and train_hashes.isdisjoint(test_hashes)
            and validation_hashes.isdisjoint(test_hashes)
        )
        acceptance = {
            "all_cells_in_selected_validation": selected_validation_summary[
                "cell_coverage"
            ]
            == len(core_cells),
            "split_seed_namespaces_disjoint": splits_disjoint,
            "split_geometry_hashes_disjoint": split_hashes_disjoint,
            "portable_actor_max_logit_error": portable_error,
            "portable_actor_matches": portable_error <= 1e-5,
            "deterministic_validation_replay": replay_matches,
            "within_wall_budget": time.time() - started <= args.max_wall_s,
            "within_rss_budget": peak_rss_mb() <= args.max_rss_mb,
        }
        acceptance["accepted"] = all(
            value
            for key, value in acceptance.items()
            if key != "portable_actor_max_logit_error"
        )
        _json_dump(run_dir / "acceptance.json", acceptance)
        _json_dump(run_dir / "validation_summary.json", selected_validation_summary)
        if test_summary is not None:
            _json_dump(run_dir / "test_summary.json", test_summary)
        manifest.update(
            {
                "status": "complete" if acceptance["accepted"] else "failed",
                "elapsed_s": time.time() - started,
                "peak_rss_mb": peak_rss_mb(),
                "selected_update": best_update,
                "selection_score": best_score,
                "consumed_training_scenarios": len(pool.consumed_scenarios),
                "validation": selected_validation_summary,
                "test": test_summary,
                "acceptance": acceptance,
            }
        )
        _json_dump(manifest_path, manifest)
        if not acceptance["accepted"]:
            raise SystemExit("adapted MAPPO acceptance checks failed")
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "elapsed_s": time.time() - started,
                "peak_rss_mb": peak_rss_mb(),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _json_dump(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
