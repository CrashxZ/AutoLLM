import numpy as np
from pathlib import Path

from marl.fleet_mappo import OBSERVATION_DIM, PRIORITY_OBSERVATION_DIM
from marl.highway_env import ACCELERATE, KEEP_LANE, LANE_LEFT, LANE_RIGHT, YIELD
from server.coordination.validator import DeterministicPlanValidator
from scripts.run_carla_paired_coordination import (
    bumper_clearance_m,
    carla_actor_observations,
    carla_joint_plan_from_actions,
    carla_validator_masked_argmax,
    build_mec_payload,
    command_conflict,
    commands_at_retry_limit,
    corridor_departures,
    diagnostic_executable_steps,
    diagnostic_step_sets,
    due_commands,
    execute_protocol_step,
    initial_conflict_evidence,
    layout_for_block,
    longitudinal_delta_m,
    lane_position,
    mappo_target_speed_kmh,
    observe_commands,
    pilot_schedule,
    protected_goal_vehicle_ids,
    RunnerBudget,
    source_file_checksums,
    transaction_executor_failure,
)


class RecordingClient:
    def __init__(self) -> None:
        self.posts = []

    def post(self, url, json):
        self.posts.append((url, json))
        return self

    def raise_for_status(self) -> None:
        return None


def row(
    veh_id: int,
    lane_id: int,
    s_m: float,
    speed: float = 35.0,
    *,
    driving_lane_ids: list[int] | None = None,
) -> dict:
    lane_ids = driving_lane_ids or (
        [-1, -2, -3, -4] if lane_id < 0 else [1, 2, 3, 4]
    )
    return {
        "ts": 100.0,
        "veh_id": veh_id,
        "road_id": 6,
        "s_m": s_m,
        "d_m": 0.0,
        "lane_id": lane_id,
        "driving_lane_ids": lane_ids,
        "driving_lane_index": lane_ids.index(lane_id),
        "speed_kmh": speed,
        "pose": {"x": 0.0, "y": s_m, "yaw": 90.0},
        "lane_change": {"state": "IDLE", "target_lane_id": None},
    }


def test_longitudinal_delta_is_forward_positive_on_both_carriageways() -> None:
    assert longitudinal_delta_m(row(1, -2, 100.0), row(2, -2, 120.0)) == 20.0
    assert longitudinal_delta_m(row(1, 4, 100.0), row(2, 4, 80.0)) == 20.0


def test_pilot_uses_disjoint_seed_namespace_and_every_configuration() -> None:
    schedule = pilot_schedule()
    assert schedule["claim_status"] == "exploratory_pilot"
    assert len(schedule["blocks"]) == 9
    assert all(block["seed"] < 7_100_000 for block in schedule["blocks"])


def test_pilot_wall_guard_preserves_sixty_second_simulation_deadline() -> None:
    budget = RunnerBudget()
    assert budget.trial_timeout_wall_s == 210.0
    assert budget.trial_timeout_sim_s == 60.0
    assert budget.ready_timeout_wall_s == 90.0
    assert budget.no_progress_retry_sim_s == 3.0
    assert budget.max_command_attempts == 10


def test_completed_command_does_not_map_stale_goal_on_a_new_road() -> None:
    telemetry = {
        "10": row(10, -1, 20.0, driving_lane_ids=[-1]),
    }
    command = {
        "veh_id": 10,
        "target_lane_id": 5,
        "activation_sim_s": 100.0,
        "completed": True,
    }

    observations = carla_actor_observations(
        telemetry,
        [10],
        {10: command},
        current_sim_s=101.0,
    )

    assert observations.shape == (1, OBSERVATION_DIM)


def test_corridor_departure_compares_against_initial_lane_domain() -> None:
    telemetry = {
        "10": row(10, 3, 20.0, driving_lane_ids=[3, 4, 5, 6]),
        "11": row(11, -1, 25.0, driving_lane_ids=[-1]),
    }

    departures = corridor_departures(
        telemetry,
        {10: (3, 4, 5, 6), 11: (3, 4, 5, 6)},
    )

    assert departures == [
        {
            "veh_id": 11,
            "road_id": 6,
            "lane_id": -1,
            "initial_driving_lane_ids": [3, 4, 5, 6],
            "observed_driving_lane_ids": [-1],
        }
    ]


def test_retry_limit_ignores_a_command_while_its_transaction_is_executing():
    commands = [
        {
            "command_id": "command-1",
            "attempts": 10,
            "completed": False,
        }
    ]

    assert commands_at_retry_limit(commands, {}, 10) == commands
    assert commands_at_retry_limit(
        commands,
        {"txn-1": {"command_id": "command-1"}},
        10,
    ) == []


def test_mec_transaction_identity_uses_review_sequence_not_lane_retries():
    command = {
        "command_id": "command-1",
        "veh_id": 10,
        "direction": "left",
        "target_lane_id": -1,
        "attempts": 0,
        "review_attempts": 3,
    }
    telemetry = {
        "10": row(10, -2, 20.0),
        "11": row(11, -1, 100.0),
    }

    payload = build_mec_payload(command, telemetry)

    assert payload["transaction_id"] == "carla-command-1-r4"


def test_lane_step_applies_validated_target_speed_before_maneuver():
    client = RecordingClient()

    action = execute_protocol_step(
        client,
        "http://test",
        10,
        {
            "action": "lane_left",
            "target_lane_id": -1,
            "target_speed_kmh": 50.0,
        },
        {"10": row(10, -2, 20.0, speed=15.0)},
    )

    assert action == "lane_left"
    assert client.posts == [
        (
            "http://test/command",
            {"cmd": "speed", "veh_id": 10, "kmh": 50.0},
        ),
        (
            "http://test/command",
            {"cmd": "lane", "veh_id": 10, "dir": "left"},
        ),
    ]


def test_longitudinal_steps_apply_validated_absolute_targets():
    client = RecordingClient()
    telemetry = {"10": row(10, -2, 20.0, speed=20.0)}

    execute_protocol_step(
        client,
        "http://test",
        10,
        {"action": "accelerate", "target_speed_kmh": 50.0},
        telemetry,
    )
    execute_protocol_step(
        client,
        "http://test",
        10,
        {"action": "yield", "target_speed_kmh": 12.0},
        telemetry,
    )

    assert client.posts == [
        (
            "http://test/command",
            {"cmd": "speed", "veh_id": 10, "kmh": 50.0},
        ),
        (
            "http://test/command",
            {"cmd": "speed", "veh_id": 10, "kmh": 12.0},
        ),
    ]


def test_joint_step_suppression_is_explicit_and_ego_only() -> None:
    steps = {
        10: {"action": "lane_left"},
        11: {"action": "accelerate"},
    }
    assert diagnostic_executable_steps(
        steps, 10, suppress_joint_participants=False
    ) == steps
    assert diagnostic_executable_steps(
        steps, 10, suppress_joint_participants=True
    ) == {10: {"action": "lane_left"}}

    execution, tracked = diagnostic_step_sets(
        steps,
        10,
        suppress_joint_participants=False,
        suppress_joint_participant_actuation=True,
    )
    assert execution == {10: {"action": "lane_left"}}
    assert tracked == steps


def test_goal_priority_only_protects_pending_commands_in_plan_horizon() -> None:
    commands = [
        {"veh_id": 10, "activation_sim_s": 100.0, "completed": False},
        {"veh_id": 11, "activation_sim_s": 108.0, "completed": False},
        {"veh_id": 12, "activation_sim_s": 108.1, "completed": False},
        {"veh_id": 13, "activation_sim_s": 99.0, "completed": True},
    ]

    assert protected_goal_vehicle_ids(commands, 100.0) == [10, 11]


def test_target_lane_dwell_does_not_reissue_directional_command() -> None:
    command = {
        "veh_id": 10,
        "target_lane_id": -2,
        "activation_sim_s": 100.0,
        "completed": False,
    }
    telemetry = {"10": row(10, -2, 20.0)}

    assert due_commands([command], 101.0, telemetry) == []

    telemetry["10"]["lane_id"] = -3
    assert due_commands([command], 101.0, telemetry) == [command]


def test_transaction_fails_immediately_on_fresh_lane_executor_abort() -> None:
    failure = transaction_executor_failure(
        {10: {"action": "lane_left"}, 11: {"action": "accelerate"}},
        {
            "10": {
                "lane_change": {
                    "last_terminal_state": "ABORT",
                    "last_terminal_reason": "no_lateral_progress",
                    "last_terminal_sim_s": 102.0,
                    "lateral_progress_m": 0.08,
                }
            }
        },
        100.0,
    )

    assert failure == {
        "veh_id": 10,
        "reason": "no_lateral_progress",
        "terminal_sim_s": 102.0,
        "lateral_progress_m": 0.08,
    }


def test_carla_actor_observation_matches_frozen_shape() -> None:
    telemetry = {"10": row(10, -3, 20.0), "11": row(11, -2, 30.0)}
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "target_lane_id": -2,
            "completed": False,
        }
    }
    observations = carla_actor_observations(
        telemetry, [10, 11], commands, current_sim_s=1.0
    )
    assert observations.shape == (2, OBSERVATION_DIM)
    assert np.all(np.isfinite(observations))
    assert observations[0, 6] == 1.0


def test_carla_priority_observation_matches_aligned_actor_shape() -> None:
    telemetry = {"10": row(10, -3, 20.0), "11": row(11, -2, 30.0)}
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "target_lane_id": -2,
            "completed": False,
        }
    }

    observations = carla_actor_observations(
        telemetry,
        [10, 11],
        commands,
        current_sim_s=1.0,
        priority_order=[1, 0],
    )

    assert observations.shape == (2, PRIORITY_OBSERVATION_DIM)
    assert observations[:, 10].tolist() == [1.0, 0.0]


def test_lane_position_and_clearance_use_road_relative_state() -> None:
    first = row(10, -3, 20.0)
    second = row(11, -2, 30.0)
    assert lane_position(first) == 2.0
    assert bumper_clearance_m(first, second) == 5.3


def test_positive_opendrive_lanes_map_to_left_to_right_policy_ordinals() -> None:
    lane_ids = [3, 4, 5, 6]

    assert lane_position(row(10, 3, 20.0, driving_lane_ids=lane_ids)) == 0.0
    assert lane_position(row(10, 5, 20.0, driving_lane_ids=lane_ids)) == 2.0
    assert lane_position(row(10, 6, 20.0, driving_lane_ids=lane_ids)) == 3.0


def test_positive_lane_goal_is_encoded_in_the_executed_direction() -> None:
    lane_ids = [3, 4, 5, 6]
    telemetry = {
        "10": row(10, 5, 20.0, driving_lane_ids=lane_ids),
        "11": row(11, 4, 80.0, driving_lane_ids=lane_ids),
    }
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "initial_lane_id": 5,
            "target_lane_id": 4,
            "direction": "left",
            "completed": False,
        }
    }

    observations = carla_actor_observations(
        telemetry, [10, 11], commands, current_sim_s=1.0
    )

    assert observations[0, 0] == 2.0 / 3.0
    assert observations[0, 2] == 1.0 / 3.0
    assert observations[0, 3] == -1.0 / 3.0


def test_positive_lane_action_validation_targets_the_lane_carla_executes() -> None:
    lane_ids = [3, 4, 5, 6]
    telemetry = {
        "10": row(10, 5, 20.0, driving_lane_ids=lane_ids),
        "11": row(11, 4, 100.0, driving_lane_ids=lane_ids),
    }

    plan = carla_joint_plan_from_actions(
        np.asarray([LANE_LEFT, KEEP_LANE]),
        telemetry,
        [10, 11],
        transaction_id="positive-lane-direction",
    )

    assert plan.plans[10].steps[0].target_lane_id == 4


def test_mappo_speed_actions_match_training_acceleration_and_restore_flow() -> None:
    assert mappo_target_speed_kmh(ACCELERATE, 50.0) == 53.6
    assert mappo_target_speed_kmh(YIELD, 50.0) == 44.6
    assert mappo_target_speed_kmh(KEEP_LANE, 72.0) == 50.0
    assert mappo_target_speed_kmh(LANE_LEFT, 72.0) == 50.0


def test_mappo_mask_rejects_positive_lane_change_into_occupied_target() -> None:
    lane_ids = [3, 4, 5, 6]
    telemetry = {
        "10": row(10, 5, 20.0, driving_lane_ids=lane_ids),
        "11": row(11, 4, 21.0, driving_lane_ids=lane_ids),
    }
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "initial_lane_id": 5,
            "target_lane_id": 4,
            "direction": "left",
            "completed": False,
        }
    }
    logits = np.zeros((2, 5), dtype=np.float64)
    logits[0, LANE_LEFT] = 5.0

    actions, masks = carla_validator_masked_argmax(
        logits,
        telemetry,
        [10, 11],
        commands,
        [0, 1],
        DeterministicPlanValidator(),
        current_sim_s=1.0,
    )

    assert not masks[0, LANE_LEFT]
    assert actions[0] != LANE_LEFT


def test_mappo_aligned_mask_allows_inactive_vehicle_gap_support() -> None:
    lane_ids = [3, 4, 5, 6]
    telemetry = {
        "10": row(10, 5, 20.0, driving_lane_ids=lane_ids),
        "11": row(11, 4, 21.0, driving_lane_ids=lane_ids),
    }
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "initial_lane_id": 5,
            "target_lane_id": 4,
            "direction": "left",
            "completed": False,
        }
    }
    logits = np.zeros((2, 5), dtype=np.float64)
    logits[1, YIELD] = 5.0

    actions, masks = carla_validator_masked_argmax(
        logits,
        telemetry,
        [10, 11],
        commands,
        [0, 1],
        DeterministicPlanValidator(),
        current_sim_s=1.0,
        allow_cooperative_support=True,
    )

    assert masks[1, YIELD]
    assert actions[1] == YIELD


def test_fcfs_conflict_detects_nearby_same_target_and_swap() -> None:
    telemetry = {"10": row(10, -3, 20.0), "11": row(11, -1, 25.0)}
    first = {
        "veh_id": 10,
        "initial_lane_id": -3,
        "target_lane_id": -2,
    }
    second = {
        "veh_id": 11,
        "initial_lane_id": -1,
        "target_lane_id": -2,
    }
    assert command_conflict(first, second, telemetry)


def test_explicit_conflict_layout_and_evidence_are_supported() -> None:
    block = {
        "executor_pattern": "close_reciprocal",
        "fleet_size": 2,
        "spawn_indices": [215, 216],
        "commands": [
            {"slot": 0, "direction": "left", "issue_offset_sim_s": 0.0},
            {"slot": 1, "direction": "right", "issue_offset_sim_s": 0.0},
        ],
    }
    layout = layout_for_block(block)
    assert layout.spawn_indices == (215, 216)

    commands = [
        {
            "command_id": "c1",
            "veh_id": 10,
            "initial_lane_id": -3,
            "target_lane_id": -2,
        },
        {
            "command_id": "c2",
            "veh_id": 11,
            "initial_lane_id": -2,
            "target_lane_id": -3,
        },
    ]
    evidence = initial_conflict_evidence(
        commands,
        {"10": row(10, -3, 20.0), "11": row(11, -2, 24.0)},
    )
    assert any(item["kind"] == "command_pair" for item in evidence)


def test_explicit_layout_allows_duplicate_indices_at_distinct_offsets() -> None:
    block = {
        "executor_pattern": "close_reciprocal",
        "fleet_size": 2,
        "spawn_indices": [40, 40],
        "spawn_longitudinal_offsets_m": [-100.0, -75.0],
        "commands": [{"slot": 0, "direction": "left"}],
    }

    layout = layout_for_block(block)

    assert layout.spawn_indices == (40, 40)


def test_explicit_layout_rejects_duplicate_physical_spawns() -> None:
    block = {
        "executor_pattern": "close_reciprocal",
        "fleet_size": 2,
        "spawn_indices": [40, 40],
        "spawn_longitudinal_offsets_m": [-100.0, -100.0],
        "commands": [{"slot": 0, "direction": "left"}],
    }

    try:
        layout_for_block(block)
    except ValueError as exc:
        assert "duplicate physical spawns" in str(exc)
    else:
        raise AssertionError("duplicate index/offset placements must fail")


def test_paired_runner_disables_unsolicited_tm_lane_changes() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_carla_paired_coordination.py"
    ).read_text(encoding="utf-8")
    assert '"tm_auto_lane_change": False' in source
    assert '"tm_route": list(block.get("tm_route") or [])' in source
    assert '"enable_cameras": enable_cameras' in source
    assert '"start_paused": True' in source
    assert 'f"{api_base}/simulation/warmup"' in source
    assert 'f"{api_base}/simulation/step"' in source
    assert '"simulation_execution_mode": "fixed_step"' in source


def test_runner_provenance_covers_executor_and_coordination_sources() -> None:
    checksums = source_file_checksums()

    assert "scripts/run_carla_paired_coordination.py" in checksums
    assert "server/lane_change.py" in checksums
    assert "server/coordination/candidate_ranking.py" in checksums
    assert all(len(checksum) == 64 for checksum in checksums.values())


def test_carla_mappo_uses_checkpoint_action_validator_contract() -> None:
    telemetry = {"10": row(10, -3, 20.0), "11": row(11, -1, 100.0)}
    commands = {
        10: {
            "activation_sim_s": 0.0,
            "target_lane_id": -4,
            "completed": False,
        }
    }
    logits = np.zeros((2, 5), dtype=np.float64)
    logits[0, LANE_RIGHT] = 5.0
    logits[1, LANE_RIGHT] = 5.0

    actions, masks = carla_validator_masked_argmax(
        logits,
        telemetry,
        [10, 11],
        commands,
        [0, 1],
        DeterministicPlanValidator(),
        current_sim_s=1.0,
    )

    assert actions.tolist() == [LANE_RIGHT, KEEP_LANE]
    assert masks[0, LANE_RIGHT]
    assert masks[1].tolist() == [True, False, False, False, False]


def test_terminal_done_must_correspond_to_planned_target_lane() -> None:
    command = {
        "veh_id": 10,
        "target_lane_id": -4,
        "initial_request_sequence": 0,
        "accepted_sequences": [],
        "completed": False,
        "terminal_done": False,
        "tracker": type("Tracker", (), {"observe": lambda self, **kwargs: False})(),
    }
    telemetry = {
        "10": {
            **row(10, -2, 20.0),
            "sim_time_s": 2.0,
            "distance_to_center": 0.1,
            "lane_change": {
                "request_sequence": 1,
                "last_terminal_state": "DONE",
                "last_terminal_sim_s": 2.0,
            },
        }
    }

    observe_commands([command], telemetry, 100.0)

    assert command["accepted_sequences"] == [1]
    assert not command["terminal_done"]
