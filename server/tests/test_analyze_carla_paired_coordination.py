from scripts.analyze_carla_paired_coordination import (
    audit_method,
    audit_pairing,
    replay_command,
    replay_gap_metrics,
    sha256_file,
)

import json


def frame(index, sim_s, vehicles):
    return {
        "sim_time_s": sim_s,
        "vehicles": {
            str(veh_id): {
                "veh_id": veh_id,
                "sim_frame": index,
                "sim_time_s": sim_s,
                "lane_id": lane,
                "distance_to_center": center,
                "pose": {"x": x, "y": y},
                "collision_count": 0,
                "lane_change": lane_change,
            }
            for veh_id, lane, center, x, y, lane_change in vehicles
        },
    }


def test_replay_command_requires_acceptance_then_centered_dwell_and_done():
    command = {
        "command_id": "c1",
        "veh_id": 10,
        "target_lane_id": -2,
        "initial_request_sequence": 0,
    }
    idle = {"request_sequence": 0, "last_terminal_sim_s": None}
    active = {"request_sequence": 1, "last_terminal_sim_s": None}
    done = {
        "request_sequence": 1,
        "last_terminal_sim_s": 2.0,
        "last_terminal_state": "DONE",
        "last_terminal_reason": "target_lane_settled",
    }
    trajectory = [
        frame(1, 0.0, [(10, -2, 0.1, 0.0, 0.0, idle)]),
        frame(2, 1.0, [(10, -3, 0.1, 1.0, 0.0, active)]),
        frame(3, 1.2, [(10, -2, 0.4, 2.0, 0.0, active)]),
        frame(4, 2.0, [(10, -2, 0.3, 3.0, 0.0, done)]),
    ]

    result = replay_command(trajectory, command)

    assert result["accepted"]
    assert result["centered_dwell_completed"]
    assert result["terminal_state"] == "DONE"
    assert result["terminal_done_in_target"]


def test_replay_command_separates_lane_id_alias_from_accepted_completion():
    command = {
        "command_id": "c1",
        "veh_id": 10,
        "target_lane_id": -3,
        "initial_request_sequence": 0,
    }
    idle = {"request_sequence": 0, "last_terminal_sim_s": None}
    trajectory = [
        frame(1, 0.0, [(10, -4, 0.1, 0.0, 0.0, idle)]),
        frame(2, 1.0, [(10, -3, 0.1, 1.0, 0.0, idle)]),
        frame(3, 2.0, [(10, -3, 0.1, 2.0, 0.0, idle)]),
    ]

    result = replay_command(trajectory, command)

    assert result["raw_target_dwell_completed"]
    assert not result["accepted"]
    assert not result["centered_dwell_completed"]


def test_audit_completion_count_is_acceptance_gated(tmp_path):
    directory = tmp_path / "FCFS_GAP"
    directory.mkdir()
    command = {
        "command_id": "c1",
        "veh_id": 10,
        "target_lane_id": -3,
        "initial_request_sequence": 0,
        "accepted_sequences": [],
    }
    idle = {"request_sequence": 0, "last_terminal_sim_s": None}
    trajectory = [
        frame(1, 0.0, [(10, -4, 0.1, 0.0, 0.0, idle)]),
        frame(2, 1.0, [(10, -3, 0.1, 1.0, 0.0, idle)]),
        frame(3, 2.0, [(10, -3, 0.1, 2.0, 0.0, idle)]),
    ]
    trajectory_path = directory / "trajectory.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory),
        encoding="utf-8",
    )
    (directory / "events.jsonl").write_text("", encoding="utf-8")
    metadata = {
        "block_id": "b1",
        "method": "FCFS_GAP",
        "scenario_family": "alias_probe",
        "fleet_size": 1,
        "seed": 1,
        "status": "timeout",
        "commands": [command],
        "completed_command_count": 0,
        "terminal_done_count": 0,
        "collision_count": 0,
        "gap_violation_samples": 0,
        "safe_task_success": False,
        "elapsed_sim_s": 2.0,
        "elapsed_wall_s": 0.1,
        "trajectory_sha256": sha256_file(trajectory_path),
        "persist_frames": False,
        "traffic_manager_auto_lane_change": False,
        "traffic_manager_collision_avoidance": False,
        "initial_conflict_required": False,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    result = audit_method(directory)

    assert result["raw_target_dwell_completion_count"] == 1
    assert result["centered_dwell_completion_count"] == 0
    assert result["checks"]["completion_count"]
    assert result["audit_complete"]


def test_replay_command_uses_later_done_after_retried_abort():
    command = {
        "command_id": "c1",
        "veh_id": 10,
        "target_lane_id": -2,
        "initial_request_sequence": 0,
    }
    abort = {
        "request_sequence": 1,
        "last_terminal_sim_s": 2.0,
        "last_terminal_state": "ABORT",
        "last_terminal_reason": "exec_timeout",
    }
    done = {
        "request_sequence": 2,
        "last_terminal_sim_s": 4.0,
        "last_terminal_state": "DONE",
        "last_terminal_reason": "target_lane_settled",
    }
    trajectory = [
        frame(1, 1.0, [(10, -3, 0.1, 0.0, 0.0, abort)]),
        frame(2, 3.0, [(10, -2, 0.2, 1.0, 0.0, done)]),
        frame(3, 3.8, [(10, -2, 0.2, 2.0, 0.0, done)]),
        frame(4, 4.0, [(10, -2, 0.2, 3.0, 0.0, done)]),
    ]

    result = replay_command(trajectory, command)

    assert result["accepted_sequence"] == 1
    assert result["terminal_state"] == "DONE"
    assert result["terminal_sim_s"] == 4.0


def test_gap_replay_counts_one_violation_sample_per_frame():
    lane_change = {"request_sequence": 0, "occupied_lane_ids": []}
    trajectory = [
        frame(
            1,
            0.0,
            [
                (10, -2, 0.0, 0.0, 0.0, lane_change),
                (11, -2, 0.0, 3.0, 0.0, lane_change),
                (12, -2, 0.0, 4.0, 0.0, lane_change),
            ],
        )
    ]

    minimum, samples = replay_gap_metrics(trajectory)

    assert minimum == 1.0
    assert samples == 1


def test_pairing_uses_declared_road_lane_position_and_speed_tolerances():
    first = {
        "paired_design_sha256": "same",
        "initial_state_signature": {
            "0": {"road_id": 6, "lane_id": -2, "s_m": 10.0, "speed_kmh": 30.0}
        },
    }
    second = {
        "paired_design_sha256": "same",
        "initial_state_signature": {
            "0": {"road_id": 6, "lane_id": -2, "s_m": 10.8, "speed_kmh": 31.4}
        },
    }

    assert audit_pairing([first, second])["paired"]
    second["initial_state_signature"]["0"]["s_m"] = 11.6
    assert not audit_pairing([first, second])["paired"]


def test_wrong_lane_done_does_not_satisfy_terminal_goal():
    command = {
        "command_id": "c1",
        "veh_id": 10,
        "target_lane_id": -4,
        "initial_request_sequence": 0,
    }
    wrong_lane_done = {
        "request_sequence": 1,
        "last_terminal_sim_s": 2.0,
        "last_terminal_state": "DONE",
        "last_terminal_reason": "target_lane_settled",
    }
    trajectory = [
        frame(1, 1.0, [(10, -3, 0.1, 0.0, 0.0, wrong_lane_done)]),
        frame(2, 2.0, [(10, -3, 0.1, 1.0, 0.0, wrong_lane_done)]),
    ]

    result = replay_command(trajectory, command)

    assert result["terminal_state"] == "DONE"
    assert not result["terminal_done_in_target"]
