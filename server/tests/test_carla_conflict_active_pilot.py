from scripts.run_carla_conflict_active_pilot import (
    FLEET_SIZES,
    HIGHWAY_TM_ROUTE,
    LEGACY_LONG_LAYOUT_PROFILE,
    LONG_LAYOUT_PROFILE,
    METHODS,
    SCENARIO_FAMILIES,
    audit_campaign_artifacts,
    build_schedule,
    conflict_layout,
    stable_provenance,
    validate_schedule,
    write_frozen_schedule,
)


def test_resume_provenance_ignores_only_creation_time() -> None:
    first = {"created_at_unix_s": 1.0, "sources": {"runner.py": "abc"}}
    second = {"created_at_unix_s": 2.0, "sources": {"runner.py": "abc"}}

    assert stable_provenance(first) == stable_provenance(second)
    second["sources"]["runner.py"] = "changed"
    assert stable_provenance(first) != stable_provenance(second)


def test_conflict_active_schedule_is_paired_balanced_and_reproducible() -> None:
    schedule = build_schedule(3)
    validate_schedule(schedule)

    assert schedule == build_schedule(3)
    assert schedule["block_count"] == 3 * len(SCENARIO_FAMILIES) * len(FLEET_SIZES)
    assert schedule["episode_count"] == schedule["block_count"] * len(METHODS)
    assert schedule["initial_target_speed_kmh"] == 50.0
    assert schedule["simulation_execution_mode"] == "fixed_step"
    assert schedule["simulation_step_ticks"] == 10
    assert schedule["traffic_manager_route"] == list(HIGHWAY_TM_ROUTE)
    assert all(
        set(block["method_order"]) == set(METHODS)
        for block in schedule["blocks"]
    )
    assert all(
        block["tm_route"] == list(HIGHWAY_TM_ROUTE)
        for block in schedule["blocks"]
    )
    assert all(block["require_initial_conflict"] for block in schedule["blocks"])


def test_schedule_can_freeze_a_shorter_straight_corridor_horizon() -> None:
    schedule = build_schedule(1, LONG_LAYOUT_PROFILE, trial_timeout_sim_s=40.0)

    validate_schedule(schedule)
    assert schedule["trial_timeout_sim_s"] == 40.0


def test_each_conflict_family_has_valid_unique_spawns_and_commands() -> None:
    for family in SCENARIO_FAMILIES:
        for fleet_size in FLEET_SIZES:
            layout = conflict_layout(family, fleet_size)
            assert len(layout["spawn_indices"]) == fleet_size
            assert len(set(layout["spawn_indices"])) == fleet_size
            assert layout["commands"]
            assert all(
                0 <= int(command["slot"]) < fleet_size
                for command in layout["commands"]
            )


def test_long_corridor_layout_uses_topology_offsets_and_distinct_transforms() -> None:
    layout = conflict_layout("close_reciprocal", 8, LONG_LAYOUT_PROFILE)

    assert layout["spawn_indices"][:4] == layout["spawn_indices"][4:]
    assert layout["spawn_longitudinal_offsets_m"] == [0.0] * 4 + [25.0] * 4
    assert len(
        set(
            zip(
                layout["spawn_indices"],
                layout["spawn_longitudinal_offsets_m"],
            )
        )
    ) == 8
    assert layout["commands"][0]["direction"] == "right"
    assert layout["commands"][1]["direction"] == "left"
    schedule = build_schedule(1, LONG_LAYOUT_PROFILE)
    validate_schedule(schedule)
    assert schedule["layout_profile"] == LONG_LAYOUT_PROFILE


def test_legacy_long_corridor_profile_remains_reproducible() -> None:
    layout = conflict_layout(
        "close_reciprocal", 8, LEGACY_LONG_LAYOUT_PROFILE
    )

    assert layout["spawn_longitudinal_offsets_m"] == [-100.0] * 4 + [-75.0] * 4
    schedule = build_schedule(1, LEGACY_LONG_LAYOUT_PROFILE)
    validate_schedule(schedule)


def test_frozen_schedule_is_written_once_with_stable_digest(tmp_path) -> None:
    schedule = build_schedule(2)
    path = tmp_path / "schedule.json"

    digest = write_frozen_schedule(schedule, path)

    assert len(digest) == 64
    assert path.read_text(encoding="utf-8").endswith("\n")
    try:
        write_frozen_schedule(schedule, path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen schedules must not be overwritten")


def test_campaign_artifact_audit_requires_raw_files_and_rejects_images(
    tmp_path,
) -> None:
    blocks = [{"block_id": "block-1"}]
    run_dir = tmp_path / "block-1" / "FCFS_GAP"
    run_dir.mkdir(parents=True)
    for filename in ("metadata.json", "trajectory.jsonl", "events.jsonl"):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")

    result = audit_campaign_artifacts(tmp_path, blocks, ["FCFS_GAP"])

    assert result["passed"] is True
    (run_dir / "unexpected.png").write_bytes(b"frame")
    try:
        audit_campaign_artifacts(tmp_path, blocks, ["FCFS_GAP"])
    except ValueError as exc:
        assert "unexpected.png" in str(exc)
    else:
        raise AssertionError("persisted images must fail the artifact audit")
