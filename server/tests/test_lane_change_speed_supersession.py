"""A terminating maneuver must not overwrite a newer coordination command."""
from unittest.mock import Mock

import pytest

from server.lane_change import LaneChangeController


@pytest.mark.parametrize("latest", [13.89, 0.0, 55.0])
def test_latest_speed_supersedes_pre_maneuver_target(latest):
    controller = object.__new__(LaneChangeController)
    controller.tm = Mock()
    controller.veh = object()
    controller._pre_desired_kmh = 50.0
    controller._get_desired_speed_kmh = lambda: latest
    controller._prev_auto_lane = False
    controller._clamped = True

    controller._restore_speed_and_auto()

    controller.tm.set_desired_speed.assert_called_once_with(controller.veh, latest)
    controller.tm.auto_lane_change.assert_called_once_with(controller.veh, False)
    assert controller._pre_desired_kmh is None
    assert controller._clamped is False


@pytest.mark.parametrize("latest", [None, float("nan")])
def test_missing_or_invalid_latest_target_uses_captured_target(latest):
    controller = object.__new__(LaneChangeController)
    controller.tm = Mock()
    controller.veh = object()
    controller._pre_desired_kmh = 40.0
    controller._get_desired_speed_kmh = lambda: latest
    controller._prev_auto_lane = False
    controller._clamped = True

    controller._restore_speed_and_auto()

    controller.tm.set_desired_speed.assert_called_once_with(controller.veh, 40.0)


@pytest.mark.parametrize("configured", [False, True])
def test_precheck_rejection_preserves_configured_auto_lane_change(configured):
    controller = object.__new__(LaneChangeController)
    controller.tm = Mock()
    controller.veh = object()
    controller._pre_desired_kmh = None
    controller._get_desired_speed_kmh = lambda: 50.0
    controller._prev_auto_lane = None
    controller._get_auto_lane_change_enabled = lambda: configured
    controller._clamped = False

    controller._restore_speed_and_auto()

    controller.tm.auto_lane_change.assert_called_once_with(controller.veh, configured)
