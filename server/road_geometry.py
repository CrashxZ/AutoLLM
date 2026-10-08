"""Road-relative lane ordering helpers independent of the CARLA runtime."""

from __future__ import annotations

from typing import Any, Optional


def _is_same_direction_driving_lane(candidate: Any, origin: Any) -> bool:
    if candidate is None:
        return False
    candidate_id = int(candidate.lane_id)
    origin_id = int(origin.lane_id)
    return bool(
        candidate_id * origin_id > 0
        and int(candidate.road_id) == int(origin.road_id)
        and int(candidate.section_id) == int(origin.section_id)
        and "driving" in str(candidate.lane_type).lower()
    )


def _adjacent_chain(origin: Any, accessor: str) -> list[Any]:
    chain: list[Any] = []
    current = origin
    visited = {
        (int(origin.road_id), int(origin.section_id), int(origin.lane_id))
    }
    while True:
        candidate: Optional[Any] = getattr(current, accessor)()
        if not _is_same_direction_driving_lane(candidate, origin):
            break
        key = (
            int(candidate.road_id),
            int(candidate.section_id),
            int(candidate.lane_id),
        )
        if key in visited:
            break
        visited.add(key)
        chain.append(candidate)
        current = candidate
    return chain


def ordered_driving_lane_ids(waypoint: Any) -> tuple[int, ...]:
    """Return same-direction driving lanes ordered from left to right.

    OpenDRIVE lane IDs are signed identifiers, not zero-based lane ordinals.
    Their first driving ID can also vary when shoulder or auxiliary lanes lie
    between the road reference line and the carriageway. Traversing CARLA's
    adjacent-lane links avoids assuming either an ID origin or a sign.
    """

    left = _adjacent_chain(waypoint, "get_left_lane")
    right = _adjacent_chain(waypoint, "get_right_lane")
    ordered = [*reversed(left), waypoint, *right]
    return tuple(int(candidate.lane_id) for candidate in ordered)
