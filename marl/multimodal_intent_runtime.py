"""Runtime adapter for the frozen temporal multimodal intent policy."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from marl.multimodal_intent import (
    INTENT_ACTIONS,
    MAX_NEIGHBOURS,
    REQUEST_ACTIONS,
    ego_feature_vector,
    goal_sentence,
    lane_index,
    neighbour_feature_matrix,
    ordered_neighbour_ids,
    protocol_lane_id,
)
from marl.multimodal_intent_model import encode_goal
from marl.multimodal_intent_model_v4 import (
    TemporalMultimodalIntentPolicyV4,
    decode_v4,
)
from server.coordination.models import (
    Action,
    CooperationRequest,
    Goal,
    GoalKind,
    IntentProposal,
    PlanStep,
    VehiclePlan,
    VehicleState,
)


TEMPORAL_DELTA_S = 0.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def back_project_states(
    states: Mapping[int, VehicleState], delta_s: float = TEMPORAL_DELTA_S
) -> Dict[int, VehicleState]:
    """Build the constant-velocity prior frame used by V5 during training."""
    if delta_s <= 0.0:
        raise ValueError("temporal delta must be positive")
    output = {}
    for veh_id, state in states.items():
        prior_s = state.longitudinal_position_m() - state.speed_mps * delta_s
        output[int(veh_id)] = state.model_copy(
            update={"x_m": prior_s, "s_m": prior_s}
        )
    return output


def render_topdown_jpeg(
    states: Mapping[int, VehicleState],
    ego_veh_id: int,
    *,
    lane_width_m: float = 3.5,
) -> bytes:
    """Render the frozen V5 top-down image without writing frames to disk."""
    if ego_veh_id not in states:
        raise KeyError(f"missing ego state: {ego_veh_id}")
    width, height = 512, 256
    road_top, road_bottom = 32, 224
    lane_height = (road_bottom - road_top) / 4.0
    image = Image.new("RGB", (width, height), (52, 107, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, road_top, width, road_bottom), fill=(58, 61, 67))
    for lane in range(1, 4):
        y = road_top + lane * lane_height
        for x in range(0, width, 28):
            draw.line(
                (x, y, min(x + 15, width), y),
                fill=(230, 230, 220),
                width=2,
            )
    ego_s = states[ego_veh_id].longitudinal_position_m()
    for lane in range(4):
        draw.text(
            (5, int(road_top + (lane + 0.5) * lane_height - 7)),
            f"{-lane - 1}",
            fill="white",
        )
    for veh_id, state in sorted(states.items()):
        x = int(width / 2 + (state.longitudinal_position_m() - ego_s) * 4.0)
        x = min(width - 18, max(18, x))
        lane_position = (
            float(state.d_m) / lane_width_m
            if state.d_m is not None
            else lane_index(state.lane_id)
        )
        y = int(road_top + (lane_position + 0.5) * lane_height)
        colour = (66, 153, 225) if veh_id == ego_veh_id else (235, 87, 87)
        draw.rounded_rectangle(
            (x - 16, y - 9, x + 16, y + 9), radius=3, fill=colour
        )
        draw.text((x - 16, y - 25), str(veh_id), fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def temporal_image_tensor(
    previous_jpeg: bytes,
    current_jpeg: bytes,
    *,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    frames = []
    for image_bytes in (previous_jpeg, current_jpeg):
        with Image.open(BytesIO(image_bytes)) as raw_image:
            image = raw_image.convert("RGB").resize(
                (image_width, image_height), Image.Resampling.BILINEAR
            )
        frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    temporal = np.concatenate(frames, axis=2)
    return torch.from_numpy(np.transpose(temporal, (2, 0, 1)).copy())


@dataclass(frozen=True)
class RuntimeIntentResult:
    proposal: IntentProposal
    trace: dict


class FrozenMultimodalIntentPolicy:
    """Load frozen V5 artifacts and emit protocol-native vehicle proposals."""

    source = "frozen-multimodal-intent-v5"

    def __init__(
        self,
        checkpoint_path: Path,
        calibration_path: Path,
        *,
        max_consecutive_holds: Optional[int] = None,
        enforce_goal_direction: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.calibration_path = Path(calibration_path).resolve()
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.calibration_sha256 = file_sha256(self.calibration_path)
        calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        if calibration.get("status") != "complete":
            raise ValueError("V5 calibration artifact is not complete")
        if calibration.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("V5 checkpoint does not match calibration artifact")
        self.request_threshold = float(calibration["selected_threshold"])

        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if int(checkpoint.get("format_version", 0)) != 4:
            raise ValueError("V5 runtime requires a format-version 4 checkpoint")
        self.model = TemporalMultimodalIntentPolicyV4(
            **checkpoint["model_config"]
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.vocabulary = dict(checkpoint["vocabulary"])
        self.max_goal_tokens = int(checkpoint["max_goal_tokens"])
        self.image_width = int(checkpoint["image_width"])
        self.image_height = int(checkpoint["image_height"])
        self.max_neighbours = int(
            checkpoint["model_config"].get("max_neighbours", MAX_NEIGHBOURS)
        )
        if max_consecutive_holds is not None and max_consecutive_holds < 1:
            raise ValueError("max consecutive holds must be positive")
        self.max_consecutive_holds = max_consecutive_holds
        self.enforce_goal_direction = bool(enforce_goal_direction)
        self._consecutive_holds: Dict[int, int] = {}
        self._last_lane_index: Dict[int, int] = {}

    def propose(
        self,
        *,
        states: Mapping[int, VehicleState],
        ego_veh_id: int,
        goal_lane_index: int,
        route_exit: bool,
        deadline_s: Optional[float],
        now_s: float,
        flow_speed_kmh: float = 50.0,
        lane_change_duration_s: float = 3.0,
        exit_id: Optional[str] = None,
    ) -> RuntimeIntentResult:
        if ego_veh_id not in states:
            raise KeyError(f"missing ego state: {ego_veh_id}")
        started = time.perf_counter()
        ego = states[ego_veh_id]
        current_lane_index = lane_index(ego.lane_id)
        goal_text = goal_sentence(
            goal_lane_index=goal_lane_index,
            route_exit=route_exit,
            deadline_s=deadline_s,
        )
        neighbour_ids = ordered_neighbour_ids(
            states, ego_veh_id, limit=self.max_neighbours
        )
        ego_features = ego_feature_vector(
            speed_mps=ego.speed_mps,
            current_lane_index=current_lane_index,
            goal_lane_index=goal_lane_index,
            route_exit=route_exit,
            deadline_s=deadline_s,
            fleet_count=len(states),
        )
        neighbour_features = neighbour_feature_matrix(
            states,
            ego_veh_id,
            neighbour_ids,
            limit=self.max_neighbours,
        )
        current_jpeg = render_topdown_jpeg(states, ego_veh_id)
        previous_jpeg = render_topdown_jpeg(
            back_project_states(states), ego_veh_id
        )
        image = temporal_image_tensor(
            previous_jpeg,
            current_jpeg,
            image_width=self.image_width,
            image_height=self.image_height,
        ).unsqueeze(0)
        goal_tokens = torch.tensor(
            encode_goal(goal_text, self.vocabulary, self.max_goal_tokens),
            dtype=torch.long,
        ).unsqueeze(0)
        ego_tensor = torch.tensor(ego_features, dtype=torch.float32).unsqueeze(0)
        neighbour_tensor = torch.tensor(
            neighbour_features, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                image, ego_tensor, neighbour_tensor, goal_tokens
            )
            decoded = decode_v4(outputs, self.request_threshold)
            action_probabilities = torch.softmax(
                outputs["action_logits"], dim=1
            )[0]
            request_probability = torch.sigmoid(
                outputs["request_required_logit"]
            )[0]
            request_type_probabilities = torch.softmax(
                outputs["request_type_logits"], dim=1
            )[0]
            slot_probabilities = torch.softmax(outputs["slot_logits"], dim=1)[0]

        action_index = int(decoded["action"][0])
        request_index = int(decoded["request"][0])
        slot_index = int(decoded["slot"][0])
        decoded_action_name = INTENT_ACTIONS[action_index]
        action_name = decoded_action_name
        required_direction = (
            Action.LANE_RIGHT.value
            if goal_lane_index > current_lane_index
            else Action.LANE_LEFT.value
        )
        opposite_direction = (
            Action.LANE_LEFT.value
            if required_direction == Action.LANE_RIGHT.value
            else Action.LANE_RIGHT.value
        )
        goal_direction_corrected = bool(
            self.enforce_goal_direction
            and current_lane_index != goal_lane_index
            and decoded_action_name == opposite_direction
        )
        if goal_direction_corrected:
            action_name = required_direction
        previous_lane_index = self._last_lane_index.get(ego_veh_id)
        if previous_lane_index != current_lane_index:
            self._consecutive_holds[ego_veh_id] = 0
        self._last_lane_index[ego_veh_id] = current_lane_index
        if (
            decoded_action_name == Action.HOLD.value
            and current_lane_index != goal_lane_index
        ):
            self._consecutive_holds[ego_veh_id] = (
                self._consecutive_holds.get(ego_veh_id, 0) + 1
            )
        else:
            self._consecutive_holds[ego_veh_id] = 0
        liveness_escalated = bool(
            self.max_consecutive_holds is not None
            and self._consecutive_holds[ego_veh_id]
            >= self.max_consecutive_holds
            and current_lane_index != goal_lane_index
        )
        if liveness_escalated:
            action_name = required_direction
        action = Action(action_name)
        if action is Action.LANE_LEFT:
            target_lane_index = max(0, current_lane_index - 1)
        elif action is Action.LANE_RIGHT:
            target_lane_index = min(3, current_lane_index + 1)
        else:
            target_lane_index = current_lane_index

        request_name = REQUEST_ACTIONS[request_index]
        target_vehicle_id = (
            int(neighbour_ids[slot_index])
            if request_index > 0 and slot_index < len(neighbour_ids)
            else None
        )
        request = None
        if target_vehicle_id is not None:
            request = CooperationRequest(
                to_vehicle_ids=[target_vehicle_id],
                requested_action=Action(request_name),
                reason="Create a validator-admissible coordination gap",
                expires_at_s=now_s + 5.0,
            )

        if route_exit:
            goal = Goal(
                kind=GoalKind.ROUTE_EXIT,
                exit_id=exit_id or f"exit-{ego_veh_id}",
                target_lane_id=protocol_lane_id(goal_lane_index),
                deadline_s=deadline_s,
                semantic=goal_text,
            )
        else:
            goal = Goal(
                kind=GoalKind.TARGET_LANE,
                target_lane_id=protocol_lane_id(goal_lane_index),
                semantic=goal_text,
            )

        current_speed_kmh = ego.speed_mps * 3.6
        target_speed_kmh = (
            flow_speed_kmh
            if action in {Action.LANE_LEFT, Action.LANE_RIGHT}
            else current_speed_kmh
        )
        source_suffixes = []
        if goal_direction_corrected:
            source_suffixes.append("goal-direction-guard")
        if liveness_escalated:
            source_suffixes.append("goal-progress-monitor")
        proposal_source = self.source + "".join(
            f"+{suffix}" for suffix in source_suffixes
        )
        proposal = IntentProposal(
            ego_veh_id=ego_veh_id,
            created_at_s=now_s,
            expires_at_s=now_s + 5.0,
            observation_ts_s=now_s,
            goal=goal,
            plan=VehiclePlan(
                veh_id=ego_veh_id,
                summary=f"V5 intent: {action.value}",
                horizon_s=lane_change_duration_s,
                steps=[
                    PlanStep(
                        action=action,
                        target_lane_id=protocol_lane_id(target_lane_index),
                        target_speed_kmh=target_speed_kmh,
                        duration_s=lane_change_duration_s,
                        completion_condition=(
                            f"lane_id={protocol_lane_id(target_lane_index)}"
                        ),
                    )
                ],
            ),
            confidence=float(
                action_probabilities[INTENT_ACTIONS.index(action_name)]
            ),
            request=request,
            image_sha256=hashlib.sha256(current_jpeg).hexdigest(),
            source=proposal_source,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_slot_target = (
            int(neighbour_ids[slot_index])
            if slot_index < len(neighbour_ids)
            else None
        )
        trace = {
            "schema_version": "1.0",
            "transaction_id": proposal.transaction_id,
            "ts_s": now_s,
            "ego_veh_id": ego_veh_id,
            "source": self.source,
            "proposal_source": proposal_source,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_sha256": self.calibration_sha256,
            "request_threshold": self.request_threshold,
            "temporal_delta_s": TEMPORAL_DELTA_S,
            "goal_text": goal_text,
            "neighbour_slots": neighbour_ids,
            "image_previous_sha256": hashlib.sha256(previous_jpeg).hexdigest(),
            "image_current_sha256": proposal.image_sha256,
            "action_probabilities": {
                name: float(action_probabilities[index])
                for index, name in enumerate(INTENT_ACTIONS)
            },
            "request_required_probability": float(request_probability),
            "request_type_probabilities": {
                name: float(request_type_probabilities[index - 1])
                for index, name in enumerate(REQUEST_ACTIONS)
                if index > 0
            },
            "slot_probabilities": [float(value) for value in slot_probabilities],
            "decoded_action": decoded_action_name,
            "effective_action": action_name,
            "consecutive_hold_proposals": self._consecutive_holds[ego_veh_id],
            "max_consecutive_holds": self.max_consecutive_holds,
            "goal_progress_escalated": liveness_escalated,
            "enforce_goal_direction": self.enforce_goal_direction,
            "goal_direction_corrected": goal_direction_corrected,
            "decoded_request": request_name,
            "decoded_slot": slot_index,
            "decoded_slot_vehicle_id": raw_slot_target,
            "effective_request": request_name if request is not None else "none",
            "effective_request_vehicle_id": target_vehicle_id,
            "request_suppressed_no_target": bool(
                request_index > 0 and target_vehicle_id is None
            ),
            "inference_latency_ms": elapsed_ms,
        }
        return RuntimeIntentResult(proposal=proposal, trace=trace)
