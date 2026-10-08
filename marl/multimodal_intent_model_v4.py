"""Temporal multimodal policy with staged cooperation and pointer heads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .multimodal_intent import (
    EGO_FEATURE_NAMES,
    INTENT_ACTIONS,
    MAX_NEIGHBOURS,
    NEIGHBOUR_FEATURE_NAMES,
    REQUEST_ACTIONS,
    class_index,
)
from .multimodal_intent_model import encode_goal


REQUEST_TYPES = REQUEST_ACTIONS[1:]


class TemporalNeighbourIntentDataset(Dataset):
    """V4 temporal images with neighbour telemetry reserved for target pointing."""

    def __init__(
        self,
        root: Path,
        rows: Sequence[dict],
        vocabulary: dict[str, int],
        *,
        image_width: int = 128,
        image_height: int = 64,
        max_goal_tokens: int = 16,
        image_index_map: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.rows = list(rows)
        self.vocabulary = dict(vocabulary)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_goal_tokens = int(max_goal_tokens)
        self.image_index_map = (
            list(range(len(self.rows)))
            if image_index_map is None
            else [int(value) for value in image_index_map]
        )
        if len(self.image_index_map) != len(self.rows):
            raise ValueError("image index map length must match rows")

    def __len__(self) -> int:
        return len(self.rows)

    def _image_tensor(self, row: dict) -> torch.Tensor:
        frames = []
        for key in ("image_previous_path", "image_current_path"):
            with Image.open(self.root / row[key]) as raw_image:
                image = raw_image.convert("RGB").resize(
                    (self.image_width, self.image_height),
                    Image.Resampling.BILINEAR,
                )
            frames.append(np.asarray(image, dtype=np.float32) / 255.0)
        temporal = np.concatenate(frames, axis=2)
        return torch.from_numpy(np.transpose(temporal, (2, 0, 1)).copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image_row = self.rows[self.image_index_map[index]]
        label = row["label"]
        request_action = label["request_action"]
        request_required = request_action != "none"
        request_type = (
            class_index(REQUEST_TYPES, request_action) if request_required else 0
        )
        return {
            "sample_id": row["sample_id"],
            "image": self._image_tensor(image_row),
            "ego_features": torch.tensor(row["ego_features"], dtype=torch.float32),
            "neighbour_features": torch.tensor(
                row["neighbour_features"], dtype=torch.float32
            ),
            "goal_tokens": torch.tensor(
                encode_goal(
                    row["goal"]["text"], self.vocabulary, self.max_goal_tokens
                ),
                dtype=torch.long,
            ),
            "action_target": torch.tensor(
                class_index(INTENT_ACTIONS, label["ego_action"]), dtype=torch.long
            ),
            "request_required_target": torch.tensor(
                float(request_required), dtype=torch.float32
            ),
            "request_type_target": torch.tensor(request_type, dtype=torch.long),
            "slot_target": torch.tensor(
                int(label["request_target_slot"]), dtype=torch.long
            ),
        }


class TemporalMultimodalIntentPolicyV4(nn.Module):
    """Force global decisions through vision while using telemetry as a pointer."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        ego_feature_count: int = len(EGO_FEATURE_NAMES),
        neighbour_feature_count: int = len(NEIGHBOUR_FEATURE_NAMES),
        max_neighbours: int = MAX_NEIGHBOURS,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.ego_feature_count = int(ego_feature_count)
        self.neighbour_feature_count = int(neighbour_feature_count)
        self.max_neighbours = int(max_neighbours)
        self.embedding_dim = int(embedding_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(48 * 2 * 4, 96),
            nn.ReLU(),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.goal_embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_dim, padding_idx=0
        )
        self.goal_encoder = nn.Sequential(nn.Linear(self.embedding_dim, 32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(96 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 96),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(96, len(INTENT_ACTIONS))
        self.request_required_head = nn.Linear(96, 1)
        self.request_type_head = nn.Linear(96, len(REQUEST_TYPES))
        self.neighbour_encoder = nn.Sequential(
            nn.Linear(self.neighbour_feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.target_query = nn.Linear(96, 32)
        self.no_target_head = nn.Linear(96, 1)

    def _goal_features(self, goal_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.goal_embedding(goal_tokens)
        mask = (goal_tokens != 0).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp(min=1)
        return self.goal_encoder((embedded * mask).sum(dim=1) / denominator)

    def forward(
        self,
        image: torch.Tensor,
        ego_features: torch.Tensor,
        neighbour_features: torch.Tensor,
        goal_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_features = self.fusion(
            torch.cat(
                (
                    self.image_encoder(image),
                    self.ego_encoder(ego_features),
                    self._goal_features(goal_tokens),
                ),
                dim=1,
            )
        )
        neighbour_encoded = self.neighbour_encoder(neighbour_features)
        query = self.target_query(global_features).unsqueeze(1)
        neighbour_scores = (query * neighbour_encoded).sum(dim=2) / math.sqrt(32.0)
        present = neighbour_features[:, :, 0] > 0.5
        neighbour_scores = neighbour_scores.masked_fill(~present, -1.0e4)
        slot_logits = torch.cat(
            (neighbour_scores, self.no_target_head(global_features)), dim=1
        )
        return {
            "action_logits": self.action_head(global_features),
            "request_required_logit": self.request_required_head(global_features).squeeze(1),
            "request_type_logits": self.request_type_head(global_features),
            "slot_logits": slot_logits,
        }

    def config(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "ego_feature_count": self.ego_feature_count,
            "neighbour_feature_count": self.neighbour_feature_count,
            "max_neighbours": self.max_neighbours,
            "embedding_dim": self.embedding_dim,
        }


def decode_v4(
    outputs: dict[str, torch.Tensor], threshold: float = 0.5
) -> dict[str, torch.Tensor]:
    required = torch.sigmoid(outputs["request_required_logit"]) >= float(threshold)
    request_type = outputs["request_type_logits"].argmax(dim=1) + 1
    request = torch.where(required, request_type, torch.zeros_like(request_type))
    return {
        "action": outputs["action_logits"].argmax(dim=1),
        "request": request,
        "slot": outputs["slot_logits"].argmax(dim=1),
    }
