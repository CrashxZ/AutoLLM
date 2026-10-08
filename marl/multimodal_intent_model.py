"""Compact image, telemetry, and goal-text intent policy."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .multimodal_intent import (
    EGO_FEATURE_NAMES,
    INTENT_ACTIONS,
    MAX_NEIGHBOURS,
    REQUEST_ACTIONS,
    class_index,
)


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocabulary(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counts):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_goal(text: str, vocabulary: dict[str, int], max_tokens: int) -> list[int]:
    tokens = tokenize(text)[:max_tokens]
    encoded = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens]
    return encoded + [vocabulary[PAD_TOKEN]] * (max_tokens - len(encoded))


class IntentImageDataset(Dataset):
    """Lazy JPEG dataset backed by one immutable JSONL split."""

    def __init__(
        self,
        root: Path,
        rows: Sequence[dict],
        vocabulary: dict[str, int],
        *,
        image_width: int = 128,
        image_height: int = 64,
        max_goal_tokens: int = 16,
    ) -> None:
        self.root = Path(root)
        self.rows = list(rows)
        self.vocabulary = dict(vocabulary)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_goal_tokens = int(max_goal_tokens)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        image = image.resize((self.image_width, self.image_height), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)).copy())
        label = row["label"]
        return {
            "sample_id": row["sample_id"],
            "image": image_tensor,
            "ego_features": torch.tensor(row["ego_features"], dtype=torch.float32),
            "goal_tokens": torch.tensor(
                encode_goal(
                    row["goal"]["text"],
                    self.vocabulary,
                    self.max_goal_tokens,
                ),
                dtype=torch.long,
            ),
            "action_target": torch.tensor(
                class_index(INTENT_ACTIONS, label["ego_action"]),
                dtype=torch.long,
            ),
            "request_target": torch.tensor(
                class_index(REQUEST_ACTIONS, label["request_action"]),
                dtype=torch.long,
            ),
            "slot_target": torch.tensor(
                int(label["request_target_slot"]),
                dtype=torch.long,
            ),
        }


class CompactMultimodalIntentPolicy(nn.Module):
    """Small task-specific multimodal policy with categorical output heads."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        ego_feature_count: int = len(EGO_FEATURE_NAMES),
        max_neighbours: int = MAX_NEIGHBOURS,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.vocabulary_size = int(vocabulary_size)
        self.ego_feature_count = int(ego_feature_count)
        self.max_neighbours = int(max_neighbours)
        self.embedding_dim = int(embedding_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 4, 64),
            nn.ReLU(),
        )
        self.telemetry_encoder = nn.Sequential(
            nn.Linear(self.ego_feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.goal_embedding = nn.Embedding(
            self.vocabulary_size,
            self.embedding_dim,
            padding_idx=0,
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 96),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(96, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, len(INTENT_ACTIONS))
        self.request_head = nn.Linear(64, len(REQUEST_ACTIONS))
        self.slot_head = nn.Linear(64, self.max_neighbours + 1)

    def forward(
        self,
        image: torch.Tensor,
        ego_features: torch.Tensor,
        goal_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_encoder(image)
        telemetry_features = self.telemetry_encoder(ego_features)
        embedded = self.goal_embedding(goal_tokens)
        mask = (goal_tokens != 0).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp(min=1)
        pooled_goal = (embedded * mask).sum(dim=1) / denominator
        goal_features = self.goal_encoder(pooled_goal)
        fused = self.fusion(
            torch.cat((image_features, telemetry_features, goal_features), dim=1)
        )
        return {
            "action_logits": self.action_head(fused),
            "request_logits": self.request_head(fused),
            "slot_logits": self.slot_head(fused),
        }

    def config(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "ego_feature_count": self.ego_feature_count,
            "max_neighbours": self.max_neighbours,
            "embedding_dim": self.embedding_dim,
        }


def predict_classes(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "action": outputs["action_logits"].argmax(dim=1),
        "request": outputs["request_logits"].argmax(dim=1),
        "slot": outputs["slot_logits"].argmax(dim=1),
    }
