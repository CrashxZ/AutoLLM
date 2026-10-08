"""PyTorch networks and portable export helpers for variable-fleet MAPPO."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from .numpy_policy import write_policy_metadata


class SharedFleetActor(nn.Module):
    """One decentralized policy shared by every vehicle and fleet size."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class PooledFleetCritic(nn.Module):
    """Shared per-agent value model conditioned on pooled global context."""

    def __init__(self, observation_dim: int, context_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + context_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if observation.ndim < 2:
            raise ValueError("observation must include agent and feature axes")
        if context.shape[:-1] != observation.shape[:-2]:
            raise ValueError("context batch axes must match observation batch axes")
        expanded = context.unsqueeze(-2).expand(*observation.shape[:-1], context.shape[-1])
        value = self.network(torch.cat([observation, expanded], dim=-1))
        return value.squeeze(-1)


def export_shared_actor(
    actor: SharedFleetActor,
    output_path: str | Path,
    metadata: Dict[str, object],
) -> None:
    """Export the actor in the existing portable NumPy policy format."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    linear_layers = [module for module in actor.network if isinstance(module, nn.Linear)]
    arrays: Dict[str, np.ndarray] = {
        "layer_count": np.asarray(len(linear_layers), dtype=np.int64)
    }
    for index, layer in enumerate(linear_layers):
        arrays[f"weight_{index}"] = (
            layer.weight.detach().cpu().numpy().astype(np.float32)
        )
        arrays[f"bias_{index}"] = (
            layer.bias.detach().cpu().numpy().astype(np.float32)
        )
    np.savez_compressed(output, **arrays)
    write_policy_metadata(str(output), metadata)
