"""Deterministic command-link impairments for paired coordination studies."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelSnapshot:
    sent_commands: int
    delivered_commands: int
    dropped_commands: int
    fallback_commands: int


class DelayedLossyActionChannel:
    """Apply fixed delay and paired packet loss to per-vehicle commands.

    A command is a one-tick action. Missing commands use the supplied local
    fallback for that tick. Packet draws are a pure function of
    ``(seed, step, vehicle)`` so every paired method sees the same loss trace
    even when method execution order differs.
    """

    def __init__(
        self,
        vehicle_count: int,
        *,
        dt_s: float,
        delay_ms: int = 0,
        packet_loss_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        if vehicle_count < 1:
            raise ValueError("vehicle_count must be positive")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if not 0.0 <= packet_loss_probability <= 1.0:
            raise ValueError("packet_loss_probability must be in [0, 1]")
        self.vehicle_count = int(vehicle_count)
        self.dt_s = float(dt_s)
        self.delay_ms = int(delay_ms)
        self.delay_steps = int(math.ceil(self.delay_ms / (1000.0 * self.dt_s)))
        self.packet_loss_probability = float(packet_loss_probability)
        self.seed = int(seed)
        self._pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sent_commands = 0
        self._delivered_commands = 0
        self._dropped_commands = 0
        self._fallback_commands = 0

    def loss_mask(self, step: int) -> np.ndarray:
        """Return the reproducible packet-loss mask for one send step."""
        if step < 0:
            raise ValueError("step must be non-negative")
        values = np.empty(self.vehicle_count, dtype=np.float64)
        for vehicle in range(self.vehicle_count):
            payload = f"{self.seed}:{step}:{vehicle}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            values[vehicle] = int.from_bytes(digest[:8], "big") / float(2**64)
        return values < self.packet_loss_probability

    def transmit(
        self,
        step: int,
        proposed_actions: np.ndarray,
        fallback_actions: np.ndarray,
    ) -> np.ndarray:
        """Send current commands and return commands available this tick."""
        proposed = self._validated_actions(proposed_actions, "proposed_actions")
        fallback = self._validated_actions(fallback_actions, "fallback_actions")
        lost = self.loss_mask(step)
        delivery_step = int(step) + self.delay_steps
        self._pending[delivery_step] = (proposed.copy(), ~lost)
        self._sent_commands += self.vehicle_count
        self._dropped_commands += int(np.sum(lost))

        available = fallback.copy()
        queued = self._pending.pop(int(step), None)
        if queued is None:
            self._fallback_commands += self.vehicle_count
            return available
        actions, delivered = queued
        available[delivered] = actions[delivered]
        delivered_count = int(np.sum(delivered))
        self._delivered_commands += delivered_count
        self._fallback_commands += self.vehicle_count - delivered_count
        return available

    def snapshot(self) -> ChannelSnapshot:
        return ChannelSnapshot(
            sent_commands=self._sent_commands,
            delivered_commands=self._delivered_commands,
            dropped_commands=self._dropped_commands,
            fallback_commands=self._fallback_commands,
        )

    def trace_sha256(self, max_steps: int) -> str:
        """Hash the complete scheduled loss trace for pairing checks."""
        digest = hashlib.sha256()
        for step in range(int(max_steps)):
            digest.update(np.packbits(self.loss_mask(step)).tobytes())
        return digest.hexdigest()

    def _validated_actions(self, actions: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(actions, dtype=np.int64)
        if values.shape != (self.vehicle_count,):
            raise ValueError(f"{name} must have shape (vehicle_count,)")
        return values
