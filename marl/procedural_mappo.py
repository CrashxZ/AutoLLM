"""MAPPO reward adapter for procedural connected coordination scenarios."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .fleet_mappo import FleetMAPPOAdapter, actor_observations, centralized_context
from .procedural_scenarios import ProceduralFleetHighwayEnv, ProceduralScenarioSpec


class ProceduralMAPPOAdapter(FleetMAPPOAdapter):
    def __init__(self, env: ProceduralFleetHighwayEnv) -> None:
        super().__init__(env)
        self.env = env
        self.scenario_spec: ProceduralScenarioSpec | None = None

    def reset_procedural(
        self,
        spec: ProceduralScenarioSpec,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.scenario_spec = spec
        self.env.reset_procedural(spec)
        return actor_observations(self.env), centralized_context(self.env)

    def prepare_step(self) -> None:
        self.env.activate_due_goals()
