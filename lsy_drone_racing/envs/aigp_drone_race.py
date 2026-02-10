"""AIGP drone racing environments.

These environments are thin wrappers around :class:`~lsy_drone_racing.envs.drone_race.DroneRaceEnv`
that swap in a non-colliding gate asset. This is useful for early curriculum stages where gate
sizes can be significantly larger than the lab gate geometry.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jp

from lsy_drone_racing.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv


class _AIGPRewardMixin:
    """Reward definition for AIGP environments.

    This is a minimal, dense reward intended to bootstrap learning:
    - gate passage reward
    - progress towards current target gate
    - time penalty
    - crash penalty
    - completion bonus

    The full DronePrix reward system can be ported on top of this mixin later.
    """

    gate_passage_reward: float = 10.0
    progress_weight: float = 2.0
    crash_penalty: float = -30.0
    time_penalty: float = -0.01
    completion_bonus: float = 10.0

    def reward(self):  # noqa: ANN001 - gymnasium expects an Array-like return
        passed = self.data.passed_gate.astype(jp.float32)
        completed = self.data.completed.astype(jp.float32)
        crashed = (self.data.disabled_drones & ~self.data.completed).astype(jp.float32)
        reward = (
            self.gate_passage_reward * passed
            + self.progress_weight * self.data.progress
            + self.time_penalty
            + self.crash_penalty * crashed
            + self.completion_bonus * completed
        )
        return reward


class AIGPDroneRaceEnv(_AIGPRewardMixin, DroneRaceEnv):
    """Single-agent AIGP drone racing environment."""

    gate_spec_path = Path(__file__).parent / "assets/gate_nocollision.xml"


class VecAIGPDroneRaceEnv(_AIGPRewardMixin, VecDroneRaceEnv):
    """Vectorized single-agent AIGP drone racing environment."""

    gate_spec_path = Path(__file__).parent / "assets/gate_nocollision.xml"
