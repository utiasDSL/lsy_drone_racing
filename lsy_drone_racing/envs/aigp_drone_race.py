"""AIGP drone racing environments.

These environments are thin wrappers around :class:`~lsy_drone_racing.envs.drone_race.DroneRaceEnv`
that swap in a non-colliding gate asset. This is useful for early curriculum stages where gate
sizes can be significantly larger than the lab gate geometry.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax.numpy as jp
import numpy as np

from lsy_drone_racing.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict


AIGP_MAX_GATES = 11


def _pad_track_gates(track: "ConfigDict", *, max_gates: int) -> "ConfigDict":
    """Pad a track's gate list to a fixed maximum size.

    We keep the observation/action space constant across curriculum stages by ensuring the
    environment always contains `max_gates` gate bodies. Shorter tracks store their true gate count
    in `track.active_gate_count` and pad the remaining gates with unreachable dummy poses.
    """
    if "gates" not in track:
        raise KeyError("track must contain gates")
    track = copy.deepcopy(track)
    n_gates = len(track.gates)
    if n_gates < 1:
        raise ValueError("track must contain at least one gate")
    if n_gates > max_gates:
        raise ValueError(f"track has {n_gates} gates, exceeds max_gates={max_gates}")

    track["active_gate_count"] = n_gates
    if n_gates == max_gates:
        return track

    dummy_gate = {"pos": [1e6, 1e6, 1e6], "rpy": [0.0, 0.0, 0.0]}
    track["gates"] = list(track.gates) + [dummy_gate] * (max_gates - n_gates)
    return track


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

    def reward(self) -> Array:
        """Compute a dense reward for early-stage learning."""
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

    def __init__(
        self,
        freq: int,
        sim_config: "ConfigDict",
        track: "ConfigDict",
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: "ConfigDict | None" = None,
        randomizations: "ConfigDict | None" = None,
        seed: str | int = "random",
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
        *,
        max_gates: int = AIGP_MAX_GATES,
    ):
        """Initialize the single-agent AIGP drone racing environment.

        Args:
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: "random" for a generated seed or the random seed directly.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
            max_gates: Maximum gate bodies to include in the simulation. Tracks with fewer gates are
                padded with dummy gates and store their true gate count in
                `track.active_gate_count`.
        """
        track = _pad_track_gates(track, max_gates=max_gates)
        super().__init__(
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self._track_pool: list["ConfigDict"] | None = None
        self._track_pool_probs: np.ndarray | None = None

    def set_track_pool(
        self,
        tracks: list["ConfigDict"],
        *,
        probs: list[float] | None = None,
        max_gates: int = AIGP_MAX_GATES,
    ) -> None:
        """Set a pool of tracks to sample from at each reset.

        Note:
            This sampling currently happens once per reset call (so vectorized environments will
            share the same sampled track across all worlds for that reset).
        """
        if not tracks:
            self._track_pool = None
            self._track_pool_probs = None
            return
        padded = [_pad_track_gates(t, max_gates=max_gates) for t in tracks]
        self._track_pool = padded
        if probs is None:
            self._track_pool_probs = None
        else:
            p = np.asarray(probs, dtype=np.float64)
            if p.shape != (len(tracks),):
                raise ValueError("probs length mismatch")
            if np.any(p < 0):
                raise ValueError("probs must be non-negative")
            s = float(p.sum())
            if s <= 0:
                raise ValueError("probs must sum to > 0")
            self._track_pool_probs = p / s

    def _reset(
        self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset the environment (optionally sampling from a track pool)."""
        if self._track_pool:
            rng = getattr(self, "_np_random", None)
            if rng is None:
                rng = np.random.default_rng(self.seed)
            idx = int(rng.choice(len(self._track_pool), p=self._track_pool_probs))
            self.track = self._track_pool[idx]
        return super()._reset(seed=seed, options=options, mask=mask)


class VecAIGPDroneRaceEnv(_AIGPRewardMixin, VecDroneRaceEnv):
    """Vectorized single-agent AIGP drone racing environment."""

    gate_spec_path = Path(__file__).parent / "assets/gate_nocollision.xml"

    def __init__(
        self,
        num_envs: int,
        freq: int,
        sim_config: "ConfigDict",
        track: "ConfigDict",
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: "ConfigDict | None" = None,
        randomizations: "ConfigDict | None" = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
        *,
        max_gates: int = AIGP_MAX_GATES,
    ):
        """Initialize the vectorized AIGP drone racing environment.

        Args:
            num_envs: Number of parallel environments.
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
            max_gates: Maximum gate bodies to include in the simulation. Tracks with fewer gates are
                padded with dummy gates and store their true gate count in
                `track.active_gate_count`.
        """
        track = _pad_track_gates(track, max_gates=max_gates)
        super().__init__(
            num_envs=num_envs,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self._track_pool: list["ConfigDict"] | None = None
        self._track_pool_probs: np.ndarray | None = None

    def set_track_pool(
        self,
        tracks: list["ConfigDict"],
        *,
        probs: list[float] | None = None,
        max_gates: int = AIGP_MAX_GATES,
    ) -> None:
        """Set a pool of tracks to sample from at each reset."""
        if not tracks:
            self._track_pool = None
            self._track_pool_probs = None
            return
        padded = [_pad_track_gates(t, max_gates=max_gates) for t in tracks]
        self._track_pool = padded
        if probs is None:
            self._track_pool_probs = None
        else:
            p = np.asarray(probs, dtype=np.float64)
            if p.shape != (len(tracks),):
                raise ValueError("probs length mismatch")
            if np.any(p < 0):
                raise ValueError("probs must be non-negative")
            s = float(p.sum())
            if s <= 0:
                raise ValueError("probs must sum to > 0")
            self._track_pool_probs = p / s

    def _reset(
        self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset the environment (optionally sampling from a track pool)."""
        if self._track_pool:
            rng = getattr(self, "_np_random", None)
            if rng is None:
                rng = np.random.default_rng(self.seed)
            idx = int(rng.choice(len(self._track_pool), p=self._track_pool_probs))
            self.track = self._track_pool[idx]
        return super()._reset(seed=seed, options=options, mask=mask)
