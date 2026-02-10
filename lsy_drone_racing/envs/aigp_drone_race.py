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

from lsy_drone_racing.aigp.rewards import RewardCalculator, RewardConfig
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
    """Reward and bookkeeping helpers for AIGP environments."""

    def _init_aigp_rewards(self, reward_config: RewardConfig | dict | str | None = "swift") -> None:
        """Initialize modular rewards and stateful reward bookkeeping."""
        self.reward_calculator = RewardCalculator(reward_config)
        self._aigp_prev_action: np.ndarray | None = None
        self._aigp_has_prev_action: np.ndarray | None = None
        self._aigp_action_diff: np.ndarray | None = None
        self._last_reward_components: dict[str, Array] | None = None

    def _aigp_on_reset(self, mask: Array | None) -> None:
        """Clear any per-episode reward bookkeeping state (e.g. previous action)."""
        if self._aigp_has_prev_action is None:
            return
        if mask is None:
            self._aigp_has_prev_action[...] = False
            if self._aigp_prev_action is not None:
                self._aigp_prev_action[...] = 0.0
            if self._aigp_action_diff is not None:
                self._aigp_action_diff[...] = 0.0
            return

        m = np.asarray(mask, dtype=bool)
        self._aigp_has_prev_action[m, :] = False
        if self._aigp_prev_action is not None:
            self._aigp_prev_action[m, :, :] = 0.0
        if self._aigp_action_diff is not None:
            self._aigp_action_diff[m, :] = 0.0

    def reward(self) -> Array:
        """Compute the modular DronePrix-style reward for the current state."""
        gates_pos = self.sim.mjx_data.mocap_pos[:, self.data.gate_mj_ids]

        action_diff = None if self._aigp_action_diff is None else jp.asarray(self._aigp_action_diff)
        has_prev_action = (
            None
            if self._aigp_has_prev_action is None
            else jp.asarray(self._aigp_has_prev_action)
        )

        reward, components = self.reward_calculator.compute(
            pos=self.sim.data.states.pos,
            vel=self.sim.data.states.vel,
            quat=self.sim.data.states.quat,
            target_gate=self.data.target_gate,
            active_gate_count=self.data.active_gate_count,
            gates_pos=gates_pos,
            passed_gate=self.data.passed_gate,
            progress=self.data.progress,
            disabled_drones=self.data.disabled_drones,
            completed=self.data.completed,
            truncated=self.truncated(),
            steps=self.data.steps,
            max_episode_steps=self.data.max_episode_steps,
            freq=self.freq,
            pos_limit_low=self.data.pos_limit_low,
            pos_limit_high=self.data.pos_limit_high,
            action_diff=action_diff,
            has_prev_action=has_prev_action,
        )
        self._last_reward_components = components
        return reward

    def apply_action(self, action: Array) -> None:
        """Apply action and update reward bookkeeping (e.g. action smoothness)."""
        # NOTE: We keep a NumPy copy for smoothness; the simulator path itself still uses JAX.
        action_buf = np.reshape(action, (self.sim.n_worlds, self.sim.n_drones, -1), copy=True)
        if self._aigp_prev_action is None or self._aigp_prev_action.shape != action_buf.shape:
            self._aigp_prev_action = np.zeros_like(action_buf, dtype=np.float32)
            self._aigp_has_prev_action = np.zeros(action_buf.shape[:2], dtype=bool)
            self._aigp_action_diff = np.zeros(action_buf.shape[:2], dtype=np.float32)

        action_diff = np.linalg.norm(action_buf - self._aigp_prev_action, axis=-1)
        action_diff = np.where(self._aigp_has_prev_action, action_diff, 0.0)

        self._aigp_action_diff = action_diff.astype(np.float32, copy=False)
        self._aigp_prev_action = action_buf
        self._aigp_has_prev_action[...] = True

        super().apply_action(action)


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
        reward_config: RewardConfig | dict | str | None = "swift",
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
            reward_config: Reward configuration (preset name, config dict, RewardConfig, or None).
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
        self._init_aigp_rewards(reward_config)

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
        obs, info = super()._reset(seed=seed, options=options, mask=mask)
        self._aigp_on_reset(mask)
        return obs, info


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
        reward_config: RewardConfig | dict | str | None = "swift",
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
            reward_config: Reward configuration (preset name, config dict, RewardConfig, or None).
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
        self._init_aigp_rewards(reward_config)

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
        obs, info = super()._reset(seed=seed, options=options, mask=mask)
        self._aigp_on_reset(mask)
        return obs, info
