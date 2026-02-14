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
AIGP_MAX_OBSTACLES = 1


def _pad_track_assets(
    track: "ConfigDict", *, max_gates: int, max_obstacles: int
) -> "ConfigDict":
    """Pad a track's gates/obstacles to fixed maximum sizes.

    We keep the observation/action space constant across curriculum stages by ensuring the
    environment always contains `max_gates` gate bodies and `max_obstacles` obstacle bodies.
    Shorter tracks store their true gate count in `track.active_gate_count` and pad the remaining
    gates/obstacles with unreachable dummy poses.

    Notes:
        Stable-Baselines3 cannot reliably handle dict observations that contain zero-sized
        dimensions (e.g. `obstacles_pos` with shape `(0, 3)`). Padding obstacles to at least 1
        avoids this issue.
    """
    if "gates" not in track:
        raise KeyError("track must contain gates")
    if "obstacles" not in track:
        track = copy.deepcopy(track)
        track["obstacles"] = []
    track = copy.deepcopy(track)
    n_gates = len(track.gates)
    if n_gates < 1:
        raise ValueError("track must contain at least one gate")
    if n_gates > max_gates:
        raise ValueError(f"track has {n_gates} gates, exceeds max_gates={max_gates}")

    active_gate_count = int(track.get("active_gate_count", n_gates))
    if active_gate_count < 1 or active_gate_count > n_gates:
        raise ValueError(f"Invalid active_gate_count={active_gate_count} for n_gates={n_gates}")
    track["active_gate_count"] = active_gate_count

    if n_gates < max_gates:
        dummy_gate = {"pos": [1e6, 1e6, 1e6], "rpy": [0.0, 0.0, 0.0]}
        track["gates"] = list(track.gates) + [dummy_gate] * (max_gates - n_gates)

    n_obstacles = len(track.obstacles)
    if n_obstacles > max_obstacles:
        raise ValueError(
            f"track has {n_obstacles} obstacles, exceeds max_obstacles={max_obstacles}"
        )
    if n_obstacles < max_obstacles:
        dummy_obstacle = {"pos": [1e6, 1e6, 1e6]}
        track["obstacles"] = list(track.obstacles) + [dummy_obstacle] * (
            max_obstacles - n_obstacles
        )

    return track


def _track_name_from_config(track: "ConfigDict", *, default: str) -> str:
    """Resolve a stable, human-readable track name from a config object."""
    for key in ("_track_name", "name", "id"):
        raw = track.get(key)
        if raw is not None:
            value = str(raw).strip()
            if value:
                return value
    return default


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
        action_buf = np.asarray(action).reshape((self.sim.n_worlds, self.sim.n_drones, -1)).copy()
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
        max_obstacles: int = AIGP_MAX_OBSTACLES,
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
            max_obstacles: Maximum obstacle bodies to include in the simulation. Tracks with fewer
                obstacles are padded with dummy obstacles to avoid zero-sized observation shapes.
        """
        track = _pad_track_assets(track, max_gates=max_gates, max_obstacles=max_obstacles)
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
        self._track_names: list[str] = [_track_name_from_config(track, default="track_00")]
        self._track_idx_per_world = np.zeros((self.sim.n_worlds,), dtype=np.int32)
        self._init_aigp_rewards(reward_config)

    def set_track_pool(
        self,
        tracks: list["ConfigDict"],
        *,
        probs: list[float] | None = None,
        max_gates: int = AIGP_MAX_GATES,
        max_obstacles: int = AIGP_MAX_OBSTACLES,
    ) -> None:
        """Set a pool of tracks to sample from at each reset.

        Note:
            This sampling currently happens once per reset call (so vectorized environments will
            share the same sampled track across all worlds for that reset).
        """
        if not tracks:
            self._track_pool = None
            self._track_pool_probs = None
            self._track_names = [_track_name_from_config(self.track, default="track_00")]
            return
        padded: list["ConfigDict"] = []
        names: list[str] = []
        for i, t in enumerate(tracks):
            t2 = _pad_track_assets(t, max_gates=max_gates, max_obstacles=max_obstacles)
            name = _track_name_from_config(t2, default=f"track_{i:02d}")
            t2["_track_id"] = int(i)
            t2["_track_name"] = name
            padded.append(t2)
            names.append(name)
        self._track_pool = padded
        self._track_names = names
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
            self._track_idx_per_world[...] = idx
        else:
            self._track_idx_per_world[...] = 0
        obs, info = super()._reset(seed=seed, options=options, mask=mask)
        self._aigp_on_reset(mask)
        return obs, info

    def info(self) -> dict:
        """Return base info plus optional track attribution metadata."""
        out = super().info()
        n_worlds, n_drones = self.sim.n_worlds, self.sim.n_drones
        track_idx = np.broadcast_to(self._track_idx_per_world[:, None], (n_worlds, n_drones))
        out["track_id"] = jp.asarray(track_idx, dtype=jp.int32)
        max_len = max((len(name) for name in self._track_names), default=1)
        track_names = np.empty((n_worlds, n_drones), dtype=f"<U{max_len}")
        for i in range(n_worlds):
            idx = int(self._track_idx_per_world[i])
            name = (
                self._track_names[idx]
                if 0 <= idx < len(self._track_names)
                else f"track_{idx:02d}"
            )
            track_names[i, :] = name
        out["track_name"] = track_names
        return out


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
        max_obstacles: int = AIGP_MAX_OBSTACLES,
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
            max_obstacles: Maximum obstacle bodies to include in the simulation. Tracks with fewer
                obstacles are padded with dummy obstacles to avoid zero-sized observation shapes.
        """
        track = _pad_track_assets(track, max_gates=max_gates, max_obstacles=max_obstacles)
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
        self._track_names: list[str] = [_track_name_from_config(track, default="track_00")]
        self._track_idx_per_world = np.zeros((self.sim.n_worlds,), dtype=np.int32)
        self._init_aigp_rewards(reward_config)

    def set_track_pool(
        self,
        tracks: list["ConfigDict"],
        *,
        probs: list[float] | None = None,
        max_gates: int = AIGP_MAX_GATES,
        max_obstacles: int = AIGP_MAX_OBSTACLES,
    ) -> None:
        """Set a pool of tracks to sample from at each reset."""
        if not tracks:
            self._track_pool = None
            self._track_pool_probs = None
            self._track_names = [_track_name_from_config(self.track, default="track_00")]
            return
        padded: list["ConfigDict"] = []
        names: list[str] = []
        for i, t in enumerate(tracks):
            t2 = _pad_track_assets(t, max_gates=max_gates, max_obstacles=max_obstacles)
            name = _track_name_from_config(t2, default=f"track_{i:02d}")
            t2["_track_id"] = int(i)
            t2["_track_name"] = name
            padded.append(t2)
            names.append(name)
        self._track_pool = padded
        self._track_names = names
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
        """Reset the environment (optionally sampling tracks per world)."""
        if not self._track_pool:
            self._track_idx_per_world[...] = 0
            obs, info = super()._reset(seed=seed, options=options, mask=mask)
            self._aigp_on_reset(mask)
            return obs, info

        world_mask = (
            np.ones((self.sim.n_worlds,), dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if not world_mask.any():
            obs, info = self.obs(), self.info()
            self._aigp_on_reset(mask)
            return obs, info

        rng = getattr(self, "_np_random", None)
        if rng is None:
            rng = np.random.default_rng(self.seed)

        sampled = np.asarray(
            rng.choice(
                len(self._track_pool),
                size=int(world_mask.sum()),
                p=self._track_pool_probs,
            ),
            dtype=np.int32,
        )
        self._track_idx_per_world[world_mask] = sampled

        seed_local = seed
        obs: dict[str, Array] | None = None
        info: dict | None = None
        unique_idx = np.unique(sampled)
        world_indices = np.flatnonzero(world_mask)
        for idx in unique_idx:
            idx_int = int(idx)
            per_world_mask = np.zeros((self.sim.n_worlds,), dtype=bool)
            per_world_mask[world_indices[sampled == idx_int]] = True
            self.track = self._track_pool[idx_int]
            obs, info = super()._reset(
                seed=seed_local,
                options=options,
                mask=jp.asarray(per_world_mask),
            )
            seed_local = None

        if obs is None or info is None:
            obs, info = self.obs(), self.info()
        self._aigp_on_reset(mask)
        return obs, info

    def info(self) -> dict:
        """Return base info plus optional track attribution metadata."""
        out = super().info()
        n_worlds, n_drones = self.sim.n_worlds, self.sim.n_drones
        track_idx = np.broadcast_to(self._track_idx_per_world[:, None], (n_worlds, n_drones))
        out["track_id"] = jp.asarray(track_idx, dtype=jp.int32)
        max_len = max((len(name) for name in self._track_names), default=1)
        track_names = np.empty((n_worlds, n_drones), dtype=f"<U{max_len}")
        for i in range(n_worlds):
            idx = int(self._track_idx_per_world[i])
            name = (
                self._track_names[idx]
                if 0 <= idx < len(self._track_names)
                else f"track_{idx:02d}"
            )
            track_names[i, :] = name
        out["track_name"] = track_names
        return out
