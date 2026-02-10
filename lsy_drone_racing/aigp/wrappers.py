"""AIGP sim-to-real wrappers.

These wrappers are designed to work with gymnasium *vector* environments.
They keep all modifications in Python (outside the JAX sim pipeline) so they are easy to iterate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from numpy.typing import NDArray

Obs = dict[str, NDArray[np.generic]]
Info = dict[str, Any]
ResetReturn = tuple[Obs, Info]
StepReturn = tuple[Obs, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], Info]


class ActionLatencyWrapper(VectorWrapper):
    """Apply N-step action latency using a per-env FIFO buffer."""

    def __init__(
        self,
        env: VectorEnv,
        *,
        latency_steps: int | tuple[int, int] = 0,
        seed: int | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: Vectorized environment.
            latency_steps: Either a fixed integer latency or an inclusive (min, max) range sampled
                per environment instance.
            seed: Seed for the wrapper's RNG.
        """
        super().__init__(env)
        self.latency_steps = latency_steps
        self._rng = np.random.default_rng(seed)

        self._latency_per_env: NDArray[np.int_] | None = None  # (num_envs,)
        self._buf: NDArray[np.floating] | None = None  # (T, num_envs, *action_shape)
        self._buf_head: int = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetReturn:
        """Reset the environment and initialize latency buffers."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._init_state()
        return obs, info

    def _init_state(self) -> None:
        num_envs = int(self.env.num_envs)
        if isinstance(self.latency_steps, tuple):
            lo, hi = int(self.latency_steps[0]), int(self.latency_steps[1])
            if hi < lo:
                lo, hi = hi, lo
            self._latency_per_env = self._rng.integers(lo, hi + 1, size=(num_envs,), dtype=np.int32)
        else:
            self._latency_per_env = np.full((num_envs,), int(self.latency_steps), dtype=np.int32)

        max_latency = int(np.max(self._latency_per_env)) if num_envs > 0 else 0
        # Use the single action space for buffer shape.
        act_shape = tuple(self.env.single_action_space.shape)
        self._buf = np.zeros((max_latency + 1, num_envs, *act_shape), dtype=np.float32)
        self._buf_head = 0

    def step(self, actions: Any) -> StepReturn:
        """Step the environment using latency-delayed actions."""
        if self._buf is None or self._latency_per_env is None:
            self._init_state()

        actions_np = np.asarray(actions, dtype=np.float32)

        # Store current actions.
        self._buf[self._buf_head, :, ...] = actions_np

        # Select delayed actions per env.
        buf_len = self._buf.shape[0]
        idx = (self._buf_head - self._latency_per_env) % buf_len  # (num_envs,)
        delayed = self._buf[idx, np.arange(actions_np.shape[0]), ...]

        self._buf_head = (self._buf_head + 1) % buf_len

        obs, reward, terminated, truncated, info = self.env.step(delayed)

        done = np.asarray(terminated) | np.asarray(truncated)
        if done.any():
            # Clear the buffer for completed envs so the next episode doesn't replay old actions.
            self._buf[:, done, ...] = 0.0
            if isinstance(self.latency_steps, tuple):
                lo, hi = int(self.latency_steps[0]), int(self.latency_steps[1])
                if hi < lo:
                    lo, hi = hi, lo
                self._latency_per_env[done] = self._rng.integers(
                    lo, hi + 1, size=int(done.sum()), dtype=np.int32
                )

        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class ImuNoiseConfig:
    """IMU-style bias + noise config for velocity and angular velocity observations."""

    vel_bias_std: float = 0.0
    ang_vel_bias_std: float = 0.0
    vel_noise_std: float = 0.0
    ang_vel_noise_std: float = 0.0
    bias_drift_std: float = 0.0  # random walk std per step


class ImuBiasNoiseWrapper(VectorWrapper):
    """Add IMU-like bias/noise to `vel` and `ang_vel` observations."""

    def __init__(self, env: VectorEnv, cfg: ImuNoiseConfig, *, seed: int | None = None):
        """Initialize the wrapper."""
        super().__init__(env)
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._vel_bias: NDArray[np.floating] | None = None  # (num_envs, n_drones, 3)
        self._ang_bias: NDArray[np.floating] | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetReturn:
        """Reset the environment and (re-)sample IMU biases."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_biases_from_obs(obs)
        obs = self._apply(obs)
        return obs, info

    def _reset_biases_from_obs(self, obs: dict[str, Any]) -> None:
        num_envs = int(self.env.num_envs)
        # Infer n_drones from observation shape if possible, otherwise default to 1.
        n_drones = int(np.asarray(obs["vel"]).shape[1]) if "vel" in obs else 1
        self._vel_bias = self._rng.normal(
            0.0, float(self.cfg.vel_bias_std), size=(num_envs, n_drones, 3)
        ).astype(np.float32)
        self._ang_bias = self._rng.normal(
            0.0, float(self.cfg.ang_vel_bias_std), size=(num_envs, n_drones, 3)
        ).astype(np.float32)

    def _apply(self, obs: Obs) -> Obs:
        if self._vel_bias is None or self._ang_bias is None:
            return obs

        out = dict(obs)
        if "vel" in out:
            vel = np.asarray(out["vel"], dtype=np.float32)
            vel = vel + self._vel_bias + self._rng.normal(
                0.0, float(self.cfg.vel_noise_std), size=vel.shape
            ).astype(np.float32)
            out["vel"] = vel
        if "ang_vel" in out:
            ang = np.asarray(out["ang_vel"], dtype=np.float32)
            ang = ang + self._ang_bias + self._rng.normal(
                0.0, float(self.cfg.ang_vel_noise_std), size=ang.shape
            ).astype(np.float32)
            out["ang_vel"] = ang
        return out

    def step(self, actions: Any) -> StepReturn:
        """Step the environment and apply IMU-like noise to observations."""
        obs, reward, terminated, truncated, info = self.env.step(actions)
        if self._vel_bias is None or self._ang_bias is None:
            self._reset_biases_from_obs(obs)

        # Drift biases.
        if float(self.cfg.bias_drift_std) > 0.0:
            self._vel_bias = self._vel_bias + self._rng.normal(
                0.0, float(self.cfg.bias_drift_std), size=self._vel_bias.shape
            ).astype(np.float32)
            self._ang_bias = self._ang_bias + self._rng.normal(
                0.0, float(self.cfg.bias_drift_std), size=self._ang_bias.shape
            ).astype(np.float32)

        done = np.asarray(terminated) | np.asarray(truncated)
        if done.any():
            # Resample biases for envs that ended.
            self._vel_bias[done, :, :] = self._rng.normal(
                0.0, float(self.cfg.vel_bias_std), size=self._vel_bias[done, :, :].shape
            ).astype(np.float32)
            self._ang_bias[done, :, :] = self._rng.normal(
                0.0, float(self.cfg.ang_vel_bias_std), size=self._ang_bias[done, :, :].shape
            ).astype(np.float32)

        obs = self._apply(obs)
        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class VioFailureConfig:
    """VIO failure model for position/velocity observations."""

    failure_prob: float = 0.0  # per-step probability to start a failure
    max_hold_steps: int = 0
    mode: str = "hold"  # "hold" or "zero"


class VioFailureWrapper(VectorWrapper):
    """Simulate VIO dropouts by holding or zeroing selected observation keys."""

    def __init__(
        self,
        env: VectorEnv,
        cfg: VioFailureConfig,
        *,
        keys: tuple[str, ...] = ("pos", "vel", "quat", "ang_vel"),
        seed: int | None = None,
    ):
        """Initialize the wrapper."""
        super().__init__(env)
        self.cfg = cfg
        self.keys = keys
        self._rng = np.random.default_rng(seed)
        self._hold_steps: NDArray[np.int_] | None = None  # (num_envs,)
        self._last: dict[str, NDArray] = {}

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetReturn:
        """Reset the environment and clear failure state."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._hold_steps = np.zeros((int(self.env.num_envs),), dtype=np.int32)
        self._last = {k: np.asarray(obs[k]).copy() for k in self.keys if k in obs}
        return obs, info

    def step(self, actions: Any) -> StepReturn:
        """Step the environment, occasionally dropping VIO-like observation updates."""
        obs, reward, terminated, truncated, info = self.env.step(actions)

        if self._hold_steps is None:
            self._hold_steps = np.zeros((int(self.env.num_envs),), dtype=np.int32)

        done = np.asarray(terminated) | np.asarray(truncated)
        if done.any():
            self._hold_steps[done] = 0

        # Start new failures.
        if float(self.cfg.failure_prob) > 0.0 and int(self.cfg.max_hold_steps) > 0:
            starts = (self._hold_steps == 0) & (
                self._rng.random(size=self._hold_steps.shape) < float(self.cfg.failure_prob)
            )
            if starts.any():
                self._hold_steps[starts] = self._rng.integers(
                    1, int(self.cfg.max_hold_steps) + 1, size=int(starts.sum()), dtype=np.int32
                )

        # Apply failure to observations.
        if self._hold_steps is not None and (self._hold_steps > 0).any():
            failing = self._hold_steps > 0
            out = dict(obs)
            for k in self.keys:
                if k not in out:
                    continue
                arr = np.asarray(out[k])
                if self.cfg.mode == "zero":
                    arr = arr.copy()
                    arr[failing, ...] = 0.0
                else:
                    # hold last value
                    if k in self._last:
                        arr = arr.copy()
                        arr[failing, ...] = self._last[k][failing, ...]
                out[k] = arr
            obs = out

            self._hold_steps[failing] -= 1

        # Update last observations for non-failing envs.
        if self._hold_steps is not None:
            ok = self._hold_steps == 0
            for k in self.keys:
                if k in obs:
                    arr = np.asarray(obs[k])
                    if k not in self._last:
                        self._last[k] = arr.copy()
                    else:
                        self._last[k][ok, ...] = arr[ok, ...]

        return obs, reward, terminated, truncated, info
