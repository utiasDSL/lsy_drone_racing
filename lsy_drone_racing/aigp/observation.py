"""Observation adapters for AIGP training/evaluation.

This module provides a lightweight observation-mode switch for curriculum training:

- ``privileged``: keep the environment's native observation dictionary.
- ``competition_proxy``: remove privileged absolute state fields and expose only
  telemetry/proxy perception style features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

try:
    from gymnasium.vector import VectorEnv, VectorWrapper
except ImportError:  # gymnasium<1.0
    from gymnasium.vector import VectorEnv
    from gymnasium.vector import VectorEnvWrapper as VectorWrapper
from numpy.typing import NDArray

Obs = dict[str, NDArray[np.generic]]
Info = dict[str, Any]
ResetReturn = tuple[Obs, Info]
StepReturn = tuple[Obs, NDArray[np.floating], NDArray[np.bool_], NDArray[np.bool_], Info]


def _as_numpy_obs(obs: dict[str, Any]) -> Obs:
    out: Obs = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if not arr.flags.writeable:
            arr = arr.copy()
        out[str(key)] = arr
    return out


def build_competition_proxy_observation_space(source: spaces.Space) -> spaces.Dict:
    """Build the competition-proxy observation space from a source dict-space."""
    if not isinstance(source, spaces.Dict):
        raise TypeError("competition proxy observation requires a Dict observation space")

    expected = (
        "pos",
        "quat",
        "vel",
        "ang_vel",
        "target_gate",
        "gates_pos",
        "gates_visited",
        "obstacles_pos",
        "obstacles_visited",
    )
    missing = [k for k in expected if k not in source.spaces]
    if missing:
        raise KeyError(f"missing required observation keys for competition proxy: {missing!r}")

    gates_pos_space = source.spaces["gates_pos"]
    obstacles_pos_space = source.spaces["obstacles_pos"]
    if not isinstance(gates_pos_space, spaces.Box) or not isinstance(
        obstacles_pos_space, spaces.Box
    ):
        raise TypeError("gates_pos/obstacles_pos must be Box spaces")

    def _float_box(shape: tuple[int, ...]) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    return spaces.Dict(
        {
            "quat": _float_box((4,)),
            "vel": _float_box((3,)),
            "ang_vel": _float_box((3,)),
            "target_gate": source.spaces["target_gate"],
            "target_gate_rel_pos": _float_box((3,)),
            "gates_rel_pos": _float_box(tuple(gates_pos_space.shape)),
            "gates_visited": source.spaces["gates_visited"],
            "obstacles_rel_pos": _float_box(tuple(obstacles_pos_space.shape)),
            "obstacles_visited": source.spaces["obstacles_visited"],
        }
    )


def project_competition_proxy_observation(obs: dict[str, Any]) -> Obs:
    """Project privileged observations into competition-proxy observations."""
    obs_np = _as_numpy_obs(obs)
    pos = np.asarray(obs_np["pos"], dtype=np.float32)
    quat = np.asarray(obs_np["quat"], dtype=np.float32)
    vel = np.asarray(obs_np["vel"], dtype=np.float32)
    ang_vel = np.asarray(obs_np["ang_vel"], dtype=np.float32)
    target_gate = np.asarray(obs_np["target_gate"], dtype=np.int32)
    gates_pos = np.asarray(obs_np["gates_pos"], dtype=np.float32)
    obstacles_pos = np.asarray(obs_np["obstacles_pos"], dtype=np.float32)
    gates_visited = np.asarray(obs_np["gates_visited"], dtype=bool)
    obstacles_visited = np.asarray(obs_np["obstacles_visited"], dtype=bool)

    # Relative geometry removes absolute world-frame privileged position from the policy input.
    gates_rel_pos = gates_pos - pos[..., None, :]
    obstacles_rel_pos = obstacles_pos - pos[..., None, :]

    n_gates = int(gates_pos.shape[-2]) if gates_pos.ndim >= 2 else 1
    target_idx = np.clip(target_gate, 0, max(0, n_gates - 1))
    gather_idx = target_idx[..., None, None]
    target_gate_rel_pos = np.take_along_axis(gates_rel_pos, gather_idx, axis=-2)[..., 0, :]
    target_gate_rel_pos = np.asarray(target_gate_rel_pos, dtype=np.float32)
    target_gate_rel_pos[target_gate < 0] = 0.0

    return {
        "quat": quat,
        "vel": vel,
        "ang_vel": ang_vel,
        "target_gate": target_gate,
        "target_gate_rel_pos": target_gate_rel_pos,
        "gates_rel_pos": np.asarray(gates_rel_pos, dtype=np.float32),
        "gates_visited": gates_visited,
        "obstacles_rel_pos": np.asarray(obstacles_rel_pos, dtype=np.float32),
        "obstacles_visited": obstacles_visited,
    }


class CompetitionProxyObsWrapper(VectorWrapper):
    """Vector-env wrapper that emits competition-proxy observations."""

    def __init__(self, env: VectorEnv):
        """Initialize wrapper and replace the exposed observation space."""
        super().__init__(env)
        source_space = getattr(env, "single_observation_space", env.observation_space)
        proxy_space = build_competition_proxy_observation_space(source_space)
        self.single_observation_space = proxy_space
        self.observation_space = proxy_space

    def _transform_obs(self, obs: dict[str, Any]) -> Obs:
        return project_competition_proxy_observation(obs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetReturn:
        """Reset environment and project observations into competition-proxy space."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform_obs(obs), info

    def _reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
        mask: Any | None = None,
    ) -> ResetReturn:
        if hasattr(self.env, "_reset"):
            obs, info = self.env._reset(seed=seed, options=options, mask=mask)  # type: ignore[attr-defined]
        else:
            obs, info = self.env.reset(seed=seed, options=options)
        return self._transform_obs(obs), info

    def step(self, actions: Any) -> StepReturn:
        """Step environment and project observations into competition-proxy space."""
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._transform_obs(obs), reward, terminated, truncated, info
