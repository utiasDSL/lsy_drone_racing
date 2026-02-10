"""Stable-Baselines3 compatibility for Gymnasium vector environments.

`lsy_drone_racing` exposes fast simulators as Gymnasium `VectorEnv`s (e.g. `VecDroneRaceEnv`).
Stable-Baselines3 (SB3) uses its own `VecEnv` abstraction, so this module provides a small adapter
to bridge the two.

This module is import-safe without SB3 installed. SB3 is only imported when you call
`make_sb3_vec_env(...)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

Obs = dict[str, NDArray[np.generic]]


def _as_numpy_obs(obs: dict[str, Any]) -> Obs:
    out: Obs = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        # JAX-to-NumPy conversions can yield read-only views. SB3/PyTorch expects writable buffers
        # (and we also mutate obs in-place when applying partial resets).
        if not arr.flags.writeable:
            arr = arr.copy()
        out[k] = arr
    return out


def _split_info(info: dict[str, Any], *, num_envs: int) -> list[dict[str, Any]]:
    """Convert a dict-of-batched-arrays info into SB3's list-of-dicts format."""
    infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    for k, v in info.items():
        arr = np.asarray(v)
        if arr.shape[:1] == (num_envs,):
            for i in range(num_envs):
                infos[i][k] = arr[i]
        else:
            # Broadcast scalars / non-batched values to all env lanes.
            for i in range(num_envs):
                infos[i][k] = v
    return infos


def _squeeze_single_drone(batch: dict[str, Any], *, num_envs: int) -> dict[str, Any]:
    """Convert `(n_envs, 1, ...)` arrays into `(n_envs, ...)` for single-drone envs."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        arr = np.asarray(v)
        if arr.ndim >= 2 and arr.shape[0] == num_envs and arr.shape[1] == 1:
            out[k] = arr[:, 0]
        else:
            out[k] = arr
    return out


def _partial_reset(env: "VectorEnv", *, done: NDArray[np.bool_]) -> tuple[Obs, dict[str, Any]]:
    """Reset only `done` lanes, using `RaceCoreEnv._reset(mask=...)` when available."""
    num_envs = int(env.num_envs)
    if done.all() or not hasattr(env, "_reset"):
        obs, info = env.reset()
        return _as_numpy_obs(obs), _squeeze_single_drone(info, num_envs=num_envs)

    obs_jax, info_jax = env._reset(mask=jp.asarray(done))  # type: ignore[attr-defined]
    obs = _as_numpy_obs(_squeeze_single_drone(obs_jax, num_envs=num_envs))
    info = _squeeze_single_drone(info_jax, num_envs=num_envs)
    return obs, info


def make_sb3_vec_env(env: "VectorEnv") -> Any:  # noqa: ANN401
    """Wrap a Gymnasium `VectorEnv` so it can be used with Stable-Baselines3.

    Args:
        env: Gymnasium vector environment. For best results, this should be a single-agent env and
            expose `single_action_space` / `single_observation_space`.

    Returns:
        An instance of `stable_baselines3.common.vec_env.VecEnv`.
    """
    try:
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Stable-Baselines3 is required for make_sb3_vec_env(). "
            "Install it (and torch) to use this training adapter."
        ) from exc

    try:
        from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
    except Exception:  # pragma: no cover - version differences
        VecEnvObs = Any  # type: ignore[misc,assignment]
        VecEnvStepReturn = Any  # type: ignore[misc,assignment]

    class _SB3Adapter(VecEnv):  # type: ignore[misc]
        """SB3 VecEnv adapter around a Gymnasium VectorEnv."""

        def __init__(self, wrapped: VectorEnv):
            self.wrapped = wrapped
            self._pending_actions: Any | None = None

            if hasattr(self.wrapped, "autoreset"):
                self.wrapped.autoreset = False  # type: ignore[attr-defined]

            num_envs = int(self.wrapped.num_envs)
            obs_space = getattr(
                self.wrapped, "single_observation_space", self.wrapped.observation_space
            )
            act_space = getattr(self.wrapped, "single_action_space", self.wrapped.action_space)
            super().__init__(num_envs=num_envs, observation_space=obs_space, action_space=act_space)

        def reset(self) -> VecEnvObs:
            obs, info = self.wrapped.reset()
            obs_np = _as_numpy_obs(obs)
            info_np = _squeeze_single_drone(info, num_envs=int(self.wrapped.num_envs))
            self.reset_infos = _split_info(info_np, num_envs=int(self.wrapped.num_envs))  # type: ignore[attr-defined]
            return obs_np

        def step_async(self, actions: Any) -> None:
            self._pending_actions = actions

        def step_wait(self) -> VecEnvStepReturn:
            assert self._pending_actions is not None
            obs, rewards, terminated, truncated, info = self.wrapped.step(self._pending_actions)
            self._pending_actions = None

            obs_np = _as_numpy_obs(obs)
            rewards_np = np.asarray(rewards, dtype=np.float32)
            term_np = np.asarray(terminated, dtype=bool)
            trunc_np = np.asarray(truncated, dtype=bool)
            dones = term_np | trunc_np

            info_np = _squeeze_single_drone(info, num_envs=int(self.wrapped.num_envs))
            infos = _split_info(info_np, num_envs=int(self.wrapped.num_envs))
            for i in range(int(self.wrapped.num_envs)):
                infos[i]["TimeLimit.truncated"] = bool(trunc_np[i] and not term_np[i])

            if dones.any():
                done_idx = np.flatnonzero(dones)
                for i in done_idx:
                    infos[int(i)]["terminal_observation"] = {
                        k: np.asarray(v[int(i)]).copy() for k, v in obs_np.items()
                    }

                reset_obs, _reset_info = _partial_reset(self.wrapped, done=dones.astype(np.bool_))
                for k, v in obs_np.items():
                    v_reset = reset_obs[k]
                    v = np.asarray(v)
                    v[dones, ...] = v_reset[dones, ...]
                    obs_np[k] = v

            return obs_np, rewards_np, dones, infos

        def close(self) -> None:
            self.wrapped.close()

        def render(self, mode: str = "human") -> Any:  # noqa: ANN401
            return self.wrapped.render()

        def get_images(self) -> list[Any]:  # noqa: ANN401
            return []

        def seed(self, seed: int | None = None) -> list[int | None]:
            obs, info = self.wrapped.reset(seed=seed)
            _ = obs, info
            return [seed] * int(self.wrapped.num_envs)

        def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:  # noqa: ANN401
            if indices is None:
                indices = range(int(self.wrapped.num_envs))
            val = getattr(self.wrapped, attr_name)
            return [val for _ in indices]

        def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:  # noqa: ANN401
            if indices is None:
                setattr(self.wrapped, attr_name, value)
                return
            setattr(self.wrapped, attr_name, value)

        def env_method(
            self,
            method_name: str,
            *method_args: Any,
            indices: Any = None,
            **method_kwargs: Any,
        ) -> list[Any]:  # noqa: ANN401
            if indices is None:
                indices = range(int(self.wrapped.num_envs))
            method = getattr(self.wrapped, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in indices]

        def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:  # noqa: ANN401
            if indices is None:
                indices = range(int(self.wrapped.num_envs))
            _ = wrapper_class
            return [False for _ in indices]

    return _SB3Adapter(env)
