"""AIGP evaluation helpers.

These utilities run rollouts on Gymnasium *vector* environments and aggregate the
`RaceCoreEnv.info()` metrics into a compact :class:`~lsy_drone_racing.aigp.curriculum.EvalSummary`.

They are intentionally RL-library agnostic: you provide a policy function.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Protocol

import jax.numpy as jp
import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.aigp.curriculum import EvalSummary

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

logger = logging.getLogger(__name__)

Obs = dict[str, NDArray[np.generic]]
Actions = NDArray[np.floating]


class PolicyFn(Protocol):
    """Protocol for policy functions used by the evaluator."""

    def __call__(self, obs: Obs) -> Actions:  # pragma: no cover - structural typing
        """Compute a batch of actions from a batch of observations."""


def make_predict_policy(model: Any, *, deterministic: bool = True) -> PolicyFn:
    """Wrap a model with a SB3-like `predict()` method into a `PolicyFn`.

    This is intentionally duck-typed to avoid a hard dependency on Stable-Baselines3.
    """

    def _policy(obs: Obs) -> Actions:
        action, _state = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return _policy


def _as_numpy_obs(obs: dict[str, Any]) -> Obs:
    """Convert (possibly JAX) batched observations into NumPy arrays."""
    out: Obs = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        # JAX-to-NumPy conversions can yield read-only views; we patch done lanes in-place.
        if not arr.flags.writeable:
            arr = arr.copy()
        out[k] = arr
    return out


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


def _partial_reset(
    env: "VectorEnv", *, done: NDArray[np.bool_]
) -> tuple[Obs, dict[str, Any], bool]:
    """Reset only the environments indicated by `done`.

    Notes:
        This relies on the underlying env exposing a `._reset(mask=...)` method (as
        :class:`~lsy_drone_racing.envs.race_core.RaceCoreEnv` does). If it's missing, this falls
        back to resetting all environments.

    Returns:
        `(obs, info, reset_all)` where `reset_all=True` indicates a full reset fallback.
    """
    num_envs = int(env.num_envs)
    if done.all():
        obs, info = env.reset()
        return _as_numpy_obs(obs), _squeeze_single_drone(info, num_envs=num_envs), True

    if not hasattr(env, "_reset"):
        obs, info = env.reset()
        return _as_numpy_obs(obs), _squeeze_single_drone(info, num_envs=num_envs), True

    # `RaceCoreEnv._reset()` expects a JAX array mask in most code paths.
    obs_jax, info_jax = env._reset(mask=jp.asarray(done))  # type: ignore[attr-defined]
    obs = _as_numpy_obs(_squeeze_single_drone(obs_jax, num_envs=num_envs))
    info = _squeeze_single_drone(info_jax, num_envs=num_envs)
    return obs, info, False


def evaluate_vec_env(
    env: "VectorEnv",
    policy: PolicyFn,
    *,
    n_episodes: int,
    max_episode_steps: int | None = None,
) -> EvalSummary:
    """Evaluate a policy on a vector environment.

    Args:
        env: A Gymnasium vector environment.
        policy: Policy function mapping batched observations to batched actions.
        n_episodes: Total number of episodes to collect across all vector lanes.
        max_episode_steps: Optional hard cap; if set, episodes are force-terminated at this length
            for evaluation purposes.

    Returns:
        Aggregated evaluation summary.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")

    # Disable env-side autoreset if supported; we do explicit partial resets so we can compute
    # clean episode metrics.
    if hasattr(env, "autoreset"):
        env.autoreset = False  # type: ignore[attr-defined]

    num_envs = int(env.num_envs)
    obs, _info = env.reset()
    obs_np = _as_numpy_obs(obs)

    ep_returns = np.zeros((num_envs,), dtype=np.float64)
    ep_lengths = np.zeros((num_envs,), dtype=np.int32)

    successes: list[bool] = []
    completion_fracs: list[float] = []
    lap_times_s: list[float] = []

    while len(successes) < n_episodes:
        actions = policy(obs_np)
        obs2, reward, terminated, truncated, info = env.step(actions)

        reward_np = np.asarray(reward, dtype=np.float64)
        terminated_np = np.asarray(terminated, dtype=bool)
        truncated_np = np.asarray(truncated, dtype=bool)
        done = terminated_np | truncated_np

        ep_returns += reward_np
        ep_lengths += 1

        info_np = {k: np.asarray(v) for k, v in info.items()}
        if max_episode_steps is not None:
            forced = ep_lengths >= int(max_episode_steps)
            if forced.any():
                done = done | forced
                truncated_np = truncated_np | forced

        if done.any():
            # Prefer `success`/`completion_fraction` from `RaceCoreEnv.info()`, but fall back to a
            # conservative default if keys are missing.
            sr = info_np.get("success")
            if sr is None:
                success_arr = terminated_np & ~truncated_np
            else:
                success_arr = np.asarray(sr, dtype=bool)

            completion = info_np.get("completion_fraction")
            if completion is None:
                completion_arr = np.zeros_like(reward_np, dtype=np.float32)
            else:
                completion_arr = np.asarray(completion, dtype=np.float32)

            lap_time_s = info_np.get("lap_time_s")
            lap_time_arr = (
                np.asarray(lap_time_s, dtype=np.float32)
                if lap_time_s is not None
                else ep_lengths.astype(np.float32)
            )

            for i in np.flatnonzero(done):
                successes.append(bool(success_arr[i]))
                completion_fracs.append(float(completion_arr[i]))
                lap_times_s.append(float(lap_time_arr[i]))
                if len(successes) >= n_episodes:
                    break

            ep_returns[done] = 0.0
            ep_lengths[done] = 0

            reset_obs, _reset_info, reset_all = _partial_reset(env, done=done.astype(np.bool_))
            obs2_np = _as_numpy_obs(obs2)
            replace_mask = np.ones_like(done, dtype=bool) if reset_all else done.astype(np.bool_)
            # Overwrite done lanes with freshly reset observations.
            for k, v in obs2_np.items():
                v_reset = reset_obs[k]
                v = np.asarray(v)
                v[replace_mask, ...] = v_reset[replace_mask, ...]
                obs2_np[k] = v
            obs_np = obs2_np
        else:
            obs_np = _as_numpy_obs(obs2)

    success_rate = float(np.mean(successes)) if successes else 0.0
    completion_mean = float(np.mean(completion_fracs)) if completion_fracs else 0.0
    completion_std = float(np.std(completion_fracs)) if completion_fracs else 0.0
    lap_time_s_median = None
    if lap_times_s and any(successes):
        lap_times_success = [
            t for t, s in zip(lap_times_s, successes, strict=True) if s
        ]
        lap_time_s_median = float(np.median(lap_times_success))

    return EvalSummary(
        n_episodes=int(n_episodes),
        success_rate=success_rate,
        completion_mean=completion_mean,
        completion_std=completion_std,
        lap_time_s_median=lap_time_s_median,
    )


def evaluate_sb3_vec_env(
    env: Any,  # noqa: ANN401
    policy: PolicyFn,
    *,
    n_episodes: int,
    max_episode_steps: int | None = None,
) -> EvalSummary:
    """Evaluate a policy on a Stable-Baselines3-style VecEnv.

    Notes:
        This expects the standard SB3 vector API:
        - `obs = env.reset()`
        - `obs, reward, done, infos = env.step(actions)`
        where `infos` is a list of dicts.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")

    num_envs = int(env.num_envs)
    obs = env.reset()
    obs_np = _as_numpy_obs(obs if not isinstance(obs, tuple) else obs[0])

    ep_lengths = np.zeros((num_envs,), dtype=np.int32)
    successes: list[bool] = []
    completion_fracs: list[float] = []
    lap_times_s: list[float] = []

    while len(successes) < n_episodes:
        actions = policy(obs_np)
        obs2, _reward, done, infos = env.step(actions)
        done_np = np.asarray(done, dtype=bool)
        ep_lengths += 1

        if done_np.any():
            done_idx = np.flatnonzero(done_np)
            for i in done_idx:
                info_i = infos[int(i)] if isinstance(infos, list) else {}

                success_raw = info_i.get("success", False)
                success_arr = np.asarray(success_raw).reshape(-1)
                success = bool(success_arr[0]) if success_arr.size else bool(success_raw)

                completion_raw = info_i.get("completion_fraction", 0.0)
                completion_arr = np.asarray(completion_raw).reshape(-1)
                completion = (
                    float(completion_arr[0]) if completion_arr.size else float(completion_raw)
                )

                lap_raw = info_i.get("lap_time_s")
                if lap_raw is None:
                    lap_time_s = float(ep_lengths[int(i)])
                else:
                    lap_arr = np.asarray(lap_raw).reshape(-1)
                    lap_time_s = float(lap_arr[0]) if lap_arr.size else float(lap_raw)

                successes.append(success)
                completion_fracs.append(completion)
                lap_times_s.append(lap_time_s)
                ep_lengths[int(i)] = 0

                if len(successes) >= n_episodes:
                    break

        # `max_episode_steps` is a compatibility parameter to mirror `evaluate_vec_env`.
        # SB3 VecEnv episodes should already be capped by env-side max_episode_steps.
        _ = max_episode_steps
        obs_np = _as_numpy_obs(obs2 if not isinstance(obs2, tuple) else obs2[0])

    success_rate = float(np.mean(successes)) if successes else 0.0
    completion_mean = float(np.mean(completion_fracs)) if completion_fracs else 0.0
    completion_std = float(np.std(completion_fracs)) if completion_fracs else 0.0
    lap_time_s_median = None
    if lap_times_s and any(successes):
        lap_times_success = [
            t for t, s in zip(lap_times_s, successes, strict=True) if s
        ]
        lap_time_s_median = float(np.median(lap_times_success))

    return EvalSummary(
        n_episodes=int(n_episodes),
        success_rate=success_rate,
        completion_mean=completion_mean,
        completion_std=completion_std,
        lap_time_s_median=lap_time_s_median,
    )


def aggregate_track_eval_summaries(
    track_summaries: list[tuple[int, str, EvalSummary]],
    *,
    bottomk_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build per-track + aggregate metrics for stratified evaluation."""
    rows = sorted(track_summaries, key=lambda x: int(x[0]))
    tracks_payload: list[dict[str, Any]] = []
    success_rates: list[float] = []
    tracks_covered = 0

    for track_id, track_name, summary in rows:
        sr = float(summary.success_rate)
        tracks_payload.append(
            {
                "track_id": int(track_id),
                "track_name": str(track_name),
                "n_episodes": int(summary.n_episodes),
                "success_rate": sr,
                "completion_mean": float(summary.completion_mean),
                "completion_std": float(summary.completion_std),
                "lap_time_s_median": (
                    None
                    if summary.lap_time_s_median is None
                    else float(summary.lap_time_s_median)
                ),
            }
        )
        if int(summary.n_episodes) > 0:
            tracks_covered += 1
            success_rates.append(sr)

    success_rate_min = float(np.min(success_rates)) if success_rates else None
    success_rate_bottom20_mean = None
    if success_rates:
        frac = float(bottomk_fraction)
        frac = min(max(frac, 0.0), 1.0)
        k = max(1, int(math.ceil(frac * len(success_rates))))
        success_rate_bottom20_mean = float(np.mean(sorted(success_rates)[:k]))

    return {
        "tracks": tracks_payload,
        "_aggregate": {
            "tracks_total": int(len(rows)),
            "tracks_covered": int(tracks_covered),
            "success_rate_min": success_rate_min,
            "success_rate_bottom20_mean": success_rate_bottom20_mean,
        },
    }
