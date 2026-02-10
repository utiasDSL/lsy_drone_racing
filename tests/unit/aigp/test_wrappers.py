import numpy as np
import pytest
from gymnasium import Env, spaces
from gymnasium.vector import SyncVectorEnv
from numpy.typing import NDArray

from lsy_drone_racing.aigp.wrappers import (
    ActionLatencyWrapper,
    ImuBiasNoiseWrapper,
    ImuNoiseConfig,
    VioFailureConfig,
    VioFailureWrapper,
)


class _ToyObsEnv(Env):
    def __init__(self, *, horizon: int = 100):
        super().__init__()
        self._t = 0
        self._horizon = int(horizon)
        self._last_action = 0.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "last_action": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32),
                "quat": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4), dtype=np.float32),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3), dtype=np.float32),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, NDArray[np.floating]], dict]:
        super().reset(seed=seed)
        self._t = 0
        self._last_action = 0.0
        return self._obs(), {}

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, bool, dict]:
        act = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._t += 1
        self._last_action = act

        terminated = self._t >= self._horizon
        truncated = False
        reward = 0.0
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> dict[str, NDArray[np.floating]]:
        last_action = np.array([self._last_action], dtype=np.float32)
        pos = np.array([[float(self._t), 0.0, 0.0]], dtype=np.float32)
        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[self._last_action, 0.0, 0.0]], dtype=np.float32)
        ang_vel = np.array([[0.0, self._last_action, 0.0]], dtype=np.float32)
        return {
            "last_action": last_action,
            "pos": pos,
            "quat": quat,
            "vel": vel,
            "ang_vel": ang_vel,
        }


@pytest.mark.unit
def test_action_latency_wrapper_fixed_latency():
    env = SyncVectorEnv([lambda: _ToyObsEnv() for _ in range(2)])
    env = ActionLatencyWrapper(env, latency_steps=2, seed=0)
    obs, _ = env.reset()
    assert obs["last_action"].shape == (2, 1)

    obs, *_ = env.step(np.array([[1.0], [2.0]], dtype=np.float32))
    assert np.allclose(obs["last_action"].squeeze(-1), np.array([0.0, 0.0], dtype=np.float32))

    obs, *_ = env.step(np.array([[3.0], [4.0]], dtype=np.float32))
    assert np.allclose(obs["last_action"].squeeze(-1), np.array([0.0, 0.0], dtype=np.float32))

    obs, *_ = env.step(np.array([[5.0], [6.0]], dtype=np.float32))
    assert np.allclose(obs["last_action"].squeeze(-1), np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.unit
def test_imu_bias_noise_wrapper_adds_bias_and_drifts():
    env = SyncVectorEnv([lambda: _ToyObsEnv() for _ in range(2)])
    cfg = ImuNoiseConfig(
        vel_bias_std=1.0,
        ang_vel_bias_std=1.0,
        vel_noise_std=0.0,
        ang_vel_noise_std=0.0,
        bias_drift_std=0.1,
    )
    env = ImuBiasNoiseWrapper(env, cfg, seed=0)

    obs0, _ = env.reset()
    vel0 = obs0["vel"].copy()
    assert float(np.sum(np.abs(vel0))) > 0.0

    obs1, *_ = env.step(np.zeros((2, 1), dtype=np.float32))
    vel1 = obs1["vel"].copy()
    assert not np.allclose(vel0, vel1)


@pytest.mark.unit
def test_vio_failure_wrapper_hold_mode_holds_last_value():
    env = SyncVectorEnv([lambda: _ToyObsEnv() for _ in range(2)])
    cfg = VioFailureConfig(failure_prob=1.0, max_hold_steps=1, mode="hold")
    env = VioFailureWrapper(env, cfg, keys=("pos",), seed=0)

    obs0, _ = env.reset()
    assert np.allclose(obs0["pos"][..., 0], 0.0)

    obs1, *_ = env.step(np.zeros((2, 1), dtype=np.float32))
    # Underlying env moved to t=1, but dropout holds t=0.
    assert np.allclose(obs1["pos"][..., 0], 0.0)
