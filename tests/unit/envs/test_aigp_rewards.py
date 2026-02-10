from pathlib import Path

import gymnasium
import numpy as np
import pytest
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config


@pytest.mark.unit
@pytest.mark.parametrize(
    "preset",
    ["swift", "grandprix", "grandprix_lite", "minimal", "minimal_curiosity"],
)
def test_aigp_reward_is_finite(preset: str):
    """Smoke-test AIGP reward computation for all main presets."""
    config = load_config(Path(__file__).parents[3] / "config/aigp_stage0_single_gate.toml")
    env = gymnasium.make(
        "AIGPDroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
        reward_config=preset,
    )
    env = JaxToNumpy(env.unwrapped)

    env.reset()
    for _ in range(5):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward), f"Reward must be finite, got {reward} for preset={preset}"

    components = env.unwrapped._last_reward_components
    assert components is not None
    for key in [
        "gate_passage",
        "progress",
        "progress_velocity",
        "speed_bonus",
        "boundary",
        "crash",
        "time_penalty",
        "completion_bonus",
    ]:
        assert key in components

    env.close()


@pytest.mark.unit
def test_vec_aigp_reward_shape_and_finite():
    """Vector env reward must be finite and have shape (num_envs,)."""
    config = load_config(Path(__file__).parents[3] / "config/aigp_stage0_single_gate.toml")
    env = gymnasium.make_vec(
        "AIGPDroneRacing-v0",
        num_envs=3,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
        reward_config="swift",
    )
    env = gymnasium.wrappers.vector.JaxToNumpy(env)

    env.reset()
    for _ in range(3):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert reward.shape == (3,)
        assert np.isfinite(reward).all()

    env.close()

