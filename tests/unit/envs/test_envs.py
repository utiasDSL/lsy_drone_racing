import warnings
from pathlib import Path

import gymnasium
import pytest
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from lsy_drone_racing.utils import load_config


@pytest.mark.unit
@pytest.mark.parametrize("action_space", ["state", "attitude"])
def test_passive_checker_wrapper_warnings(action_space: str):
    """Check passive env checker wrapper warnings.

    We disable the passive env checker by default. This test ensures that unexpected warnings are
    still seen, and even raises them to an exception.
    """
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True) as w:
        env = gymnasium.make(
            "DroneRacing-v0",
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            random_resets=config.env.random_resets,
            seed=config.env.seed,
            disable_env_checker=False,
        )
        env_reset_passive_checker(env)
        env_step_passive_checker(env, env.action_space.sample())
        # Filter out any warnings about 2D Box observation spaces.
        w = list(filter(lambda i: "neither an image, nor a 1D vector" not in i.message.args[0], w))
        assert len(w) == 0, f"No warnings should be raised, got: {[i.message.args[0] for i in w]}"


@pytest.mark.unit
@pytest.mark.parametrize("action_space", ["state", "attitude"])
def test_vector_passive_checker_wrapper_warnings(action_space: str):
    """Check passive env checker wrapper warnings.

    We disable the passive env checker by default. This test ensures that unexpected warnings are
    still seen, and even raises them to an exception.
    """
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True) as w:
        env = gymnasium.make_vec(
            "DroneRacing-v0",
            num_envs=2,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            random_resets=config.env.random_resets,
            seed=config.env.seed,
        )
        env_reset_passive_checker(env)
        env_step_passive_checker(env, env.action_space.sample())
        # Filter out any warnings about 2D Box observation spaces.
        w = list(filter(lambda i: "neither an image, nor a 1D vector" not in i.message.args[0], w))
        assert len(w) == 0, f"No warnings should be raised, got: {[i.message.args[0] for i in w]}"
