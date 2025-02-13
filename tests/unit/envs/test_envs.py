import warnings
from pathlib import Path

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import JaxToNumpy

from lsy_drone_racing.utils import load_config


@pytest.mark.unit
@pytest.mark.parametrize("action_space", ["state", "attitude"])
def test_passive_checker_wrapper_warnings(action_space: str):
    """Check passive env checker wrapper warnings.

    We disable the passive env checker by default. This test ensures that unexpected warnings are
    still seen, and even raises them to an exception.
    """
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make(
            "DroneRacing-v0",
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            action_space=action_space,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            random_resets=config.env.random_resets,
            seed=config.env.seed,
            disable_env_checker=False,
        )
        check_env(JaxToNumpy(env))
