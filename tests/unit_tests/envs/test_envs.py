import warnings
from pathlib import Path

import gymnasium
import pytest
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.utils import load_config


@pytest.mark.parametrize("env", ["DroneRacing-v0", "DroneRacingThrust-v0"])
@pytest.mark.unit
def test_passive_checker_wrapper_warnings(env: str):
    """Check passive env checker wrapper warnings.

    We disable the passive env checker by default. This test ensures that unexpected warnings are
    still seen, and even raises them to an exception.
    """
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    config.sim.physics = PhysicsMode.DEFAULT
    with warnings.catch_warnings(record=True) as w:
        env = gymnasium.make(env, config=config, disable_env_checker=False)
        env_reset_passive_checker(env)
        env_step_passive_checker(env, env.action_space.sample())
        # Filter out any warnings about 2D Box observation spaces.
        w = list(filter(lambda i: "neither an image, nor a 1D vector" not in i.message.args[0], w))
        assert len(w) == 0, f"No warnings should be raised, got: {[i.message.args[0] for i in w]}"
