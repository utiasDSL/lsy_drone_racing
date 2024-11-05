from pathlib import Path

import gymnasium
import pytest
import toml
from munch import munchify

from lsy_drone_racing.sim.physics import PhysicsMode

CONFIG_FILES = ["level0.toml", "level1.toml", "level2.toml", "level3.toml"]
ENVIRONMENTS = ["DroneRacing-v0", "DroneRacingThrust-v0"]


@pytest.mark.parametrize("physics", PhysicsMode)
@pytest.mark.parametrize("config_file", CONFIG_FILES)
@pytest.mark.parametrize("env_id", ENVIRONMENTS)
@pytest.mark.integration
def test_sim(physics: PhysicsMode, config_file: str, env_id: str):
    """Test the simulation environments with different physics modes and config files."""
    with open(Path(__file__).parents[2] / "config" / config_file) as f:
        config = munchify(toml.load(f))
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"
    config.env.id = env_id  # override environment ID

    if physics == PhysicsMode.SYS_ID and env_id == "DroneRacing-v0":
        pytest.skip("System identification model not supported for full state control interface")

    env = gymnasium.make(env_id, config=config)
    env.reset()

    for _ in range(10):  # Run for 10 steps or until episode ends
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    env.close()
