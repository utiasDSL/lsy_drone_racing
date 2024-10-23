from pathlib import Path

import gymnasium
import pytest

from lsy_drone_racing.utils import load_config

env_ids = ["DroneRacing-v0", "DroneRacingThrust-v0"]


@pytest.mark.parametrize(
    "config_file", ["level0.toml", "level1.toml", "level2.toml", "level3.toml"]
)
@pytest.mark.parametrize("env_id", env_ids)
@pytest.mark.unit
def test_config_load_and_env_creation(config_file: str, env_id: str):
    """Test if config files can be loaded and used to create a functioning environment."""
    # Load the config
    config_path = Path(__file__).parents[3] / "config" / config_file
    config = load_config(config_path)

    # Check if the current env_id is valid. We are not using it, but it should be valid.
    assert config["env"]["id"] in env_ids, f"Env ID {config['env']['id']} is invalid"
    env = gymnasium.make(env_id, config=config)
    env.reset()
    env.step(env.action_space.sample())
