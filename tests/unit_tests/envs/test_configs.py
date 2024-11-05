from pathlib import Path

import gymnasium
import pytest

from lsy_drone_racing.utils import load_config


@pytest.mark.parametrize(
    "config_file", ["level0.toml", "level1.toml", "level2.toml", "level3.toml"]
)
@pytest.mark.unit
def test_config_load_and_env_creation(config_file: str):
    """Test if config files can be loaded and used to create a functioning environment."""
    # Load the config
    config_path = Path(__file__).parents[3] / "config" / config_file
    config = load_config(config_path)

    env = gymnasium.make(config.env.id, config=config)
    env.reset()
    env.step(env.action_space.sample())
