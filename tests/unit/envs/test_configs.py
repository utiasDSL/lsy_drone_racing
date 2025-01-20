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

    gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        random_resets=config.env.random_resets,
        seed=config.env.seed,
    )
