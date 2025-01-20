from pathlib import Path

import gymnasium
import pytest

import lsy_drone_racing  # noqa: F401, environment registrations
from lsy_drone_racing.utils import load_config

CONFIG_FILES = ["level0.toml", "level1.toml", "level2.toml", "level3.toml"]
MULTI_CONFIG_FILES = ["multi_level0.toml", "multi_level3.toml"]


@pytest.mark.parametrize("physics", ["analytical", "sys_id"])
@pytest.mark.parametrize("config_file", CONFIG_FILES)
@pytest.mark.integration
def test_envs(physics: str, config_file: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"
    config.env.id = "DroneRacing-v0"  # override environment ID

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
    )
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("physics", ["analytical", "sys_id"])
@pytest.mark.parametrize("config_file", MULTI_CONFIG_FILES)
@pytest.mark.integration
def test_vec_envs(physics: str, config_file: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"
    config.env.id = "MultiDroneRacing-v0"  # override environment ID

    env = gymnasium.make_vec(
        "MultiDroneRacing-v0",
        num_envs=2,
        n_drones=config.env.n_drones,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        random_resets=config.env.random_resets,
        seed=config.env.seed,
    )
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
