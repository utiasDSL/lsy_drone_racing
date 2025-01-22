from pathlib import Path

import gymnasium
import pytest

import lsy_drone_racing  # noqa: F401, environment registrations
from lsy_drone_racing.utils import load_config

CONFIG_FILES = {
    "DroneRacing-v0": ["level0.toml", "level1.toml", "level2.toml", "level3.toml"],
    "MultiDroneRacing-v0": ["multi_level0.toml", "multi_level3.toml"],
}
ENV_IDS = ["DroneRacing-v0", "MultiDroneRacing-v0"]


@pytest.mark.parametrize("physics", ["analytical", "sys_id"])
@pytest.mark.parametrize(
    ("env_id", "config_file"),
    [(env_id, config_file) for env_id in ENV_IDS for config_file in CONFIG_FILES[env_id]],
)
@pytest.mark.integration
def test_single_drone_envs(env_id: str, config_file: str, physics: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"

    kwargs = {
        "freq": config.env.freq,
        "sim_config": config.sim,
        "sensor_range": config.env.sensor_range,
        "track": config.env.track,
        "disturbances": config.env.get("disturbances"),
        "randomizations": config.env.get("randomizations"),
        "random_resets": config.env.random_resets,
        "seed": config.env.seed,
    }
    if "n_drones" in config.env:
        kwargs["n_drones"] = config.env.n_drones

    env = gymnasium.make(env_id, **kwargs)
    env.reset()
    for _ in range(100):
        _, _, _, _, _ = env.step(env.action_space.sample())
    env.close()

    kwargs["num_envs"] = 2
    env = gymnasium.make_vec(env_id, **kwargs)
    env.reset()
    for _ in range(100):
        _, _, _, _, _ = env.step(env.action_space.sample())
    env.close()
