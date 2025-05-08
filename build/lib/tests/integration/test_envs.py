from pathlib import Path

import gymnasium
import jax
import pytest

import lsy_drone_racing  # noqa: F401, environment registrations
from lsy_drone_racing.utils import load_config

CONFIG_FILES = {
    "DroneRacing-v0": ["level0.toml", "level1.toml", "level2.toml"],
    "MultiDroneRacing-v0": ["multi_level0.toml", "multi_level3.toml"],
}
DEVICES = ["cpu", "gpu"]


def available_backends() -> list[str]:
    """Return list of available JAX backends."""
    backends = []
    for backend in ["tpu", "gpu", "cpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


def skip_unavailable_device(device: str):
    if device not in available_backends():
        pytest.skip(f"{device} device not available")


@pytest.mark.parametrize("physics", ["analytical", "sys_id"])
@pytest.mark.parametrize("config_file", CONFIG_FILES["DroneRacing-v0"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.integration
def test_single_drone_envs(config_file: str, physics: str, device: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics  # override physics mode
    assert hasattr(config.env, "id"), "Environment ID is not set"
    skip_unavailable_device(device)

    kwargs = {
        "freq": config.env.freq,
        "sim_config": config.sim,
        "sensor_range": config.env.sensor_range,
        "track": config.env.track,
        "disturbances": config.env.get("disturbances"),
        "randomizations": config.env.get("randomizations"),
        "seed": config.env.seed,
        "device": device,
    }
    if "n_drones" in config.env:
        kwargs["n_drones"] = config.env.n_drones

    device = jax.devices(device)[0]
    env = gymnasium.make("DroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()

    kwargs["num_envs"] = 2
    env = gymnasium.make_vec("DroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()


@pytest.mark.parametrize("physics", ["analytical", "sys_id"])
@pytest.mark.parametrize("config_file", CONFIG_FILES["MultiDroneRacing-v0"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.integration
def test_multi_drone_envs(config_file: str, physics: str, device: str):
    """Test the simulation environments with different physics modes and config files."""
    config = load_config(Path(__file__).parents[2] / "config" / config_file)
    assert hasattr(config.sim, "physics"), "Physics mode is not set"
    config.sim.physics = physics
    assert hasattr(config.env, "id"), "Environment ID is not set"
    skip_unavailable_device(device)

    kwargs = {
        "freq": config.env.kwargs[0]["freq"],
        "sim_config": config.sim,
        "sensor_range": config.env.kwargs[0]["sensor_range"],
        "track": config.env.track,
        "disturbances": config.env.get("disturbances"),
        "randomizations": config.env.get("randomizations"),
        "seed": config.env.seed,
        "device": device,
    }
    if "n_drones" in config.env:
        kwargs["n_drones"] = config.env.n_drones

    device = jax.devices(device)[0]
    env = gymnasium.make("MultiDroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()
    kwargs["num_envs"] = 2
    env = gymnasium.make_vec("MultiDroneRacing-v0", **kwargs)
    env.reset()
    for _ in range(2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["pos"].device == device, f"Expected device {device}, but got {obs['pos'].device}"
    env.close()
