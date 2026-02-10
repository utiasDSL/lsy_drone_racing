import warnings
from pathlib import Path

import gymnasium
import jax.numpy as jp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config


@pytest.mark.unit
@pytest.mark.parametrize("control_mode", ["state", "attitude"])
def test_passive_checker_wrapper_warnings(control_mode: str):
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
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
            disable_env_checker=False,
        )
        check_env(JaxToNumpy(env.unwrapped))
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/multi_level0.toml")
    with warnings.catch_warnings(record=True):
        env = gymnasium.make(
            "MultiDroneRacing-v0",
            freq=config.env.kwargs[0]["freq"],
            sim_config=config.sim,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
            disable_env_checker=False,
        )
        check_env(JaxToNumpy(env.unwrapped))
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make_vec(
            "DroneRacing-v0",
            num_envs=2,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        env = gymnasium.wrappers.vector.JaxToNumpy(env)
        # Check vector env specific attributes
        assert hasattr(env, "num_envs")
        assert isinstance(env.num_envs, int)
        assert env.num_envs == 2
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        obs, info = env.reset()
        assert obs in env.observation_space
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
    env.close()

    config = load_config(Path(__file__).parents[3] / "config/multi_level0.toml")
    with warnings.catch_warnings(record=True):  # Catch unnecessary warnings from gymnasium
        env = gymnasium.make_vec(
            "MultiDroneRacing-v0",
            num_envs=3,
            freq=config.env.kwargs[0]["freq"],
            sim_config=config.sim,
            sensor_range=config.env.kwargs[0]["sensor_range"],
            control_mode=control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
        env = gymnasium.wrappers.vector.JaxToNumpy(env)
        # Check vector env specific attributes
        assert hasattr(env, "num_envs")
        assert isinstance(env.num_envs, int)
        assert env.num_envs == 3
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        obs, info = env.reset()
        assert obs in env.observation_space
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        assert reward.shape == (3, 2)  # 2 envs, 3 drones
        assert terminated.shape == (3, 2)  # 2 envs, 3 drones
        assert truncated.shape == (3, 2)  # 2 envs, 3 drones
    env.close()


@pytest.mark.unit
def test_level2_randomization():
    """Test that level2 config properly randomizes gate and obstacle positions."""
    config = load_config(Path(__file__).parents[3] / "config/level2.toml")
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=100.0,  # Large sensor range to immediately see all changes
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
    )
    obs, info = env.reset()
    gate_pos_1, obstacle_pos_1 = obs["gates_pos"], obs["obstacles_pos"]
    gate_quat_1 = obs["gates_quat"]

    obs, info = env.reset()
    gate_pos_2, obstacle_pos_2 = obs["gates_pos"], obs["obstacles_pos"]
    gate_quat_2 = obs["gates_quat"]
    env.close()

    assert not (gate_pos_1 == gate_pos_2).all(), "Gate positions unchanged after reset."
    assert not (obstacle_pos_1 == obstacle_pos_2).all(), "Obstacle positions unchanged after reset."
    assert not (gate_quat_1 == gate_quat_2).all(), "Gate rotations unchanged after reset."


@pytest.mark.unit
def test_contact_mask():
    """Test that collision mask is properly generated and not all zeros."""
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    mask = env.unwrapped.data.contact_masks
    assert not jp.all(mask == 0), "Contact mask is all zeros."


@pytest.mark.unit
def test_masked_reset_only_resets_masked_worlds():
    """Regression test: `RaceCoreEnv._reset(mask=...)` must not overwrite unmasked worlds."""
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    env = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=2,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
    )
    env.reset(seed=0)

    # Overwrite world 1 with sentinel values. A buggy masked reset would clobber all worlds.
    data = env.unwrapped.sim.data
    sentinel_pos = data.states.pos.at[1].set(
        jp.asarray([[9.0, 8.0, 7.0]], dtype=data.states.pos.dtype)
    )
    env.unwrapped.sim.data = data.replace(states=data.states.replace(pos=sentinel_pos))

    mjx_data = env.unwrapped.sim.mjx_data
    sentinel_mocap_pos = mjx_data.mocap_pos.at[1, 0].set(
        jp.asarray([6.0, 5.0, 4.0], dtype=mjx_data.mocap_pos.dtype)
    )
    env.unwrapped.sim.mjx_data = mjx_data.replace(mocap_pos=sentinel_mocap_pos)

    pos_before = np.asarray(env.unwrapped.sim.data.states.pos[1])
    mocap_before = np.asarray(env.unwrapped.sim.mjx_data.mocap_pos[1, 0])

    env.unwrapped._reset(mask=jp.asarray([True, False]))

    pos_after = np.asarray(env.unwrapped.sim.data.states.pos[1])
    mocap_after = np.asarray(env.unwrapped.sim.mjx_data.mocap_pos[1, 0])
    assert np.array_equal(pos_after, pos_before), "Unmasked world state changed on masked reset."
    assert np.array_equal(mocap_after, mocap_before), (
        "Unmasked world track changed on masked reset."
    )
    env.close()
