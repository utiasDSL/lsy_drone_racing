from __future__ import annotations

import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

load_config_code = f"""
from pathlib import Path

from lsy_drone_racing.utils import load_config

config = load_config(Path('{Path(__file__).parents[1] / "config/level3.toml"}'))
"""

env_setup_code = """
import gymnasium

import lsy_drone_racing

env = gymnasium.make(
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
env.reset()
env.step(env.action_space.sample())  # JIT compile
env.reset()
env.action_space.seed(42)
action = env.action_space.sample()
"""

attitude_env_setup_code = """
import gymnasium

import lsy_drone_racing

env = gymnasium.make('DroneRacingAttitude-v0',
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
env.reset()
env.step(env.action_space.sample())  # JIT compile
env.reset()
env.action_space.seed(42)
action = env.action_space.sample()
"""

load_multi_drone_config_code = f"""
from pathlib import Path

from lsy_drone_racing.utils import load_config

config = load_config(Path('{Path(__file__).parents[1] / "config/multi_level3.toml"}'))
"""

multi_drone_env_setup_code = """
import gymnasium
import jax

import lsy_drone_racing

env = gymnasium.make('MultiDroneRacing-v0',
    n_envs=1,  # TODO: Remove this for single-world envs
    n_drones=config.env.n_drones,
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    random_resets=config.env.random_resets,
    seed=config.env.seed,
    device='cpu',
)

env.reset()
# JIT step
env.step(env.action_space.sample())
jax.block_until_ready(env.unwrapped.data)
# JIT masked reset (used in autoreset)
mask = env.unwrapped.data.marked_for_reset
mask = mask.at[0].set(True)
env.unwrapped.reset(mask=mask)
jax.block_until_ready(env.unwrapped.data)
env.action_space.seed(2)
"""


def time_sim_reset(n_tests: int = 10, number: int = 1) -> NDArray[np.floating]:
    setup = load_config_code + env_setup_code
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_sim_step(
    n_tests: int = 10, number: int = 1, physics_mode: str = "analytical"
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.physics = '{physics_mode}'\n"""
    setup = load_config_code + modify_config_code + env_setup_code + "\nenv.reset()"
    stmt = """env.step(action)"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_sim_attitude_step(n_tests: int = 10, number: int = 1) -> NDArray[np.floating]:
    setup = load_config_code + attitude_env_setup_code + "\nenv.reset()"
    stmt = """env.step(action)"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_multi_drone_reset(n_tests: int = 10, number: int = 1) -> NDArray[np.floating]:
    setup = load_multi_drone_config_code + multi_drone_env_setup_code + "\nenv.reset()"
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))


def time_multi_drone_step(
    n_tests: int = 10, number: int = 100, physics_mode: str = "analytical"
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.physics = '{physics_mode}'\n"""
    setup = (
        load_multi_drone_config_code
        + modify_config_code
        + multi_drone_env_setup_code
        + "\nenv.reset()"
    )
    stmt = """env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=n_tests))
