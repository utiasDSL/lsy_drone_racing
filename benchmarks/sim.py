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
"""

load_multi_drone_config_code = f"""
from pathlib import Path

from lsy_drone_racing.utils import load_config

config = load_config(Path('{Path(__file__).parents[1] / "config/multi_level0.toml"}'))
"""

multi_drone_env_setup_code = """
import gymnasium

import lsy_drone_racing

env = gymnasium.make('MultiDroneRacing-v0',
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
env.step(env.action_space.sample())  # JIT compile
env.reset()
"""


def time_sim_reset(n_tests: int = 10) -> NDArray[np.floating]:
    setup = load_config_code + env_setup_code
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))


def time_sim_step(
    n_tests: int = 10, sim_steps: int = 100, physics_mode: str = "analytical"
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.physics = '{physics_mode}'\n"""
    setup = load_config_code + modify_config_code + env_setup_code + "\nenv.reset()"
    stmt = f"""
for _ in range({sim_steps}):
    env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))


def time_sim_attitude_step(n_tests: int = 10, sim_steps: int = 100) -> NDArray[np.floating]:
    setup = load_config_code + attitude_env_setup_code + "\nenv.reset()"
    stmt = f"""
for _ in range({sim_steps}):
    env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))


def time_multi_drone_reset(n_tests: int = 10) -> NDArray[np.floating]:
    setup = load_multi_drone_config_code + multi_drone_env_setup_code + "\nenv.reset()"
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))


def time_multi_drone_step(
    n_tests: int = 10, sim_steps: int = 100, physics_mode: str = "analytical"
) -> NDArray[np.floating]:
    modify_config_code = f"""config.sim.physics = '{physics_mode}'\n"""
    setup = (
        load_multi_drone_config_code
        + modify_config_code
        + multi_drone_env_setup_code
        + "\nenv.reset()"
    )
    stmt = f"""
for _ in range({sim_steps}):
    env.step(env.action_space.sample())"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))
