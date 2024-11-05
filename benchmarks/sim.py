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

config = load_config(Path('{Path(__file__).parents[1] / "config/level0.toml"}'))

"""
env_setup_code = """
import gymnasium

import lsy_drone_racing

env = gymnasium.make('DroneRacing-v0', config=config)

"""
attitude_env_setup_code = """
import gymnasium

import lsy_drone_racing

env = gymnasium.make('DroneRacingThrust-v0', config=config)
"""


def time_sim_reset(n_tests: int = 10) -> NDArray[np.floating]:
    setup = load_config_code + env_setup_code
    stmt = """env.reset()"""
    return np.array(timeit.repeat(stmt=stmt, setup=setup, number=1, repeat=n_tests))


def time_sim_step(
    n_tests: int = 10, sim_steps: int = 100, physics_mode: str = "pyb"
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
