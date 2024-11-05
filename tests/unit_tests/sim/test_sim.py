from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import load_config


def create_sim(physics: PhysicsMode) -> Sim:
    config = load_config(Path(__file__).parents[3] / "config/level3.toml")
    return Sim(
        track=config.track,
        sim_freq=config.sim.sim_freq,
        ctrl_freq=config.sim.ctrl_freq,
        disturbances=config.env.disturbances,
        randomization=config.env.randomization,
        gui=False,
        physics=physics,
    )


@pytest.mark.parametrize("physics", PhysicsMode)
@pytest.mark.unit
def test_sim_seed(physics: PhysicsMode):
    """Test if the simulation environment is deterministic with the same seed."""
    config = load_config(Path(__file__).parents[3] / "config/level3.toml")
    sim = Sim(
        track=config.env.track,
        sim_freq=config.sim.sim_freq,
        ctrl_freq=config.sim.ctrl_freq,
        disturbances=config.sim.disturbances,
        randomization=config.env.randomization,
        gui=False,
        physics=physics,
    )
    seed = 42
    sim.seed(seed)
    sim.reset()

    # Perform some actions and record the states
    states_1 = []
    for _ in range(5):
        action = sim.action_space.sample()
        if physics == PhysicsMode.SYS_ID:
            sim.step_sys_id(action[0], action[1:], 1 / sim.settings.ctrl_freq)
        else:
            sim.step(action)
        states_1.append(
            np.concatenate([sim.drone.pos, sim.drone.rpy, sim.drone.vel, sim.drone.ang_vel])
        )

    # Reset the simulation and set the same seed
    sim.seed(seed)
    sim.reset()

    # Perform the same actions and record the states
    states_2 = []
    for _ in range(5):
        action = sim.action_space.sample()
        if physics == PhysicsMode.SYS_ID:
            sim.step_sys_id(action[0], action[1:], 1 / sim.settings.ctrl_freq)
        else:
            sim.step(action)
        states_2.append(
            np.concatenate([sim.drone.pos, sim.drone.rpy, sim.drone.vel, sim.drone.ang_vel])
        )

    # Check if the recorded states are the same
    assert np.array_equal(states_1, states_2)
