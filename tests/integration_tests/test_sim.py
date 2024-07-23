from pathlib import Path

import gymnasium
import pytest
import toml
from munch import munchify

from lsy_drone_racing.sim.physics import PhysicsMode


@pytest.mark.parametrize("physics", PhysicsMode)
@pytest.mark.integration
def test_sim(physics: PhysicsMode):
    """Test the simulation environment with different physics modes."""
    with open(Path(__file__).parents[1] / "config/test.toml") as f:
        config = munchify(toml.load(f))
    config.sim.physics = physics  # override physics mode
    env = gymnasium.make("DroneRacing-v0", config=config)
    env.reset()
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
