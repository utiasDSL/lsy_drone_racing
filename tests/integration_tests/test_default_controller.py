from pathlib import Path

import gymnasium
import pytest

from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
def test_default_controller():
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    ctrl_cls = load_controller(Path(__file__).parents[2] / "examples/trajectory_controller.py")
    env = gymnasium.make("DroneRacing-v0", config=config)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info)
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_learn(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert info["target_gate"] == -1, "Example controller failed to complete the track"
