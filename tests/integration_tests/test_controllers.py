from pathlib import Path

import gymnasium
import pytest

from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
@pytest.mark.parametrize("controller_file", ["trajectory_controller.py", "ppo_controller.py"])
def test_controllers(controller_file: str):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / f"lsy_drone_racing/control/{controller_file}"
    )
    env = gymnasium.make("DroneRacing-v0", config=config)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info)
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    # No assertion for finishing the race


@pytest.mark.integration
def test_thrust_controller():
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / "lsy_drone_racing/control/thrust_controller.py"
    )
    # Change the action space to collective thrust
    env = gymnasium.make("DroneRacingThrust-v0", config=config)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info)
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert info["target_gate"] == -1, "Thrust controller failed to complete the track"


@pytest.mark.integration
def test_trajectory_controller_finish():
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / "lsy_drone_racing/control/trajectory_controller.py"
    )
    env = gymnasium.make("DroneRacing-v0", config=config)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info)
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert info["target_gate"] == -1, "Trajectory controller failed to complete the track"
