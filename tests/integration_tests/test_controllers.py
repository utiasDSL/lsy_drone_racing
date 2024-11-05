import importlib
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from lsy_drone_racing.sim.physics import PhysicsMode
from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.integration
@pytest.mark.parametrize("controller_file", ["trajectory_controller.py", "ppo_controller.py"])
def test_controllers(controller_file: str):
    if controller_file == "ppo_controller.py" and not importlib.util.find_spec("stable_baselines3"):
        pytest.skip("Requires the stable baselines3 library")

    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.physics = PhysicsMode.DEFAULT
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
@pytest.mark.parametrize("physics", PhysicsMode)
def test_thrust_controller(physics: PhysicsMode):
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.gui = False
    config.sim.physics = physics
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
    assert obs["target_gate"] == -1, "Thrust controller failed to complete the track"


@pytest.mark.integration
@pytest.mark.parametrize("yaw", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
@pytest.mark.parametrize("physics", ["pyb", "dyn"])
def test_trajectory_controller_finish(yaw: float, physics: str):
    """Test if the trajectory controller can finish the track.

    To catch bugs that only occur with orientations other than the unit quaternion, we test if the
    controller can finish the track with different desired yaws.
    """
    config = load_config(Path(__file__).parents[2] / "config/level0.toml")
    config.sim.physics = physics
    config.sim.gui = False
    ctrl_cls = load_controller(
        Path(__file__).parents[2] / "lsy_drone_racing/control/trajectory_controller.py"
    )
    env = gymnasium.make("DroneRacing-v0", config=config)
    obs, info = env.reset()
    ctrl = ctrl_cls(obs, info)
    while True:
        action = ctrl.compute_control(obs, info)
        action[9] = yaw  # Quadrotor should be able to finish the track regardless of yaw
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    assert obs["target_gate"] == -1, "Trajectory controller failed to complete the track"
