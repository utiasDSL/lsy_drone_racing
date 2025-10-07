from pathlib import Path

import numpy as np
import pytest
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.envs.utils import gate_passed
from lsy_drone_racing.utils import load_config, load_controller


@pytest.mark.unit
def test_load_config():
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    assert isinstance(config, ConfigDict), f"Config file is not a ConfigDict: {type(config)}"


@pytest.mark.unit
def test_load_controller():
    c = load_controller(
        Path(__file__).parents[3] / "lsy_drone_racing/control/trajectory_controller.py"
    )
    assert issubclass(c, Controller), f"Controller {c} is not a subclass of `Controller`"


@pytest.mark.unit
def test_gate_pass():
    # TODO: Check accelerated function in RaceCore instead
    gate_pos = np.array([0, 0, 0])
    gate_quat = R.identity().as_quat()
    gate_size = np.array([1, 1])
    # Test passing through the gate
    drone_pos, last_drone_pos = np.array([0, 1, 0]), np.array([0, -1, 0])
    assert gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, gate_size)
    # Test passing outside the gate boundaries
    drone_pos, last_drone_pos = np.array([2, 1, 0]), np.array([2, -1, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, gate_size)
    # Test passing close to the gate
    drone_pos, last_drone_pos = np.array([0.51, 1, 0]), np.array([0.51, -1, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, gate_size)
    # Test passing opposite direction
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, gate_size)
    # Test with rotated gate
    rotated_gate_quat = R.from_euler("xyz", [0, np.pi / 4, 0]).as_quat()
    drone_pos, last_drone_pos = np.array([0.5, 0.5, 0]), np.array([-0.5, -0.5, 0])
    assert gate_passed(drone_pos, last_drone_pos, gate_pos, rotated_gate_quat, gate_size)
    # Test with moved gate
    moved_gate_pos = np.array([1, 1, 1])
    drone_pos, last_drone_pos = np.array([1, 2, 1]), np.array([1, 0, 1])
    assert gate_passed(drone_pos, last_drone_pos, moved_gate_pos, gate_quat, gate_size)
    # Test not crossing the plane
    drone_pos, last_drone_pos = np.array([0, -0.5, 0]), np.array([0, -1, 0])
    assert not gate_passed(drone_pos, last_drone_pos, gate_pos, gate_quat, gate_size)
