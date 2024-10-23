from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import BaseController
from lsy_drone_racing.utils import check_gate_pass, load_config, load_controller, map2pi


@pytest.mark.unit
def test_load_config():
    config = load_config(Path(__file__).parents[3] / "config/level0.toml")
    assert isinstance(config, dict), f"Config file is not a dictionary: {config}"


@pytest.mark.unit
def test_load_controller():
    c = load_controller(
        Path(__file__).parents[3] / "lsy_drone_racing/control/trajectory_controller.py"
    )
    assert issubclass(c, BaseController), f"Controller {c} is not a subclass of BaseController"


@pytest.mark.unit
def test_map2pi():
    assert map2pi(0) == 0
    assert map2pi(np.pi) == -np.pi
    assert map2pi(-np.pi) == -np.pi
    assert map2pi(2 * np.pi) == 0
    assert map2pi(-2 * np.pi) == 0
    assert np.allclose(map2pi(np.arange(10) * 2 * np.pi), np.zeros(10))
    assert np.max(map2pi(np.linspace(-100, 100, num=1000))) <= np.pi
    assert np.min(map2pi(np.linspace(-100, 100, num=1000))) >= -np.pi


@pytest.mark.unit
def test_check_gate_pass():
    gate_pos = np.array([0, 0, 0])
    gate_rot = R.from_euler("xyz", [0, 0, 0])
    gate_size = np.array([1, 1])
    # Test passing through the gate
    assert check_gate_pass(gate_pos, gate_rot, gate_size, np.array([0, 1, 0]), np.array([0, -1, 0]))
    # Test passing outside the gate boundaries
    assert not check_gate_pass(
        gate_pos, gate_rot, gate_size, np.array([2, 1, 0]), np.array([2, -1, 0])
    )
    # Test passing close to the gate
    assert not check_gate_pass(
        gate_pos, gate_rot, gate_size, np.array([0.51, 1, 0]), np.array([0.51, -1, 0])
    )
    # Test passing opposite direction
    assert not check_gate_pass(
        gate_pos, gate_rot, gate_size, np.array([0, -1, 0]), np.array([0, 1, 0])
    )
    # Test with rotated gate
    rotated_gate = R.from_euler("xyz", [0, np.pi / 4, 0])
    assert check_gate_pass(
        gate_pos, rotated_gate, gate_size, np.array([0.5, 0.5, 0]), np.array([-0.5, -0.5, 0])
    )
    # Test with moved gate
    moved_gate_pos = np.array([1, 1, 1])
    assert check_gate_pass(
        moved_gate_pos, gate_rot, gate_size, np.array([1, 2, 1]), np.array([1, 0, 1])
    )
    # Test not crossing the plane
    assert not check_gate_pass(
        gate_pos, gate_rot, gate_size, np.array([0, -0.5, 0]), np.array([0, -1, 0])
    )
