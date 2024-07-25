from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import load_config, load_controller, map2pi


@pytest.mark.unit
def test_load_config():
    config = load_config(Path(__file__).parents[2] / "config/test.toml")
    assert isinstance(config, dict), f"Config file is not a dictionary: {config}"


@pytest.mark.unit
def test_load_controller():
    c = load_controller(Path(__file__).parents[3] / "examples/controller.py")
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
