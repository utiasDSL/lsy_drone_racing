import numpy as np
import pytest

from lsy_drone_racing.utils.rotations import map2pi


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
