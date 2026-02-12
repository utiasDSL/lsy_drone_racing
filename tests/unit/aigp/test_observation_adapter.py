import numpy as np
import pytest
from gymnasium import spaces

from lsy_drone_racing.aigp.observation import (
    build_competition_proxy_observation_space,
    project_competition_proxy_observation,
)


@pytest.mark.unit
def test_competition_proxy_projection_excludes_privileged_absolute_keys():
    obs = {
        "pos": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        "quat": np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        "vel": np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32),
        "ang_vel": np.array([[[0.01, 0.02, 0.03]]], dtype=np.float32),
        "target_gate": np.array([[1]], dtype=np.int32),
        "gates_pos": np.array([[[[2.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]], dtype=np.float32),
        "gates_visited": np.array([[[True, False]]], dtype=bool),
        "obstacles_pos": np.array([[[[1.5, 2.0, 2.5]]]], dtype=np.float32),
        "obstacles_visited": np.array([[[False]]], dtype=bool),
    }

    out = project_competition_proxy_observation(obs)
    assert "pos" not in out
    assert "gates_pos" not in out
    assert "obstacles_pos" not in out
    assert "target_gate_rel_pos" in out
    np.testing.assert_allclose(out["gates_rel_pos"][0, 0, 1], np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(out["target_gate_rel_pos"][0, 0], np.array([2.0, 2.0, 2.0]))


@pytest.mark.unit
def test_competition_proxy_target_rel_pos_zeroed_when_completed():
    obs = {
        "pos": np.zeros((1, 1, 3), dtype=np.float32),
        "quat": np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        "vel": np.zeros((1, 1, 3), dtype=np.float32),
        "ang_vel": np.zeros((1, 1, 3), dtype=np.float32),
        "target_gate": np.array([[-1]], dtype=np.int32),
        "gates_pos": np.array([[[[9.0, 9.0, 9.0]]]], dtype=np.float32),
        "gates_visited": np.array([[[True]]], dtype=bool),
        "obstacles_pos": np.zeros((1, 1, 1, 3), dtype=np.float32),
        "obstacles_visited": np.array([[[False]]], dtype=bool),
    }
    out = project_competition_proxy_observation(obs)
    np.testing.assert_allclose(out["target_gate_rel_pos"], np.zeros((1, 1, 3), dtype=np.float32))


@pytest.mark.unit
def test_competition_proxy_observation_space_builds_expected_keys():
    source = spaces.Dict(
        {
            "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "target_gate": spaces.Discrete(4, start=-1),
            "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3), dtype=np.float32),
            "gates_visited": spaces.Box(low=0, high=1, shape=(4,), dtype=bool),
            "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
            "obstacles_visited": spaces.Box(low=0, high=1, shape=(2,), dtype=bool),
        }
    )
    proxy = build_competition_proxy_observation_space(source)
    assert isinstance(proxy, spaces.Dict)
    assert "target_gate_rel_pos" in proxy.spaces
    assert "gates_rel_pos" in proxy.spaces
    assert "obstacles_rel_pos" in proxy.spaces
