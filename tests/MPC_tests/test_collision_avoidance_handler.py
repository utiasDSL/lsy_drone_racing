from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from casadi import MX
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.collision_avoidance import CollisionAvoidanceHandler


@pytest.fixture
def handler() -> CollisionAvoidanceHandler:
    """Create a standard CollisionAvoidanceHandler instance for testing."""
    return CollisionAvoidanceHandler(
        num_gates=2,
        num_obstacles=5,
        gate_length=2.0,
        ellipsoid_length=1.0,
        ellipsoid_radius=0.5,
        obstacle_radius=0.3,
        ignored_obstacle_indices=[1, 3],
    )


@pytest.fixture
def empty_handler() -> CollisionAvoidanceHandler:
    """Create a CollisionAvoidanceHandler with no gates or obstacles for edge case testing."""
    return CollisionAvoidanceHandler(
        num_gates=0,
        num_obstacles=0,
        gate_length=2.0,
        ellipsoid_length=1.0,
        ellipsoid_radius=0.5,
        obstacle_radius=0.3,
    )


def test_get_active_obstacle_indices(handler: CollisionAvoidanceHandler) -> None:
    """Test that active obstacle indices are correctly calculated."""
    assert handler.get_active_obstacle_indices() == [0, 2, 4]


@pytest.mark.parametrize(
    "ignored_indices,expected_active",
    [(None, [0, 1, 2, 3, 4]), ([], [0, 1, 2, 3, 4]), ([0, 1, 2, 3, 4], []), ([0, 4], [1, 2, 3])],
)
def test_get_active_obstacle_indices_parametrized(
    ignored_indices: List[int], expected_active: List[int]
) -> None:
    """Test active obstacle indices with different ignored indices configurations."""
    handler = CollisionAvoidanceHandler(
        num_gates=1,
        num_obstacles=5,
        gate_length=2.0,
        ellipsoid_length=1.0,
        ellipsoid_radius=0.5,
        obstacle_radius=0.3,
        ignored_obstacle_indices=ignored_indices,
    )
    assert handler.get_active_obstacle_indices() == expected_active


def test_get_obstacle_cylinders(handler: CollisionAvoidanceHandler) -> None:
    """Test that obstacle cylinders are correctly generated."""
    handler.obstacle_positions = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]])

    result = handler.get_obstacle_cylinders()
    expected_pos = np.array([[0, 0, 0], [2, 2, 0], [4, 4, 0]])
    expected_radius = np.array([0.3, 0.3, 0.3])

    np.testing.assert_array_equal(result["pos"], expected_pos)
    np.testing.assert_array_equal(result["radius"], expected_radius)


def test_get_obstacle_cylinders_empty(empty_handler: CollisionAvoidanceHandler) -> None:
    """Test that empty obstacle cylinders are handled correctly."""
    empty_handler.obstacle_positions = np.array([])

    result = empty_handler.get_obstacle_cylinders()
    assert len(result["pos"]) == 0
    assert len(result["radius"]) == 0


def test_get_gate_ellipsoids(handler: CollisionAvoidanceHandler) -> None:
    """Test that gate ellipsoids are correctly generated."""
    handler.gate_positions = np.array([[0, 0, 0], [1, 1, 1]])
    quat = Rotation.from_euler("z", [0, np.pi / 2]).as_quat()
    handler.gate_rotations = Rotation.from_quat(quat).as_euler("xyz")

    result = handler.get_gate_ellipsoids()
    assert result["pos"].shape == (8, 3)
    assert result["axes"].shape == (8, 3)
    assert result["rot"].shape == (8, 3, 3)

    # Verify that rotations are correctly applied
    # First gate (no rotation)
    assert np.allclose(result["pos"][0], np.array([1.0, 0.0, 0.0]), atol=1e-5)
    # Second gate (rotated 90 degrees around z)
    assert np.allclose(result["pos"][4], np.array([1.0, 1.0 + 1.0, 1.0]), atol=1e-5)


def test_create_obstacle_expressions(handler: CollisionAvoidanceHandler) -> None:
    """Test that obstacle expressions are correctly created."""
    drone_pos = MX.sym("x", 3)
    params, h_expr = handler._create_obstacle_expressions(drone_pos)

    assert len(params) == 3  # 3 active obstacles
    assert len(h_expr) == 3  # 3 expressions for the obstacles


def test_create_obstacle_expressions_none_ignored(empty_handler: CollisionAvoidanceHandler) -> None:
    """Test obstacle expressions when no obstacles are ignored."""
    # Update empty handler to have obstacles but none ignored
    empty_handler.num_obstacles = 3
    empty_handler.num_active_obstacles = 3
    empty_handler.ignored_obstacle_indices = set()

    drone_pos = MX.sym("x", 3)
    params, h_expr = empty_handler._create_obstacle_expressions(drone_pos)

    assert len(params) == 3
    assert len(h_expr) == 3


def test_create_gate_expressions(handler: CollisionAvoidanceHandler) -> None:
    """Test that gate expressions are correctly created."""
    drone_pos = MX.sym("x", 3)
    params, h_expr = handler._create_gate_expressions(drone_pos)

    assert len(params) == 2  # 2 gates
    assert len(h_expr) == 8  # 2 gates × 4 ellipsoids


def test_setup_model(handler: CollisionAvoidanceHandler) -> None:
    """Test that the model setup correctly adds collision avoidance constraints."""
    mock_model = MagicMock()
    mock_model.x = MX.sym("x", 3)
    mock_model.p = MX()

    # Keep track of the original model p attribute
    original_p = mock_model.p

    handler.setup_model(mock_model)

    # Check that constraint expressions were added to the model
    assert hasattr(mock_model, "con_h_expr")

    # Verify p was modified (not the same as the original)
    assert mock_model.p is not original_p


def test_setup_ocp(handler: CollisionAvoidanceHandler) -> None:
    """Test that OCP is correctly set up with collision avoidance constraints."""
    mock_ocp = MagicMock()

    # Mock model.p.rows() to return something valid for parameter initialization
    mock_ocp.model.p.rows.return_value = 10

    # Run the method
    handler.setup_ocp(mock_ocp)

    # Check that the attributes are set correctly
    assert mock_ocp.dims.nsh == handler.num_constraints
    assert len(mock_ocp.constraints.lh) == handler.num_constraints
    assert len(mock_ocp.constraints.uh) == handler.num_constraints
    assert len(mock_ocp.constraints.idxsh) == handler.num_constraints

    # Verify slack variables are properly set
    assert np.array_equal(mock_ocp.constraints.idxsh, np.arange(handler.num_constraints))

    # Check that parameter values were initialized correctly
    assert mock_ocp.parameter_values is not None


def test_update_parameters(handler: CollisionAvoidanceHandler) -> None:
    """Test that parameters are correctly updated in the OCP solver."""
    ocp_solver = MagicMock()
    obs = {
        "obstacles_pos": np.array([[0, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0], [5, 5, 0]]),
        "gates_pos": np.array([[1, 1, 1], [2, 2, 2]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 1, 0]]),
    }

    handler.update_parameters(ocp_solver, N_horizon=5, obs=obs)

    # Check that the OCP solver was updated for each time step in the horizon
    assert ocp_solver.set.call_count == 5

    # Check that the internal state was updated correctly
    np.testing.assert_array_equal(handler.obstacle_positions, obs["obstacles_pos"])
    np.testing.assert_array_equal(handler.gate_positions, obs["gates_pos"])

    # Verify that gate rotations were converted from quaternions to euler angles
    expected_rotations = Rotation.from_quat(obs["gates_quat"]).as_euler("xyz", degrees=False)
    np.testing.assert_array_almost_equal(handler.gate_rotations, expected_rotations)


def test_update_parameters_with_empty_obstacles(handler: CollisionAvoidanceHandler) -> None:
    """Test parameter updates when there are no active obstacles."""
    # Set all obstacles to be ignored
    handler.ignored_obstacle_indices = set(range(handler.num_obstacles))
    handler.num_active_obstacles = 0

    ocp_solver = MagicMock()
    obs = {
        "obstacles_pos": np.array([[0, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0], [5, 5, 0]]),
        "gates_pos": np.array([[1, 1, 1], [2, 2, 2]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 1, 0]]),
    }

    # Should not raise any errors even with all obstacles ignored
    handler.update_parameters(ocp_solver, N_horizon=3, obs=obs)

    assert ocp_solver.set.call_count == 3


def test_constructor_defaults() -> None:
    """Test that default values in the constructor work correctly."""
    handler = CollisionAvoidanceHandler(
        num_gates=1,
        num_obstacles=3,
        gate_length=1.0,
        ellipsoid_length=0.5,
        ellipsoid_radius=0.2,
        obstacle_radius=0.1,
        # ignored_obstacle_indices not provided
    )

    assert handler.ignored_obstacle_indices == set()
    assert handler.num_active_obstacles == 3
    assert handler.num_constraints == 7  # 1 gate (4 ellipsoids) + 3 obstacles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
