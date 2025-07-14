"""Direct tests for TrajectoryPlanner class."""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest

from lsy_drone_racing.control.smooth_trajectory_planner import TrajectoryPlanner


class TestTrajectoryPlanner:
    """Direct tests for TrajectoryPlanner functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock()
        config.env.freq = 50
        config.env.track = {
            "gates": [
                {"pos": [1.0, 0.0, 1.0], "rpy": [0, 0, 0], "height": 0.5, "width": 0.5},
                {"pos": [2.0, 1.0, 1.0], "rpy": [0, 0, 1.57], "height": 0.8, "width": 0.8},
                {"pos": [1.0, 2.0, 1.2], "rpy": [0, 0, 3.14], "height": 0.6, "width": 0.6},
                {"pos": [0.0, 1.0, 1.0], "rpy": [0, 0, -1.57], "height": 0.7, "width": 0.7},
            ]
        }
        return config

    @pytest.fixture
    def logger(self):
        """Create mock logger."""
        logger = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_trajectory_update = Mock()
        return logger

    @pytest.fixture
    def planner(self, config, logger):
        """Create configured trajectory planner."""
        planner = TrajectoryPlanner(config, logger, N=30, T_HORIZON=1.5)

        # Set parameters that MPController would normally set
        planner.approach_dist = [0.2, 0.3, 0.2, 0.1]
        planner.exit_dist = [0.4, 0.15, 0.2, 5.0]
        planner.approach_height_offset = [0.01, 0.1, 0.0, 0.0]
        planner.exit_height_offset = [0.1, 0.0, 0.05, 0.0]
        planner.default_approach_dist = 0.1
        planner.default_exit_dist = 0.5
        planner.default_approach_height_offset = 0.1
        planner.default_exit_height_offset = 0.0

        return planner

    @pytest.fixture
    def sample_obs(self):
        """Create sample observation."""
        return {
            "pos": np.array([0.0, -0.5, 1.0]),
            "vel": np.array([0.5, 0.0, 0.0]),
            "gates_pos": np.array(
                [[1.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.2], [0.0, 1.0, 1.0]]
            ),
            "target_gate": 0,
        }

    def test_gate_info_valid(self, planner):
        """Test gate info extraction with valid data."""
        gate = {"pos": [1.0, 0.5, 1.2], "rpy": [0, 0, 0.5], "height": 0.6, "width": 0.8}

        info = planner._get_gate_info(gate)

        assert np.allclose(info["pos"], [1.0, 0.5, 1.2])
        assert info["height"] == 0.6
        assert info["width"] == 0.8
        assert isinstance(info["normal"], np.ndarray)
        assert len(info["normal"]) == 3

    def test_gate_info_missing_fields(self, planner):
        """Test gate info with missing fields uses defaults."""
        gate = {"pos": [2.0, 1.0, 1.0]}  # Missing height, width, rpy

        info = planner._get_gate_info(gate)

        assert np.allclose(info["pos"], [2.0, 1.0, 1.0])
        assert info["height"] == 0.5  # Default
        assert info["width"] == 0.5  # Default

    def test_gate_info_invalid_position(self, planner):
        """Test gate info with invalid position falls back to default."""
        invalid_gates = [
            {"pos": None},
            {"pos": "invalid"},
            {},  # No pos key
        ]

        for gate in invalid_gates:
            info = planner._get_gate_info(gate)
            assert np.allclose(info["pos"], [0.0, 0.0, 1.0])  # Default fallback

    def test_waypoint_generation_basic(self, planner, sample_obs):
        """Test basic waypoint generation."""
        waypoints = planner.generate_waypoints(sample_obs, start_gate_idx=0)

        assert len(waypoints) > 1
        assert np.allclose(waypoints[0], sample_obs["pos"])

    def test_waypoint_generation_elevated(self, planner, sample_obs):
        """Test waypoint generation with elevated start."""
        waypoints = planner.generate_waypoints(sample_obs, elevated_start=True)

        assert len(waypoints) >= 2
        # Check that some point is elevated
        max_z = np.max(waypoints[:, 2])
        assert max_z > sample_obs["pos"][2]

    def test_trajectory_from_waypoints_basic(self, planner):
        """Test trajectory generation from waypoints."""
        waypoints = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 1.0, 1.0]])

        x_des, y_des, z_des = planner.generate_trajectory_from_waypoints(
            waypoints, target_gate_idx=0, tick=0
        )

        assert len(x_des) > 0
        assert len(y_des) > 0
        assert len(z_des) > 0
        assert len(x_des) == len(y_des) == len(z_des)

    def test_trajectory_velocity_aware(self, planner):
        """Test velocity-aware trajectory generation."""
        waypoints = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 1.0, 1.0]])
        current_vel = np.array([1.0, 0.0, 0.0])

        x_des, y_des, z_des = planner.generate_trajectory_from_waypoints(
            waypoints, target_gate_idx=1, use_velocity_aware=True, current_vel=current_vel, tick=50
        )

        assert len(x_des) > 0
        assert planner.current_trajectory is not None

    def test_adaptive_speeds(self, planner):
        """Test adaptive speed calculation."""
        waypoints = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

        planner.gate_indices = [1, 2, 3]  # Set gate indices for speed calculation
        speeds = planner.calculate_adaptive_speeds(waypoints, current_target_gate=0)

        assert len(speeds) == len(waypoints)
        assert all(speed > 0 for speed in speeds)

    def test_optimal_gate_crossing(self, planner):
        """Test optimal gate crossing calculation."""
        original = np.array([1.0, 0.0, 1.0])
        moved = np.array([1.1, 0.0, 1.0])  # Gate moved 10cm

        optimal = planner._calculate_optimal_gate_crossing(original, moved)

        # Should be between original and moved position
        assert 1.0 <= optimal[0] <= 1.1

    def test_file_saving(self, planner):
        """Test trajectory file saving."""
        # Generate a trajectory first
        waypoints = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        planner.generate_trajectory_from_waypoints(waypoints, 0, tick=0)

        # Add some drone position data
        planner.drone_positions = [[0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
        planner.drone_timestamps = [0.0, 0.02]
        planner.drone_ticks = [0, 1]

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_traj.npz")

            success = planner.save_trajectories_to_file(filename)

            assert success == True
            assert os.path.exists(filename)

            # Verify file contents
            data = np.load(filename, allow_pickle=True)
            assert "traj_0" in data
            assert "drone_actual_positions" in data

    def test_smooth_replanning_waypoints(self, planner, sample_obs):
        """Test smooth replanning waypoint generation."""
        current_vel = np.array([1.0, 0.0, 0.0])
        remaining_gates = planner.config.env.track["gates"][1:]  # Skip first gate

        waypoints = planner.generate_smooth_replanning_waypoints(
            sample_obs, current_vel, updated_gate_idx=1, remaining_gates=remaining_gates
        )

        assert len(waypoints) > 1
        assert isinstance(waypoints, np.ndarray)
        assert waypoints.shape[1] == 3  # 3D coordinates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
