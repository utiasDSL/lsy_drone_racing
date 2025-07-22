"""Integration tests for MPC controller that test end-to-end functionality."""

from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lsy_drone_racing.control.attitude_mpc_combined import MPController
from lsy_drone_racing.utils import load_config


class TestMPCIntegration:
    """Integration tests for complete MPC controller workflows."""

    @pytest.fixture
    def real_config(self) -> Union[Mock, Any]:
        """Load actual config file if available, otherwise create mock.

        Returns:
            Union[Mock, Any]: Either a loaded configuration object or a mock configuration.
        """
        try:
            # Try to load real config - adjust path as needed
            config_path = Path(__file__).parents[2] / "config/level0.toml"
            if config_path.exists():
                return load_config(config_path)
        except Exception:
            pass

        # Fallback to mock config
        config = Mock()
        config.env.freq = 50
        config.env.track = {
            "gates": [
                {"pos": [1.0, 0.0, 1.0], "rpy": [0, 0, 0]},
                {"pos": [2.0, 1.0, 1.0], "rpy": [0, 0, 1.57]},
                {"pos": [1.0, 2.0, 1.2], "rpy": [0, 0, 3.14]},
                {"pos": [0.0, 1.0, 1.0], "rpy": [0, 0, -1.57]},
            ]
        }
        return config

    @pytest.fixture
    def flight_sequence(self) -> List[Dict[str, Any]]:
        """Create a sequence of observations simulating a flight.

        Returns:
            List[Dict[str, Any]]: List of observation dictionaries representing a flight sequence.
        """
        sequence = []

        # Starting position
        base_pos = [0.0, -1.0, 1.0]
        base_vel = [0.0, 0.5, 0.0]

        for i in range(20):  # 20 time steps
            # Simulate drone moving forward
            pos = [base_pos[0], base_pos[1] + i * 0.1, base_pos[2]]
            vel = [base_vel[0], base_vel[1], base_vel[2]]

            # Determine current target gate based on position
            if pos[1] < 0.5:
                target_gate = 0
            elif pos[1] < 1.5:
                target_gate = 1
            else:
                target_gate = 2

            obs = {
                "pos": np.array(pos),
                "vel": np.array(vel),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "gates_pos": np.array(
                    [[1.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.2], [0.0, 1.0, 1.0]]
                ),
                "obstacles_pos": np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 0.8]]),
                "target_gate": target_gate,
            }
            sequence.append(obs)

        return sequence

    @pytest.mark.integration
    def test_full_control_loop_sequence(
        self, real_config: Union[Mock, Any], flight_sequence: List[Dict[str, Any]]
    ) -> None:
        """Test running full control loop for multiple time steps.

        Args:
            real_config: Configuration object (real or mock).
            flight_sequence: List of observation dictionaries for flight simulation.
        """
        # Mock the heavy Acados dependencies
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            # Set up mock solver
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0  # Success
                            mock_solver.get.return_value = np.array(
                                [
                                    0.0,
                                    0.0,
                                    1.0,  # position
                                    0.0,
                                    0.0,
                                    0.0,  # velocity
                                    0.0,
                                    0.0,
                                    0.0,  # roll, pitch, yaw
                                    0.3,
                                    0.3,  # f_collective, f_cmd
                                    0.0,
                                    0.0,
                                    0.0,  # rpy_cmd
                                ]
                            )
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            # Initialize controller
                            controller = MPController(flight_sequence[0], {}, real_config)

                            # Run through flight sequence
                            actions = []
                            for i, obs in enumerate(flight_sequence):
                                action = controller.compute_control(obs)
                                actions.append(action)

                                # Simulate step callback
                                controller.step_callback(
                                    action=action,
                                    obs=obs,
                                    reward=1.0,
                                    terminated=False,
                                    truncated=False,
                                    info={},
                                )

                            # Verify results
                            assert len(actions) == len(flight_sequence)
                            assert all(action.shape == (4,) for action in actions)
                            assert controller._tick == len(flight_sequence)

    @pytest.mark.integration
    def test_gate_progression_tracking(self, real_config: Union[Mock, Any]) -> None:
        """Test gate progression tracking through multiple gates.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0
                            mock_solver.get.return_value = np.zeros(14)
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            # Create initial observation
                            obs = {
                                "pos": np.array([0.0, 0.0, 1.0]),
                                "vel": np.array([0.0, 0.0, 0.0]),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                                "gates_pos": np.array(
                                    [
                                        [1.0, 0.0, 1.0],
                                        [2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.2],
                                        [0.0, 1.0, 1.0],
                                    ]
                                ),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]]),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)

                            # Test progression through gates
                            for gate in range(4):
                                obs["target_gate"] = gate
                                controller.compute_control(obs)
                                assert controller.gates_passed == gate

                            # Test completion
                            obs["target_gate"] = -1
                            controller.compute_control(obs)
                            assert controller.gates_passed == 4
                            assert controller.flight_successful

    @pytest.mark.integration
    def test_replanning_integration(self, real_config: Union[Mock, Any]) -> None:
        """Test integration of replanning functionality.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0
                            mock_solver.get.return_value = np.zeros(
                                14, dtype=np.float64
                            )  # Explicit dtype
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            # Initial observation - use explicit dtypes
                            obs = {
                                "pos": np.array([0.0, -0.5, 1.0], dtype=np.float64),
                                "vel": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                                "gates_pos": np.array(
                                    [
                                        [1.0, 0.0, 1.0],
                                        [2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.2],
                                        [0.0, 1.0, 1.0],
                                    ],
                                    dtype=np.float64,
                                ),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)

                            # First control call
                            controller.compute_control(obs)
                            assert len(controller.updated_gates) == 0

                            # Set up for replanning
                            controller._tick = 1
                            controller.last_replanning_tick = 0
                            controller.replanning_frequency = 1

                            # Create new observation with moved gate - explicit dtypes
                            obs_moved = {
                                "pos": np.array([0.0, -0.5, 1.0], dtype=np.float64),
                                "vel": np.array(
                                    [0.5, 0.0, 0.0], dtype=np.float64
                                ),  # Moving toward gate
                                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                                "gates_pos": np.array(
                                    [
                                        [1.2, 0.0, 1.0],  # Moved gate
                                        [2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.2],
                                        [0.0, 1.0, 1.0],
                                    ],
                                    dtype=np.float64,
                                ),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
                                "target_gate": 0,
                            }

                            # Mock trajectory generation with explicit dtypes
                            with patch.object(
                                controller.trajectory_planner,
                                "generate_smooth_replanning_waypoints",
                            ) as mock_waypoints:
                                with patch.object(
                                    controller.trajectory_planner,
                                    "generate_trajectory_from_waypoints",
                                ) as mock_traj:
                                    mock_waypoints.return_value = np.array(
                                        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64
                                    )
                                    mock_traj.return_value = (
                                        np.array([0.0, 1.0], dtype=np.float64),
                                        np.array([0.0, 0.0], dtype=np.float64),
                                        np.array([1.0, 1.0], dtype=np.float64),
                                    )

                                    # This should trigger replanning
                                    controller.compute_control(obs_moved)

                            # Should trigger replanning
                            assert 0 in controller.updated_gates

    @pytest.mark.integration
    def test_weight_adjustment_integration(self, real_config: Union[Mock, Any]) -> None:
        """Test weight adjustment integration during replanning.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0
                            mock_solver.get.return_value = np.zeros(14)
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            obs = {
                                "pos": np.array([0.0, 0.0, 1.0]),
                                "vel": np.array([0.0, 0.0, 0.0]),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                                "gates_pos": np.array([[1.0, 0.0, 1.0]]),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]]),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)
                            original_q_pos = controller.original_mpc_weights["Q_pos"]

                            # Trigger weight adjustment
                            controller._activate_replanning_weights_gradual()

                            # Run several control steps
                            for i in range(10):
                                controller.compute_control(obs)
                                controller._tick = i

                            # Weights should be adjusted during this period
                            assert controller.mpc_weights["Q_pos"] >= original_q_pos

    @pytest.mark.integration
    def test_episode_reset_integration(self, real_config: Union[Mock, Any]) -> None:
        """Test full episode reset integration.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0
                            mock_solver.get.return_value = np.zeros(14)
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            obs = {
                                "pos": np.array([0.0, 0.0, 1.0]),
                                "vel": np.array([0.0, 0.0, 0.0]),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                                "gates_pos": np.array([[1.0, 0.0, 1.0]]),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]]),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)

                            # Modify controller state
                            for i in range(50):
                                controller.compute_control(obs)
                                controller.step_callback(np.zeros(4), obs, 1.0, False, False, {})

                            controller.gates_passed = 2
                            controller.finished = True
                            controller.updated_gates.add(1)
                            controller._activate_replanning_weights_gradual()

                            # Mock the file operations that happen in episode_reset
                            with patch.object(controller, "x_traj", []):  # Mock trajectory data
                                with patch("numpy.savez"):  # Mock file saving
                                    # Reset episode
                                    controller.episode_reset()

                            # Verify complete reset
                            assert controller._tick == 0
                            assert controller.gates_passed == 0
                            assert not controller.finished
                            assert len(controller.updated_gates) == 0
                            assert not controller.weights_adjusted
                            assert controller.mpc_weights == controller.original_mpc_weights

    @pytest.mark.integration
    def test_trajectory_planning_integration(self, real_config: Union[Mock, Any]) -> None:
        """Test integration between controller and trajectory planner.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            mock_solver = Mock()
                            mock_solver.solve.return_value = 0
                            mock_solver.get.return_value = np.zeros(14)
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            obs = {
                                "pos": np.array([0.0, 0.0, 1.0]),
                                "vel": np.array([0.5, 0.0, 0.0]),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                                "gates_pos": np.array(
                                    [
                                        [1.0, 0.0, 1.0],
                                        [2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.2],
                                        [0.0, 1.0, 1.0],
                                    ]
                                ),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]]),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)

                            # Verify trajectory planner is properly initialized
                            assert controller.trajectory_planner is not None
                            assert hasattr(controller, "x_des")
                            assert hasattr(controller, "y_des")
                            assert hasattr(controller, "z_des")

                            # Verify trajectory data exists
                            assert len(controller.x_des) > 0
                            assert len(controller.y_des) > 0
                            assert len(controller.z_des) > 0

                            # Test path retrieval
                            path = controller.get_path()
                            assert path.shape[1] == 3  # 3D coordinates

                            # Test predicted trajectory
                            pred_traj, full_traj = controller.get_predicted_trajectory()
                            assert pred_traj.shape[0] == controller.N
                            assert pred_traj.shape[1] == 3

    @pytest.mark.integration
    def test_error_handling_integration(self, real_config: Union[Mock, Any]) -> None:
        """Test error handling in integrated system.

        Args:
            real_config: Configuration object (real or mock).
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver"
        ) as mock_solver_class:
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            # Mock solver that sometimes fails
                            mock_solver = Mock()
                            mock_solver.solve.side_effect = [
                                1,
                                0,
                                1,
                                0,
                            ]  # Alternate success/failure
                            mock_solver.get.return_value = np.zeros(14)
                            mock_solver.set = Mock()
                            mock_solver.cost_set = Mock()
                            mock_solver_class.return_value = mock_solver

                            obs = {
                                "pos": np.array([0.0, 0.0, 1.0]),
                                "vel": np.array([0.0, 0.0, 0.0]),
                                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                                "gates_pos": np.array([[1.0, 0.0, 1.0]]),
                                "obstacles_pos": np.array([[0.5, 0.5, 0.5]]),
                                "target_gate": 0,
                            }

                            controller = MPController(obs, {}, real_config)

                            # Should not crash despite solver failures
                            for i in range(4):
                                action = controller.compute_control(obs)
                                assert action.shape == (4,)
                                assert np.all(np.isfinite(action))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
