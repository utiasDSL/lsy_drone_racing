"""MPC Controller for drone racing using attitude control interface.

This module contains the main MPC controller class that coordinates trajectory planning,
gate detection, and control execution for autonomous drone racing.

micromamba activate /home/kstandard/.local/share/mamba/envs/DroneRace
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosOcpSolver

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.acados_model import setup_ocp, export_quadrotor_ode_model
from lsy_drone_racing.control.logging_setup import FlightLogger
from lsy_drone_racing.control.smooth_trajectory_planner import TrajectoryPlanner
from lsy_drone_racing.control.collision_avoidance import CollisionAvoidanceHandler
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.control.warm_start import x_initial, u_initial

if TYPE_CHECKING:
    from numpy.typing import NDArray

OBSTACLE_RADIUS = 0.13  # Radius of the obstacles in meters
GATE_LENGTH = 0.50  # Length of the gate in meters
ELLIPSOID_RADIUS = 0.12  # Diameter of the ellipsoid in meters
ELLIPSOID_LENGTH = 0.7  # Length of the ellipsoid in meters


class MPController(Controller):
    """MPC using the collective thrust and attitude interface for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller."""
        super().__init__(obs, info, config)

        # Setup logging
        self.flight_logger = FlightLogger("MPController")

        # Basic configuration
        self.freq = config.env.freq
        self._tick = 0
        self.start_pos = obs["pos"]
        self.config = config

        # MPC parameters
        self.N = 60
        self.T_HORIZON = 2.0
        self.dt = self.T_HORIZON / self.N

        # Replanning configuration
        self.replanning_frequency = 1
        self.last_replanning_tick = 0

        # Approach parameters for different gates
        self.approach_dist = [0.2, 0.3, 0.3, 0.1]
        self.exit_dist = [0.5, 0.15, 0.35, 5.0]
        self.default_approach_dist = 0.1
        self.default_exit_dist = 0.5

        # Height offset parameters
        self.approach_height_offset = [0.01, 0.01, -0.1, 0.0]
        self.exit_height_offset = [0.1, 0.01, 0.1, 0.0]
        self.default_approach_height_offset = 0.1
        self.default_exit_height_offset = 0.0

        # Initialize target gate tracking
        self.current_target_gate_idx = 0

        # Setup collision avoidance
        num_gates = len(obs["gates_pos"])
        num_obstacles = len(obs["obstacles_pos"])
        self.collision_avoidance_handler = CollisionAvoidanceHandler(
            num_gates,
            num_obstacles,
            GATE_LENGTH,
            ELLIPSOID_LENGTH,
            ELLIPSOID_RADIUS,
            OBSTACLE_RADIUS,
        )

        # Create the MPC solver
        self.mpc_weights = {
            "Q_pos": 8,  # Position tracking weight -10
            "Q_vel": 0.01,  # Velocity tracking weight -0.01
            "Q_rpy": 0.01,  # Attitude (roll/pitch/yaw) weight
            "Q_thrust": 0.01,  # Collective thrust weight
            "Q_cmd": 0.01,  # Command tracking weight
            "R": 0.01,  # Control input regularization weight -0.007
        }
        # Setup the acados model and solver
        model = export_quadrotor_ode_model()
        self.collision_avoidance_handler.setup_model(model)
        ocp = setup_ocp(model, self.T_HORIZON, self.N, self.mpc_weights)
        self.collision_avoidance_handler.setup_ocp(ocp)
        self.acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file="lsy_example_mpc.json", verbose=False
        )

        # Warm start the solver
        for i in range(self.N):
            # Set initial state and control inputs
            self.acados_ocp_solver.set(i, "x", x_initial[i])
            self.acados_ocp_solver.set(i, "u", u_initial[i])

        # Controller state variables
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.finished = False

        # Trajectory tracking
        self.updated_gates = set()
        self.updated_speeds = set()
        self.saved_trajectories = []
        self.trajectory_metadata = []
        self.current_trajectory = None

        # Episode statistics
        self.gates_passed = 0
        self.total_gates = len(config.env.track["gates"])
        self.flight_successful = False

        # Initialize trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            self.config, self.flight_logger, self.N, self.T_HORIZON
        )

        # Store original weights for restoration
        self.original_mpc_weights = self.mpc_weights.copy()

        # Define replanning weights (higher path following)
        self.replanning_mpc_weights = self.mpc_weights.copy()
        self.replanning_mpc_weights["Q_pos"] *= 2.0  #  position tracking

        # Weight adjustment tracking
        self.weights_adjusted = False
        self.weight_adjustment_start_tick = 0
        self.weight_adjustment_duration = int(0.7 * config.env.freq)

        # Store parameters directly in the trajectory planner
        self.trajectory_planner.freq = self.freq
        self.trajectory_planner.approach_dist = self.approach_dist
        self.trajectory_planner.exit_dist = self.exit_dist
        self.trajectory_planner.approach_height_offset = self.approach_height_offset
        self.trajectory_planner.exit_height_offset = self.exit_height_offset
        self.trajectory_planner.default_approach_dist = self.default_approach_dist
        self.trajectory_planner.default_exit_dist = self.default_exit_dist
        self.trajectory_planner.default_approach_height_offset = self.default_approach_height_offset
        self.trajectory_planner.default_exit_height_offset = self.default_exit_height_offset

        # Log initialization parameters
        controller_params = {
            "MPC_N": self.N,
            "MPC_T_HORIZON": self.T_HORIZON,
            "MPC_dt": self.dt,
            "MPC_weights": self.mpc_weights,
            "approach_distances": self.approach_dist,
            "exit_distances": self.exit_dist,
            "approach_height_offsets": self.approach_height_offset,
            "exit_height_offsets": self.exit_height_offset,
            "start_position": self.start_pos.tolist(),
            "total_gates": self.total_gates,
            "freq": self.freq,
            "replanning_frequency": self.replanning_frequency,
            "default_approach_dist": self.default_approach_dist,
            "default_exit_dist": self.default_exit_dist,
            "default_approach_height_offset": self.default_approach_height_offset,
            "default_exit_height_offset": self.default_exit_height_offset,
        }
        self.flight_logger.log_initialization(controller_params, self._tick)

        # Initialize trajectory
        self.waypoints = self.trajectory_planner.generate_waypoints(obs, 0, elevated_start=True)
        x_des, y_des, z_des = self.trajectory_planner.generate_trajectory_from_waypoints(
            self.waypoints,
            0,
            use_velocity_aware=False,
            tick=0,  # Add tick parameter
        )
        self._trajectory_start_tick = 0

        # Store trajectory references
        self.x_des = x_des
        self.y_des = y_des
        self.z_des = z_des

        # Initialize current_trajectory to prevent None errors
        self.current_trajectory = {
            "tick": 0,
            "waypoints": self.waypoints.copy(),
            "x": x_des.copy(),
            "y": y_des.copy(),
            "z": z_des.copy(),
            "timestamp": time.time(),
        }

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Execute MPC control with replanning capabilities."""
        # Store current velocity for smooth trajectory transitions
        self._current_vel = obs["vel"].copy()

        # Track gate progress
        current_target_gate = obs.get("target_gate", 0)
        if isinstance(current_target_gate, np.ndarray):
            current_target_gate = int(current_target_gate.item())
        else:
            current_target_gate = int(current_target_gate)

        # Update gates passed tracking
        if current_target_gate == -1:  # Finished all gates
            self.gates_passed = self.total_gates
            self.flight_successful = True
            self.flight_logger.log_gate_progress(
                self.gates_passed, self.total_gates, current_target_gate, self._tick
            )
        elif current_target_gate > self.gates_passed:
            self.gates_passed = current_target_gate
            self.flight_logger.log_gate_progress(
                self.gates_passed, self.total_gates, current_target_gate, self._tick
            )

        # UPDATE WEIGHTS
        self._update_weights()

        # Check for replanning
        just_replanned = self._check_and_execute_replanning(obs, current_target_gate)

        self.collision_avoidance_handler.update_parameters(self.acados_ocp_solver, self.N, obs)

        # Execute MPC control
        return self._execute_mpc_control(obs, current_target_gate, just_replanned)

    def _check_and_execute_replanning(self, obs: dict, current_target_gate: int) -> bool:
        """Enhanced replanning with smooth transitions."""
        just_replanned = False
        should_replan = False

        # Check for replanning based on gate observations
        if self._tick - self.last_replanning_tick >= self.replanning_frequency:
            self.last_replanning_tick = self._tick

            # Chek if variables are initialized
            if "gates_pos" in obs and obs["gates_pos"] is not None and "target_gate" in obs:
                # Check if the target gate hasn't been updated yet
                if (
                    current_target_gate < len(self.config.env.track["gates"])
                    and current_target_gate < len(obs["gates_pos"])
                    and current_target_gate not in self.updated_gates
                ):
                    config_pos = np.array(
                        self.config.env.track["gates"][current_target_gate]["pos"]
                    )
                    observed_pos = np.array(obs["gates_pos"][current_target_gate])

                    # Check distance gate has moved
                    diff = np.linalg.norm(config_pos - observed_pos)

                    if diff > 0.05 and self._is_drone_approaching_gate(
                        obs, current_target_gate
                    ):  # cm threshold and drone is approaching the gate
                        replan_info = {
                            "gate_idx": current_target_gate,
                            "diff": diff,
                            "config_pos": f"[{config_pos[0]:.3f}, {config_pos[1]:.3f}]",
                            "observed_pos": f"[{observed_pos[0]:.3f}, {observed_pos[1]:.3f}]",
                        }
                        self.flight_logger.log_replanning_event(replan_info, self._tick)

                        should_replan = True
                        just_replanned = True
                        self.updated_gates.add(current_target_gate)

            if should_replan:
                self._activate_replanning_weights_gradual()

                # Generate NEW waypoints with momentum preservation
                new_waypoints = self.trajectory_planner.generate_smooth_replanning_waypoints(
                    obs,
                    obs["vel"] if "vel" in obs else np.zeros(3),
                    current_target_gate,
                    self.config.env.track["gates"][current_target_gate:],
                )

                # Generate trajectory with blending
                self.x_des, self.y_des, self.z_des = (
                    self.trajectory_planner.generate_trajectory_from_waypoints(
                        new_waypoints,
                        current_target_gate,
                        use_velocity_aware=True,
                        current_vel=obs["vel"] if "vel" in obs else np.zeros(3),
                        tick=self._tick,
                    )
                )

                self._trajectory_start_tick = self._tick

                trajectory_info = {
                    "mode": "enhanced-velocity-aware-with-blending",
                    "waypoints": len(new_waypoints),
                    "target_gate": current_target_gate,
                    "weights_boosted": True,
                    "momentum_preserved": True,
                }

                self.flight_logger.log_trajectory_update(trajectory_info, self._tick)
                just_replanned = True

                self.current_trajectory = {
                    "tick": 0,
                    "waypoints": self.waypoints.copy(),
                    "x": self.x_des.copy(),
                    "y": self.y_des.copy(),
                    "z": self.z_des.copy(),
                    "timestamp": time.time(),
                }

        return just_replanned

    def _is_drone_approaching_gate(self, obs: dict, gate_idx: int) -> bool:
        """Check if drone is approaching the gate (not moving away from it)."""
        try:
            # Get drone position and velocity
            drone_pos = obs["pos"]
            drone_vel = obs["vel"]

            # Get gate position (use observed if available, otherwise config)
            if "gates_pos" in obs and gate_idx < len(obs["gates_pos"]):
                gate_pos = np.array(obs["gates_pos"][gate_idx])
            else:
                gate_pos = np.array(self.config.env.track["gates"][gate_idx]["pos"])

            # Vector from drone to gate
            to_gate = gate_pos - drone_pos

            # Check if drone velocity is generally toward the gate
            if np.linalg.norm(drone_vel) > 0.1:  # Only check if drone is moving
                vel_normalized = drone_vel / np.linalg.norm(drone_vel)
                to_gate_normalized = to_gate / max(np.linalg.norm(to_gate), 1e-6)

                # Dot product > 0 means moving toward gate
                approach_alignment = np.dot(vel_normalized, to_gate_normalized)

                # Also check distance - if very close (< 1m), might be passing through
                distance_to_gate = np.linalg.norm(to_gate)

                # Approaching if:
                # 1. Moving toward gate (dot product > 0.3, about 70° cone)
                # 2. Not too close (> 0.5m) OR moving fast toward it
                is_moving_toward = approach_alignment > 0.3
                is_reasonable_distance = distance_to_gate > 0.7 or approach_alignment > 0.7

                result = is_moving_toward and is_reasonable_distance

                # Debug logging for problematic cases
                if not result and self._tick % 50 == 0:
                    print(
                        f"Not approaching gate {gate_idx}: alignment={approach_alignment:.2f}, distance={distance_to_gate:.2f}"
                    )

                return result
            else:
                # If drone is not moving, assume it's valid to replan
                return True

        except Exception as e:
            # If any error occurs, err on the side of allowing replanning
            print(f"Error in _is_drone_approaching_gate: {e}")
            return True

    def _execute_mpc_control(
        self, obs: dict, current_target_gate: int, just_replanned: bool
    ) -> NDArray[np.floating]:
        """Execute the MPC optimization and return control commands."""
        # Execute MPC control with trajectory offset
        trajectory_offset = getattr(self, "_trajectory_start_tick", 0)
        trajectory_index = max(0, self._tick - trajectory_offset)

        i = min(trajectory_index, len(self.x_des) - 1)
        if trajectory_index >= len(self.x_des):
            self.finished = True
            self.flight_logger.log_warning(
                f"Trajectory finished - reached end of trajectory at tick {self._tick}", self._tick
            )

        # Get current state and setup
        q = obs["quat"]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)
        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )

        # Set current state for MPC
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # Calculate tracking error for adaptive cost weights
        current_pos = obs["pos"]
        traj_len = len(self.x_des)
        i = min(i, traj_len - 1)
        target_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
        tracking_error = np.linalg.norm(current_pos - target_pos)

        # Set MPC references
        self._set_mpc_references(i, traj_len, just_replanned, tracking_error)

        # Solve MPC and get control
        status = self.acados_ocp_solver.solve()

        if status != 0:
            self.flight_logger.log_warning(
                f"MPC solver failed with status {status} at tick {self._tick}", self._tick
            )

        x1 = self.acados_ocp_solver.get(1, "x")
        _ = self.acados_ocp_solver.get(0, "u")

        # Smooth control updates
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        # Log tracking performance occasionally
        if self._tick % 100 == 0 or just_replanned:
            tracking_info = {
                "error": tracking_error,
                "gates_passed": self.gates_passed,
                "total_gates": self.total_gates,
                "just_replanned": just_replanned,
                "solver_status": status,
                "weights_adjusted": self.weights_adjusted,
                "current_Q_pos": self.mpc_weights["Q_pos"],
                "current_Q_vel": self.mpc_weights["Q_vel"],
                "current_R": self.mpc_weights["R"],
            }
            self.flight_logger.log_tracking_performance(tracking_info, self._tick)

        # Return the real control commands
        cmd = x1[10:14]  # [f_collective_cmd, roll_cmd, pitch_cmd, yaw_cmd]
        return cmd

    def _set_mpc_references(
        self, i: int, traj_len: int, just_replanned: bool, tracking_error: float
    ):
        """Set MPC reference trajectory with conservative tracking."""
        # CONSERVATIVE reference tracking
        for j in range(self.N):
            idx = i + j
            if idx < traj_len:
                x_ref = self.x_des[idx]
                y_ref = self.y_des[idx]
                z_ref = self.z_des[idx]
            else:
                x_ref = self.x_des[-1]
                y_ref = self.y_des[-1]
                z_ref = self.z_des[-1]

            # GENTLE reference with conservative position tracking
            if just_replanned or tracking_error > 0.25:  # Higher threshold
                # Gentle velocity references toward trajectory
                if j < self.N - 1:
                    next_idx = min(idx + 1, traj_len - 1)
                    desired_vel = (
                        np.array(
                            [
                                self.x_des[next_idx] - x_ref,
                                self.y_des[next_idx] - y_ref,
                                self.z_des[next_idx] - z_ref,
                            ]
                        )
                        * self.freq
                    )  # Convert to velocity

                    # Conservative scaling for gentle tracking
                    vel_scale = 1.2 if just_replanned else 1.0
                    desired_vel *= vel_scale

                    # Conservative velocity limit
                    max_vel = 2.0
                    vel_norm = np.linalg.norm(desired_vel)
                    if vel_norm > max_vel:
                        desired_vel = desired_vel / vel_norm * max_vel
                else:
                    desired_vel = np.zeros(3)
            else:
                # Very light velocity bias toward trajectory for normal tracking
                if j < self.N - 1:
                    next_idx = min(idx + 1, traj_len - 1)
                    desired_vel = (
                        np.array(
                            [
                                self.x_des[next_idx] - x_ref,
                                self.y_des[next_idx] - y_ref,
                                self.z_des[next_idx] - z_ref,
                            ]
                        )
                        * self.freq
                        * 0.3  # Very light influence
                    )
                else:
                    desired_vel = np.zeros(3)

            # Base reference with gentle velocity tracking
            yref = np.array(
                [
                    x_ref,
                    y_ref,
                    z_ref,  # Position (3)
                    desired_vel[0],
                    desired_vel[1],
                    desired_vel[2],  # Gentle velocity references (3)
                    0.0,
                    0.0,
                    0.0,  # RPY (3)
                    0.35,
                    0.35,  # f_collective, f_collective_cmd (2)
                    0.0,
                    0.0,
                    0.0,  # rpy_cmd (3)
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # Control inputs (4)
                ]
            )

            self.acados_ocp_solver.set(j, "yref", yref)

        # Terminal cost (14 dimensions: states only)
        terminal_idx = min(i + self.N, traj_len - 1)
        yref_N = np.array(
            [
                self.x_des[terminal_idx],
                self.y_des[terminal_idx],
                self.z_des[terminal_idx],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter and log step information."""
        self._tick += 1

        # Log important step information occasionally
        if self._tick % 100 == 0:  # Every 100 steps
            current_target_gate = obs.get("target_gate", 0)
            if isinstance(current_target_gate, np.ndarray):
                current_target_gate = int(current_target_gate.item())
            else:
                current_target_gate = int(current_target_gate)

            step_info = {
                "reward": reward,
                "gates_passed": self.gates_passed,
                "total_gates": self.total_gates,
                "target_gate": current_target_gate,
                "terminated": terminated,
                "truncated": truncated,
            }
            self.flight_logger.log_step_info(step_info, self._tick)

        return self.finished

    def episode_callback(self):
        """Reset controller state for new episode and log final statistics."""
        # Log final episode statistics WITH weight information
        summary = {
            "flight_successful": self.flight_successful,
            "gates_passed": self.gates_passed,
            "total_gates": self.total_gates,
            "finished": self.finished,
            "freq": self.freq,
            "controller_params": {
                "MPC_N": self.N,
                "MPC_T_HORIZON": self.T_HORIZON,
                "MPC_weights": self.mpc_weights,
                "original_weights": self.original_mpc_weights,
                "replanning_weights": self.replanning_mpc_weights,
                "weight_adjustment_duration": self.weight_adjustment_duration,
                "approach_distances": self.approach_dist,
                "exit_distances": self.exit_dist,
                "replanning_frequency": self.replanning_frequency,
            },
        }
        self.flight_logger.log_episode_summary(summary, self._tick)

        # Reset for next episode
        self._tick = 0
        self.finished = False
        self.updated_gates = set()
        self.upadted_speeds = set()
        self._trajectory_start_tick = 0
        self.gates_passed = 0
        self.flight_successful = False

        # Reset weight adjustment state
        self.weights_adjusted = False
        self.weight_adjustment_start_tick = 0
        self.mpc_weights = self.original_mpc_weights.copy()  # Ensure clean start

        # Clear drone position history
        self.trajectory_planner.drone_positions = []
        self.trajectory_planner.drone_timestamps = []
        self.trajectory_planner.drone_ticks = []

    def episode_reset(self):
        """Reset controller state for new episode (called from sim.py)."""
        self.episode_callback()

    def get_predicted_trajectory(self) -> np.ndarray:
        """Return the MPC's predicted trajectory over the horizon."""
        pred_traj = []
        for i in range(self.N):
            x = self.acados_ocp_solver.get(i, "x")
            pred_traj.append(x[:3])
        if self.current_trajectory is not None:
            full_trajectory = np.column_stack(
                [
                    self.current_trajectory["x"],
                    self.current_trajectory["y"],
                    self.current_trajectory["z"],
                ]
            )
        else:
            full_trajectory = np.array([])

        return np.array(pred_traj), full_trajectory

    def _activate_replanning_weights_gradual(self):
        """Activate higher path following weights with gradual transition."""
        if not self.weights_adjusted:
            self.weights_adjusted = True
            self.weight_adjustment_start_tick = self._tick

    def _update_weights(self):
        """Gradually transition between weight sets."""
        weights_changed = False

        if self.weights_adjusted:
            ticks_since_start = self._tick - self.weight_adjustment_start_tick
            if ticks_since_start < self.weight_adjustment_duration:
                # Full replanning weights active
                self.mpc_weights = self.replanning_mpc_weights.copy()
            else:
                # Restore original weights
                self.mpc_weights = self.original_mpc_weights.copy()
                self.weights_adjusted = False

            weights_changed = True

        if weights_changed:
            try:
                # Create Q matrix (14x14) for states
                Q = np.diag(
                    [
                        self.mpc_weights["Q_pos"],
                        self.mpc_weights["Q_pos"],
                        self.mpc_weights["Q_pos"],  # Position
                        self.mpc_weights["Q_vel"],
                        self.mpc_weights["Q_vel"],
                        self.mpc_weights["Q_vel"],  # Velocity
                        self.mpc_weights["Q_rpy"],
                        self.mpc_weights["Q_rpy"],
                        self.mpc_weights["Q_rpy"],  # RPY
                        self.mpc_weights["Q_thrust"],
                        self.mpc_weights["Q_cmd"],  # f_collective, f_cmd
                        self.mpc_weights["Q_cmd"],
                        self.mpc_weights["Q_cmd"],
                        self.mpc_weights["Q_cmd"],  # rpy_cmd
                    ]
                )

                # Create R matrix (4x4) for controls
                R = np.diag(
                    [
                        self.mpc_weights["R"],
                        self.mpc_weights["R"],
                        self.mpc_weights["R"],
                        self.mpc_weights["R"],
                    ]
                )

                # Create the full 18x18 cost matrix using block_diag
                W = scipy.linalg.block_diag(Q, R)

                # Update all stages
                for i in range(self.N):
                    self.acados_ocp_solver.cost_set(i, "W", W)

                # Update terminal cost (only states, 14x14)
                self.acados_ocp_solver.cost_set(self.N, "W", Q)

                try:
                    # Read back the cost matrix from stage 0
                    _ = self.acados_ocp_solver.cost_get(0, "W")

                except Exception as verify_e:
                    print(f"---Could not verify weights: {verify_e}")

            except Exception as e:
                print(f"----Failed to update OCP weights: {e}")
                pass

    def get_path(self) -> np.ndarray:
        """Get the current path of the drone."""
        return np.array([self.x_des[:-100], self.y_des[:-100], self.z_des[:-100]]).T

    def get_predicted_path(self) -> np.ndarray:
        """Get the predicted path from the MPC solver."""
        pred_traj = []
        for i in range(self.N + 1):
            x = self.acados_ocp_solver.get(i, "x")
            pred_traj.append(x[:3])
        return np.array(pred_traj)

    def get_obstacle_positions(self) -> NDArray[np.floating]:
        """Get the positions of obstacles in the environment."""
        return self.obs["obstacles_pos"]
