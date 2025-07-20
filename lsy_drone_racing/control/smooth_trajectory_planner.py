"""Trajectory planning module for drone racing.

This module handles waypoint generation, trajectory planning through gates,
and velocity-aware trajectory updates for smooth flight paths.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


from lsy_drone_racing.utils.controller_config import (
    get_optimization_params,
    get_parameter,
    get_speed_params,
    get_trajectory_params,
)


class TrajectoryPlanner:
    """Handles trajectory planning and waypoint generation for drone racing."""

    def __init__(
        self,
        config: dict[str, Any],
        logger: Any,
        N: int = None,
        T_HORIZON: float = None,
        start_time: str = "00_00_00",
    ) -> None:
        """Initialize the trajectory planner.

        Args:
            config: Configuration dictionary containing environment settings
            logger: Logger instance for recording events
            N: Number of trajectory points (default: loaded from config or 30)
            T_HORIZON: Horizon time for trajectory planning (default: loaded from config or 1.5)
            start_time: Start time string for logging (default: "00_00_00")
        """
        self.start_time = start_time
        self.config = config
        self.logger = logger  # FlightLogger instance

        # Load constants from config
        self.N = N if N is not None else get_parameter("trajectory_planner.N_default")
        self.T_HORIZON = (
            T_HORIZON
            if T_HORIZON is not None
            else get_parameter("trajectory_planner.T_HORIZON_default")
        )

        # Load trajectory parameters
        traj_params = get_trajectory_params()
        self.approach_dist = traj_params["approach_dist"]
        self.exit_dist = traj_params["exit_dist"]
        self.approach_height_offset = traj_params["approach_height_offset"]
        self.exit_height_offset = traj_params["exit_height_offset"]
        self.default_approach_dist = traj_params["default_approach_dist"]
        self.default_exit_dist = traj_params["default_exit_dist"]
        self.default_approach_height_offset = traj_params["default_approach_height_offset"]
        self.default_exit_height_offset = traj_params["default_exit_height_offset"]

        # Load optimization parameters
        optimization_params = get_optimization_params()
        self.drone_clearance_horizontal = optimization_params["drone_clearance_horizontal"]
        self.gate_half_width = optimization_params["gate_half_width"]
        self.position_diff_threshold = optimization_params["position_diff_threshold"]
        self.replanning_threshold = optimization_params["replanning_threshold"]

        # Load speed parameters
        speed_params = get_speed_params()
        self.base_speed = speed_params["base_speed"]
        self.high_speed = speed_params["high_speed"]
        self.approach_speed = speed_params["approach_speed"]
        self.exit_speed = speed_params["exit_speed"]

        # Load generation parameters
        self.min_speed_threshold = get_parameter(
            "trajectory_planner.generation.min_speed_threshold"
        )
        self.min_gate_duration = get_parameter("trajectory_planner.generation.min_gate_duration")
        self.extra_points_final_gate = get_parameter(
            "trajectory_planner.generation.extra_points_final_gate"
        )
        self.extra_points_normal = get_parameter(
            "trajectory_planner.generation.extra_points_normal"
        )

        # Load smoothing parameters
        self.momentum_time = get_parameter("trajectory_planner.smoothing.momentum_time")
        self.max_momentum_distance = get_parameter(
            "trajectory_planner.smoothing.max_momentum_distance"
        )
        self.velocity_threshold = get_parameter("trajectory_planner.smoothing.velocity_threshold")
        self.transition_factor = get_parameter("trajectory_planner.smoothing.transition_factor")

        # Current target gate tracking
        self.current_target_gate_idx = 0
        self.gate_indices: list[int] = []

        # Trajectory storage
        self.current_trajectory: dict[str, Any] | None = None
        self.freq = config.env.freq

        # Drone position tracking
        self.drone_positions: list[NDArray[np.floating]] = []
        self.drone_timestamps: list[float] = []
        self.drone_ticks: list[int] = []

    def generate_trajectory_from_waypoints(
        self,
        waypoints: NDArray[np.floating],
        target_gate_idx: int,
        use_velocity_aware: bool = False,
        current_vel: NDArray[np.floating] | None = None,
        tick: int = 0,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Generate trajectory from waypoints with smooth continuation after last gate.

        Args:
            waypoints: Array of waypoint coordinates (shape: N x 3)
            target_gate_idx: Index of the target gate
            use_velocity_aware: Whether to use velocity-aware trajectory generation
            current_vel: Current velocity vector (default: None)
            tick: Current simulation tick (default: 0)

        Returns:
            Tuple of (x_trajectory, y_trajectory, z_trajectory) arrays
        """
        if len(waypoints) < 2:
            self.logger.log_warning("Not enough waypoints to generate trajectory", tick)
            return np.array([]), np.array([]), np.array([])

        speeds = self.calculate_adaptive_speeds(waypoints, target_gate_idx)

        # Time parameterization
        time_points = [0.0]
        for i in range(1, len(waypoints)):
            segment_distance = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            avg_speed = (speeds[i - 1] + speeds[i]) / 2
            segment_time = segment_distance / max(avg_speed, 0.1)
            time_points.append(time_points[-1] + segment_time)

        total_time = time_points[-1]
        if self.gate_indices and len(self.gate_indices) > 0:
            # Gate indices are stored as [approach, center, exit, approach, center, exit, ...]
            num_gates = len(self.gate_indices) // 3
        else:
            # Fallback: estimate based on waypoints
            num_gates = max(1, (len(waypoints) - 2) // 3)  # Subtract momentum/transition points
        min_duration = max(num_gates * 3.0, total_time)

        # ADD SMOOTH CONTINUATION POINTS FOR LAST GATE
        if target_gate_idx >= 3 or target_gate_idx == -1:
            # Calculate continuation direction from last two waypoints
            if len(waypoints) >= 2:
                last_direction = waypoints[-1] - waypoints[-2]
                last_direction_norm = last_direction / max(np.linalg.norm(last_direction), 1e-6)

                # Add continuation waypoints that extend naturally
                continuation_distances = [0.5, 1.0, 1.5, 2.0]  # Progressive distances
                for dist in continuation_distances:
                    continuation_point = waypoints[-1] + last_direction_norm * dist
                    waypoints = np.vstack([waypoints, continuation_point])

                    # Add corresponding time points
                    continuation_time = dist / 1.6  # Continue at 1.6 m/s
                    time_points.append(time_points[-1] + continuation_time)

                total_time = time_points[-1]
                min_duration = max(min_duration, total_time + 1.0)  # Extra time buffer

        if total_time < min_duration:
            time_points.append(min_duration)
            waypoints = np.vstack([waypoints, waypoints[-1]])

        # Spline generation with optional velocity boundary conditions
        time_normalized = np.array(time_points) / max(time_points[-1], 1.0)

        if use_velocity_aware and current_vel is not None and np.linalg.norm(current_vel) > 0.3:
            try:
                # For last gate, set terminal velocity to continue forward
                if target_gate_idx >= 3 or target_gate_idx == -1:
                    if len(waypoints) >= 2:
                        terminal_vel = (waypoints[-1] - waypoints[-2]) / max(
                            time_points[-1] - time_points[-2], 0.1
                        )
                    else:
                        terminal_vel = current_vel * 0.5  # Reduce speed gradually
                else:
                    terminal_vel = np.zeros(3)  # Stop for intermediate gates

                cs_x = CubicSpline(
                    time_normalized,
                    waypoints[:, 0],
                    bc_type=((1, current_vel[0]), (1, terminal_vel[0])),
                )
                cs_y = CubicSpline(
                    time_normalized,
                    waypoints[:, 1],
                    bc_type=((1, current_vel[1]), (1, terminal_vel[1])),
                )
                cs_z = CubicSpline(
                    time_normalized,
                    waypoints[:, 2],
                    bc_type=((1, current_vel[2]), (1, terminal_vel[2])),
                )
            except Exception as e:
                self.logger.log_warning(f"Velocity boundary conditions failed: {str(e)}", tick)
                cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type="natural")
                cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type="natural")
                cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type="natural")
        else:
            cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type="natural")
            cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type="natural")
            cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type="natural")

        # Generate trajectory points
        min_points = max(int(self.freq * min_duration), tick + 3 * self.N)
        ts = np.linspace(0, 1, min_points)
        x_des = cs_x(ts)
        y_des = cs_y(ts)
        z_des = cs_z(ts)

        # Minimal terminal extension - spline already handles continuation
        if target_gate_idx >= 3 or target_gate_idx == -1:
            extra_points = self.extra_points_final_gate  # Minimal for MPC horizon
        else:
            extra_points = self.extra_points_normal  # Normal extension for earlier gates

        x_des = np.concatenate((x_des, [x_des[-1]] * extra_points))
        y_des = np.concatenate((y_des, [y_des[-1]] * extra_points))
        z_des = np.concatenate((z_des, [z_des[-1]] * extra_points))

        # Save trajectory
        self.current_trajectory = {
            "tick": tick,
            "waypoints": waypoints.copy(),
            "x": x_des.copy(),
            "y": y_des.copy(),
            "z": z_des.copy(),
            "timestamp": time.time(),
        }
        if not use_velocity_aware:
            self.save_trajectories_to_file(
                f"flight_logs/-1_planned_trajectories_{self.start_time}.npz"
            )
        else:
            self.save_trajectories_to_file(
                f"flight_logs/{target_gate_idx}_planned_trajectories_{self.start_time}.npz"
            )

        return x_des, y_des, z_des

    def generate_waypoints(
        self,
        obs: dict[str, NDArray[np.floating]],
        start_gate_idx: int = 0,
        elevated_start: bool = False,
    ) -> NDArray[np.floating]:
        """Generate waypoints that navigate through all gates.

        Args:
            obs: Observation dictionary containing position and gate information
            start_gate_idx: Starting gate index (default: 0)
            elevated_start: Whether to start at elevated position (default: False)

        Returns:
            Array of waypoints for trajectory generation
        """
        # Start with the current position as a single-element array
        start_point = np.array(obs["pos"]).reshape(1, 3)

        # Generate gate waypoints
        gates = self._get_gates_with_observations(obs, start_gate_idx)
        gate_waypoints = self.loop_path_gen(gates, obs, elevated_start)

        # Concatenate the arrays properly
        return np.concatenate((start_point, gate_waypoints))

    def loop_path_gen(
        self,
        gates: list[dict[str, Any]],
        obs: dict[str, NDArray[np.floating]],
        elevated_start: bool = False,
    ) -> NDArray[np.floating]:
        """Generate a loop path through all gates with height offsets.

        Args:
            gates: List of gate dictionaries containing position and orientation
            obs: Observation dictionary containing current position
            elevated_start: Whether to start at elevated position (default: False)

        Returns:
            Array of waypoints for gate traversal
        """
        waypoints = []
        if elevated_start:
            # Start with an elevated position if specified
            start_point = np.array(obs["pos"]).reshape(1, 3)
            start_point[0, 2] += 0.1
            waypoints.append(start_point[0])

        # Create waypoints for each gate
        self.gate_indices = []
        for gate_idx, gate in enumerate(gates):
            gate_info = self._get_gate_info(gate)
            original_gate_idx = gate_idx + (4 - len(gates))

            # Set current target gate index for proximity checking
            self.current_target_gate_idx = original_gate_idx

            # Use individual distances if available, otherwise fall back to defaults
            if original_gate_idx < len(self.approach_dist):
                approach_dist = self.approach_dist[original_gate_idx]
            else:
                approach_dist = self.default_approach_dist

            if original_gate_idx < len(self.exit_dist):
                exit_dist = self.exit_dist[original_gate_idx]
            else:
                exit_dist = self.default_exit_dist

            # Get height offsets for this gate
            if original_gate_idx < len(self.approach_height_offset):
                approach_z_offset = self.approach_height_offset[original_gate_idx]
            else:
                approach_z_offset = self.default_approach_height_offset

            if original_gate_idx < len(self.exit_height_offset):
                exit_z_offset = self.exit_height_offset[original_gate_idx]
            else:
                exit_z_offset = self.default_exit_height_offset

            # Create approach, center, and exit points with individual distances and height offsets
            approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
            approach_point[2] += approach_z_offset  # Add height offset

            exit_point = gate_info["center"] - gate_info["normal"] * exit_dist
            exit_point[2] += exit_z_offset  # Add height offset

            # Add approach point with height offset
            waypoints.append(approach_point)
            self.gate_indices.append(len(waypoints) - 1)

            # Always add gate center point
            gate_point = gate_info["center"].copy()
            waypoints.append(gate_point)
            self.gate_indices.append(len(waypoints) - 1)

            # Add exit point with height offset
            waypoints.append(exit_point)
            self.gate_indices.append(len(waypoints) - 1)

            if self.current_target_gate_idx == 2:
                approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
                approach_point[0] -= 0.05
                approach_point[2] += 0.05
                waypoints.append(approach_point)

        return np.array(waypoints)

    def _get_gate_info(self, gate: dict[str, Any]) -> dict[str, Any]:
        """Extract gate information including position, orientation, and dimensions.

        Args:
            gate: Gate dictionary containing position and orientation information

        Returns:
            Dictionary containing processed gate information including center, normal, dimensions
        """
        try:
            # Ensure we're working with a valid dictionary
            if not isinstance(gate, dict):
                self.logger.log_warning(f"Gate is not a dictionary: {gate}", 0)
                gate = {"pos": [0.0, 0.0, 1.0]}

            # Extract position
            if "pos" not in gate:
                self.logger.log_warning(f"Gate missing position key: {gate}", 0)
                pos = np.array([0.0, 0.0, 1.0])
            elif gate["pos"] is None:
                self.logger.log_warning(f"Gate position is None: {gate}", 0)
                pos = np.array([0.0, 0.0, 1.0])
            else:
                # Handle different position formats
                raw_pos = gate["pos"]
                if isinstance(raw_pos, dict) and "pos" in raw_pos:
                    # Handle nested dictionary case
                    self.logger.log_warning(f"Converting nested gate dictionary: {raw_pos}", 0)
                    raw_pos = raw_pos["pos"]

                if isinstance(raw_pos, (list, tuple, np.ndarray)) and len(raw_pos) == 3:
                    pos = np.array(raw_pos)
                elif np.isscalar(raw_pos):
                    self.logger.log_warning(
                        f"Gate position is scalar {raw_pos}, using as z-coordinate", 0
                    )
                    pos = np.array([0.0, 0.0, float(raw_pos)])
                else:
                    self.logger.log_warning(
                        f"Invalid position format: {type(raw_pos)}, value: {raw_pos}", 0
                    )
                    pos = np.array([0.0, 0.0, 1.0])

            # Extract gate dimensions
            gate_height = gate.get("height", 0.5)
            gate_width = gate.get("width", 0.5)

            # Determine target height based on gate dimensions
            is_short_gate = gate_height <= 0.6
            target_height = pos[2] if is_short_gate else pos[2] + 0.2

            # Get orientation
            if "rpy" in gate and gate["rpy"] is not None:
                rpy = np.array(gate["rpy"])
                rotation = R.from_euler("xyz", rpy)
            else:
                rotation = R.from_euler("xyz", [0, 0, 0])
                rpy = np.zeros(3)

            # Calculate normal vector and adjusted gate center
            temp_normal = rotation.apply([1.0, 0.0, 0.0])
            normal = np.array([temp_normal[1], -temp_normal[0], temp_normal[2]])
            y_dir = rotation.apply([0.0, 1.0, 0.0])
            z_dir = rotation.apply([0.0, 0.0, 1.0])

            # Adjust center point height
            gate_center = pos.copy()
            gate_center[2] = target_height

            # SPEED OPTIMIZATION: Shift gate center for racing line
            if hasattr(self, "current_target_gate_idx"):
                current_gate = self.current_target_gate_idx

                # Create racing line by shifting gate centers
                if current_gate > 0:
                    # Get previous gate for approach direction
                    prev_gate = self.config.env.track["gates"][current_gate - 1]
                    prev_pos = np.array(prev_gate["pos"])

                    # Shift gate center forward along approach direction
                    approach_direction = gate_center - prev_pos
                    if np.linalg.norm(approach_direction) > 0:
                        approach_norm = approach_direction / np.linalg.norm(approach_direction)
                        gate_center += approach_norm * 0.05  # 5cm forward shift

            return {
                "pos": pos,
                "center": gate_center,
                "normal": normal,
                "y_dir": y_dir,
                "z_dir": z_dir,
                "rotation": rotation,
                "height": gate_height,
                "width": gate_width,
                "is_short": is_short_gate,
                "rpy": rpy,
            }
        except Exception as e:
            self.logger.log_error(f"Error processing gate: {e}", 0)
            # Return safe default values
            default_pos = np.array([0.0, 0.0, 1.0])
            return {
                "pos": default_pos,
                "center": default_pos,
                "normal": np.array([1.0, 0.0, 0.0]),
                "y_dir": np.array([0.0, 1.0, 0.0]),
                "z_dir": np.array([0.0, 0.0, 1.0]),
                "rotation": R.from_euler("xyz", [0, 0, 0]),
                "height": 0.5,
                "width": 0.5,
                "is_short": True,
                "rpy": np.zeros(3),
            }

    def _get_gates_with_observations(
        self, obs: dict[str, Any], start_idx: int
    ) -> list[dict[str, Any]]:
        """Get gate info using observed positions when available.

        Args:
            obs: Observation dictionary containing gate positions
            start_idx: Starting gate index

        Returns:
            List of gate dictionaries with observed or configured positions
        """
        gates = []
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations:
            for i in range(start_idx, len(self.config.env.track["gates"])):
                gates.append(self._get_gate_with_observation(obs, i))

        return gates

    def _get_gate_with_observation(self, obs: dict[str, Any], gate_idx: int) -> dict[str, Any]:
        """Get a single gate with observed position if available.

        Args:
            obs: Observation dictionary containing gate positions
            gate_idx: Index of the gate to retrieve

        Returns:
            Gate dictionary with observed or configured position
        """
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations and gate_idx < len(obs["gates_pos"]):
            # Use observed position but keep original orientation
            original_gate = self.config.env.track["gates"][gate_idx]
            observed_pos = obs["gates_pos"][gate_idx]

            # Extract position from original gate dict
            if isinstance(original_gate, dict) and "pos" in original_gate:
                original_pos = np.array(original_gate["pos"])
            else:
                # Fallback if pos key is missing
                original_pos = np.array([0.0, 0.0, 1.0])

            # Convert observed position to numpy array for comparison
            observed_pos_array = np.array(observed_pos)

            # Calculate the difference
            position_difference = observed_pos_array - original_pos
            distance_moved = np.linalg.norm(position_difference)

            gate_info = self.config.env.track["gates"][gate_idx].copy()

            # Get orientation
            if "rpy" in gate_info and gate_info["rpy"] is not None:
                rpy = np.array(gate_info["rpy"])
                rotation = R.from_euler("xyz", rpy)
            else:
                rotation = R.from_euler("xyz", [0, 0, 0])
                rpy = np.zeros(3)

            # Calculate normal vector and adjusted gate center
            temp_normal = rotation.apply([1.0, 0.0, 0.0])
            gate_info["normal"] = np.array([temp_normal[1], -temp_normal[0], temp_normal[2]])
            if distance_moved > 0.05:  # cm threshold
                gate_info["pos"] = self._calculate_optimal_gate_crossing(
                    original_pos, observed_pos_array
                )
            else:
                gate_info["pos"] = obs["gates_pos"][gate_idx]
            return gate_info
        else:
            # Fall back to static configuration
            return self.config.env.track["gates"][gate_idx]

    def _calculate_optimal_gate_crossing(
        self, original_gate: NDArray[np.floating], new_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Calculate the optimal gate crossing point that lies between the two gate centers.

        Minimizes adjustment while ensuring safe passage.

        Args:
            original_gate: Original gate position
            new_pos: New observed gate position

        Returns:
            Optimal gate crossing position
        """
        # Define drone safety clearance (use instance variables loaded from config)
        drone_clearance_horizontal = self.drone_clearance_horizontal
        gate_half_width = self.gate_half_width

        # Vector from original to new gate center
        gate_displacement = new_pos - original_gate
        gate_distance = np.linalg.norm(gate_displacement)

        if gate_distance == 0:
            return original_gate  # No displacement, use original

        # Required distance from new gate center to ensure clearance from gate margins
        required_distance_from_new_gate = gate_half_width - drone_clearance_horizontal

        if required_distance_from_new_gate < 0:
            # Safety clearance too large for gate, use midpoint
            optimal_point = (original_gate + new_pos) / 2
        else:
            # Calculate the point on the line that is required_distance_from_new_gate away from new_pos
            # Point = new_pos - (required_distance / total_distance) * displacement_vector
            direction_to_original = (
                -gate_displacement / gate_distance
            )  # Unit vector toward original
            optimal_point = new_pos + direction_to_original * required_distance_from_new_gate

            # Ensure the point is actually between the two gate centers
            # Check if it's beyond the original gate
            distance_to_original = np.linalg.norm(optimal_point - original_gate)
            if distance_to_original > gate_distance:
                # Point is beyond original gate, use original gate position
                optimal_point = original_gate

        return optimal_point

    def save_trajectories_to_file(self, filename: str = "trajectories.npz") -> bool:
        """Save the current trajectory and drone positions to a file for later analysis.

        Args:
            filename: Name of the output file (default: "trajectories.npz")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we have a trajectory to save
            if self.current_trajectory is None:
                self.logger.log_warning("No trajectory to save", 0)
                return False

            # Create data structure with only one trajectory
            waypoints = self.current_trajectory["waypoints"]

            # Handle different waypoint formats
            if waypoints is not None:
                # Check if waypoints is a dictionary (from horizon trajectory)
                if isinstance(waypoints, dict):
                    # Extract coordinate arrays from dictionary
                    if "x" in waypoints and "y" in waypoints and "z" in waypoints:
                        # Convert dictionary format to numpy array
                        x_coords = np.array(waypoints["x"])
                        y_coords = np.array(waypoints["y"])
                        z_coords = np.array(waypoints["z"])

                        # Take a subset for waypoints (every 10th point for visualization)
                        step = max(1, len(x_coords) // 10)
                        waypoints_x = x_coords[::step]
                        waypoints_y = y_coords[::step]
                        waypoints_z = z_coords[::step]
                    else:
                        # Empty arrays if dictionary doesn't have expected keys
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
                elif isinstance(waypoints, np.ndarray):
                    # Check if it's a proper 2D array
                    if waypoints.ndim == 2 and waypoints.shape[1] >= 3:
                        waypoints_x = waypoints[:, 0]
                        waypoints_y = waypoints[:, 1]
                        waypoints_z = waypoints[:, 2]
                    elif waypoints.ndim == 1 and len(waypoints) >= 3:
                        # Single waypoint case
                        waypoints_x = np.array([waypoints[0]])
                        waypoints_y = np.array([waypoints[1]])
                        waypoints_z = np.array([waypoints[2]])
                    else:
                        # Invalid array format
                        self.logger.log_warning(
                            f"Invalid waypoints array shape: {waypoints.shape}", 0
                        )
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
                else:
                    # Try to convert to numpy array
                    try:
                        waypoints_array = np.array(waypoints)

                        if waypoints_array.ndim == 2 and waypoints_array.shape[1] >= 3:
                            waypoints_x = waypoints_array[:, 0]
                            waypoints_y = waypoints_array[:, 1]
                            waypoints_z = waypoints_array[:, 2]
                        else:
                            waypoints_x = np.array([])
                            waypoints_y = np.array([])
                            waypoints_z = np.array([])
                    except Exception:
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
            else:
                # Empty arrays if no waypoints
                waypoints_x = np.array([])
                waypoints_y = np.array([])
                waypoints_z = np.array([])

            # Convert drone positions to numpy arrays
            if self.drone_positions:
                drone_positions_array = np.array(self.drone_positions)
                drone_pos_x = drone_positions_array[:, 0]
                drone_pos_y = drone_positions_array[:, 1]
                drone_pos_z = drone_positions_array[:, 2]
                drone_timestamps_array = np.array(self.drone_timestamps)
                drone_ticks_array = np.array(self.drone_ticks)
            else:
                drone_pos_x = np.array([])
                drone_pos_y = np.array([])
                drone_pos_z = np.array([])
                drone_timestamps_array = np.array([])
                drone_ticks_array = np.array([])

            data = {
                "traj_0": {
                    "tick": self.current_trajectory["tick"],
                    "x": self.current_trajectory["x"],
                    "y": self.current_trajectory["y"],
                    "z": self.current_trajectory["z"],
                    "waypoints_x": waypoints_x,
                    "waypoints_y": waypoints_y,
                    "waypoints_z": waypoints_z,
                },
                # Add drone position data
                "drone_actual_positions": {
                    "x": drone_pos_x,
                    "y": drone_pos_y,
                    "z": drone_pos_z,
                    "timestamps": drone_timestamps_array,
                    "ticks": drone_ticks_array,
                    "num_positions": len(self.drone_positions),
                },
            }

            # Add metadata about number of trajectories
            data["num_trajectories"] = 1  # Always one trajectory

            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            np.savez(filename, **data)

            # Log successful save with position count
            save_info = {
                "filename": filename,
                "trajectory_points": len(self.current_trajectory["x"]),
                "drone_positions_saved": len(self.drone_positions),
                "waypoints_saved": len(waypoints_x),
            }
            self.logger.log_trajectory_update(save_info, 0)

            return True
        except Exception as e:
            self.logger.log_error(f"Error saving trajectory and positions: {e}", 0)
            return False

    def generate_smooth_replanning_waypoints(
        self,
        obs: dict[str, Any],
        current_vel: NDArray[np.floating],
        updated_gate_idx: int,
        remaining_gates: list[dict[str, Any]],
    ) -> NDArray[np.floating]:
        """Generate waypoints that preserve momentum and avoid backward motion.

        Args:
            obs: Observation dictionary containing current state
            current_vel: Current velocity vector
            updated_gate_idx: Index of the updated gate
            remaining_gates: List of remaining gates to traverse

        Returns:
            Array of waypoints for smooth replanning
        """
        current_pos = obs["pos"]
        current_speed = np.linalg.norm(current_vel)

        waypoints = []
        self.gate_indices = []

        # Safety check
        if not remaining_gates:
            self.logger.log_warning("No remaining gates for replanning", 0)
            return np.array([current_pos])

        # 1. MOMENTUM PRESERVATION PHASE
        if current_speed > self.velocity_threshold:
            momentum_time = self.momentum_time
            momentum_distance = min(current_speed * momentum_time, self.max_momentum_distance)

            vel_normalized = current_vel / current_speed
            momentum_point = current_pos + vel_normalized * momentum_distance
            waypoints.append(momentum_point)

            # 2. SMOOTH TRANSITION PHASE
            # Get the actual observed gate position
            observed_gate = self._get_gate_with_observation(obs, updated_gate_idx)
            gate_info = self._get_gate_info(observed_gate)

            # Get configured approach distance and height offset
            if updated_gate_idx < len(self.approach_dist):
                approach_dist = self.approach_dist[updated_gate_idx]
            else:
                approach_dist = self.default_approach_dist

            if updated_gate_idx < len(self.approach_height_offset):
                approach_z_offset = self.approach_height_offset[updated_gate_idx]
            else:
                approach_z_offset = self.default_approach_height_offset

            # Calculate proper approach point using observed gate position
            proper_approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
            proper_approach_point[2] += approach_z_offset

            # Create transition toward proper approach point
            approach_vector = proper_approach_point - momentum_point
            transition_point = momentum_point + self.transition_factor * approach_vector
            waypoints.append(transition_point)

        # 3. GATE APPROACH PHASE WITH JUMP PREVENTION
        for relative_idx, gate in enumerate(remaining_gates):
            absolute_gate_idx = updated_gate_idx + relative_idx
            observed_gate = self._get_gate_with_observation(obs, absolute_gate_idx)
            gate_info = self._get_gate_info(observed_gate)  # uses observed position

            if absolute_gate_idx >= len(self.config.env.track["gates"]):
                break

            self.current_target_gate_idx = absolute_gate_idx

            # Get distances and offsets
            if absolute_gate_idx < len(self.approach_dist):
                approach_dist = self.approach_dist[absolute_gate_idx]
            else:
                approach_dist = self.default_approach_dist

            if absolute_gate_idx < len(self.exit_dist):
                exit_dist = self.exit_dist[absolute_gate_idx]
            else:
                exit_dist = self.default_exit_dist

            if absolute_gate_idx < len(self.approach_height_offset):
                approach_z_offset = self.approach_height_offset[absolute_gate_idx]
            else:
                approach_z_offset = self.default_approach_height_offset

            if absolute_gate_idx < len(self.exit_height_offset):
                exit_z_offset = self.exit_height_offset[absolute_gate_idx]
            else:
                exit_z_offset = self.default_exit_height_offset

            # Calculate approach point using observed gate position
            configured_approach = gate_info["center"] + gate_info["normal"] * approach_dist
            configured_approach[2] += approach_z_offset

            # Add the main gate waypoints using observed positions
            waypoints.append(configured_approach)
            self.gate_indices.append(len(waypoints) - 1)

            gate_point = gate_info["center"].copy()
            waypoints.append(gate_point)
            self.gate_indices.append(len(waypoints) - 1)

            exit_point = gate_info["center"] - gate_info["normal"] * exit_dist
            exit_point[2] += exit_z_offset
            waypoints.append(exit_point)
            self.gate_indices.append(len(waypoints) - 1)

            # Gate 2 special case
            if absolute_gate_idx == 2:
                extra_approach_point = gate_info["center"] + gate_info["normal"] * 0.1
                extra_approach_point[0] -= 0.05
                extra_approach_point[2] += 0.05
                waypoints.append(extra_approach_point)

        waypoints_array = np.array(waypoints)

        return waypoints_array

    def calculate_adaptive_speeds(
        self, waypoints: NDArray[np.floating], current_target_gate: int
    ) -> list[float]:
        """Calculate adaptive speeds based on gate proximity and flight phase using gate_indices.

        Args:
            waypoints: Array of waypoints for trajectory
            current_target_gate: Index of the current target gate

        Returns:
            List of speeds corresponding to each waypoint
        """
        speeds = []

        # Speed configuration (use instance variables loaded from config)
        base_speed = self.base_speed
        high_speed = self.high_speed
        approach_speed = self.approach_speed
        exit_speed = self.exit_speed

        for i, _ in enumerate(waypoints):
            # Default to base speed
            speed = base_speed

            # Check if this waypoint index is in gate_indices
            if i in self.gate_indices:
                # Find which position in gate_indices this is
                gate_position = self.gate_indices.index(i)

                # Every 3 indices represent: approach, center, exit for each gate
                waypoint_type = gate_position % 3  # 0=approach, 1=center, 2=exit
                gate_number = gate_position // 3  # Which gate (0, 1, 2, 3)
                absolute_gate_idx = current_target_gate + gate_number

                if waypoint_type == 0:  # Approach point
                    speed = approach_speed
                elif waypoint_type == 1:  # Gate center
                    speed = approach_speed
                elif waypoint_type == 2:  # Exit point
                    speed = exit_speed

                # Special handling for Gate 1 (reduce speeds)
                if absolute_gate_idx == 0:
                    if waypoint_type in [0, 1]:
                        speed = approach_speed * 0.7
                elif absolute_gate_idx == 1:
                    if waypoint_type in [0, 1]:  # Approach and center
                        speed = approach_speed * 0.8  # Slower for Gate 1
                    # Exit speed remains high for Gate 1
                elif absolute_gate_idx == 2:
                    if waypoint_type == 0:
                        speed = approach_speed * 0.5
                    elif waypoint_type == 2:
                        speed = exit_speed * 0.8

            else:
                # Non-gate waypoints (momentum, transition, intermediate points)
                if i <= 1:
                    # Early waypoints (momentum/transition) - maintain reasonable speed
                    speed = base_speed
                else:
                    # Intermediate waypoints between gates - use high speed
                    speed = high_speed

            speeds.append(speed)

        print(f"Final speeds: {speeds}")
        return speeds
