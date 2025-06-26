"""Trajectory planning module for drone racing.

This module handles waypoint generation, trajectory planning through gates,
and velocity-aware trajectory updates for smooth flight paths.
"""

import os
import time

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R


class TrajectoryPlanner:
    """Handles trajectory planning and waypoint generation for drone racing."""

    def __init__(self, config: dict):
        """Initialize the trajectory planner."""
        self.config = config

        # Current target gate tracking
        self.current_target_gate_idx = 0

        # Trajectory storage
        self.current_trajectory = None
        self.freq = config.env.freq

    def generate_waypoints(
        self, obs: dict[str, NDArray[np.floating]], start_gate_idx: int = 0
    ) -> np.ndarray:
        """Generate waypoints that navigate through all gates."""
        # Start with the current position as a single-element array
        start_point = np.array(obs["pos"]).reshape(1, 3)

        # Generate gate waypoints
        gates = self._get_gates_with_observations(obs, start_gate_idx)
        gate_waypoints = self.loop_path_gen(gates, obs)

        # Concatenate the arrays properly
        return np.concatenate((start_point, gate_waypoints))

    def generate_trajectory_from_waypoints(
        self, waypoints: np.ndarray, tick: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate trajectory with optional velocity-aware smooth transitions."""
        if len(waypoints) < 2:
            self.logger.log_warning("Not enough waypoints to generate trajectory", tick)
            return np.array([]), np.array([]), np.array([])

        # Standard trajectory generation (for initial and velocity-aware)
        # Calculate distances between waypoints
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

        # Standard speeds for initial trajectory
        speeds = [0.8, 0.8, 1.0, 1.3, 1.5, 1.5, 1.3, 1.5, 0.8, 0.8, 1.3, 0.8, 1.3, 1.3]
        print(speeds)

        # Calculate time points
        time_points = [0.0]
        for i in range(1, len(waypoints)):
            segment_distance = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            avg_speed = (speeds[i - 1] + speeds[i]) / 2
            segment_time = segment_distance / max(avg_speed, 0.1)
            time_points.append(time_points[-1] + segment_time)
        print(time_points[-1])

        total_time = time_points[-1]
        min_duration = max(8.0, total_time)

        if total_time < min_duration:
            time_points.append(min_duration)
            waypoints = np.vstack([waypoints, waypoints[-1]])

        # Normalize time for spline
        time_normalized = np.array(time_points) / max(time_points[-1], 1.0)

        # Use natural splines for initial trajectory
        cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type="natural")
        cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type="natural")
        cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type="natural")

        # Generate trajectory points
        trajectory_freq = self.freq
        min_points = max(int(trajectory_freq * min_duration), tick + 3 * 30)  # Assuming N=30

        ts = np.linspace(0, 1, min_points)
        x_des = cs_x(ts)
        y_des = cs_y(ts)
        z_des = cs_z(ts)

        dx_des = cs_x.derivative()(ts)
        dy_des = cs_y.derivative()(ts)
        dz_des = cs_z.derivative()(ts)

        # Add terminal points for stability
        extra_points = 3 * 30 + 1  # Assuming N=30
        x_des = np.concatenate((x_des, [x_des[-1]] * extra_points))
        y_des = np.concatenate((y_des, [y_des[-1]] * extra_points))
        z_des = np.concatenate((z_des, [z_des[-1]] * extra_points))
        dx_des = np.concatenate((dx_des, [dx_des[-1]] * extra_points))
        dy_des = np.concatenate((dy_des, [dy_des[-1]] * extra_points))
        dz_des = np.concatenate((dz_des, [dz_des[-1]] * extra_points))

        # Save trajectory
        trajectory = {
            "tick": tick,
            "waypoints": waypoints.copy(),
            "x": x_des.copy(),
            "y": y_des.copy(),
            "z": z_des.copy(),
            "timestamp": time.time(),
        }

        self.current_trajectory = trajectory

        return x_des, y_des, z_des, dx_des, dy_des, dz_des

    def _get_gates_with_observations(self, obs: dict, start_idx: int) -> list[dict]:
        """Get gate info using observed positions when available."""
        gates = []
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations:
            for i in range(start_idx, len(self.config.env.track["gates"])):
                gates.append(self._get_gate_with_observation(obs, i))

        return gates

    def _get_gate_with_observation(self, obs: dict, gate_idx: int) -> dict:
        """Get a single gate with observed position if available."""
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations and gate_idx < len(obs["gates_pos"]):
            # Use observed position but keep original orientation
            gate_info = self.config.env.track["gates"][gate_idx].copy()
            gate_info["pos"] = obs["gates_pos"][gate_idx]
            return gate_info
        else:
            # Fall back to static configuration
            return self.config.env.track["gates"][gate_idx]

    def loop_path_gen(self, gates: list[dict], obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Generate a loop path through all gates with corridor constraints and height offsets."""
        waypoints = []

        # Create waypoints for each gate
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

            # Always add gate center point
            waypoints.append(gate_info["center"])

            # Add exit point with height offset
            waypoints.append(exit_point)

            if self.current_target_gate_idx == 2:
                approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
                approach_point[2] += 0.8
                waypoints.append(approach_point)

        return np.array(waypoints)

    def _get_gate_info(self, gate: dict) -> dict:
        """Extract gate information including position, orientation, and dimensions."""
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
