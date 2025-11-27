"""
Trajectory-Following Controller with Dynamic Replanning for Drone Racing

  This controller uses cubic spline interpolation to generate smooth trajectories through waypoints.
It supports dynamic replanning when gates or obstacles change position during flight.
  This controller is a refactored and extended version of the MyStateController 
from the LSY Drone Racing project by Yuming Li (TUM).
It is used solely for learning and research purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

# Import the new interactive visualizer
try:
    from lsy_drone_racing.utils.trajectory_visualizer import TrajectoryVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("Warning: TrajectoryVisualizer not available. Visualization disabled.")
    VISUALIZER_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """Trajectory-following controller for drone racing.
    
    This controller plans a smooth trajectory through predefined waypoints and tracks it
    over time. It can dynamically replan when the environment changes.
    """

    # Class constants
    TRAJECTORY_DURATION = 20.0  # Total trajectory duration in seconds
    STATE_DIMENSION = 13  # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
    OBSTACLE_SAFETY_DISTANCE = 0.3  # Minimum distance to obstacles in meters
    VISUALIZATION_SAMPLES = 100  # Number of points for trajectory visualization
    LOG_INTERVAL = 100  # Print debug info every N ticks

    # Waypoint generation parameters
    APPROACH_DISTANCE = 0.6  # Distance before/after gate center for waypoints
    NUM_INTERMEDIATE_POINTS = 5  # Number of waypoints around each gate

    # Detour parameters
    ANGLE_THRESHOLD = 120.0  # Angle threshold in degrees for detecting backtracking
    DETOUR_DISTANCE = 0.65  # Distance from gate center for detour waypoint

    def __init__(
        self, 
        obs: dict[str, NDArray[np.floating]], 
        info: dict, 
        config: dict
    ):
        """Initialize the controller.

        Args:
            obs: Initial observation containing drone state, gates, and obstacles.
            info: Initial environment information from reset.
            config: Race configuration with environment frequency and settings.
        """
        super().__init__(obs, info, config)
        
        # Controller state
        self._time_step = 0
        self._control_frequency = config.env.freq
        self._is_finished = False
        
        # Environment state tracking for change detection
        self._last_gate_flags = None
        self._last_obstacle_flags = None

        # Real gate positions tracking (for visualization)
        num_gates = len(obs['gates_pos'])
        self._gate_real_positions = np.full((num_gates, 3), np.nan)  # Initially unknown
        self._gate_detected_flags = np.zeros(num_gates, dtype=bool)  # Initially all undetected

        # === DEBUG: Initialize debug attributes ===
        self._debug_detour_analysis = []  # Will store analysis for each gate pair
        self._debug_detour_summary = {}   # Will store overall summary
        self._debug_detour_waypoints_added = []
        self._debug_waypoints_initial = None
        self._debug_waypoints_after_detour = None
        self._debug_waypoints_final = None

        # Extract gate information
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
        self._extract_gate_coordinate_frames(obs['gates_quat'])
        
        # Extract obstacle information
        self.obstacle_positions = obs['obstacles_pos']
        
        # Initial drone position
        self.initial_position = obs['pos']
        
        # Enable visualization (trajectory plotting)
        self.visualization = False

        # Calculate waypoints 
        waypoints = self.calc_waypoints_from_gates(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS
        )
        print(f"Initial waypoints count: {len(waypoints)}")
        print(f"Initial waypoints:\n{waypoints}")
                
        # === DEBUG: Save initial waypoints ===
        self._debug_waypoints_initial = waypoints.copy()
        
        # Step 2: Add detour waypoints for backtracking gates
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
            angle_threshold=self.ANGLE_THRESHOLD,
            detour_distance=self.DETOUR_DISTANCE
        )
        print(f"Waypoints after detour: {len(waypoints)}")

        # === DEBUG: Save waypoints after detour ===
        self._debug_waypoints_after_detour = waypoints.copy()
        
        # Apply collision avoidance
        time_params, waypoints = self._avoid_collisions(
            waypoints, 
            self.obstacle_positions,
            self.OBSTACLE_SAFETY_DISTANCE
        )

        # === DEBUG: Save final waypoints ===
        self._debug_waypoints_final = waypoints.copy()
        
        # Generate smooth trajectory
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        # Initialize the new interactive visualizer
        self.visualizer = None
        if self.visualization and VISUALIZER_AVAILABLE:
            self.visualizer = TrajectoryVisualizer(
                width=1400,
                height=1000,
                title="Drone Racing - Interactive 3D Trajectory"
            )
            # Create initial visualization
            self.visualizer.visualize(
                gate_positions=self.gate_positions,
                gate_normals=self.gate_normals,
                obstacle_positions=self.obstacle_positions,
                trajectory=self.trajectory,
                trajectory_duration=self.TRAJECTORY_DURATION,
                waypoints=waypoints,
                drone_position=obs['pos'],
                gate_detected_status=self._gate_detected_flags,  # Pass initial detection status
                show=True  # Display in browser/notebook
            )
            
            # Optionally save initial state as HTML
            # self.visualizer.save_html("trajectory_initial.html")

        print("=== Available info keys ===")
        print(info.keys())
        print("\n=== Available obs keysduide ===")
        print(obs.keys())


    def _extract_gate_normals(self, gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Extract gate normal vectors from quaternions.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [w, x, y, z].
            
        Returns:
            Array of gate normal vectors (first column of rotation matrices).
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        return rotation_matrices[:, :, 0]  # Extract first column (x-axis / normal)

    def calc_waypoints_from_gates(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: float = 0.5,
        num_intermediate_points: int = 5
    ) -> NDArray[np.floating]:
        """Automatically generate waypoints based on gate positions.
        
        Creates multiple waypoints around each gate to ensure smooth passage.
        
        Args:
            initial_position: Starting position of the drone.
            gate_positions: Positions of all gates.
            gate_normals: Normal vectors of all gates.
            approach_distance: Distance before/after gate center for waypoints.
            num_intermediate_points: Number of points to place around each gate.
            
        Returns:
            Array of waypoints including initial position.
        """
        num_gates = gate_positions.shape[0]
        
        # Create waypoints before and after each gate
        waypoints_per_gate = []
        for i in range(num_intermediate_points):
            # Interpolate from -approach_distance to +approach_distance
            offset = -approach_distance + (i / (num_intermediate_points - 1)) * 2 * approach_distance
            waypoints_per_gate.append(gate_positions + offset * gate_normals)
        
        # Reshape to (num_gates * num_intermediate_points, 3)
        waypoints = np.concatenate(waypoints_per_gate, axis=1)
        waypoints = waypoints.reshape(num_gates, num_intermediate_points, 3).reshape(-1, 3)
        
        # Prepend initial position
        waypoints = np.vstack([initial_position, waypoints])
        
        return waypoints

    def _generate_trajectory(
        self, 
        duration: float, 
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """Generate a cubic spline trajectory through waypoints.
        
        Uses arc-length parameterization for more uniform velocity distribution.
        
        Args:
            duration: Total time duration for the trajectory.
            waypoints: Array of 3D waypoints.
            
        Returns:
            CubicSpline object for trajectory evaluation.
        """
        # Calculate segment lengths
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Cumulative arc length
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Parameterize time by arc length for uniform velocity
        time_parameters = cumulative_arc_length / cumulative_arc_length[-1] * duration
        
        return CubicSpline(time_parameters, waypoints)

    def _avoid_collisions(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
        safety_distance: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Modify trajectory to avoid collisions with obstacles.
        
        Detects trajectory segments that pass too close to obstacles and inserts
        new waypoints to steer around them.
        
        Args:
            waypoints: Original waypoints.
            obstacle_positions: Positions of cylindrical obstacles.
            safety_distance: Minimum safe distance from obstacles.
            
        Returns:
            Tuple of (time_parameters, modified_waypoints).
        """
        # Generate initial trajectory
        trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        # Sample trajectory at high resolution
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION, 
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        trajectory_points = trajectory(time_samples)
        
        # Process each obstacle
        for obstacle_position in obstacle_positions:
            collision_free_times = []
            collision_free_waypoints = []
            
            is_inside_obstacle = False
            entry_index = None
            
            for i, point in enumerate(trajectory_points):
                # Check distance in XY plane only (cylindrical obstacles)
                distance_xy = np.linalg.norm(obstacle_position[:2] - point[:2])
                
                if distance_xy < safety_distance:
                    if not is_inside_obstacle:
                        # First entry into obstacle zone
                        is_inside_obstacle = True
                        entry_index = i
                        
                elif is_inside_obstacle:
                    # Exiting obstacle zone - compute avoidance waypoint
                    exit_index = i
                    is_inside_obstacle = False
                    
                    # Compute new waypoint direction (average of entry and exit directions)
                    entry_point = trajectory_points[entry_index]
                    exit_point = trajectory_points[exit_index]
                    
                    entry_direction = entry_point[:2] - obstacle_position[:2]
                    exit_direction = exit_point[:2] - obstacle_position[:2]
                    avoidance_direction = entry_direction + exit_direction
                    avoidance_direction /= np.linalg.norm(avoidance_direction)
                    
                    # Place new waypoint at safety distance
                    new_position_xy = obstacle_position[:2] + avoidance_direction * safety_distance
                    new_position_z = (entry_point[2] + exit_point[2]) / 2
                    new_waypoint = np.concatenate([new_position_xy, [new_position_z]])
                    
                    # Add at midpoint time
                    collision_free_times.append((time_samples[entry_index] + time_samples[exit_index]) / 2)
                    collision_free_waypoints.append(new_waypoint)
                    
                else:
                    # Point is safe, keep it
                    collision_free_times.append(time_samples[i])
                    collision_free_waypoints.append(point)
            
            # Update for next obstacle
            time_samples = np.array(collision_free_times)
            trajectory_points = np.array(collision_free_waypoints)
        
        return time_samples, trajectory_points

    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect if any gate or obstacle position has changed.
        
        Uses the 'visited' flags to detect when environment elements are triggered,
        which may indicate position changes in randomized scenarios.
        Also records real gate positions when gates are detected.
        
        Args:
            obs: Current observation with 'gates_visited' and 'obstacles_visited' flags.
            
        Returns:
            True if a change was detected, False otherwise.
        """
        # Initialize on first call
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs['gates_visited'], dtype=bool)
            self._last_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)
            return False
        
        # Check for newly visited gates or obstacles
        current_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        current_obstacle_flags = np.array(obs['obstacles_visited'], dtype=bool)
        
        gate_newly_visited = np.any((~self._last_gate_flags) & current_gate_flags)
        obstacle_newly_visited = np.any((~self._last_obstacle_flags) & current_obstacle_flags)
        
        # Update gate detection status and record real positions
        for i, is_visited in enumerate(current_gate_flags):
            if is_visited and not self._gate_detected_flags[i]:
                # Gate newly detected - record its real position
                self._gate_detected_flags[i] = True
                self._gate_real_positions[i] = obs['gates_pos'][i]
                print(f"[GATE DETECTED] Gate {i+1} at real position: "
                      f"[{obs['gates_pos'][i][0]:.3f}, {obs['gates_pos'][i][1]:.3f}, {obs['gates_pos'][i][2]:.3f}]")
        
        # Update stored flags
        self._last_gate_flags = current_gate_flags
        self._last_obstacle_flags = current_obstacle_flags
        
        return gate_newly_visited or obstacle_newly_visited

    def _replan_trajectory(self, obs: dict[str, NDArray[np.floating]], current_time: float) -> None:
        """Replan trajectory when environment changes."""
        print(f"\n[REPLANNING] Time: {current_time:.2f}s")
        
        # Update gate information with complete coordinate frames
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
            self._extract_gate_coordinate_frames(obs['gates_quat'])
        
        # Step 1: Generate new waypoints
        waypoints = self.calc_waypoints_from_gates(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=self.APPROACH_DISTANCE,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS
        )
        print(f"New waypoints count: {len(waypoints)}")
        
        # Step 2: Add detour waypoints
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=self.NUM_INTERMEDIATE_POINTS,
            angle_threshold=self.ANGLE_THRESHOLD,
            detour_distance=self.DETOUR_DISTANCE
        )
        
        # Step 3: Apply collision avoidance
        _, waypoints = self._avoid_collisions(
            waypoints,
            obs['obstacles_pos'],
            self.OBSTACLE_SAFETY_DISTANCE
        )
        
        # Step 4: Generate new trajectory
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        # Update visualization with new trajectory
        if self.visualization and self.visualizer is not None:
            self.visualizer.update(
                trajectory=self.trajectory,
                trajectory_duration=self.TRAJECTORY_DURATION
            )

    def _extract_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extract complete local coordinate frames for each gate.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [w, x, y, z].
            
        Returns:
            Tuple of (normals, y_axes, z_axes) where:
            - normals: Gate normal vectors (x-axis in local frame, penetration direction)
            - y_axes: Gate width direction (left-right in local frame)
            - z_axes: Gate height direction (up-down in local frame)
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]  # First column: normal (x-axis)
        y_axes = rotation_matrices[:, :, 1]   # Second column: width (y-axis)
        z_axes = rotation_matrices[:, :, 2]   # Third column: height (z-axis)
        
        return normals, y_axes, z_axes

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        """Add detour waypoints for gates that require backtracking.
        
        Simplified version: always detour to the right (+y direction) when backtracking is detected.
        
        Args:
            waypoints: Original waypoints array with shape (N, 3).
            gate_positions: Positions of all gates.
            gate_normals: Normal vectors (x-axes) of all gates.
            gate_y_axes: Y-axes (width direction, left-right) of all gates.
            gate_z_axes: Z-axes (height direction, up-down) of all gates.
            num_intermediate_points: Number of waypoints generated per gate.
            angle_threshold: Angle threshold in degrees for detecting backtracking.
            detour_distance: Distance from gate center for detour waypoint.
            
        Returns:
            Modified waypoints array with detour waypoints inserted.
        """
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)  # Convert to list for easier insertion
        
        # Track how many waypoints we've inserted (affects subsequent indices)
        inserted_count = 0
        
        # === DEBUG: Store analysis results for each gate pair ===
        self._debug_detour_analysis = []  # List of dicts with debug info for each gate pair
        
        # === DEBUG: Store all detour waypoints that were actually added ===
        self._debug_detour_waypoints_added = []  # List of (gate_index, waypoint_coords) tuples
        
        print("\n=== Detour Waypoint Analysis (Simplified: Right-side only) ===")
        
        # Check each pair of consecutive gates
        for i in range(num_gates - 1):
            # === DEBUG: Create debug dict for this gate pair ===
            debug_info = {
                'gate_i': i,
                'gate_i_plus_1': i + 1,
            }
            
            # Calculate indices accounting for previously inserted waypoints
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            # === DEBUG: Store indices ===
            debug_info['last_idx_gate_i'] = last_idx_gate_i
            debug_info['first_idx_gate_i_plus_1'] = first_idx_gate_i_plus_1
            
            # Get the two waypoints
            p1 = waypoints_list[last_idx_gate_i]          # Last point of gate i (+0.5m along normal)
            p2 = waypoints_list[first_idx_gate_i_plus_1]  # First point of gate i+1 (-0.5m along normal)
            
            # === DEBUG: Store waypoints ===
            debug_info['p1_last_of_gate_i'] = p1.copy()
            debug_info['p2_first_of_gate_i_plus_1'] = p2.copy()
            
            # Calculate vector from p1 to p2
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            # === DEBUG: Store vector ===
            debug_info['vector_p1_to_p2'] = v.copy()
            debug_info['vector_norm'] = v_norm
            
            if v_norm < 1e-6:
                print(f"\nGate {i} -> Gate {i+1}: Vector too short, skipping")
                debug_info['skipped'] = True
                debug_info['skip_reason'] = 'vector_too_short'
                self._debug_detour_analysis.append(debug_info)
                continue
            
            # Calculate angle between this vector and gate i's normal
            normal_i = gate_normals[i]
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            # === DEBUG: Store angle calculation ===
            debug_info['gate_i_normal'] = normal_i.copy()
            debug_info['cos_angle'] = cos_angle
            debug_info['angle_degrees'] = angle_deg
            debug_info['angle_threshold'] = angle_threshold
            
            print(f"\nGate {i} -> Gate {i+1}:")
            print(f"  Vector length: {v_norm:.3f}m")
            print(f"  Angle with gate {i} normal: {angle_deg:.1f}°")
            
            # Check if backtracking is detected (angle > threshold means going backwards)
            if angle_deg > angle_threshold:
                print(f"  ⚠️  BACKTRACKING detected! Determining detour direction...")
                
                # === DEBUG: Mark as needing detour ===
                debug_info['needs_detour'] = True
                
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                
                # === DEBUG: Store gate coordinate frame ===
                debug_info['gate_i_center'] = gate_center.copy()
                debug_info['gate_i_y_axis'] = y_axis.copy()
                debug_info['gate_i_z_axis'] = z_axis.copy()
                debug_info['detour_distance'] = detour_distance
                                
                # Step 1: Project vector v onto gate plane (remove normal component)
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                # === DEBUG: Store projection ===
                debug_info['v_projection_on_gate_plane'] = v_proj.copy()
                debug_info['v_projection_norm'] = v_proj_norm
                
                if v_proj_norm < 1e-6:
                    # Projection is too small, default to right side
                    print(f"  Warning: Projection too small, defaulting to RIGHT side")
                    detour_direction_vector = y_axis
                    detour_direction_name = 'right (+y_axis) [default]'
                    proj_angle_deg = 0.0
                else:
                    # Step 2: Calculate components in local coordinate system
                    v_proj_y = np.dot(v_proj, y_axis)  # Left-right component
                    v_proj_z = np.dot(v_proj, z_axis)  # Up-down component
                    
                    # === DEBUG: Store local components ===
                    debug_info['v_proj_y_component'] = v_proj_y
                    debug_info['v_proj_z_component'] = v_proj_z
                    
                    # Step 3: Calculate angle in gate plane
                    # angle = 0° means +y direction (right)
                    # angle = 90° means +z direction (up)
                    # angle = ±180° means -y direction (left)
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    # === DEBUG: Store angle ===
                    debug_info['projection_angle_degrees'] = proj_angle_deg
                    
                    # Step 4: Determine detour direction based on angle
                    if -90 <= proj_angle_deg < 45:
                        # Right side
                        detour_direction_vector = y_axis
                        detour_direction_name = 'right (+y_axis)'
                    elif 45 <= proj_angle_deg < 135:
                        # Top side
                        detour_direction_vector = z_axis
                        detour_direction_name = 'top (+z_axis)'
                    else:  # angle >= 135 or angle < -90
                        # Left side
                        detour_direction_vector = -y_axis
                        detour_direction_name = 'left (-y_axis)'
                    
                    print(f"  Projection angle: {proj_angle_deg:.1f}° → Detour direction: {detour_direction_name}")
                
                # === DEBUG: Store direction choice ===
                debug_info['detour_direction_vector'] = detour_direction_vector.copy()
                debug_info['detour_direction_name'] = detour_direction_name
                debug_info['projection_angle_degrees'] = proj_angle_deg
                
                # Step 5: Calculate detour waypoint
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                
                # === DEBUG: Store detour waypoint ===
                debug_info['detour_waypoint'] = detour_waypoint.copy()
                debug_info['detour_direction'] = detour_direction_name
                
                # Step 6: Insert the detour waypoint into the waypoints list  
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
    
                # === DEBUG: Store insertion info ===
                debug_info['insert_position'] = insert_position
                debug_info['inserted'] = True
                
                print(f"  Inserted detour waypoint at index {insert_position}")
                print(f"  Detour coords: [{detour_waypoint[0]:.3f}, {detour_waypoint[1]:.3f}, {detour_waypoint[2]:.3f}]")
            else:
                print(f"  ✓ No backtracking detected, proceeding normally")
                debug_info['needs_detour'] = False
                debug_info['inserted'] = False
            
            # === DEBUG: Store current inserted count ===
            debug_info['total_inserted_so_far'] = inserted_count
            
            # Add this gate pair's debug info to the list
            self._debug_detour_analysis.append(debug_info)
        
        # === DEBUG: Store final summary ===
        self._debug_detour_summary = {
            'total_detours_added': inserted_count,
            'original_waypoint_count': len(waypoints),
            'final_waypoint_count': len(waypoints_list),
            'num_gate_pairs_checked': num_gates - 1,
            'detour_waypoints': self._debug_detour_waypoints_added  # Also include in summary
        }
        
        print(f"\n=== Total detour waypoints added: {inserted_count} ===")
        
        # === DEBUG: Print all added detour waypoints for easy viewing ===
        if self._debug_detour_waypoints_added:
            print("\n=== Added Detour Waypoints ===")
            for idx, detour_info in enumerate(self._debug_detour_waypoints_added):
                coords = detour_info['waypoint_coords']
                gate_idx = detour_info['gate_index']
                print(f"  Detour #{idx+1} (Gate {gate_idx}): "
                    f"[{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}] - {detour_info['direction']}")
        print()
        
        return np.array(waypoints_list)
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state for the drone.
        
        Args:
            obs: Current observation of environment state.
            info: Optional additional information.
            
        Returns:
            13D state vector [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
            Only position (first 3 elements) is set; rest are zeros for low-level controller.
        """
        # Compute current time along trajectory
        current_time = min(self._time_step / self._control_frequency, self.TRAJECTORY_DURATION)
        
        # Sample target position from trajectory
        target_position = self.trajectory(current_time)
        
        # Periodic logging
        if self._time_step % self.LOG_INTERVAL == 0:
            print(f"Time: {current_time:.2f}s | "
                  f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # Check for environment changes and replan if necessary
        if self._detect_environment_change(obs):
            self._replan_trajectory(obs, current_time)
        
        # Update interactive visualization with new drone position, gate status, and real positions
        if self.visualization and self.visualizer is not None:
            self.visualizer.update(
                drone_position=obs['pos'],
                gate_detected_status=self._gate_detected_flags,
                gate_real_positions=self._gate_real_positions
            )
        # Check if trajectory is complete
        if current_time >= self.TRAJECTORY_DURATION:
            self._is_finished = True
        
        # Draw trajectory in simulation environment (if available)
        try:
            draw_line(self.env, self.trajectory(self.trajectory.x), 
                     rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except (AttributeError, TypeError):
            pass  # env not available or draw_line not supported
        
        # Return 13D state with only position filled
        return np.concatenate((target_position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        """Called after each environment step.
        
        Args:
            action: Action taken.
            obs: Resulting observation.
            reward: Reward received.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Additional information.
            
        Returns:
            True if controller is finished, False otherwise.
        """
        self._time_step += 1
        return self._is_finished

    # ==================== Utility Methods for External Use ====================
    
    def get_trajectory_function(self) -> CubicSpline:
        """Get the trajectory spline function.
        
        Returns:
            CubicSpline object representing the trajectory.
        """
        return self.trajectory

    def get_trajectory_waypoints(self) -> NDArray[np.floating]:
        """Get discrete waypoints sampled from trajectory at control frequency.
        
        Returns:
            Array of waypoints with shape (num_timesteps, 3).
        """
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        return self.trajectory(time_samples)

    def set_time_step(self, time_step: int) -> None:
        """Set the current time step (for testing/debugging).
        
        Args:
            time_step: New time step value.
        """
        self._time_step = time_step
    
    # ==================== Visualization Export Methods ====================
    
    def save_visualization_html(self, filepath: str = "trajectory_demo.html") -> None:
        """Save current visualization as interactive HTML file.
        
        This creates a standalone HTML file that can be opened in any browser
        and allows full interaction (rotate, zoom, pan) without requiring Python.
        
        Args:
            filepath: Path where to save the HTML file.
        
        Example:
            controller.save_visualization_html("my_race_demo.html")
        """
        if self.visualizer is None:
            print("Warning: No visualizer available. Enable visualization first.")
            return
        
        try:
            self.visualizer.save_html(filepath, auto_open=False)
            print(f"✓ Interactive demo saved! Open '{filepath}' in your browser to view.")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def save_trajectory_data(
        self, 
        filepath: str = "trajectory_data.json",
        format: str = 'json'
    ) -> None:
        """Save trajectory data for analysis.
        
        Args:
            filepath: Path where to save the data file.
            format: Format to use ('json' or 'csv').
        
        Example:
            controller.save_trajectory_data("trajectory.json")
            controller.save_trajectory_data("trajectory.csv", format='csv')
        """
        if self.visualizer is None:
            print("Warning: No visualizer available. Enable visualization first.")
            return
        
        try:
            self.visualizer.save_trajectory_data(filepath, format=format)
        except Exception as e:
            print(f"Error saving trajectory data: {e}")
