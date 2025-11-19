"""
Controller that follows a pre-defined trajectory with gate detection and obstacle avoidance.
It uses a cubic spline interpolation to generate a smooth trajectory through waypoints.
When gates are detected with changed positions, it updates the trajectory.
When obstacles are detected, it applies lateral shifts to avoid collisions.
PID control is used for accurate trajectory tracking.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller with gate detection, obstacle avoidance, and PID tracking."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
        self._sensor_range = config.env.sensor_range
        
        # Store initial drone position
        self._initial_pos = obs["pos"].copy()
        
        # Store nominal gate information from config
        self._gates = []
        for gate in config.env.track.gates:
            gate_pos = np.array(gate["pos"], dtype=float)
            gate_rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", gate_rpy)
            gate_normal = rot.as_matrix()[:, 0]  # x-axis points through gate
            gate_y_axis = rot.as_matrix()[:, 1]  # y-axis width direction
            gate_z_axis = rot.as_matrix()[:, 2]  # z-axis height direction
            
            self._gates.append({
                "nominal_pos": gate_pos.copy(),
                "nominal_normal": gate_normal.copy(),
                "nominal_y_axis": gate_y_axis.copy(),
                "nominal_z_axis": gate_z_axis.copy(),
                "nominal_rpy": gate_rpy.copy(),
                "detected_pos": None,
                "detected_normal": None,
                "detected_y_axis": None,
                "detected_z_axis": None,
                "passed": False,  # Track if gate has been passed
            })
        
        self._num_gates = len(self._gates)
        
        # Waypoint generation parameters
        self._approach_dist = 0.5
        self._num_intermediate_points = 5
        self._angle_threshold = 120.0  # Degrees for backtracking detection
        self._detour_distance = 0.65   # Distance for detour waypoints

        # Gate corridor parameters
        self._gate_corridor_width = 0.3
        self._gate_pole_radius = 0.08  # Radius of gate support poles
        self._gate_pole_safety_margin = 0.15  # Extra safety margin

        # Pitch lock threshold (deg + rad)
        self._pitch_lock_thresh_deg = 7.5
        self._pitch_lock_thresh = np.deg2rad(self._pitch_lock_thresh_deg)        

        # Obstacle avoidance parameters
        self._obstacle_safety_distance = 0.35  # Increased from 0.3
        
        # PID gains for trajectory tracking
        self._kp_pos = 2.8      # Increased from 2.5
        self._kd_pos = 1.8      # Increased from 1.5
        self._ki_pos = 0.15     # Increased from 0.1
        
        # PID state
        self._pos_error_integral = np.zeros(3, dtype=np.float32)
        self._integral_limit = 2.0  # Anti-windup limit
        
        # Timing
        self._t_total = 25.0  # Total flight time
        
        # Initialize tick counter BEFORE building trajectory
        self._tick = 0
        self._finished = False
        self._last_update_tick = -100  # Prevent frequent trajectory updates
        self._current_target_gate = 0  # Track which gate we're heading towards
        
        # Build initial trajectory
        self._build_trajectory()
        
        print("=" * 60)
        print("Improved Controller v2 - Enhanced Gate & Pole Avoidance")
        print(f"Gates: {self._num_gates}")
        print(f"Total waypoints: {len(self._waypoints)}")
        print(f"PID Gains - Kp: {self._kp_pos}, Kd: {self._kd_pos}, Ki: {self._ki_pos}")
        print("=" * 60)

    def _extract_gate_coordinate_frames(
        self, 
        gate_idx: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extract complete local coordinate frame for a gate.
        
        Args:
            gate_idx: Index of the gate.
            
        Returns:
            Tuple of (position, normal, y_axis, z_axis) where:
            - position: Gate center position
            - normal: Gate normal vector (x-axis, penetration direction)
            - y_axis: Gate width direction (left-right)
            - z_axis: Gate height direction (up-down)
        """
        gate = self._gates[gate_idx]
        
        # Use detected if available, else nominal
        pos = gate["detected_pos"] if gate["detected_pos"] is not None else gate["nominal_pos"]
        normal = gate["detected_normal"] if gate["detected_normal"] is not None else gate["nominal_normal"]
        y_axis = gate["detected_y_axis"] if gate["detected_y_axis"] is not None else gate["nominal_y_axis"]
        z_axis = gate["detected_z_axis"] if gate["detected_z_axis"] is not None else gate["nominal_z_axis"]
        
        return pos, normal, y_axis, z_axis

    def calc_waypoints_from_gates(
        self,
        initial_position: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Generate waypoints based on gate positions.
        
        Creates multiple waypoints around each gate to ensure smooth passage.
        
        Args:
            initial_position: Starting position of the drone.
            
        Returns:
            Array of waypoints including initial position.
        """
        waypoints_per_gate = []
        
        for i in range(self._num_intermediate_points):
            # Interpolate from -approach_distance to +approach_distance
            offset = -self._approach_dist + (i / (self._num_intermediate_points - 1)) * 2 * self._approach_dist
            
            gate_waypoints = []
            for gate_idx in range(self._num_gates):
                pos, normal, _, _ = self._extract_gate_coordinate_frames(gate_idx)
                gate_waypoints.append(pos + offset * normal)
            
            waypoints_per_gate.append(np.array(gate_waypoints))
        
        # Reshape to (num_gates * num_intermediate_points, 3)
        waypoints = np.concatenate(waypoints_per_gate, axis=1)
        waypoints = waypoints.reshape(self._num_gates, self._num_intermediate_points, 3).reshape(-1, 3)
        
        # Prepend initial position
        waypoints = np.vstack([initial_position, waypoints])
        
        return waypoints

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Add detour waypoints for gates that require backtracking.
        
        Args:
            waypoints: Original waypoints array with shape (N, 3).
            
        Returns:
            Modified waypoints array with detour waypoints inserted.
        """
        waypoints_list = list(waypoints)
        inserted_count = 0
        
        print("\n=== Detour Waypoint Analysis ===")
        
        # Check each pair of consecutive gates
        for i in range(self._num_gates - 1):
            # Calculate indices accounting for previously inserted waypoints
            last_idx_gate_i = 1 + (i + 1) * self._num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * self._num_intermediate_points + inserted_count
            
            # Get the two waypoints
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            
            # Calculate vector from p1 to p2
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-6:
                print(f"\nGate {i} -> Gate {i+1}: Vector too short, skipping")
                continue
            
            # Get gate i's coordinate frame
            gate_center, normal_i, y_axis, z_axis = self._extract_gate_coordinate_frames(i)
            
            # Calculate angle between vector and gate i's normal
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            print(f"\nGate {i} -> Gate {i+1}:")
            print(f"  Vector length: {v_norm:.3f}m")
            print(f"  Angle with gate {i} normal: {angle_deg:.1f}°")
            
            # Check if backtracking is detected
            if angle_deg > self._angle_threshold:
                print(f"  ⚠️  BACKTRACKING detected! Determining detour direction...")
                
                # Project vector v onto gate plane
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                if v_proj_norm < 1e-6:
                    # Default to right side
                    detour_direction_vector = y_axis
                    detour_direction_name = 'right (+y_axis) [default]'
                    proj_angle_deg = 0.0
                else:
                    # Calculate components in local coordinate system
                    v_proj_y = np.dot(v_proj, y_axis)
                    v_proj_z = np.dot(v_proj, z_axis)
                    
                    # Calculate angle in gate plane
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    # Determine detour direction based on angle
                    if -90 <= proj_angle_deg < 45:
                        detour_direction_vector = y_axis
                        detour_direction_name = 'right (+y_axis)'
                    elif 45 <= proj_angle_deg < 135:
                        detour_direction_vector = z_axis
                        detour_direction_name = 'top (+z_axis)'
                    else:
                        detour_direction_vector = -y_axis
                        detour_direction_name = 'left (-y_axis)'
                    
                    print(f"  Projection angle: {proj_angle_deg:.1f}° → Detour direction: {detour_direction_name}")
                
                # Calculate detour waypoint
                detour_waypoint = gate_center + self._detour_distance * detour_direction_vector
                
                # Insert the detour waypoint
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
                
                print(f"  Inserted detour waypoint at index {insert_position}")
                print(f"  Detour coords: [{detour_waypoint[0]:.3f}, {detour_waypoint[1]:.3f}, {detour_waypoint[2]:.3f}]")
            else:
                print(f"  ✓ No backtracking detected, proceeding normally")
        
        print(f"\n=== Total detour waypoints added: {inserted_count} ===\n")
        
        return np.array(waypoints_list)

    def _get_gate_pole_positions(self, gate_idx: int) -> list[NDArray[np.floating]]:
        """Get the positions of the support poles for a gate.
        
        Gates have support poles at the bottom. This function calculates their positions.
        
        Args:
            gate_idx: Index of the gate.
            
        Returns:
            List of pole positions (typically 2 poles per gate).
        """
        pos, normal, y_axis, z_axis = self._extract_gate_coordinate_frames(gate_idx)
        
        # Gate dimensions (approximate)
        gate_width = 0.45  # Half-width of gate opening
        
        # Poles are at the bottom of the gate, on either side
        # Assuming poles are at ground level (z=0) or slightly above
        pole_height = pos[2]  # Use gate center height as reference
        
        # Calculate pole positions on left and right sides
        left_pole = pos + gate_width * y_axis - (pole_height - 0.05) * z_axis
        right_pole = pos - gate_width * y_axis - (pole_height - 0.05) * z_axis
        
        return [left_pole, right_pole]

    def _avoid_gate_poles(
        self,
        waypoints: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Modify trajectory to avoid gate support poles.
        
        Args:
            waypoints: Original waypoints.
            
        Returns:
            Modified waypoints array.
        """
        # Collect all gate pole positions
        all_pole_positions = []
        for gate_idx in range(self._num_gates):
            pole_positions = self._get_gate_pole_positions(gate_idx)
            all_pole_positions.extend(pole_positions)
        
        if len(all_pole_positions) == 0:
            return waypoints
        
        pole_positions = np.array(all_pole_positions)
        
        # Generate initial trajectory
        pos_spline, _, _ = self._generate_trajectory(self._t_total, waypoints)
        
        # Sample trajectory at high resolution
        time_samples = np.linspace(0, self._t_total, int(self._freq * self._t_total))
        trajectory_points = pos_spline(time_samples)
        
        modified_points = []
        safety_distance = self._gate_pole_radius + self._gate_pole_safety_margin
        
        # Process each trajectory point
        for i, point in enumerate(trajectory_points):
            modified_point = point.copy()
            
            # Check distance to all poles
            for pole_pos in pole_positions:
                # Only check XY distance (poles are vertical)
                dist_xy = np.linalg.norm(point[:2] - pole_pos[:2])
                
                if dist_xy < safety_distance:
                    # Push point away from pole
                    direction = point[:2] - pole_pos[:2]
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 1e-6:
                        direction = direction / direction_norm
                        push_distance = safety_distance - dist_xy
                        modified_point[:2] += direction * push_distance * 1.2  # Extra push factor
            
            modified_points.append(modified_point)
        
        return np.array(modified_points)

    def _generate_trajectory(
        self, 
        duration: float, 
        waypoints: NDArray[np.floating]
    ) -> tuple[CubicSpline, CubicSpline, CubicSpline]:
        """Generate cubic spline trajectory through waypoints.
        
        Uses arc-length parameterization for more uniform velocity distribution.
        
        Args:
            duration: Total time duration for the trajectory.
            waypoints: Array of 3D waypoints.
            
        Returns:
            Tuple of (position_spline, velocity_spline, acceleration_spline).
        """
        # Calculate segment lengths
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Cumulative arc length
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Parameterize time by arc length for uniform velocity
        time_parameters = cumulative_arc_length / cumulative_arc_length[-1] * duration
        
        pos_spline = CubicSpline(time_parameters, waypoints)
        vel_spline = pos_spline.derivative(nu=1)
        acc_spline = pos_spline.derivative(nu=2)
        
        return pos_spline, vel_spline, acc_spline

    def _avoid_collisions(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Modify trajectory to avoid collisions with obstacles.
        
        Args:
            waypoints: Original waypoints.
            obstacle_positions: Positions of obstacles.
            
        Returns:
            Modified waypoints array.
        """
        if len(obstacle_positions) == 0:
            return waypoints
        
        # Generate initial trajectory
        pos_spline, _, _ = self._generate_trajectory(self._t_total, waypoints)
        
        # Sample trajectory at high resolution
        time_samples = np.linspace(0, self._t_total, int(self._freq * self._t_total))
        trajectory_points = pos_spline(time_samples)
        
        # Process each obstacle
        for obstacle_position in obstacle_positions:
            collision_free_times = []
            collision_free_waypoints = []
            
            is_inside_obstacle = False
            entry_index = None
            
            for i, point in enumerate(trajectory_points):
                # Check distance in XY plane only
                distance_xy = np.linalg.norm(obstacle_position[:2] - point[:2])
                
                if distance_xy < self._obstacle_safety_distance:
                    if not is_inside_obstacle:
                        is_inside_obstacle = True
                        entry_index = i
                        
                elif is_inside_obstacle:
                    # Exiting obstacle zone - compute avoidance waypoint
                    exit_index = i
                    is_inside_obstacle = False
                    
                    # Compute new waypoint direction
                    entry_point = trajectory_points[entry_index]
                    exit_point = trajectory_points[exit_index]
                    
                    entry_direction = entry_point[:2] - obstacle_position[:2]
                    exit_direction = exit_point[:2] - obstacle_position[:2]
                    avoidance_direction = entry_direction + exit_direction
                    avoidance_direction /= np.linalg.norm(avoidance_direction)
                    
                    # Place new waypoint at safety distance
                    new_position_xy = obstacle_position[:2] + avoidance_direction * self._obstacle_safety_distance
                    new_position_z = (entry_point[2] + exit_point[2]) / 2
                    new_waypoint = np.concatenate([new_position_xy, [new_position_z]])
                    
                    # Add at midpoint time
                    collision_free_times.append((time_samples[entry_index] + time_samples[exit_index]) / 2)
                    collision_free_waypoints.append(new_waypoint)
                    
                else:
                    # Point is safe
                    collision_free_times.append(time_samples[i])
                    collision_free_waypoints.append(point)
            
            # Update for next obstacle
            time_samples = np.array(collision_free_times)
            trajectory_points = np.array(collision_free_waypoints)
        
        return trajectory_points

    def _is_in_gate_corridor(self, p: NDArray[np.floating]) -> bool:
        """Return True if point p is inside any gate's corridor segment."""
        for i in range(self._num_gates):
            pos, normal, _, _ = self._extract_gate_coordinate_frames(i)

            # Signed distance along the gate normal
            s = float(np.dot(p - pos, normal))

            # Inside the longitudinal bounds of the corridor?
            if -self._approach_dist <= s <= self._approach_dist:
                # Lateral distance to the gate normal line
                lateral_vec = (p - pos) - s * normal
                if np.linalg.norm(lateral_vec) <= self._gate_corridor_width:
                    return True
        return False

    def _build_trajectory(self):
        """Build spline trajectory through all gates with detour waypoints."""
        # Step 1: Generate basic waypoints
        waypoints = self.calc_waypoints_from_gates(self._initial_pos)
        print(f"Initial waypoints count: {len(waypoints)}")
        
        # Step 2: Add detour waypoints for backtracking gates
        waypoints = self._add_detour_waypoints(waypoints)
        print(f"Waypoints after detour: {len(waypoints)}")
        
        # Step 3: Apply collision avoidance for obstacles
        if hasattr(self, '_obstacle_positions') and len(self._obstacle_positions) > 0:
            waypoints = self._avoid_collisions(waypoints, self._obstacle_positions)
            print(f"Waypoints after obstacle avoidance: {len(waypoints)}")
        
        # Step 4: Apply collision avoidance for gate poles
        waypoints = self._avoid_gate_poles(waypoints)
        print(f"Waypoints after gate pole avoidance: {len(waypoints)}")
        
        self._waypoints = waypoints
        
        # Step 5: Generate smooth trajectory
        self._des_pos_spline, self._des_vel_spline, self._des_acc_spline = \
            self._generate_trajectory(self._t_total, self._waypoints)
        
        print(f"[Tick {self._tick}] Trajectory built with {len(self._waypoints)} waypoints")

    def _update_current_target_gate(self, current_pos: NDArray[np.floating]):
        """Update which gate the drone is currently targeting."""
        for i in range(self._current_target_gate, self._num_gates):
            gate_pos, gate_normal, _, _ = self._extract_gate_coordinate_frames(i)
            
            # Check if we've passed this gate
            to_gate = gate_pos - current_pos
            # If we're behind the gate (negative dot product), we've passed it
            if np.dot(to_gate, gate_normal) < -0.2:
                self._gates[i]["passed"] = True
                self._current_target_gate = min(i + 1, self._num_gates - 1)
                print(f"[Tick {self._tick}] Passed gate {i}, now targeting gate {self._current_target_gate}")

    def _update_gate_detections(self, obs: dict[str, NDArray[np.floating]]):
        """Update gate positions when detected within sensor range."""
        if "gates_pos" not in obs or "gates_quat" not in obs:
            return
        
        current_pos = obs["pos"]
        trajectory_updated = False
        
        # Update current target gate
        self._update_current_target_gate(current_pos)
        
        for i, gate_pos in enumerate(obs["gates_pos"]):
            # Skip gates we've already passed
            if self._gates[i]["passed"]:
                continue
            
            # Check if gate is within sensor range
            dist = np.linalg.norm(gate_pos - current_pos)
            if dist > self._sensor_range:
                continue
            
            # Determine reference position for change detection
            if self._gates[i]["detected_pos"] is None:
                # First-time detection: compare with nominal
                reference_pos = self._gates[i]["nominal_pos"]
                is_first_detection = True
            else:
                # Subsequent detection: compare with last detected
                reference_pos = self._gates[i]["detected_pos"]
                is_first_detection = False
            
            # Calculate position change
            pos_change = np.linalg.norm(gate_pos - reference_pos)
            
            # Update only if significant change
            should_update = pos_change > 0.05  # 5cm threshold
            
            if is_first_detection and not should_update:
                should_update = True  # Force update on first detection
            
            if should_update:
                # Get gate orientation and extract complete coordinate frame
                rot = R.from_quat(obs["gates_quat"][i])
                det_euler = rot.as_euler("xyz")
                nom_pitch = self._gates[i]["nominal_rpy"][1]
                
                # Pitch lock
                if abs(det_euler[1] - nom_pitch) < self._pitch_lock_thresh:
                    det_euler[1] = nom_pitch
                    rot = R.from_euler("xyz", det_euler)
                
                rot_matrix = rot.as_matrix()
                gate_normal = rot_matrix[:, 0]
                gate_y_axis = rot_matrix[:, 1]
                gate_z_axis = rot_matrix[:, 2]
                
                # Update detection
                self._gates[i]["detected_pos"] = gate_pos.copy()
                self._gates[i]["detected_normal"] = gate_normal.copy()
                self._gates[i]["detected_y_axis"] = gate_y_axis.copy()
                self._gates[i]["detected_z_axis"] = gate_z_axis.copy()
                
                # Log the update
                if is_first_detection:
                    print(f"[Tick {self._tick}] Gate {i} FIRST detection: {pos_change:.3f}m offset")
                else:
                    print(f"[Tick {self._tick}] Gate {i} UPDATED: {pos_change:.3f}m change detected")
                
                # Mark for trajectory rebuild if significant change
                if pos_change > 0.05:
                    trajectory_updated = True
        
        # Rebuild trajectory if any gate changed significantly
        if trajectory_updated:
            print(f"[Tick {self._tick}] Rebuilding trajectory with updated gates...")
            self._build_trajectory()
            self._last_update_tick = self._tick
            self._pos_error_integral = np.zeros(3, dtype=np.float32)

    def _lateral_obstacle_avoidance(
        self, 
        des_pos: NDArray[np.floating],
        current_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply lateral shift to avoid obstacles (real-time adjustment)."""
        if len(obstacles_pos) == 0:
            return des_pos
        
        # Get trajectory direction (horizontal only)
        traj_direction = des_vel[:2].copy()
        if np.linalg.norm(traj_direction) < 0.01:
            traj_direction = (des_pos[:2] - current_pos[:2])
        
        if np.linalg.norm(traj_direction) < 0.01:
            return des_pos
        
        traj_direction = traj_direction / np.linalg.norm(traj_direction)
        
        # Perpendicular direction
        perp_direction = np.array([-traj_direction[1], traj_direction[0]])
        
        total_offset = np.zeros(2)
        avoidance_distance = 0.7  # Increased from 0.6
        obstacle_radius = 0.15
        safety_margin = 0.3  # Increased from 0.25
        
        for obs_pos in obstacles_pos:
            dist_to_drone = np.linalg.norm(obs_pos[:2] - current_pos[:2])
            dist_to_desired = np.linalg.norm(obs_pos[:2] - des_pos[:2])
            
            if min(dist_to_drone, dist_to_desired) > avoidance_distance:
                continue
            
            # Check if obstacle is along trajectory
            to_obs = obs_pos[:2] - current_pos[:2]
            projection = np.dot(to_obs, traj_direction)
            
            if projection < -0.2:
                continue
            
            # Lateral distance from trajectory line
            lateral_dist = abs(np.dot(to_obs, perp_direction))
            required_clearance = obstacle_radius + safety_margin
            
            if lateral_dist < required_clearance:
                obstacle_side = np.sign(np.dot(to_obs, perp_direction))
                avoidance_dist = required_clearance - lateral_dist
                strength = 1.5 * avoidance_dist / required_clearance
                offset = -obstacle_side * perp_direction * strength * 0.25  # Increased from 0.20
                total_offset += offset
        
        # Apply offset
        modified_pos = des_pos.copy()
        modified_pos[:2] += total_offset
        
        return modified_pos

    def _compute_pid_control(
        self,
        current_pos: NDArray[np.floating],
        current_vel: NDArray[np.floating],
        des_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        des_acc: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute PID control for trajectory tracking."""
        # Position error
        pos_error = des_pos - current_pos
        
        # Velocity error
        vel_error = des_vel - current_vel
        
        # Integral with anti-windup
        self._pos_error_integral += pos_error * self._dt
        self._pos_error_integral = np.clip(
            self._pos_error_integral,
            -self._integral_limit,
            self._integral_limit
        )
        
        # PID control law
        acc_pid = (
            des_acc +
            self._kp_pos * pos_error +
            self._kd_pos * vel_error +
            self._ki_pos * self._pos_error_integral
        )
        
        return acc_pid

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state with gate detection, obstacle avoidance, and PID."""
        
        # Store obstacle positions for trajectory building
        if "obstacles_pos" in obs:
            self._obstacle_positions = obs["obstacles_pos"]
        
        # IMPROVED: More aggressive gate detection updates
        # Check more frequently, especially when approaching gates
        dist_to_next_gate = float('inf')
        if self._current_target_gate < self._num_gates:
            gate_pos, _, _, _ = self._extract_gate_coordinate_frames(self._current_target_gate)
            dist_to_next_gate = np.linalg.norm(gate_pos - obs["pos"])
        
        # Update more frequently when close to next gate
        early_phase = self._tick < int(3.0 * self._freq)
        approaching_gate = dist_to_next_gate < 1.5
        
        if early_phase:
            update_gap = 3  # Very frequent updates in early phase
        elif approaching_gate:
            update_gap = 5  # Frequent updates when approaching gate
        else:
            update_gap = 10  # Normal updates otherwise
        
        if self._tick - self._last_update_tick > update_gap:
            self._update_gate_detections(obs)
        
        # Get desired state from spline
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True
        
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_acc = self._des_acc_spline(t)
        
        # Apply real-time lateral obstacle avoidance (outside gate corridors)
        if "obstacles_pos" in obs and len(obs["obstacles_pos"]) > 0:
            if not self._is_in_gate_corridor(des_pos):
                des_pos = self._lateral_obstacle_avoidance(
                    des_pos,
                    obs["pos"],
                    des_vel,
                    obs["obstacles_pos"]
                )
        
        # Compute PID control
        des_acc_pid = self._compute_pid_control(
            obs["pos"],
            obs["vel"],
            des_pos,
            des_vel,
            des_acc
        )
        
        # State action
        action = np.concatenate([
            des_pos,
            des_vel,
            des_acc_pid,
            np.zeros(4, dtype=np.float32)
        ], dtype=np.float32)
        
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state for new episode."""
        self._tick = 0
        self._finished = False
        self._last_update_tick = -100
        self._current_target_gate = 0
        
        # Reset PID state
        self._pos_error_integral = np.zeros(3, dtype=np.float32)
        
        # Reset gate detections
        for gate in self._gates:
            gate["detected_pos"] = None
            gate["detected_normal"] = None
            gate["detected_y_axis"] = None
            gate["detected_z_axis"] = None
            gate["passed"] = False
        
        # Rebuild trajectory
        self._build_trajectory()
        
        print("\n" + "=" * 60)
        print("Episode Reset - Trajectory rebuilt")
        print("=" * 60 + "\n")