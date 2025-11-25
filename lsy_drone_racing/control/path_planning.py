"""
Path Planning Module for Drone Racing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

try:
    from lsy_drone_racing.utils.trajectory_visualizer import TrajectoryVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("Warning: TrajectoryVisualizer not available. Visualization disabled.")
    VISUALIZER_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Data Classes for Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for path planning."""
    # Waypoint generation
    approach_distance: float = 0.5          # Distance before/after gate center
    num_intermediate_points: int = 5        # Points around each gate
    
    # Detour settings
    angle_threshold: float = 120.0          # Angle threshold for backtracking detection (degrees)
    detour_distance: float = 0.65           # Distance from gate center for detour
    
    # Obstacle avoidance
    safety_distance: float = 0.3            # Minimum distance from obstacles (meters)
    
    # Arc-length reparameterization
    arc_step: float = 0.05                  # Arc length sampling step
    arc_epsilon: float = 1e-5               # Convergence threshold
    
    # Trajectory extension
    extend_length: float = 1.0              # Extension length at trajectory end
    
    # Visualization settings
    visualization_enabled: bool = True     # Enable/disable visualization
    visualization_width: int = 1400         # Visualization window width
    visualization_height: int = 1000        # Visualization window height
    visualization_output_dir: Optional[str] = None  # Output directory for screenshots


@dataclass
class TrajectoryResult:
    """Result of trajectory planning."""
    spline: CubicSpline                     # Main trajectory spline
    arc_spline: Optional[CubicSpline]       # Arc-length parameterized spline
    waypoints: NDArray[np.floating]         # Original waypoints
    total_length: float                     # Total arc length
    gate_thetas: Optional[NDArray]          # Arc-length parameters at gates
    gate_positions: Optional[NDArray] = None  # Gate positions
    gate_normals: Optional[NDArray] = None    # Gate normals
    obstacle_positions: Optional[NDArray] = None  # Obstacle positions
    trajectory_duration: float = 30.0         # Trajectory duration


# =============================================================================
# Composite Spline for Multi-Stage Trajectories
# =============================================================================

class CompositeSpline:
    """Composite spline combining two trajectory segments."""
    
    def __init__(self, first: CubicSpline, second: CubicSpline, offset: float):
        self.trajectory_1 = first
        self.trajectory_2 = second
        self.offset = offset
        self.x = np.concatenate([first.x, second.x + offset])
    
    def __call__(self, t):
        if np.isscalar(t):
            return self.trajectory_1(t) if t < self.offset else self.trajectory_2(t - self.offset)
        return np.array([self(t_i) for t_i in t])
    
    def derivative(self, order: int = 1):
        return CompositeSpline(
            self.trajectory_1.derivative(order),
            self.trajectory_2.derivative(order),
            self.offset,
        )


# =============================================================================
# Gate Frame Utilities
# =============================================================================

class GateFrameExtractor:
    """Utility class for extracting gate coordinate frames from quaternions."""
    
    @staticmethod
    def extract_frames(
        gates_quaternions: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Extract complete local coordinate frames for each gate.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [x, y, z, w].
            
        Returns:
            Tuple of (normals, y_axes, z_axes):
            - normals: Gate normal vectors (x-axis, penetration direction)
            - y_axes: Gate width direction (left-right)
            - z_axes: Gate height direction (up-down)
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]   # First column: normal (x-axis)
        y_axes = rotation_matrices[:, :, 1]    # Second column: width (y-axis)
        z_axes = rotation_matrices[:, :, 2]    # Third column: height (z-axis)
        
        return normals, y_axes, z_axes
    
    @staticmethod
    def extract_normals(gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Extract only gate normal vectors."""
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        return rotation_matrices[:, :, 0]


# =============================================================================
# Path Planning Core
# =============================================================================

class PathPlanner:
    """
    Main path planning class combining stability and speed optimizations.
    
    This class provides methods for:
    - Generating waypoints from gate positions
    - Adding detour waypoints for backtracking gates
    - Avoiding obstacles with safety margins
    - Arc-length parameterization for uniform speed
    - Trajectory extension for MPC prediction horizon
    """
    
    def __init__(self, config: Optional[PathConfig] = None):
        """
        Initialize the path planner.
        
        Args:
            config: Path planning configuration. Uses defaults if None.
        """
        self.config = config or PathConfig()
        self._debug_info = {}
    
    # =========================================================================
    # Waypoint Generation
    # =========================================================================
    
    def generate_waypoints(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: Optional[float] = None,
        num_points: Optional[int] = None
    ) -> NDArray[np.floating]:
        """
        Generate waypoints based on gate positions.
        
        Creates multiple waypoints around each gate to ensure smooth passage.
        
        Args:
            initial_position: Starting position of the drone.
            gate_positions: Positions of all gates (N, 3).
            gate_normals: Normal vectors of all gates (N, 3).
            approach_distance: Distance before/after gate center (optional).
            num_points: Number of points around each gate (optional).
            
        Returns:
            Array of waypoints (M, 3) including initial position.
        """
        approach_dist = approach_distance or self.config.approach_distance
        n_points = num_points or self.config.num_intermediate_points
        num_gates = gate_positions.shape[0]
        
        # Create waypoints before and after each gate
        waypoints_per_gate = []
        for i in range(n_points):
            # Interpolate from -approach_distance to +approach_distance
            alpha = i / (n_points - 1) if n_points > 1 else 0.0
            offset = -approach_dist + alpha * 2 * approach_dist
            waypoints_per_gate.append(gate_positions + offset * gate_normals)
        
        # Reshape to (num_gates * n_points, 3)
        stacked = np.stack(waypoints_per_gate, axis=1)
        waypoints = stacked.reshape(-1, 3)
        
        # Prepend initial position
        return np.vstack([initial_position, waypoints])
    
    # =========================================================================
    # Detour Waypoint Insertion
    # =========================================================================
    
    def add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: Optional[int] = None,
        angle_threshold: Optional[float] = None,
        detour_distance: Optional[float] = None
    ) -> NDArray[np.floating]:
        """
        Add detour waypoints for gates that require backtracking.
        
        Detects when the path between consecutive gates requires going
        backwards through the current gate, and inserts detour waypoints
        to navigate around the gate.
        
        Args:
            waypoints: Original waypoints array (N, 3).
            gate_positions: Positions of all gates.
            gate_normals: Normal vectors (x-axes) of all gates.
            gate_y_axes: Y-axes (width direction) of all gates.
            gate_z_axes: Z-axes (height direction) of all gates.
            num_intermediate_points: Points per gate (for index calculation).
            angle_threshold: Angle threshold for backtracking detection (degrees).
            detour_distance: Distance from gate center for detour waypoint.
            
        Returns:
            Modified waypoints array with detour waypoints inserted.
        """
        n_points = num_intermediate_points or self.config.num_intermediate_points
        angle_thresh = angle_threshold or self.config.angle_threshold
        detour_dist = detour_distance or self.config.detour_distance
        
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)
        inserted_count = 0
        
        self._debug_info['detour_analysis'] = []
        
        for i in range(num_gates - 1):
            # Calculate indices accounting for previously inserted waypoints
            last_idx_gate_i = 1 + (i + 1) * n_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * n_points + inserted_count
            
            if last_idx_gate_i >= len(waypoints_list) or first_idx_gate_i_plus_1 >= len(waypoints_list):
                break
            
            # Get the two waypoints
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            
            # Calculate vector from p1 to p2
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-6:
                continue
            
            # Calculate angle between this vector and gate i's normal
            normal_i = gate_normals[i]
            cos_angle = np.clip(np.dot(v, normal_i) / v_norm, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            # Check if backtracking is detected
            if angle_deg > angle_thresh:
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                
                # Project vector onto gate plane
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                if v_proj_norm < 1e-6:
                    detour_direction = y_axis
                else:
                    v_proj_y = np.dot(v_proj, y_axis)
                    v_proj_z = np.dot(v_proj, z_axis)
                    proj_angle = np.degrees(np.arctan2(v_proj_z, v_proj_y))
                    
                    # Determine detour direction based on angle
                    if -90 <= proj_angle < 45:
                        detour_direction = y_axis
                    elif 45 <= proj_angle < 135:
                        detour_direction = z_axis
                    else:
                        detour_direction = -y_axis
                
                # Calculate and insert detour waypoint
                detour_waypoint = gate_center + detour_dist * detour_direction
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
                
                self._debug_info['detour_analysis'].append({
                    'gate_index': i,
                    'angle': angle_deg,
                    'detour_waypoint': detour_waypoint.copy()
                })
        
        return np.array(waypoints_list)
    
    # =========================================================================
    # Obstacle Avoidance
    # =========================================================================
    
    def avoid_obstacles(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
        safety_distance: Optional[float] = None,
        trajectory_duration: float = 30.0,
        sampling_freq: float = 100.0
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Modify trajectory to avoid collisions with obstacles.
        
        Detects trajectory segments that pass too close to obstacles and
        inserts new waypoints to steer around them.
        
        Args:
            waypoints: Original waypoints.
            obstacle_positions: Positions of cylindrical obstacles.
            safety_distance: Minimum safe distance from obstacles.
            trajectory_duration: Duration for initial trajectory.
            sampling_freq: Sampling frequency for collision checking.
            
        Returns:
            Tuple of (time_parameters, modified_waypoints).
        """
        safe_dist = safety_distance or self.config.safety_distance
        
        # Generate initial trajectory
        trajectory = self.create_spline(trajectory_duration, waypoints)
        
        # Sample trajectory at high resolution
        n_samples = int(sampling_freq * trajectory_duration)
        time_samples = np.linspace(0, trajectory_duration, n_samples)
        trajectory_points = trajectory(time_samples)
        
        # Process each obstacle
        for obstacle_pos in obstacle_positions:
            collision_free_times = []
            collision_free_waypoints = []
            
            is_inside_obstacle = False
            entry_index = None
            
            for i, point in enumerate(trajectory_points):
                # Check distance in XY plane only (cylindrical obstacles)
                distance_xy = np.linalg.norm(obstacle_pos[:2] - point[:2])
                
                if distance_xy < safe_dist:
                    if not is_inside_obstacle:
                        is_inside_obstacle = True
                        entry_index = i
                        
                elif is_inside_obstacle:
                    exit_index = i
                    is_inside_obstacle = False
                    
                    # Compute avoidance waypoint
                    entry_point = trajectory_points[entry_index]
                    exit_point = trajectory_points[exit_index]
                    
                    entry_direction = entry_point[:2] - obstacle_pos[:2]
                    exit_direction = exit_point[:2] - obstacle_pos[:2]
                    avoidance_direction = entry_direction + exit_direction
                    avoidance_direction /= np.linalg.norm(avoidance_direction) + 1e-6
                    
                    new_position_xy = obstacle_pos[:2] + avoidance_direction * safe_dist
                    new_position_z = (entry_point[2] + exit_point[2]) / 2
                    new_waypoint = np.concatenate([new_position_xy, [new_position_z]])
                    
                    collision_free_times.append((time_samples[entry_index] + time_samples[exit_index]) / 2)
                    collision_free_waypoints.append(new_waypoint)
                else:
                    collision_free_times.append(time_samples[i])
                    collision_free_waypoints.append(point)
            
            # Handle case where trajectory ends inside obstacle
            if is_inside_obstacle:
                collision_free_times.append(time_samples[-1])
                collision_free_waypoints.append(trajectory_points[-1])
            
            time_samples = np.array(collision_free_times)
            trajectory_points = np.array(collision_free_waypoints)
        
        return time_samples, trajectory_points
    
    # =========================================================================
    # Spline Generation
    # =========================================================================
    
    def create_spline(
        self,
        duration: float,
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """
        Generate a cubic spline trajectory through waypoints.
        
        Uses arc-length parameterization for more uniform velocity distribution.
        
        Args:
            duration: Total time duration for the trajectory.
            waypoints: Array of 3D waypoints.
            
        Returns:
            CubicSpline object for trajectory evaluation.
        """
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative_arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        time_parameters = cumulative_arc_length / (cumulative_arc_length[-1] + 1e-6) * duration
        
        return CubicSpline(time_parameters, waypoints)
    
    def reparametrize_by_arclength(
        self,
        trajectory: CubicSpline,
        arc_step: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> CubicSpline:
        """
        Reparametrize trajectory by arc length for uniform speed.
        
        Args:
            trajectory: Input trajectory spline.
            arc_step: Arc length sampling step.
            epsilon: Convergence threshold for iterative refinement.
            
        Returns:
            Arc-length parameterized spline.
        """
        step = arc_step or self.config.arc_step
        eps = epsilon or self.config.arc_epsilon
        
        total_param_range = trajectory.x[-1] - trajectory.x[0]
        
        for _ in range(99):
            n_segments = max(2, int(total_param_range / step))
            t_samples = np.linspace(0.0, total_param_range, n_segments)
            pts = trajectory(t_samples)
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_param_range = float(cum_arc[-1])
            trajectory = CubicSpline(cum_arc, pts)
            
            if np.std(seg_lengths) <= eps:
                return trajectory
        
        return trajectory
    
    def extend_spline(
        self,
        trajectory: CubicSpline,
        extend_length: Optional[float] = None
    ) -> CubicSpline:
        """
        Extend trajectory along its terminal tangent direction.
        
        Args:
            trajectory: Input trajectory spline.
            extend_length: Extension length.
            
        Returns:
            Extended trajectory spline.
        """
        ext_len = extend_length or self.config.extend_length
        
        base_knots = trajectory.x
        base_dt = min(base_knots[1] - base_knots[0], 0.2)
        p_end = trajectory(base_knots[-1])
        v_end = trajectory.derivative(1)(base_knots[-1])
        v_dir = v_end / (np.linalg.norm(v_end) + 1e-6)
        
        extra_knots = np.arange(
            base_knots[-1] + base_dt,
            base_knots[-1] + ext_len,
            base_dt,
        )
        p_extend = np.array([p_end + v_dir * (s - base_knots[-1]) for s in extra_knots])
        
        theta_new = np.concatenate([base_knots, extra_knots])
        p_new = np.vstack([trajectory(base_knots), p_extend])
        
        return CubicSpline(theta_new, p_new, axis=0)
    
    # =========================================================================
    # Path Analysis Utilities
    # =========================================================================
    
    def compute_curvature(
        self,
        spline: CubicSpline,
        t_vals: NDArray[np.floating],
        eps: float = 1e-8
    ) -> NDArray[np.floating]:
        """
        Compute curvature along the spline.
        
        Args:
            spline: Trajectory spline.
            t_vals: Parameter values for evaluation.
            eps: Small value for numerical stability.
            
        Returns:
            Curvature values at each t_val.
        """
        v = spline(t_vals, 1)
        a = spline(t_vals, 2)
        cross_term = np.cross(v, a)
        num = np.linalg.norm(cross_term, axis=1)
        den = np.linalg.norm(v, axis=1) ** 3 + eps
        return num / den
    
    def find_closest_point(
        self,
        trajectory: CubicSpline,
        position: NDArray[np.floating],
        sample_interval: float = 0.05
    ) -> Tuple[float, NDArray[np.floating]]:
        """
        Find the closest point on trajectory to a given position.
        
        Args:
            trajectory: Trajectory spline.
            position: Query position (3,).
            sample_interval: Sampling interval for search.
            
        Returns:
            Tuple of (parameter_value, closest_point).
        """
        total_length = float(trajectory.x[-1])
        t_samples = np.arange(0.0, total_length, sample_interval)
        if t_samples.size == 0:
            return 0.0, trajectory(0.0)
        
        points = trajectory(t_samples)
        dists = np.linalg.norm(points - position, axis=1)
        idx_min = int(np.argmin(dists))
        
        return idx_min * sample_interval, points[idx_min]
    
    def get_gate_parameters(
        self,
        trajectory: CubicSpline,
        gate_positions: NDArray[np.floating],
        sample_interval: float = 0.05
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get arc-length parameters corresponding to gate positions.
        
        Args:
            trajectory: Arc-length parameterized trajectory.
            gate_positions: Gate center positions.
            sample_interval: Sampling interval for search.
            
        Returns:
            Tuple of (theta_values, interpolated_positions).
        """
        theta_list = []
        pos_list = []
        
        for gate_center in gate_positions:
            theta, pos = self.find_closest_point(trajectory, gate_center, sample_interval)
            theta_list.append(theta)
            pos_list.append(pos)
        
        return np.array(theta_list), np.array(pos_list)
    
    # =========================================================================
    # Complete Path Planning Pipeline
    # =========================================================================
    
    def plan_trajectory(
        self,
        obs: dict[str, NDArray[np.floating]],
        trajectory_duration: float = 30.0,
        sampling_freq: float = 100.0,
        for_mpcc: bool = True,
        mpcc_extension_length: float = 12.0
    ) -> TrajectoryResult:
        """
        Complete path planning pipeline.
        
        Args:
            obs: Observation dictionary containing:
                - 'pos': Current drone position
                - 'gates_pos': Gate positions
                - 'gates_quat': Gate quaternions
                - 'obstacles_pos': Obstacle positions
            trajectory_duration: Total trajectory duration.
            sampling_freq: Sampling frequency for collision checking.
            for_mpcc: Whether to prepare trajectory for MPCC controller.
            mpcc_extension_length: Extension length for MPCC prediction.
            
        Returns:
            TrajectoryResult with trajectory and metadata.
        """
        # Extract data from observation
        initial_pos = obs['pos']
        gate_positions = obs['gates_pos']
        gate_quats = obs['gates_quat']
        obstacle_positions = obs['obstacles_pos']
        
        # Extract gate coordinate frames
        gate_normals, gate_y_axes, gate_z_axes = GateFrameExtractor.extract_frames(gate_quats)
        
        # Step 1: Generate initial waypoints
        waypoints = self.generate_waypoints(
            initial_pos,
            gate_positions,
            gate_normals
        )
        
        # Step 2: Add detour waypoints for backtracking gates
        waypoints = self.add_detour_waypoints(
            waypoints,
            gate_positions,
            gate_normals,
            gate_y_axes,
            gate_z_axes
        )
        
        # Step 3: Avoid obstacles
        _, waypoints = self.avoid_obstacles(
            waypoints,
            obstacle_positions,
            trajectory_duration=trajectory_duration,
            sampling_freq=sampling_freq
        )
        
        # Step 4: Create main trajectory spline
        spline = self.create_spline(trajectory_duration, waypoints)
        
        # Step 5: Prepare for MPCC if requested
        arc_spline = None
        gate_thetas = None
        total_length = trajectory_duration
        
        if for_mpcc:
            # Extend and reparametrize by arc length
            extended = self.extend_spline(spline, extend_length=mpcc_extension_length)
            arc_spline = self.reparametrize_by_arclength(extended)
            total_length = float(arc_spline.x[-1])
            
            # Get gate parameters on arc-length trajectory
            gate_thetas, _ = self.get_gate_parameters(arc_spline, gate_positions)
        
        return TrajectoryResult(
            spline=spline,
            arc_spline=arc_spline,
            waypoints=waypoints,
            total_length=total_length,
            gate_thetas=gate_thetas,
            gate_positions=gate_positions,
            gate_normals=gate_normals,
            obstacle_positions=obstacle_positions,
            trajectory_duration=trajectory_duration
        )
    
    def replan_trajectory(
        self,
        obs: dict[str, NDArray[np.floating]],
        current_position: NDArray[np.floating],
        **kwargs
    ) -> TrajectoryResult:
        """
        Replan trajectory from current position.
        
        Args:
            obs: Updated observation dictionary.
            current_position: Current drone position.
            **kwargs: Additional arguments for plan_trajectory.
            
        Returns:
            Updated TrajectoryResult.
        """
        # Update initial position to current position
        obs = obs.copy()
        obs['pos'] = current_position
        
        return self.plan_trajectory(obs, **kwargs)


# =============================================================================
# Trajectory Visualizer Wrapper
# =============================================================================

class PathVisualizer:
    """
    Wrapper class for trajectory visualization.
    
    Provides easy integration with PathPlanner for visualizing
    planned trajectories, gates, obstacles, and drone position.
    """
    
    def __init__(
        self,
        width: int = 1400,
        height: int = 1000,
        title: str = "Drone Racing - Trajectory Visualization",
        output_dir: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize the visualizer.
        
        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            title: Window title.
            output_dir: Directory for saving screenshots.
            enabled: Whether visualization is enabled.
        """
        self.enabled = enabled and VISUALIZER_AVAILABLE
        self.visualizer = None
        self.width = width
        self.height = height
        self.title = title
        self.output_dir = output_dir
        
        # State tracking
        self._gate_detected_flags = None
        self._gate_real_positions = None
        
        if self.enabled:
            self._create_visualizer()
        elif enabled and not VISUALIZER_AVAILABLE:
            print("[PathVisualizer] Warning: Visualization requested but TrajectoryVisualizer not available.")
    
    def _create_visualizer(self):
        """Create the underlying visualizer."""
        if VISUALIZER_AVAILABLE:
            self.visualizer = TrajectoryVisualizer(
                width=self.width,
                height=self.height,
                title=self.title,
                output_dir=self.output_dir
            )
    
    def visualize_trajectory(
        self,
        result: TrajectoryResult,
        drone_position: Optional[NDArray[np.floating]] = None,
        gate_detected_status: Optional[NDArray[np.bool_]] = None,
        show: bool = True
    ):
        """
        Visualize a trajectory result.
        
        Args:
            result: TrajectoryResult from PathPlanner.
            drone_position: Current drone position.
            gate_detected_status: Gate detection status array.
            show: Whether to display immediately.
        """
        if not self.enabled or self.visualizer is None:
            return
        
        # Initialize gate tracking
        if gate_detected_status is not None:
            self._gate_detected_flags = gate_detected_status.copy()
        elif result.gate_positions is not None:
            num_gates = len(result.gate_positions)
            self._gate_detected_flags = np.zeros(num_gates, dtype=bool)
        
        if result.gate_positions is not None:
            num_gates = len(result.gate_positions)
            self._gate_real_positions = np.full((num_gates, 3), np.nan)
        
        # Create visualization
        self.visualizer.visualize(
            gate_positions=result.gate_positions,
            gate_normals=result.gate_normals,
            obstacle_positions=result.obstacle_positions,
            trajectory=result.spline,
            trajectory_duration=result.trajectory_duration,
            waypoints=result.waypoints,
            drone_position=drone_position,
            gate_detected_status=self._gate_detected_flags,
            show=show
        )
    
    def update(
        self,
        drone_position: Optional[NDArray[np.floating]] = None,
        trajectory: Optional[CubicSpline] = None,
        trajectory_duration: float = 30.0,
        gate_detected_status: Optional[NDArray[np.bool_]] = None,
        gate_real_positions: Optional[NDArray[np.floating]] = None
    ):
        """
        Update the visualization.
        
        Args:
            drone_position: Current drone position.
            trajectory: Updated trajectory spline.
            trajectory_duration: Duration of trajectory.
            gate_detected_status: Gate detection status array.
            gate_real_positions: Real gate positions when detected.
        """
        if not self.enabled or self.visualizer is None:
            return
        
        # Update gate tracking
        if gate_detected_status is not None:
            self._gate_detected_flags = gate_detected_status.copy()
        
        if gate_real_positions is not None:
            self._gate_real_positions = gate_real_positions.copy()
        
        self.visualizer.update(
            drone_position=drone_position,
            trajectory=trajectory,
            trajectory_duration=trajectory_duration,
            gate_detected_status=self._gate_detected_flags,
            gate_real_positions=self._gate_real_positions
        )
    
    def update_gate_detection(
        self,
        gate_index: int,
        is_detected: bool,
        real_position: Optional[NDArray[np.floating]] = None
    ):
        """
        Update detection status of a specific gate.
        
        Args:
            gate_index: Index of the gate.
            is_detected: Whether the gate is detected.
            real_position: Real position of the gate if detected.
        """
        if not self.enabled or self.visualizer is None:
            return
        
        # Update internal tracking
        if self._gate_detected_flags is not None and gate_index < len(self._gate_detected_flags):
            self._gate_detected_flags[gate_index] = is_detected
        
        if real_position is not None and self._gate_real_positions is not None:
            if gate_index < len(self._gate_real_positions):
                self._gate_real_positions[gate_index] = real_position
        
        self.visualizer.update_gate_detection(gate_index, is_detected, real_position)
    
    def save_screenshot(self, filepath: Optional[str] = None, dpi: int = 150):
        """
        Save current view as screenshot.
        
        Args:
            filepath: Path to save the image. If None, auto-generates name.
            dpi: Image resolution.
        """
        if not self.enabled or self.visualizer is None:
            print("[PathVisualizer] Cannot save screenshot: visualizer not available.")
            return
        
        if filepath is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"trajectory_{timestamp}.png"
        
        self.visualizer.save_image(filepath, dpi=dpi)
    
    def save_trajectory_data(self, filepath: str, format: str = 'json'):
        """
        Save trajectory data to file.
        
        Args:
            filepath: Path to save the data.
            format: Format ('json' or 'csv').
        """
        if not self.enabled or self.visualizer is None:
            print("[PathVisualizer] Cannot save data: visualizer not available.")
            return
        
        self.visualizer.save_trajectory_data(filepath, format=format)
    
    def close(self):
        """Close the visualizer."""
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
    
    @property
    def is_available(self) -> bool:
        """Check if visualization is available."""
        return self.enabled and self.visualizer is not None