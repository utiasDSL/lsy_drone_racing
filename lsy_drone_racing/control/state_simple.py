"""Controller that follows a pre-defined trajectory with gate detection and obstacle avoidance.
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
        
        # Store nominal gate information from config
        self._gates = []
        for gate in config.env.track.gates:
            gate_pos = np.array(gate["pos"], dtype=float)
            gate_rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", gate_rpy)
            gate_normal = rot.as_matrix()[:, 0]  # x-axis points through gate
            
            self._gates.append({
                "nominal_pos": gate_pos.copy(),
                "nominal_normal": gate_normal.copy(),
                "nominal_rpy": gate_rpy.copy(), # keep nominal rpy
                "detected_pos": None,
                "detected_normal": None,
            })
        
        self._num_gates = len(self._gates)
        
        # Gate traversal parameters
        self._approach_dist = 0.45   # meters before gate center
        self._exit_dist = 0.55       # meters after gate center

        # Gate2 special handling
        self._gate2_idx = 2
        self._exit_dist_gate2 = 0.3
        self._climb_alt_gate2 = 1.25

        # gate corridor half-width
        self._gate_corridor_width = 0.3

        # pitch lock threshold (deg + rad)
        self._pitch_lock_thresh_deg = 7.5
        self._pitch_lock_thresh = np.deg2rad(self._pitch_lock_thresh_deg)        

        # Obstacle avoidance parameters
        self._avoidance_distance = 0.6
        self._obstacle_radius = 0.15
        self._safety_margin = 0.25
        
        # PID gains for trajectory tracking
        self._kp_pos = 2.5      # Position proportional gain
        self._kd_pos = 1.5      # Position derivative gain
        self._ki_pos = 0.1      # Position integral gain
        
        # PID state
        self._pos_error_integral = np.zeros(3, dtype=np.float32)
        self._integral_limit = 2.0  # Anti-windup limit
        
        # Timing
        self._t_total = 20.0  # Total flight time
        
        # Initialize tick counter BEFORE building trajectory
        self._tick = 0
        self._finished = False
        self._last_update_tick = -100  # Prevent frequent trajectory updates
        
        # Build initial trajectory
        self._build_trajectory()
        
        print("=" * 60)
        print("Level 2 Controller with Gate Detection & Obstacle Avoidance")
        print(f"Gates: {self._num_gates}")
        print(f"Total waypoints: {len(self._waypoints)}")
        print(f"PID Gains - Kp: {self._kp_pos}, Kd: {self._kd_pos}, Ki: {self._ki_pos}")
        print("=" * 60)

    def _is_in_gate_corridor(self, p: NDArray[np.floating]) -> bool:
        """Return True if point p is inside any gate's corridor segment."""
        for i, g in enumerate(self._gates):
            pos    = g["detected_pos"]    if g["detected_pos"]    is not None else g["nominal_pos"]
            normal = g["detected_normal"] if g["detected_normal"] is not None else g["nominal_normal"]

            # Signed distance along the gate normal (s < 0: before gate, s > 0: after gate)
            s = float(np.dot(p - pos, normal))

            # Corridor length for this gate (gate2 has shorter exit segment)
            exit_len = self._exit_dist_gate2 if i == self._gate2_idx else self._exit_dist

            # Inside the longitudinal bounds of the corridor?
            if -self._approach_dist <= s <= exit_len:
                # Lateral distance to the gate normal line (in 3D)
                lateral_vec = (p - pos) - s * normal
                if np.linalg.norm(lateral_vec) <= self._gate_corridor_width:
                    return True
        return False

    def _get_gate_waypoints(self, gate_idx: int) -> list:
        gate = self._gates[gate_idx]
        # Use detected position if available, else nominal
        pos    = gate["detected_pos"]    if gate["detected_pos"] is not None    else gate["nominal_pos"]
        normal = gate["detected_normal"] if gate["detected_normal"] is not None else gate["nominal_normal"]

        approach   = pos - self._approach_dist * normal
        center     = pos.copy()

        if gate_idx == self._gate2_idx:
            exit_short = pos + self._exit_dist_gate2 * normal
            climb_point = exit_short.copy()     # Climb point after gate2
            climb_point[2] = self._climb_alt_gate2
            # return [approach, center, exit_short, exit_short.copy(), climb_point]
            return [approach, center, exit_short, climb_point]

        exit_point = pos + self._exit_dist * normal
        return [approach, center, exit_point]

    def _build_trajectory(self):
        """Build spline trajectory through all gates."""
        waypoints = []
        
        # Starting position
        waypoints.append(np.array([-1.5, 0.75, 0.05], dtype=float))
        
        # Add smooth transition to first gate
        first_gate_wps = self._get_gate_waypoints(0)
        mid_to_first = (waypoints[0] + first_gate_wps[0]) / 2
        waypoints.append(mid_to_first)
        
        # Add waypoints for each gate
        for i in range(self._num_gates):
            gate_wps = self._get_gate_waypoints(i)
            waypoints.extend(gate_wps)
            
            # Add smooth transition between gates
            if i < self._num_gates - 1:
                next_gate_wps = self._get_gate_waypoints(i + 1)
                mid_point = (gate_wps[-1] + next_gate_wps[0]) / 2
                waypoints.append(mid_point)
        
        # Final hover point after last gate
        last_exit = waypoints[-1]
        waypoints.append(last_exit + np.array([0.5, 0.0, 0.0], dtype=float))
        
        self._waypoints = np.array(waypoints, dtype=float)
        
        # Create splines
        n = len(self._waypoints)
        t = np.linspace(0, self._t_total, n)
        self._des_pos_spline = CubicSpline(t, self._waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative(nu=1)
        self._des_acc_spline = self._des_pos_spline.derivative(nu=2)
        
        print(f"[Tick {self._tick}] Trajectory built with {len(self._waypoints)} waypoints")

    def _update_gate_detections(self, obs: dict[str, NDArray[np.floating]]):
        """Update gate positions when detected within sensor range."""
        if "gates_pos" not in obs or "gates_quat" not in obs:
            return
        
        current_pos = obs["pos"]
        trajectory_updated = False
        
        for i, gate_pos in enumerate(obs["gates_pos"]):
            # Check if gate is within sensor range
            dist = np.linalg.norm(gate_pos - current_pos)
            if dist > self._sensor_range:
                continue
            
            # First time detecting this gate
            if self._gates[i]["detected_pos"] is None:
                # Get gate orientation
                rot = R.from_quat(obs["gates_quat"][i])
                det_euler = rot.as_euler("xyz")
                nom_pitch = self._gates[i]["nominal_rpy"][1]
                if abs(det_euler[1] - nom_pitch) < self._pitch_lock_thresh:
                    det_euler[1] = nom_pitch  # lock pitch to nominal
                    rot = R.from_euler("xyz", det_euler)

                gate_normal = rot.as_matrix()[:, 0]
                
                # Check if position changed significantly from nominal
                pos_change = np.linalg.norm(gate_pos - self._gates[i]["nominal_pos"])
                
                # Store detected position
                self._gates[i]["detected_pos"] = gate_pos.copy()
                self._gates[i]["detected_normal"] = gate_normal.copy()
                
                if pos_change > 0.05:  # Significant change (>5cm)
                    print(f"[Tick {self._tick}] Gate {i} detected with {pos_change:.3f}m offset")
                    trajectory_updated = True
                else:
                    print(f"[Tick {self._tick}] Gate {i} detected (nominal position)")
        
        # Rebuild trajectory if any gate changed
        if trajectory_updated:
            print(f"[Tick {self._tick}] Rebuilding trajectory with updated gates...")
            self._build_trajectory()
            self._last_update_tick = self._tick
            # Reset integral on trajectory change to prevent windup
            self._pos_error_integral = np.zeros(3, dtype=np.float32)

    def _lateral_obstacle_avoidance(
        self, 
        des_pos: NDArray[np.floating],
        current_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply lateral shift to avoid obstacles."""
        if len(obstacles_pos) == 0:
            return des_pos
        
        # Get trajectory direction (horizontal only)
        traj_direction = des_vel[:2].copy()
        if np.linalg.norm(traj_direction) < 0.01:
            traj_direction = (des_pos[:2] - current_pos[:2])
        
        if np.linalg.norm(traj_direction) < 0.01:
            return des_pos
        
        traj_direction = traj_direction / np.linalg.norm(traj_direction)
        
        # Perpendicular direction (rotate 90 degrees)
        perp_direction = np.array([-traj_direction[1], traj_direction[0]])
        
        total_offset = np.zeros(2)
        
        for obs_pos in obstacles_pos:
            # Distance checks
            dist_to_drone = np.linalg.norm(obs_pos[:2] - current_pos[:2])
            dist_to_desired = np.linalg.norm(obs_pos[:2] - des_pos[:2])
            
            if min(dist_to_drone, dist_to_desired) > self._avoidance_distance:
                continue
            
            # Check if obstacle is along trajectory
            to_obs = obs_pos[:2] - current_pos[:2]
            projection = np.dot(to_obs, traj_direction)
            
            if projection < -0.2:  # Behind us
                continue
            
            # Lateral distance from trajectory line
            lateral_dist = abs(np.dot(to_obs, perp_direction))
            required_clearance = self._obstacle_radius + self._safety_margin
            
            if lateral_dist < required_clearance:
                # Determine which side obstacle is on
                obstacle_side = np.sign(np.dot(to_obs, perp_direction))
                
                # Calculate avoidance offset
                avoidance_distance = required_clearance - lateral_dist
                strength = 1.5 * avoidance_distance / required_clearance
                offset = -obstacle_side * perp_direction * strength * 0.20
                
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
        """Compute PID control for trajectory tracking.
        
        Returns:
            Desired acceleration including feedforward and feedback terms
        """
        # Position error
        pos_error = des_pos - current_pos
        
        # Velocity error
        vel_error = des_vel - current_vel
        
        # Integral of position error (with anti-windup)
        self._pos_error_integral += pos_error * self._dt
        self._pos_error_integral = np.clip(
            self._pos_error_integral,
            -self._integral_limit,
            self._integral_limit
        )
        
        # PID control law
        # Feedforward: desired acceleration from trajectory
        # Feedback: P term on position error + D term on velocity error + I term on integrated error
        acc_pid = (
            des_acc +                                    # Feedforward
            self._kp_pos * pos_error +                   # Proportional
            self._kd_pos * vel_error +                   # Derivative
            self._ki_pos * self._pos_error_integral      # Integral
        )
        
        return acc_pid

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state with gate detection, obstacle avoidance, and PID."""
        
        # Update gate detections (but not too frequently to avoid trajectory jumps)
        early_phase = self._tick < int(5.0 * self._freq)
        update_gap  = 5 if early_phase else 20 
        if self._tick - self._last_update_tick > update_gap:
            self._update_gate_detections(obs)
        
        # Get desired position, velocity, and acceleration from spline
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True
        
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_acc = self._des_acc_spline(t)
        
        # Apply lateral obstacle avoidance
        if "obstacles_pos" in obs and len(obs["obstacles_pos"]) > 0:
            # skip (or weaken) avoidance when inside a gate corridor
            if not self._is_in_gate_corridor(des_pos):
                des_pos = self._lateral_obstacle_avoidance(
                    des_pos,
                    obs["pos"],
                    des_vel,
                    obs["obstacles_pos"]
                )
        
        # Compute PID control for better tracking
        des_acc_pid = self._compute_pid_control(
            obs["pos"],
            obs["vel"],
            des_pos,
            des_vel,
            des_acc
        )
        
        # State action: [pos(3), vel(3), acc(3), yaw(1), rrate(1), prate(1), yrate(1)]
        action = np.concatenate([
            des_pos,
            des_vel,
            des_acc_pid,  # Use PID-computed acceleration
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
        
        # Reset PID state
        self._pos_error_integral = np.zeros(3, dtype=np.float32)
        
        # Reset gate detections
        for gate in self._gates:
            gate["detected_pos"] = None
            gate["detected_normal"] = None
        
        # Rebuild trajectory from nominal positions
        self._build_trajectory()
        
        print("\n" + "=" * 60)
        print("Episode Reset - Trajectory rebuilt from nominal positions")
        print("=" * 60 + "\n")
