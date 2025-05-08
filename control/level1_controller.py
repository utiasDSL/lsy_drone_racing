"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from gate positions defined in the config file.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from crazyflow.constants import MASS
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

class Level1Controller(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.drone_mass = MASS
        self.control_mode = config.env.control_mode
        
        # Adaptive PID gains for robustness
        self.base_kp = np.array([4.0, 4.0, 8.0])    # Increased base position gains
        self.base_ki = np.array([0.8, 0.8, 1.5])    # Increased base integral gains
        self.base_kd = np.array([1.5, 1.5, 4.0])    # Increased base derivative gains
        
        # Initialize adaptive gains
        self.kp = self.base_kp.copy()
        self.ki = self.base_ki.copy()
        self.kd = self.base_kd.copy()
        
        # Anti-windup and integral limits
        self.ki_range = np.array([3.0, 3.0, 0.6])  # Increased integral limits
        self.i_error = np.zeros(3)
        
        # Disturbance rejection
        self.disturbance_estimate = np.zeros(3)
        self.disturbance_gain = 0.1  # Gain for disturbance estimation
        
        # Adaptive control parameters
        self.error_history = []
        self.max_history = 10
        self.g = 9.81
        self._tick = 0
        
        # Get gate positions and obstacles from config
        self.gate_positions = self._get_gate_positions(config)
        self.obstacle_positions = self._get_obstacle_positions(config)
        
        # Generate waypoints including intermediate points for smooth transitions
        waypoints = self._generate_waypoints(self.gate_positions, self.obstacle_positions)
        
        # Create smooth trajectory with more points for better control
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])
        
        # Fixed completion time for consistent performance
        self.des_completion_time = 15.0  # Fixed time for consistent performance
        ts = np.linspace(0, 1, int(self.freq * self.des_completion_time))
        
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)
        
        if self.control_mode == "state":
            # Compute velocities and accelerations with proper scaling
            self.vx_des = cs_x.derivative(1)(ts) * (1/self.des_completion_time)
            self.vy_des = cs_y.derivative(1)(ts) * (1/self.des_completion_time)
            self.vz_des = cs_z.derivative(1)(ts) * (1/self.des_completion_time)
            self.ax_des = cs_x.derivative(2)(ts) * (1/self.des_completion_time**2)
            self.ay_des = cs_y.derivative(2)(ts) * (1/self.des_completion_time**2)
            self.az_des = cs_z.derivative(2)(ts) * (1/self.des_completion_time**2)
        
        self._finished = False

    def _get_gate_positions(self, config) -> List[Tuple[float, float, float]]:
        """Extract gate positions from config file."""
        gate_positions = []
        for gate in config.env.track.gates:
            gate_positions.append(tuple(gate.pos))
        return gate_positions

    def _get_obstacle_positions(self, config) -> List[Tuple[float, float, float]]:
        """Extract obstacle positions from config file."""
        obstacle_positions = []
        for obstacle in config.env.track.obstacles:
            obstacle_positions.append(tuple(obstacle.pos))
        return obstacle_positions

    def _generate_waypoints(self, gate_positions: List[Tuple[float, float, float]], 
                          obstacle_positions: List[Tuple[float, float, float]]) -> np.ndarray:
        """Generate waypoints including intermediate points for smooth transitions and obstacle avoidance."""
        waypoints = []
        
        # Add start position
        start_pos = tuple(config.env.track.drones[0].pos)
        waypoints.append(start_pos)
        
        # Add gate positions with intermediate points
        for i, gate_pos in enumerate(gate_positions):
            if i > 0:
                # Add intermediate point before gate
                prev_gate = gate_positions[i-1]
                intermediate = self._get_intermediate_point(prev_gate, gate_pos, obstacle_positions)
                waypoints.append(intermediate)
            
            # Add gate position
            waypoints.append(gate_pos)
            
            if i < len(gate_positions) - 1:
                # Add intermediate point after gate
                next_gate = gate_positions[i+1]
                intermediate = self._get_intermediate_point(gate_pos, next_gate, obstacle_positions)
                waypoints.append(intermediate)
        
        return np.array(waypoints)

    def _get_intermediate_point(self, pos1: Tuple[float, float, float], 
                              pos2: Tuple[float, float, float],
                              obstacle_positions: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Generate an intermediate point between two positions for smooth transitions and obstacle avoidance."""
        # Calculate midpoint
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2
        mid_z = (pos1[2] + pos2[2]) / 2
        
        # Add offset for smoother trajectory and obstacle avoidance
        offset = 0.3  # Increased offset for better obstacle avoidance
        
        # Check for obstacles between points
        for obs_pos in obstacle_positions:
            # Calculate distance to obstacle
            dist_to_obs = np.sqrt((mid_x - obs_pos[0])**2 + (mid_y - obs_pos[1])**2)
            
            # If obstacle is close to the path, adjust the intermediate point
            if dist_to_obs < 1.0:  # Obstacle avoidance threshold
                # Calculate vector from obstacle to midpoint
                dx = mid_x - obs_pos[0]
                dy = mid_y - obs_pos[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Normalize and scale the offset
                if dist > 0:
                    mid_x += (dx/dist) * offset
                    mid_y += (dy/dist) * offset
                
                # Adjust height to avoid obstacle
                mid_z = max(mid_z, obs_pos[2] + 0.5)  # Stay above obstacle
        
        # Add directional offset for smoother trajectory
        if pos2[0] < pos1[0]:  # Moving left
            mid_x += offset
        elif pos2[0] > pos1[0]:  # Moving right
            mid_x -= offset
            
        if pos2[1] < pos1[1]:  # Moving backward
            mid_y += offset
        elif pos2[1] > pos1[1]:  # Moving forward
            mid_y -= offset
            
        return (mid_x, mid_y, mid_z)

    def _update_adaptive_gains(self, pos_error, vel_error):
        """Update PID gains based on error history and current errors."""
        # Store error history
        self.error_history.append(np.linalg.norm(pos_error))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Calculate error trend
        if len(self.error_history) > 1:
            error_trend = np.mean(np.diff(self.error_history))
            
            # Adjust gains based on error trend
            if error_trend > 0:  # Error is increasing
                self.kp = self.base_kp * 1.2  # Increase proportional gain
                self.kd = self.base_kd * 1.2  # Increase derivative gain
            else:  # Error is decreasing
                self.kp = self.base_kp * 0.9  # Decrease proportional gain
                self.kd = self.base_kd * 0.9  # Decrease derivative gain
            
            # Keep gains within reasonable bounds
            self.kp = np.clip(self.kp, self.base_kp * 0.5, self.base_kp * 2.0)
            self.kd = np.clip(self.kd, self.base_kd * 0.5, self.base_kd * 2.0)

    def _update_disturbance_estimate(self, pos_error, vel_error):
        """Update disturbance estimate using error information."""
        # Simple disturbance observer
        self.disturbance_estimate = (1 - self.disturbance_gain) * self.disturbance_estimate + \
                                  self.disturbance_gain * (pos_error + vel_error)

    def compute_control(self, obs, info=None):
        i = min(self._tick, len(self.x_des) - 1)
        if i == len(self.x_des) - 1:
            self._finished = True

        if self.control_mode == "state":
            # For state control, return full state command with proper scaling
            return np.array([
                self.x_des[i], self.y_des[i], self.z_des[i],          # Position
                self.vx_des[i], self.vy_des[i], self.vz_des[i],       # Velocity
                self.ax_des[i], self.ay_des[i], self.az_des[i],       # Acceleration
                0.0, 0.0, 0.0, 0.0                                     # Yaw and rates
            ], dtype=np.float32)
        else:  # attitude control
            des_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
            
            # Calculate errors with proper scaling
            pos_error = des_pos - obs["pos"]
            vel_error = -obs["vel"]  # Desired velocity is zero
            
            # Update adaptive gains and disturbance estimate
            self._update_adaptive_gains(pos_error, vel_error)
            self._update_disturbance_estimate(pos_error, vel_error)
            
            # Update integral error with anti-windup
            self.i_error += pos_error * (1 / self.freq)
            self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)
            
            # Compute target thrust with gravity compensation and disturbance rejection
            target_thrust = np.zeros(3)
            target_thrust += self.kp * pos_error
            target_thrust += self.ki * self.i_error
            target_thrust += self.kd * vel_error
            target_thrust += self.disturbance_estimate  # Add disturbance compensation
            target_thrust[2] += self.drone_mass * self.g
            
            # Get current orientation
            z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
            
            # Compute desired thrust with proper limits
            thrust_desired = target_thrust.dot(z_axis)
            thrust_desired = np.clip(thrust_desired, 
                                   0.8 * self.drone_mass * self.g,  # Increased minimum thrust
                                   4.0 * self.drone_mass * self.g)  # Increased maximum thrust
            
            # Compute desired orientation with smoother transitions
            z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
            
            # Compute yaw to point towards next waypoint with smoothing
            des_yaw = 0.0
            if i < len(self.x_des) - 1:
                dx = self.x_des[i + 1] - self.x_des[i]
                dy = self.y_des[i + 1] - self.y_des[i]
                des_yaw = np.arctan2(dy, dx)
                
                # Smooth yaw transitions
                current_yaw = R.from_quat(obs["quat"]).as_euler("xyz")[2]
                yaw_diff = des_yaw - current_yaw
                if yaw_diff > np.pi:
                    yaw_diff -= 2 * np.pi
                elif yaw_diff < -np.pi:
                    yaw_diff += 2 * np.pi
                des_yaw = current_yaw + 0.1 * yaw_diff  # Smooth yaw transitions
            
            x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
            y_axis_desired = np.cross(z_axis_desired, x_c_des)
            y_axis_desired /= np.linalg.norm(y_axis_desired)
            x_axis_desired = np.cross(y_axis_desired, z_axis_desired)
            
            R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
            euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
            
            return np.concatenate([[thrust_desired], euler_desired], dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self.i_error[:] = 0