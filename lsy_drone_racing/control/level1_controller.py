"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

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
        
        # More aggressive PID gains for better tracking
        self.kp = np.array([1.0, 1.0, 2.0])    # Increased position gains
        self.ki = np.array([0.1, 0.1, 0.2])    # Increased integral gains
        self.kd = np.array([0.4, 0.4, 0.4])    # Increased derivative gains
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0
        
        # Optimized waypoints for Level 1 with proper gate orientations
        waypoints = np.array([
            [1.0, 1.5, 0.07],      # Start
            [0.45, -0.5, 0.56],    # Gate 1
            [1.0, -1.05, 1.11],    # Gate 2
            [0.0, 1.0, 0.56],      # Gate 3
            [-0.5, 0.0, 1.11],     # Gate 4
        ])
        
        # Create smooth trajectory with more points for better control
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])
        
        # Shorter completion time for more aggressive flight
        des_completion_time = 12
        ts = np.linspace(0, 1, int(self.freq * des_completion_time))
        
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)
        
        if self.control_mode == "state":
            # Compute velocities and accelerations with proper scaling
            self.vx_des = cs_x.derivative(1)(ts) * (1/des_completion_time)
            self.vy_des = cs_y.derivative(1)(ts) * (1/des_completion_time)
            self.vz_des = cs_z.derivative(1)(ts) * (1/des_completion_time)
            self.ax_des = cs_x.derivative(2)(ts) * (1/des_completion_time**2)
            self.ay_des = cs_y.derivative(2)(ts) * (1/des_completion_time**2)
            self.az_des = cs_z.derivative(2)(ts) * (1/des_completion_time**2)
        
        self._finished = False

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
        else:  # attitude control mode
            des_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
            
            # Calculate errors with proper scaling
            pos_error = des_pos - obs["pos"]
            vel_error = -obs["vel"]  # Desired velocity is zero
            
            # Update integral error with anti-windup
            self.i_error += pos_error * (1 / self.freq)
            self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)
            
            # Compute target thrust with gravity compensation
            target_thrust = np.zeros(3)
            target_thrust += self.kp * pos_error
            target_thrust += self.ki * self.i_error
            target_thrust += self.kd * vel_error
            target_thrust[2] += self.drone_mass * self.g
            
            # Get current orientation
            z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
            
            # Compute desired thrust with proper limits
            thrust_desired = target_thrust.dot(z_axis)
            thrust_desired = np.clip(thrust_desired, 
                                   0.5 * self.drone_mass * self.g,  # Increased minimum thrust
                                   2.0 * self.drone_mass * self.g)  # Increased maximum thrust
            
            # Compute desired orientation
            z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
            
            # Compute yaw to point towards next waypoint
            des_yaw = 0.0
            if i < len(self.x_des) - 1:
                dx = self.x_des[i + 1] - self.x_des[i]
                dy = self.y_des[i + 1] - self.y_des[i]
                des_yaw = np.arctan2(dy, dx)
            
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