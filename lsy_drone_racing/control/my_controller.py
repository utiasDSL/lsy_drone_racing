from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]


        # In __init__: Use these balanced gains
# 1. Softer P and D gains to prevent high-frequency shaking when inertia randomly drops
        self.kp = np.array([0.8, 0.8, 1.5])
        self.kd = np.array([0.4, 0.4, 0.6])
        
        # 2. Stronger I gain to quickly "learn" and adapt to the unknown random mass
        self.ki = np.array([0.25, 0.25, 0.4]) 
        
        # 3. Increase the ki_range (windup limit) so it is actually allowed 
        # to push hard enough to carry the heaviest random mass spawns
        self.ki_range = np.array([4.0, 4.0, 2.5])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._t_total = 12  # s
        # --- DYNAMIC TRAJECTORY GENERATION ---

        self.nominal_gates = [np.array(g["pos"]) for g in config.env.track.gates]
        self.nominal_obstacles = [np.array(o["pos"]) for o in config.env.track.obstacles]
        
        # Track which gates we have already updated our trajectory for
        self.gates_discovered = np.zeros(len(self.nominal_gates), dtype=bool)
        self.obstacles_discovered = np.zeros(len(self.nominal_obstacles), dtype=bool)

        self.spawn_pos = obs["pos"].copy()

        

        # Generate initial trajectory guess
        self._update_trajectory(obs["pos"], self.nominal_gates, self.nominal_obstacles)
        
        self._tick = 0
        self._finished = False

    def _update_trajectory(self, current_pos, target_gates, target_obstacles):
        R_DRONE = 0.10
        R_OBS = 0.03 / 2.0
        GATE_HALF_WIDTH = 0.4 / 2
        
        MARGIN = 0.10
        SAFE_RADIUS = R_DRONE + R_OBS + MARGIN 
        MAX_GATE_SHIFT = GATE_HALF_WIDTH - R_DRONE - 0.02 
        GATE_DEPTH = 0.5      
        
        waypoints_list = [current_pos]
        takeoff_pos = current_pos.copy()
        takeoff_pos[2] += 0.5
        waypoints_list.append(takeoff_pos)
        
        for i, gate_pos in enumerate(target_gates):
            prev_pos = waypoints_list[-1]
            path_len_2d = np.linalg.norm(gate_pos[0:2] - prev_pos[0:2])
            
            if path_len_2d < 1e-4:
                continue 

            direction = (gate_pos - prev_pos) / np.linalg.norm(gate_pos - prev_pos)
            right_shift = np.array([direction[1], -direction[0], 0.0])
            
            # --- PHASE 1: THE APPROACH DODGE ---
            for obs_pos in target_obstacles:
                dist_to_gate = np.linalg.norm(gate_pos[0:2] - obs_pos[0:2])
                dist_from_prev = np.linalg.norm(obs_pos[0:2] - prev_pos[0:2])
                stretch_threshold = np.sqrt(path_len_2d**2 + 4 * (SAFE_RADIUS**2)) - path_len_2d
                
                if (dist_to_gate + dist_from_prev) < (path_len_2d + stretch_threshold):
                    detour_pos = obs_pos.copy()
                    detour_pos[2] = gate_pos[2] 
                    detour_pos += right_shift * SAFE_RADIUS 
                    detour_pos -= direction * 0.25
                    waypoints_list.append(detour_pos)
                    break 
            
            # --- PHASE 2: ADD GATE ---
            waypoints_list.append(gate_pos)

            # --- PHASE 3: TWO-STAGE EXIT DODGE ---
            phantom_pos = gate_pos + (direction * GATE_DEPTH)
            post_dodge_pos = None
            blocking_obs_dist = None
            closest_obs_pos = None
            
            # Step 1: Scan for the closest blocking obstacle behind the gate
            for obs_pos in target_obstacles:
                obs_dir = obs_pos[0:2] - gate_pos[0:2]
                forward_dist = np.dot(obs_dir, direction[0:2])
                
                # Check if it is within 1.5m in front of us
                if 0 < forward_dist < 1.5:
                    lateral_dist = np.linalg.norm(obs_dir - forward_dist * direction[0:2])
                    
                    # If it is in our flight path
                    if lateral_dist < SAFE_RADIUS:
                        if blocking_obs_dist is None or forward_dist < blocking_obs_dist:
                            blocking_obs_dist = forward_dist
                            closest_obs_pos = obs_pos
                            
            # Step 2: Calculate variable GATE_DEPTH
            if blocking_obs_dist is not None:
                # Scale depth to 40% of the distance to the obstacle, 
                # but never less than 0.15m (to clear the gate frame)
                GATE_DEPTH = max(0.15, blocking_obs_dist * 0.4)
            else:
                # If the track is clear, use a smooth 0.8m straight exit
                GATE_DEPTH = 0.8 
                
            # Step 3: Create the waypoints
            phantom_pos = gate_pos + (direction * GATE_DEPTH)
            post_dodge_pos = None
            
            if blocking_obs_dist is not None:
                # Determine which way to shift
                cross_p = direction[0]*(closest_obs_pos[1]-gate_pos[1]) - direction[1]*(closest_obs_pos[0]-gate_pos[0])
                shift_dir = right_shift if cross_p > 0 else -right_shift
                
                # Stage 1: Shift the phantom position slightly to start the dodge
                phantom_pos += shift_dir * min(SAFE_RADIUS, MAX_GATE_SHIFT)
                
                # Stage 2: Full sweep placed perfectly parallel to the obstacle
                post_dodge_pos = closest_obs_pos.copy()
                post_dodge_pos[2] = gate_pos[2]
                post_dodge_pos += shift_dir * (SAFE_RADIUS * 2.5)
            
            # Append waypoints
            phantom_pos[2] = gate_pos[2]
            waypoints_list.append(phantom_pos)
            
            if post_dodge_pos is not None:
                waypoints_list.append(post_dodge_pos)
            
        waypoints = np.array(waypoints_list)
        
        diffs = np.diff(waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cum_dist = np.insert(np.cumsum(distances), 0, 0.0)
        
        if cum_dist[-1] > 0:
            t = (cum_dist / cum_dist[-1]) * self._t_total
        else:
            t = np.linspace(0, self._t_total, len(waypoints))

        self._des_pos_spline = CubicSpline(t, waypoints)
        #self._des_pos_spline = PchipInterpolator(t, waypoints, axis=0)
        self._des_vel_spline = self._des_pos_spline.derivative()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        
        visited_mask_gates = obs["gates_visited"]
        newly_discovered_gates = visited_mask_gates & ~self.gates_discovered

        visited_mask_obstacles = obs["obstacles_visited"]
        newly_discovered_obstacles = visited_mask_obstacles & ~self.obstacles_discovered

        if np.any(newly_discovered_gates) or np.any(newly_discovered_obstacles):
            # We found the true position of a gate! 
            # Update our known gate array with the true positions from obs
            current_known_gates = []
            for i in range(len(self.nominal_gates)):
                if visited_mask_gates[i]:
                    current_known_gates.append(obs["gates_pos"][i]) # True pos
                else:
                    current_known_gates.append(self.nominal_gates[i]) # Still guessing
            
            current_known_obstacles = []
            for i in range(len(self.nominal_obstacles)):
                if visited_mask_obstacles[i]:
                    current_known_obstacles.append(obs["obstacles_pos"][i]) # True pos
                else:
                    current_known_obstacles.append(self.nominal_obstacles[i]) # Still guessing
            
            # Re-calculate the CubicSpline trajectory using the new information
            self._update_trajectory(self.spawn_pos, current_known_gates, current_known_obstacles)
            self.gates_discovered = visited_mask_gates.copy()
            self.obstacles_discovered = visited_mask_obstacles.copy()

        # 2. Proceed with your normal Attitude PID control using the updated spline
        t = min(self._tick / self._freq, self._t_total)
        
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g


        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

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
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
