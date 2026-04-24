"""Fast trajectory controller for Level 2 with Graphical Debugging.

Uses a cubic spline with distance-based time allocation. 
- Gate Threading: Brackets gates with 'pre' and 'post' waypoints.
- Obstacle Repulsion: Radially pushes waypoints away from obstacles.
- Graphical Debugging: Renders the danger zones, waypoints, and the actual flight path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_REPLAN_THRESHOLD = 0.05
_GATE_MARGIN = 0.160      # Meters before and after the gate for a straight approach
_OBSTACLE_MARGIN = 0.160  # Meters of safe clearance radius around the vertical obstacles

# Nominal track layout
_NOMINAL_GATE_POS = np.array(
    [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]], dtype=np.float64
)
_NOMINAL_GATE_YAW = np.array([-0.78, 2.35, 3.14, 0.0], dtype=np.float64)

_NOMINAL_OBSTACLE_POS = np.array(
    [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]], dtype=np.float64
)


class StateController(Controller):
    """Fast state controller with dynamic gate threading, obstacle repulsion, and visual debug."""

    MAX_VELOCITY = 100  
    MAX_ACCELERATION = 100 

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._t_total = 10.0

        self._waypoints_list = []
        self._gate_indices = {}  

        self._waypoints_list.append([-1.5, 0.75, 0.05])  # Start
        self._waypoints_list.append([-1.0, 0.55, 0.4])   # Intermediate

        self._add_gate_waypoints(gate_id=0)
        self._add_gate_waypoints(gate_id=1, intermediate_point=[1.3, -0.15, 0.9])
        self._add_gate_waypoints(gate_id=2, intermediate_point=[-0.5, -0.05, 0.7])
        self._waypoints_list.append([-1.2, -0.2, 1.2])   # Intermediate
        self._add_gate_waypoints(gate_id=3)
        self._waypoints_list.append([0.5, -0.75, 1.2])   # End

        self._base_waypoints = np.array(self._waypoints_list, dtype=np.float64)
        self._waypoints = self._base_waypoints.copy()
        
        self._planned_gates_pos = np.array(obs.get("gates_pos", _NOMINAL_GATE_POS), dtype=np.float64)
        self._planned_obstacles_pos = np.array(obs.get("obstacles_pos", _NOMINAL_OBSTACLE_POS), dtype=np.float64)
        self._replanned_gates: set[int] = set()

        # --- VISUAL DEBUGGING STATE ---
        self._path_history = []  # Stores the drone's actual flown path
        # ------------------------------

        self._des_pos_spline: CubicSpline | None = None
        self._des_vel_spline: CubicSpline | None = None
        self._des_acc_spline: CubicSpline | None = None
        
        self._build_spline()
        self._tick = 0
        self._finished = False

    def _add_gate_waypoints(self, gate_id: int, intermediate_point: list[float] | None = None):
        if intermediate_point:
            self._waypoints_list.append(intermediate_point)
            
        pos = _NOMINAL_GATE_POS[gate_id]
        yaw = _NOMINAL_GATE_YAW[gate_id]
        
        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        prev_wp = np.array(self._waypoints_list[-1])
        travel_vec = pos - prev_wp
        if np.dot(travel_vec, normal) < 0:
            normal = -normal  
            
        pre_wp = pos - _GATE_MARGIN * normal
        post_wp = pos + _GATE_MARGIN * normal
        
        pre_idx = len(self._waypoints_list)
        self._waypoints_list.append(pre_wp.tolist())
        post_idx = len(self._waypoints_list)
        self._waypoints_list.append(post_wp.tolist())
        
        self._gate_indices[gate_id] = (pre_idx, post_idx)

    def _build_spline(self) -> None:
        """Applies Obstacle Shielding, Prunes clashing points, and builds splines."""
        safe_wps = []
        
        # 1. Obstacle Repulsion Logic (Same as before)
        for i in range(len(self._waypoints)):
            wp = self._waypoints[i].copy()
            pushed = False

            for obs_pos in self._planned_obstacles_pos:
                dist_xy = np.linalg.norm(wp[:2] - obs_pos[:2])
                
                if dist_xy < _OBSTACLE_MARGIN:
                    push_vec = wp[:2] - obs_pos[:2]
                    if np.linalg.norm(push_vec) < 1e-3:
                        push_vec = np.array([1.0, 0.0]) 
                    push_vec = push_vec / np.linalg.norm(push_vec) 

                    safe_wp = wp.copy()
                    safe_wp[:2] = obs_pos[:2] + push_vec * _OBSTACLE_MARGIN

                    tangent = np.array([-push_vec[1], push_vec[0], 0.0])
                    shield_dist = 0.18  
                    
                    wp_pre = safe_wp + tangent * shield_dist
                    wp_post = safe_wp - tangent * shield_dist

                    if len(safe_wps) > 0:
                        travel_dir = safe_wp - safe_wps[-1]
                        if np.dot(travel_dir, tangent) < 0:
                            wp_pre, wp_post = wp_post, wp_pre 

                    safe_wps.extend([wp_pre, safe_wp, wp_post])
                    pushed = True
                    break 

            if not pushed:
                safe_wps.append(wp)

        # --- 2. THE PRUNING FILTER (NEW) ---
        # Remove waypoints that are squished too close together to prevent spline knots
        pruned_wps = [safe_wps[0]]
        for i in range(1, len(safe_wps)):
            dist_to_prev = np.linalg.norm(safe_wps[i] - pruned_wps[-1])
            
            # Keep the point IF it is > 15cm away, OR if it's the final finish-line point
            if dist_to_prev > 0.15 or i == len(safe_wps) - 1:
                pruned_wps.append(safe_wps[i])
        # -----------------------------------

        self._active_path_wps = np.array(pruned_wps)

        # 3. Distance-Based Time Allocation
        differences = np.diff(self._active_path_wps, axis=0)
        distances = np.linalg.norm(differences, axis=1)

        cum_distances = np.concatenate(([0], np.cumsum(distances)))
        total_distance = cum_distances[-1]

        t = (cum_distances / total_distance) * self._t_total

        self._des_pos_spline = CubicSpline(t, self._active_path_wps)
        self._des_vel_spline = self._des_pos_spline.derivative(nu=1)
        self._des_acc_spline = self._des_pos_spline.derivative(nu=2)

    def _check_and_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Updates internal state with revealed gates/obstacles and triggers rebuild."""
        needs_rebuild = False

        # --- 1. GATE REPLANNING (Permanently alters self._waypoints) ---
        target_gate = int(obs["target_gate"])
        if target_gate >= 0 and target_gate not in self._replanned_gates:
            new_gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=np.float64)
            new_yaw = obs["gates_rpy"][target_gate][2] if "gates_rpy" in obs else _NOMINAL_GATE_YAW[target_gate]
                
            delta_pos = new_gate_pos - self._planned_gates_pos[target_gate]
            
            if np.linalg.norm(delta_pos) > _REPLAN_THRESHOLD:
                normal = np.array([np.cos(new_yaw), np.sin(new_yaw), 0.0])
                
                pre_idx, post_idx = self._gate_indices[target_gate]
                prev_wp = self._waypoints[pre_idx - 1]
                if np.dot(new_gate_pos - prev_wp, normal) < 0:
                    normal = -normal
                    
                self._waypoints[pre_idx] = new_gate_pos - _GATE_MARGIN * normal
                self._waypoints[post_idx] = new_gate_pos + _GATE_MARGIN * normal
                
                self._planned_gates_pos[target_gate] = new_gate_pos
                self._replanned_gates.add(target_gate)
                needs_rebuild = True

        # --- 2. OBSTACLE UPDATE (Only updates state, _build_spline handles the math) ---
        if "obstacles_pos" in obs:
            current_obs_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)
            for i in range(len(current_obs_pos)):
                delta = current_obs_pos[i] - self._planned_obstacles_pos[i]
                if np.linalg.norm(delta) > _REPLAN_THRESHOLD:
                    self._planned_obstacles_pos[i] = current_obs_pos[i]
                    needs_rebuild = True
                    
                    # --- NEW: DYNAMIC BRAKING ---
                    # If an obstacle suddenly appears in our way, give the drone 
                    # an extra 0.3 seconds to safely navigate the complex dodge!
                    self._t_total += 0.3
                    print(f"Obstacle Dodged! Relaxing flight time to {self._t_total:.2f}s")
                    # ----------------------------

        if needs_rebuild:
            self._build_spline()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        # --- RECORD FLIGHT PATH ---
        if "pos" in obs:
            self._path_history.append(obs["pos"].copy())
        # --------------------------

        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True
            
        self._check_and_replan(obs)
        
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_acc = self._des_acc_spline(t)

        vel_magnitude = np.linalg.norm(des_vel)
        if vel_magnitude > self.MAX_VELOCITY:
            des_vel = des_vel * (self.MAX_VELOCITY / vel_magnitude)

        acc_magnitude = np.linalg.norm(des_acc)
        if acc_magnitude > self.MAX_ACCELERATION:
            des_acc = des_acc * (self.MAX_ACCELERATION / acc_magnitude)

        return np.concatenate((des_pos, des_vel, des_acc, np.zeros(4)), dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self._waypoints = self._base_waypoints.copy()
        self._replanned_gates = set()
        self._path_history = []  # Clear the trail on reset!
        self._build_spline()

    def render_callback(self, sim: Sim):
        t = min(self._tick / self._freq, self._t_total)
        
        # 1. DRAW ACTUAL FLOWN PATH (Orange Trail)
        if len(self._path_history) > 1:
            # We slice the array or draw it to show where the drone physically went
            path_array = np.array(self._path_history)
            draw_line(sim, path_array, rgba=(1.0, 0.5, 0.0, 1.0))
        
        # 2. DRAW TARGET WAYPOINTS (Magenta dots)
        draw_points(sim, self._waypoints, rgba=(1.0, 0.0, 1.0, 1.0), size=0.03)

        # 3. DRAW DANGER ZONES (Red Obstacle Circles & Pillars)
        theta = np.linspace(0, 2 * np.pi, 21) # 21 points creates a closed 20-sided polygon
        for obs_pos in self._planned_obstacles_pos:
            # Draw the red repulsion circle at z=1.0m (roughly drone flight height)
            z_height = 1.0
            circle_points = []
            for th in theta:
                circle_points.append([obs_pos[0] + _OBSTACLE_MARGIN * np.cos(th),
                                      obs_pos[1] + _OBSTACLE_MARGIN * np.sin(th),
                                      z_height])
            draw_line(sim, np.array(circle_points), rgba=(1.0, 0.0, 0.0, 0.8))
            
            # Draw a vertical line representing the center of the obstacle pole
            pole_line = np.array([[obs_pos[0], obs_pos[1], 0.0], [obs_pos[0], obs_pos[1], 1.55]])
            draw_line(sim, pole_line, rgba=(1.0, 0.0, 0.0, 0.4))

        # 4. DRAW IDEAL TRAJECTORY (Green Line)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        # 5. DRAW CURRENT SETPOINT (Red Dot)
        setpoint = self._des_pos_spline(t).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        
        # 6. DRAW FEEDFORWARD VELOCITY VECTORS (Blue Ticks)
        time_samples = np.linspace(0, self._t_total, 15)
        pos_samples = self._des_pos_spline(time_samples)
        vel_samples = self._des_vel_spline(time_samples)
        
        vel_magnitude_array = np.linalg.norm(vel_samples, axis=1, keepdims=True)
        vel_magnitude_array[vel_magnitude_array == 0] = 1.0
        vel_normalized = vel_samples / vel_magnitude_array * 0.2
        
        for pos, vel in zip(pos_samples, vel_normalized):
            end_point = pos + vel
            draw_line(sim, np.array([pos, end_point]), rgba=(0.0, 0.5, 1.0, 0.7))