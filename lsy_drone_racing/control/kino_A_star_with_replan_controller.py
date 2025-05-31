"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False


from typing import List, Dict, Tuple, Set, Union
import numpy as np
from scipy.interpolate import CubicSpline, BSpline
from scipy.interpolate import interp1d

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.fresssack_controller import FresssackController
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

from lsy_drone_racing.tools.ext_tools import TransformTool, LinAlgTool
from lsy_drone_racing.tools.race_objects import Gate, Obstacle
from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D
from lsy_drone_racing.tools.planners.kino_A_star import KinoDynamicAStarPlanner
from lsy_drone_racing.tools.planners.b_spline_optimizer import UniformBSpline, BsplineOptimizer
from heapq import heappush, heappop


class KinoAStarWithReplanController(FresssackController):
    """A-star trajectory controller."""


    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        
        
        self.obs_init_radius = [0.25,0.2,0.1,0.15]
        self.obs_corrupt_radius = [0.15, 0.2, 0.1, 0.15]

        self.init_gates(obs = obs,
                         gate_inner_size = [0.2,0.2,0.2,0.3],
                         gate_outer_size = [1.0,1.0,1.0,0.8],
                         gate_safe_radius = [0.4,0.4,0.4,0.4],
                         entry_offset = [0.1,0.4,0.2,0.1],
                         exit_offset = [0.5,0.4,0.2,0.4],
                        #  entry_offset = [0.3,0.7,0.3,0.2],
                        #  exit_offset = [0.5,0.1,0.1,0.3],
                         thickness = [0.4, 0.2, 0.2, 0.05],
                         vel_limit = [1.0, 1.0, 0.2, 1.0])        
        self.init_obstacles(obs = obs,
                            obs_safe_radius = self.obs_init_radius)
        self.init_states(obs = obs)
        self.compensate_gate_pos(idx = [0,1,2,3])
        self.compensate_obs_pos()

        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Here starts A-star planner initilization
        self.occ_map_res = 0.05
        self.occ_map_xlim =  [-1.0, 1.5]
        self.occ_map_ylim =[-1.9, 1.6]
        self.occ_map_zlim = [0.1, 1.5]

        self.use_tube_map = True

        # Get the planning tube.
        # if self.use_tube_map:
        #     self.tube_radius = 0.4
        #     self.tube_map = OccupancyMap3D(xlim = self.occ_map_xlim, ylim = self.occ_map_ylim, zlim = self.occ_map_zlim , resolution=self.occ_map_res, init_val = 1)
        #     paths = [r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv",
        #             r"lsy_drone_racing/planned_trajectories/param_c_6_sec_bigger_pillar.csv"]
        #     for path in paths:
        #         t, pos, vel = FresssackController.read_trajectory(path) 
        #         spline = CubicSpline(t, pos)
        #         self.tube_map.add_trajectory_tube(spline = spline, radius = self.tube_radius)
        # self.tube_map.save_to_file(path = r'lsy_drone_racing/saved_map/tube_map_medium.npz')
        if self.use_tube_map:
            self.tube_map = OccupancyMap3D.from_file(path = r'lsy_drone_racing/saved_map/tube_map_0_3.npz')
        # Build the occupancy map
        self.rebuild_occ_map(use_tube_map = self.use_tube_map)

        # Get local path
        self.use_b_spline = False
        self.fore_see_gates = 4
        t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv") 
        self.dt = t[1] - t[0]
        t = np.array(t)
        if self.use_b_spline:
            self.current_spline = UniformBSpline()
            self.current_spline.parameter_2_bspline_uniform(pos,v_start = vel[0], v_end = vel[-1], dt = self.dt)
            self.initial_trajactory = self.current_spline.b_spline
            if LOCAL_MODE:
                self.current_spline.visualize_B_spline(self.fig, self.ax)
        else:
            self.current_spline = CubicSpline(t * 1, pos)
            if LOCAL_MODE:
                self.visualize_spline_trajectory(self.fig, self.ax,
                                                  self.current_spline,
                                                  t_start = self.current_spline.x[0],
                                                    t_end = self.current_spline.x[-1])
            self.initial_trajactory = self.current_spline
        
        
        self.trajectory = self.initial_trajactory
        self.planner = None

        

        

            
    def replan(self, foresee_gates : Union[int, None] = None) -> Tuple[List[NDArray[np.floating]],List[NDArray[np.floating]]]:
        if foresee_gates is None:
            foresee_gates = self.fore_see_gates
        
        i = self.next_gate
        skip_entrance : bool = KinoAStarWithReplanController.in_entry_region(pos = self.pos, gate = self.gates[i])

        skip_last_exit : bool = False
        if i > 0:
            skip_last_exit = True

            # skip_last_exit = not KinoAStarWithReplanController.in_exit_region(pos = self.pos, gate = self.gates[i - 1])
        else:
            skip_last_exit = True

        dt = self.dt

        start_vel = self.vel
        start_pos = self.pos

        path_raw : List[NDArray[np.floating]] = []
        vel_raw : List[NDArray[np.floating]]= []

        if i != 0 and (not((skip_last_exit))):
            print(f'Planning to gate {i - 1} exit')
            goal_position = self.gates[i-1].pos + self.gates[i-1].exit_offset * self.gates[i-1].norm_vec
            goal_vel = self.gates[i-1].norm_vec * self.gates[i-1].vel_limit
            
            def close_enough(current_pos: NDArray[np.floating],
                    current_vel : Union[NDArray[np.floating],List[NDArray[np.floating]]]) -> bool:
                x = goal_position
                dx = x - current_pos
                very_close = 0.1
                return ((self.gates[i - 1].in_gate_cylinder(current_pos)) and
                            (-0.5 * self.gates[i - 1].exit_offset  < np.dot(self.gates[i - 1].norm_vec, dx) < self.gates[i - 1].exit_offset * 0.5)) or (np.linalg.norm(dx) <= very_close)

            
            plan_success, elapsed_time = self.planner.search(
                        start_pt = start_pos,
                        start_v = start_vel,
                        end_pt = goal_position,
                        end_v = goal_vel,
                        time_out = 1000,
                        fig = self.fig,
                        ax = self.ax,
                        close_enough = close_enough,
                        soft_collision_constraint = False)
            
            if not plan_success:
                print("The planner did not succeed in planning a path")
                return None, None

            a_star_path , a_star_vel = self.planner.get_sample_path(dt = dt)
            path_raw += a_star_path
            vel_raw += a_star_vel

            if len(a_star_path) != 0:
                start_pos = a_star_path[-1]
                start_vel = a_star_vel[-1]

        while i < min(len(self.gates), self.next_gate + foresee_gates):
            if i != self.next_gate or not(skip_entrance):
                print(f'Planning to gate entrance {i}')
                goal_position = self.gates[i].pos - self.gates[i].norm_vec * self.gates[i].entry_offset
                goal_vel = self.gates[i].norm_vec * self.gates[i].vel_limit

                def close_enough(current_pos: NDArray[np.floating],
                                current_vel : Union[NDArray[np.floating],List[NDArray[np.floating]]]) -> bool:
                    x = goal_position
                    dx = x - current_pos
                    upper_bound = 0.2
                    lower_bound = -0.1
                    very_close = 0.1
                    return ((self.gates[i].in_gate_cylinder(current_pos)) and
                                (lower_bound  < np.dot(self.gates[i].norm_vec, dx) < upper_bound)) or (np.linalg.norm(dx, ord = np.inf) < very_close)
                
                plan_success, elapsed_time = self.planner.search(
                                    start_pt = start_pos,
                                    start_v = start_vel,
                                    end_pt = goal_position,
                                    end_v = goal_vel,
                                    time_out = 1000,
                                    fig = self.fig,
                                    ax = self.ax,
                                    close_enough = close_enough,
                                    soft_collision_constraint = False)      
                if not plan_success:
                    print("The planner did not succeed in planning a path")
                    return None, None

                a_star_path , a_star_vel = self.planner.get_sample_path(dt = dt)
                path_raw += a_star_path
                vel_raw += a_star_vel

                if len(a_star_path) != 0:
                    start_pos = a_star_path[-1]
                    start_vel = a_star_vel[-1]
            
            print(f'Planning to gate {i} exit')
            goal_position = self.gates[i].pos + self.gates[i].exit_offset * self.gates[i].norm_vec
            goal_vel = self.gates[i].norm_vec * self.gates[i].vel_limit
            
            def close_enough(current_pos: NDArray[np.floating],
                    current_vel : Union[NDArray[np.floating],List[NDArray[np.floating]]]) -> bool:
                x = goal_position
                dx = x - current_pos
                very_close = 0.1
                return ((self.gates[i].in_gate_cylinder(current_pos)) and
                            (-0.5 * self.gates[i].exit_offset  < np.dot(self.gates[i].norm_vec, dx) < self.gates[i].exit_offset * 0.5)) or (np.linalg.norm(dx) <= very_close)

            
            plan_success, elapsed_time = self.planner.search(
                        start_pt = start_pos,
                        start_v = start_vel,
                        end_pt = goal_position,
                        end_v = goal_vel,
                        time_out = 1000,
                        fig = self.fig,
                        ax = self.ax,
                        close_enough = close_enough,
                        soft_collision_constraint = False)
            
            if not plan_success:
                print("The planner did not succeed in planning a path")
                return None, None

            a_star_path , a_star_vel = self.planner.get_sample_path(dt = dt)
            path_raw += a_star_path
            vel_raw += a_star_vel

            if len(a_star_path) != 0:
                start_pos = a_star_path[-1]
                start_vel = a_star_vel[-1]
            i += 1
        
        return path_raw, vel_raw

    def compensate_gate_pos(self, idx : List[int] = None):
        if idx is None:
            self.gates[1].pos = self.gates[1].pos + np.array([0, 0, 0.3])
            self.gates[2].pos = self.gates[2].pos - np.array([0, 0, 0.1])
            u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
            self.gates[0].pos = self.gates[0].pos - u * 0.1
            return
        
        if 0 in idx:
            u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
            self.gates[0].pos = self.gates[0].pos - u * 0.1
        if 1 in idx:
            self.gates[1].pos = self.gates[1].pos + np.array([0, 0, 0.15])
        if True or (2 in idx) :
            self.gates[2].pos = self.gates[2].pos - np.array([0, 0, 0.05])
        


    def compensate_obs_pos(self, idx : List[int] = None):
        if idx is None:
            idx = [i for i in range(len(self.obstacles))]
        for index, obstacle in enumerate(self.obstacles):
            if index in idx:
                self.obstacles[index].safe_radius = self.obs_init_radius[index]
            else:
                self.obstacles[index].safe_radius = self.obs_corrupt_radius[index]

        # self.obstacles[0].pos = self.obstacles[0].pos + np.array([0, 0.05, 0])
        # self.obstacles[1].pos = self.obstacles[1].pos + np.array([0.0, 0.05, 0])
        # self.obstacles[2].pos = self.obstacles[2].pos + np.array([0, 0, 0])
        # self.obstacles[3].pos = self.obstacles[3].pos + np.array([0, 0.15, 0])
        pass

    def in_entry_region(pos : NDArray[np.floating], gate : Gate)-> bool:
        dx = gate.pos - pos
        return ((gate.in_gate_cylinder(pos = pos)) and
                            (0 <= np.dot(gate.norm_vec, dx) < gate.entry_offset + 0.2))
    
    def in_exit_region(pos:NDArray[np.floating], gate : Gate)-> bool:
        dx = gate.pos - pos
        return ((gate.in_gate_cylinder(pos = pos)) and
                            (-gate.exit_offset-0.2 < np.dot(gate.norm_vec, dx) <= 0))    

    def rebuild_occ_map(self, use_tube_map : bool = True):
        if hasattr(self, 'occ_map'):
            self.occ_map.clear_visualization()
        if use_tube_map:
            self.occ_map = self.tube_map.copy()
        else:
            self.occ_map = OccupancyMap3D(xlim = self.occ_map_xlim, ylim = self.occ_map_ylim, zlim = self.occ_map_zlim , resolution=self.occ_map_res, init_val = 0)
        
        for gate in self.gates:
            self.occ_map.add_gate_object(gate, obs_val = 1, free_val = 0)
            # self.occ_map.add_gate(center = gate.pos, inner_size = gate.inner_height,outer_size=gate.outer_height,  thickness = gate.thickness, norm_vec = gate.norm_vec)
        for idx, cylinder in enumerate(self.obstacles):
            self.occ_map.add_vertical_cylinder(center = cylinder.pos, radius = cylinder.safe_radius)
        if LOCAL_MODE:
            self.occ_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
                                                free_color = "LightBlue",
                                                free_alpha = 0.5, 
                                                occupied_color = None, 
                                                #  occupied_color = "MistyRose", 
                                                #  occupied_alpha = 0.1
                                                 )



       
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:  
        # if LOCAL_MODE and self._tick % 10 == 0:
        #     FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = self.pos)
        need_gate_update, update_flags = self.check_gate_change(obs['gates_visited'])
        if need_gate_update:
            idx = []
            for index, flag in enumerate(update_flags):
                if not obs['gates_visited'][index]:
                    idx.append(index)
                gate_pos = obs['gates_pos'][index]
                gate_quat = obs['gates_quat'][index]
                self.update_gates(gate_idx = index, gate_pos=gate_pos, gate_quat = gate_quat )   
            self.compensate_gate_pos(idx = idx)

        need_obs_update, update_flags = self.check_obs_change(obs['obstacles_visited'])
        if need_obs_update:
            idx = []
            for index, flag in enumerate(update_flags):
                if not obs['obstacles_visited'][index]:
                    idx.append(index)
                obs_pos = obs['obstacles_pos'][index]
                self.update_obstacles(obst_idx=index, obst_pnt = obs_pos)
            self.compensate_obs_pos(idx = idx)

        if need_gate_update or need_obs_update:
            # Rebuild OC Map
            self.rebuild_occ_map(use_tube_map = self.use_tube_map)
            if LOCAL_MODE:
                if (self.planner is not None) and (self.planner._open_set_plot is not None):
                    try:
                        self.planner._open_set_plot.remove()
                    except:
                        pass
                    
            self.planner = KinoDynamicAStarPlanner(map = self.occ_map)
            # self.planner.set_param(w_time = 100,
            #                    max_vel = 5.0,
            #                    max_acc = 10.0,
            #                    tie_breaker = 1.0 + 1.0 / 10000,
            #                    acc_resolution = 5.0,
            #                    time_resolution= 0.05,
            #                    max_duration = 0.15,
            #                    safety_check_res = 0.02,
            #                    lambda_heu = 5.0
            #                    )
            self.planner.set_param(w_time = 100,
                               max_vel = 5.0,
                               max_acc = 10.0,
                               tie_breaker = 1.0 + 1.0 / 10000,
                               acc_resolution = 5.0,
                               time_resolution= 0.05,
                               max_duration = 0.2,
                               safety_check_res = 0.02,
                               lambda_heu = 5.0
                               )

        next_gate = self.update_next_gate()
        if need_gate_update or need_obs_update:
            # Replan using A star
            if need_gate_update or need_obs_update:
                self.path_raw, self.vel_raw = self.replan(foresee_gates = 2)
                if self.path_raw is not None:
                    if self.use_b_spline:
                        if self.current_spline is not None and hasattr(self.current_spline,'_last_plot') and self.current_spline._last_plot is not None:
                            try:
                                self.current_spline._last_plot.remove()
                            except:
                                pass
                        self.current_spline = UniformBSpline()
                        self.current_spline.parameter_2_bspline_uniform(self.path_raw, v_start = self.vel, v_end = np.array([0,0,0]), dt = self.dt, offset = self.current_t)
                        self.trajectory = self.current_spline.b_spline
                        if LOCAL_MODE:
                            self.current_spline.visualize_B_spline(self.fig, self.ax)
                    else:
                        t = np.linspace(self.current_t - 3 * self.dt, self.current_t + self.dt * (len(self.path_raw) - 3), len(self.path_raw))
                        self.current_spline= CubicSpline(t, self.path_raw)
                        if LOCAL_MODE:
                            self.visualize_spline_trajectory(self.fig, self.ax,
                                                            self.current_spline,
                                                            t_start = self.current_spline.x[0],
                                                                t_end = self.current_spline.x[-1])
                        self.trajectory = self.current_spline
                
                
    
        return np.concatenate((self.trajectory(self.current_t), np.zeros(10)), dtype=np.float32)

    
    def get_trajectory_tracking_target(self, lookahead: float = 0.3, interval: float = 0.02, forward_window: float = 2.0) -> NDArray[np.floating]:
        t_min = max(self.track_t - 3.0, self.current_t)
        t_max = min(self.trajectory.x[-1], t_min + forward_window)
        if t_min >= self.trajectory.x[-1]:
            return np.concatenate([self.trajectory(self.trajectory.x[-1]), np.zeros(3)])

        t_samples = np.arange(t_min, t_max + 1e-6, interval)
        traj_points = self.trajectory(t_samples)
        dists_to_drone = np.linalg.norm(traj_points - self.pos, axis=1)

        idx_closest = np.argmin(dists_to_drone)
        # if dists_to_drone[idx_closest] > lookahead
        #     t_target = t_samples[idx_closest]
        if True:
            cum_dist = 0.0
            for i in range(idx_closest, len(t_samples) - 1):
                step = np.linalg.norm(traj_points[i + 1] - traj_points[i])
                cum_dist += step
                if cum_dist >= lookahead:
                    t_target = t_samples[i + 1]
                    break
            else:
                t_target = t_samples[-1]

        self.track_t = t_target

        tangent = self.trajectory(t_target, nu=1)
        tangent_dir = LinAlgTool.normalize(tangent)
        target_vel = tangent_dir * self.velocity
        target_pos =  self.trajectory(t_target)
        if LOCAL_MODE and self._tick % 10 == 0:
            FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = target_pos, color = 'purple')
            FresssackController.draw_drone_vel(fig = self.fig, ax = self.ax, pos = target_pos, vel = target_vel, color = 'purple')

        return np.concatenate([target_pos, target_vel])



    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self.step_update(obs = obs)
        return self._finished
        """Reset the time step counter."""
        self._tick = 0

