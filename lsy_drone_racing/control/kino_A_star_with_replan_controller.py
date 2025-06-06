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


from typing import List, Dict, Tuple, Set, Union, Callable, Optional
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
from lsy_drone_racing.tools.planners.kino_A_star import KinoDynamicAStarPlanner, KinoAStarPlannerConfig
from lsy_drone_racing.tools.planners.b_spline_optimizer import UniformBSpline, BsplineOptimizer
from lsy_drone_racing.tools.planners.tube_map import TubeMap, PathSegment

from heapq import heappush, heappop

def gate_observation_func(gates : List[Gate], obs : Dict[str, np.ndarray]) -> None:  
    raw_gate_pos = obs['gates_pos']
    raw_gate_quat = obs['gates_quat']
    observed = obs['gates_visited']

    if len(gates) != len(raw_gate_pos):
        # If the gates are not initilized, initilize with None objects
        gates.clear()
        for i in range(len(raw_gate_pos)):
            gates.append(None)

    gate_inner_size_unobserved = [0.2,0.2,0.2,0.3]
    gate_outer_size_unobserved = [1.0,1.0,1.0,0.8]
    gate_safe_radius_unobserved = [0.4,0.4,0.4,0.4]
    entry_offset_unobserved = [0.3,0.2,0.2,0.1]
    exit_offset_unobserved = [0.7,0.4,0.2,0.4]
    thickness_unobserved = [0.4, 0.2, 0.2, 0.05]
    vel_limit_unobserved = [1.0, 1.0, 0.2, 1.0]

    gate_inner_size_observed = [0.2,0.2,0.2,0.3]
    gate_outer_size_observed = [1.0,1.0,1.0,0.8]
    gate_safe_radius_observed = [0.4,0.4,0.4,0.4]
    entry_offset_observed = [0.3,0.2,0.2,0.1]
    exit_offset_observed = [0.7,0.4,0.2,0.4]
    thickness_observed = [0.4, 0.2, 0.2, 0.05]
    vel_limit_observed = [1.0, 1.0, 0.2, 1.0]

    for index in range(len(gates)):
        if not observed[index]:
            gates[index] = Gate(
                pos = raw_gate_pos[index],
                quat = raw_gate_quat[index],
                inner_width=gate_inner_size_unobserved[index],
                inner_height=gate_inner_size_unobserved[index],
                outer_width=gate_outer_size_unobserved[index],
                outer_height=gate_outer_size_unobserved[index],
                safe_radius=gate_safe_radius_unobserved[index],
                entry_offset=entry_offset_unobserved[index],
                exit_offset=exit_offset_unobserved[index],
                thickness=thickness_unobserved[index],
                vel_limit=vel_limit_unobserved[index]
            )

            if index == 0:
                # u = LinAlgTool.normalize(np.cross(gates[0].norm_vec, np.array([0,0,1])))
                # gates[index].pos = gates[index].pos - u * 0.1
                pass
            elif index == 1:
                gates[index].pos = gates[index].pos + np.array([0, 0, 0.15])
            elif index == 2 :
                gates[index].pos = gates[index].pos - np.array([0, 0, 0.05])
        else:
            gates[index] = Gate(
                pos = raw_gate_pos[index],
                quat = raw_gate_quat[index],
                inner_width=gate_inner_size_observed[index],
                inner_height=gate_inner_size_observed[index],
                outer_width=gate_outer_size_observed[index],
                outer_height=gate_outer_size_observed[index],
                safe_radius=gate_safe_radius_observed[index],
                entry_offset=entry_offset_observed[index],
                exit_offset=exit_offset_observed[index],
                thickness=thickness_observed[index],
                vel_limit=vel_limit_observed[index]
            )
            if index == 2 :
                gates[index].pos = gates[index].pos - np.array([0, 0, 0.05])

def obstacle_observation_func(obstacles : List[Obstacle], obs : Dict[str, np.ndarray]) -> None:
    raw_obs_pos = obs['obstacles_pos']
    visited = obs['obstacles_visited']

    obs_unobserved_radius = [0.25,0.2,0.1,0.15]
    obs_observed_radius = [0.15, 0.15, 0.1, 0.15]

    if len(obstacles) != len(raw_obs_pos):
        obstacles.clear()
        for i in range(len(raw_obs_pos)):
            # If the obstacles are not initilized, initilize with None obejects
            obstacles.append(None)

    for index in range(len(obstacles)):
        if not visited[index]:
            obstacles[index] = Obstacle(raw_obs_pos[index], obs_unobserved_radius[index])
            # obstacles[index].pos = raw_obs_pos[index]
            # obstacles[index].safe_radius = obs_unobserved_radius[index]
        else:
            obstacles[index] = Obstacle(raw_obs_pos[index], obs_observed_radius[index])

            # obstacles[index].pos = raw_obs_pos[index]
            # obstacles[index].safe_radius = obs_observed_radius[index]



class ReplanControllerConfig:
    gate_observation_function: Callable[[List[int], Dict[str, np.ndarray]], None]
    obstacle_observation_function: Callable[[List[int], Dict[str, np.ndarray]], None]
    
    original_traj_path : str

    planner_config : KinoAStarPlannerConfig
    fore_see_gates : int
    b_spline_optimization : bool

    use_tube_map : bool
    tube_map_dir : List[str]
    generate_new_tube : bool
    tube_radius : np.floating
    save_tube_to : str
    
    

class KinoAStarWithReplanController(FresssackController):
    """A-star trajectory controller."""
    planner : KinoDynamicAStarPlanner
    replan_controller_config : ReplanControllerConfig
    tube_map : OccupancyMap3D
    occ_map : OccupancyMap3D

    fig : figure.Figure
    ax: axes.Axes

    def init_tube_map(self):
        self.tube_map = None
        if self.replan_controller_config.use_tube_map:
            if not self.replan_controller_config.generate_new_tube:
                self.tube_maps = TubeMap.read_from_file(path = self.replan_controller_config.tube_map_dir[0])
            else:
                pass

    def init_gate_obstacle_observation(self, obs: Dict[str, NDArray]):
        self.replan_controller_config.gate_observation_function(self.gates, obs)
        self.replan_controller_config.obstacle_observation_function(self.obstacles, obs)

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env = None,  individual_node : bool = True):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)

        # Setup visualizations
        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')

        
        if individual_node:
            self.occ_map_res = 0.05
            self.occ_map_xlim =  [-1.0, 1.5]
            self.occ_map_ylim =[-1.9, 1.6]
            self.occ_map_zlim = [0.1, 1.5]

        

        # Replan controller configuration initilization
        if individual_node:
            # Initilize everything related to replanning
            self.replan_controller_config : ReplanControllerConfig = ReplanControllerConfig()
            self.replan_controller_config.obstacle_observation_function = obstacle_observation_func
            self.replan_controller_config.gate_observation_function = gate_observation_func

            self.replan_controller_config.original_traj_path = r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv"

            self.replan_controller_config.planner_config = KinoAStarPlannerConfig(
                w_time = 100,
                max_vel = 5.0,
                max_acc = 10.0,
                tie_breaker = 1.0 + 1.0 / 10000,
                acc_resolution = 5.0,
                time_resolution= 0.05,
                max_duration = 0.15,
                safety_check_res = 0.02,
                lambda_heu = 5.0
            )

            self.replan_controller_config.use_tube_map = True

            self.replan_controller_config.generate_new_tube = False
            # self.replan_controller_config.tube_map_dir = [r'lsy_drone_racing/saved_map/tube_map_0_3.npz']
            self.replan_controller_config.tube_map_dir = [r'lsy_drone_racing/saved_map/tube_maps_medium.pkl']
            # self.replan_controller_config.generate_new_tube = True
            # self.replan_controller_config.tube_map_dir =[
            #         r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets_segmented.csv",
            #         r"lsy_drone_racing/planned_trajectories/param_b_6_sec_bigger_pillar_segmented.csv"]
            
            # self.replan_controller_config.save_tube_to = r'lsy_drone_racing/saved_map/tube_map_medium.npz'
            # self.replan_controller_config.tube_radius = 0.3

            # self.tube_maps = TubeMap()
            # self.tube_maps.generate_tube_map(
            #     paths=self.replan_controller_config.tube_map_dir,
            #     num_gates = 4,
            #     occ_map_xlim=self.occ_map_xlim,
            #     occ_map_ylim= self.occ_map_ylim,
            #     occ_map_zlim=self.occ_map_zlim,
            #     occ_map_res=self.occ_map_res,
            #     tube_radius=self.replan_controller_config.tube_radius,
            #     save_to = r'lsy_drone_racing/saved_map/tube_maps_tight.pkl'
            # )

            # # self.tube_maps = TubeMap.read_from_file(r'lsy_drone_racing/saved_map/tube_maps_tight.pkl')

            # temp_map = OccupancyMap3D.merge([self.tube_maps.tubes[(0,0)], self.tube_maps.tubes[(0,1)], self.tube_maps.tubes[(0,2)]], mode = 'intersection')
            # temp_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
            #                                     free_color = "LightBlue",
            #                                     free_alpha = 0.5, 
            #                                     occupied_color = None, 
            #                                     #  occupied_color = "MistyRose", 
            #                                     #  occupied_alpha = 0.1
            #                                      )
            # input("Stop here")
            # temp_map = OccupancyMap3D.merge([self.tube_maps.tubes[(1,0)], self.tube_maps.tubes[(1,1)], self.tube_maps.tubes[(1,2)]], mode = 'intersection')
            # temp_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
            #                                     free_color = "LightBlue",
            #                                     free_alpha = 0.5, 
            #                                     occupied_color = None, 
            #                                     #  occupied_color = "MistyRose", 
            #                                     #  occupied_alpha = 0.1
            #                                      )
            # input("Stop here")
            # temp_map = OccupancyMap3D.merge([self.tube_maps.tubes[(2,0)], self.tube_maps.tubes[(2,1)], self.tube_maps.tubes[(2,2)]], mode = 'intersection')
            # temp_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
            #                                     free_color = "LightBlue",
            #                                     free_alpha = 0.5, 
            #                                     occupied_color = None, 
            #                                     #  occupied_color = "MistyRose", 
            #                                     #  occupied_alpha = 0.1
            #                                      )
            # input("Stop here")
            # temp_map = OccupancyMap3D.merge([self.tube_maps.tubes[(3,0)], self.tube_maps.tubes[(3,1)], self.tube_maps.tubes[(3,2)]], mode = 'intersection')
            # temp_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
            #                                     free_color = "LightBlue",
            #                                     free_alpha = 0.5, 
            #                                     occupied_color = None, 
            #                                     #  occupied_color = "MistyRose", 
            #                                     #  occupied_alpha = 0.1
            #                                      )
            # input("Stop here")

            self.replan_controller_config.fore_see_gates = 2
            self.replan_controller_config.b_spline_optimization = False
        
            self.init_tube_map()
            self.init_gate_obstacle_observation(obs = obs)
            self.rebuild_occ_map(use_tube_map = self.replan_controller_config.use_tube_map)

            # Get local path

            t, pos, vel = FresssackController.read_trajectory(self.replan_controller_config.original_traj_path) 
        
            self.traj_dt = t[1] - t[0]
            t = np.array(t)
            if self.replan_controller_config.b_spline_optimization:
                self.current_spline = UniformBSpline()
                self.current_spline.parameter_2_bspline_uniform(pos,v_start = vel[0], v_end = vel[-1], dt = self.traj_dt)
                self.initial_trajactory = self.current_spline
                if LOCAL_MODE:
                    self.current_spline.visualize_B_spline(self.fig, self.ax)
            else:
                self.current_spline = CubicSpline(t, pos)
                if LOCAL_MODE:
                    self.visualize_spline_trajectory(self.fig, self.ax,
                                                    self.current_spline,
                                                    t_start = self.current_spline.x[0],
                                                        t_end = self.current_spline.x[-1])
                self.initial_trajactory = self.current_spline
            
            
            self.trajectory = self.initial_trajactory

            self.planner_config : KinoAStarPlannerConfig = KinoAStarPlannerConfig(
                w_time = 100,
                max_vel = 5.0,
                max_acc = 10.0,
                tie_breaker = 1.0 + 1.0 / 10000,
                acc_resolution = 5.0,
                time_resolution= 0.05,
                max_duration = 0.15,
                safety_check_res = 0.02,
                lambda_heu = 5.0
            )

            self.planner : KinoDynamicAStarPlanner = None

        

        
    def generate_tube_from_path(self, 
                                paths : List[str],
                                tube_radius : np.floating = 0.4,
                                save_to : str = None) -> OccupancyMap3D:
        result = OccupancyMap3D(xlim = self.occ_map_xlim,
                                ylim = self.occ_map_ylim,
                                zlim = self.occ_map_zlim ,
                                resolution=self.occ_map_res,
                                init_val = 1)
        for path in paths:
            t, pos, vel = FresssackController.read_trajectory(path) 
            spline = CubicSpline(t, pos)
            result.add_trajectory_tube(spline = spline, radius = tube_radius)
        if save_to is not None:
            result.save_to_file(path = save_to)
        return result
    
    def replan(self,
               foresee_gates : Optional[int] = None,
               current_pos : Optional[NDArray] = None,
               current_vel : Optional[NDArray] = None,
               skip_all_entrance : Optional[bool] = False
               ) -> Tuple[List[NDArray[np.floating]],List[NDArray[np.floating]]]:
        if foresee_gates is None:
            foresee_gates = self.replan_controller_config.fore_see_gates
        
        i = self.next_gate
        (skip_first_entrance) : bool = KinoAStarWithReplanController.in_entry_region(pos = self.pos, gate = self.gates[i])

        skip_last_exit : bool = False
        if i > 0:
            skip_last_exit = True
            # skip_last_exit = not KinoAStarWithReplanController.in_exit_region(pos = self.pos, gate = self.gates[i - 1])
        else:
            skip_last_exit = True

        dt = self.traj_dt
        
        start_vel = self.vel if current_vel is None else current_vel
        start_pos = self.pos if current_pos is None else current_pos

        path_raw : List[NDArray[np.floating]] = [start_pos]
        vel_raw : List[NDArray[np.floating]]= [start_vel]

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
            path_raw += a_star_path[1:]
            vel_raw += a_star_vel[1:]

            if len(a_star_path) != 0:
                start_pos = a_star_path[-1]
                start_vel = a_star_vel[-1]

        while i < min(len(self.gates), self.next_gate + foresee_gates):
            temp_map = self.rebuild_occ_map_short(next_gate = i, use_tube_map = True, visualize = True)
            self.planner = KinoDynamicAStarPlanner(map = temp_map)
            self.planner.setup_param(params = self.replan_controller_config.planner_config)

            if (not (i == self.next_gate and skip_first_entrance)) and (not skip_all_entrance):
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
                path_raw += a_star_path[1:]
                vel_raw += a_star_vel[1:]

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
                            (-0.1  < np.dot(self.gates[i].norm_vec, dx) < 0.3)) or (np.linalg.norm(dx) <= very_close)

            
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
            path_raw += a_star_path[1:]
            vel_raw += a_star_vel[1:]

            if len(a_star_path) != 0:
                start_pos = a_star_path[-1]
                start_vel = a_star_vel[-1]
            i += 1
        
        return path_raw, vel_raw

    # def compensate_gate_pos(self, idx : List[int] = None):
    #     if idx is None:
    #         self.gates[1].pos = self.gates[1].pos + np.array([0, 0, 0.3])
    #         self.gates[2].pos = self.gates[2].pos - np.array([0, 0, 0.1])
    #         u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
    #         self.gates[0].pos = self.gates[0].pos - u * 0.1
    #         return
    #     if 0 in idx:
    #         u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
    #         self.gates[0].pos = self.gates[0].pos - u * 0.1
    #     if 1 in idx:
    #         self.gates[1].pos = self.gates[1].pos + np.array([0, 0, 0.15])
    #     if True or (2 in idx) :
    #         self.gates[2].pos = self.gates[2].pos - np.array([0, 0, 0.05])
        
    # def compensate_obs_pos(self, idx : List[int] = None):
    #     if idx is None:
    #         idx = [i for i in range(len(self.obstacles))]
    #     for index, obstacle in enumerate(self.obstacles):
    #         if index in idx:
    #             self.obstacles[index].safe_radius = self.obs_init_radius[index]
    #         else:
    #             self.obstacles[index].safe_radius = self.obs_corrupt_radius[index]

        # self.obstacles[0].pos = self.obstacles[0].pos + np.array([0, 0.05, 0])
        # self.obstacles[1].pos = self.obstacles[1].pos + np.array([0.0, 0.05, 0])
        # self.obstacles[2].pos = self.obstacles[2].pos + np.array([0, 0, 0])
        # self.obstacles[3].pos = self.obstacles[3].pos + np.array([0, 0.15, 0])

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
            tube_list = []
            for gate_idx in range(self.next_gate, self.next_gate + self.replan_controller_config.fore_see_gates):
                if not(gate_idx == self.next_gate and KinoAStarWithReplanController.in_entry_region(self.pos, self.gates[gate_idx])):
                    tube_list.append(self.tube_maps.tubes[(gate_idx, PathSegment.OUTER_ZONE)])
                tube_list.append(self.tube_maps.tubes[(gate_idx, PathSegment.ENTRANCE_ZONE)])
                tube_list.append(self.tube_maps.tubes[(gate_idx, PathSegment.EXIT_ZONE)])
            self.occ_map = OccupancyMap3D.merge(tube_list, mode = 'intersection')
            # self.occ_map = self.tube_map.copy()
        else:
            self.occ_map = OccupancyMap3D(xlim = self.occ_map_xlim,
                                        ylim = self.occ_map_ylim,
                                        zlim = self.occ_map_zlim ,
                                        resolution=self.occ_map_res,
                                        init_val = 0)
        for gate in self.gates:
            self.occ_map.add_gate_object(gate, obs_val = 1, free_val = 0)
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
    def rebuild_occ_map_short(self, next_gate : int, use_tube_map : bool = True, visualize : bool = True) -> OccupancyMap3D:
        if hasattr(self, 'occ_map'):
            self.occ_map.clear_visualization()
        
        if use_tube_map:
            tube_list = []
            # if next_gate != 0:
            #     tube_list.append(self.tube_maps.tubes[(next_gate - 1, PathSegment.EXIT_ZONE)])
            tube_list.append(self.tube_maps.tubes[(next_gate, PathSegment.OUTER_ZONE)])
            tube_list.append(self.tube_maps.tubes[(next_gate, PathSegment.ENTRANCE_ZONE)])
            tube_list.append(self.tube_maps.tubes[(next_gate, PathSegment.EXIT_ZONE)])
            occ_map = OccupancyMap3D.merge(tube_list, mode = 'intersection')
        else:
            occ_map = OccupancyMap3D(xlim = self.occ_map_xlim,
                                        ylim = self.occ_map_ylim,
                                        zlim = self.occ_map_zlim ,
                                        resolution=self.occ_map_res,
                                        init_val = 0)
        if next_gate != 0:
            occ_map.add_gate_object(self.gates[next_gate - 1], obs_val = 1, free_val = 0)
        occ_map.add_gate_object(self.gates[next_gate], obs_val = 1, free_val = 0)

        for idx, cylinder in enumerate(self.obstacles):
            occ_map.add_vertical_cylinder(center = cylinder.pos, radius = cylinder.safe_radius)
        if LOCAL_MODE:
            if visualize:
                occ_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False,
                                                free_color = "LightBlue",
                                                free_alpha = 0.5, 
                                                occupied_color = None, 
                                                #  occupied_color = "MistyRose", 
                                                #  occupied_alpha = 0.1
                                                 )
        return occ_map
    
    def update_obj_states(self, obs : Dict[str, NDArray], compensate_gates: bool = True ,compensate_obs : bool = True) -> Tuple[bool, bool]:
        need_gate_update, update_flags = self.check_gate_change(obs['gates_visited'])
        if need_gate_update:
            self.replan_controller_config.gate_observation_function(self.gates, obs)
            # idx = []
            # for index, flag in enumerate(update_flags):
            #     if not obs['gates_visited'][index]:
            #         idx.append(index)
            #     gate_pos = obs['gates_pos'][index]
            #     gate_quat = obs['gates_quat'][index]
            #     self.update_gates(gate_idx = index, gate_pos=gate_pos, gate_quat = gate_quat )   
            # if compensate_gates:
            #     self.compensate_gate_pos(idx = idx)

        need_obs_update, update_flags = self.check_obs_change(obs['obstacles_visited'])

        if need_obs_update:
            self.replan_controller_config.obstacle_observation_function(self.obstacles, obs)
            # idx = []
            # for index, flag in enumerate(update_flags):
            #     if not obs['obstacles_visited'][index]:
            #         idx.append(index)
            #     obs_pos = obs['obstacles_pos'][index]
            #     self.update_obstacles(obst_idx=index, obst_pnt = obs_pos)
            # if compensate_obs:
            #     self.compensate_obs_pos(idx = idx)

        return need_gate_update, need_obs_update

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:  
        # if LOCAL_MODE and self._tick % 10 == 0:
        #     FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = self.pos)
        need_gate_update, need_obs_update = self.update_obj_states(obs= obs)
        # need_gate_update, update_flags = self.check_gate_change(obs['gates_visited'])
        # if need_gate_update:
        #     idx = []
        #     for index, flag in enumerate(update_flags):
        #         if not obs['gates_visited'][index]:
        #             idx.append(index)
        #         gate_pos = obs['gates_pos'][index]
        #         gate_quat = obs['gates_quat'][index]
        #         self.update_gates(gate_idx = index, gate_pos=gate_pos, gate_quat = gate_quat )   
        #     self.compensate_gate_pos(idx = idx)

        # need_obs_update, update_flags = self.check_obs_change(obs['obstacles_visited'])
        # if need_obs_update:
        #     idx = []
        #     for index, flag in enumerate(update_flags):
        #         if not obs['obstacles_visited'][index]:
        #             idx.append(index)
        #         obs_pos = obs['obstacles_pos'][index]
        #         self.update_obstacles(obst_idx=index, obst_pnt = obs_pos)
        #     self.compensate_obs_pos(idx = idx)

        if need_gate_update or need_obs_update:
            # Rebuild OC Map
            # self.rebuild_occ_map(use_tube_map = self.replan_controller_config.use_tube_map)
            if LOCAL_MODE:
                if self.planner is not None:
                    self.planner.remove_plot()
                    
            # self.planner = KinoDynamicAStarPlanner(map = self.occ_map)
            # self.planner.setup_param(self.planner_config)
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
            # self.planner.set_param(w_time = 100,
            #                    max_vel = 5.0,
            #                    max_acc = 10.0,
            #                    tie_breaker = 1.0 + 1.0 / 10000,
            #                    acc_resolution = 5.0,
            #                    time_resolution= 0.05,
            #                    max_duration = 0.2,
            #                    safety_check_res = 0.02,
            #                    lambda_heu = 5.0
            #                    )


            # Replan using A star
            self.path_raw, self.vel_raw = self.replan(foresee_gates = self.replan_controller_config.fore_see_gates,
                                                       skip_all_entrance = False)
            if self.path_raw is not None:
                if self.replan_controller_config.b_spline_optimization:
                    if self.current_spline is not None:
                            self.current_spline.remove_plot()
                    self.current_spline = UniformBSpline()
                    self.current_spline.parameter_2_bspline_uniform(self.path_raw, v_start = self.vel, v_end = np.array([0,0,0]), dt = self.traj_dt, offset = self.current_t)
                    self.trajectory = self.current_spline
                    if LOCAL_MODE:
                        self.current_spline.visualize_B_spline(self.fig, self.ax)
                else:
                    t = np.linspace(self.current_t - 2 * self.traj_dt, self.current_t + self.traj_dt * (len(self.path_raw) - 2), len(self.path_raw))
                    self.current_spline= CubicSpline(t, self.path_raw)
                    if LOCAL_MODE:
                        self.visualize_spline_trajectory(self.fig, self.ax,
                                                        self.current_spline,
                                                        t_start = self.current_spline.x[0],
                                                            t_end = self.current_spline.x[-1])
                    self.trajectory = self.current_spline
            else:
                pass
        return np.concatenate((self.trajectory(self.current_t), np.zeros(10)), dtype=np.float32)





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
        self.update_next_gate()
        return self._finished
        """Reset the time step counter."""
        self._tick = 0

