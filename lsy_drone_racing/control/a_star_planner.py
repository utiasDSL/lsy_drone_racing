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
except:
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

from heapq import heappush, heappop
class AStarPlanner3D:
    def __init__(self, occ_map: OccupancyMap3D):
        self.map = occ_map
        self.directions = [(dx, dy, dz) for dx in [-1, 0, 1]
                           for dy in [-1, 0, 1]
                           for dz in [-1, 0, 1]
                           if not (dx == dy == dz == 0)]

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    

    def plan(self,
             start_pos: NDArray[np.floating],
             goal_pos: NDArray[np.floating],
             parent_dir: NDArray[np.floating] = None,
             beta: np.floating = 2.0,
             gamma : np.floating = 50.0) -> List[NDArray[np.floating]]:

        start = tuple(self.map.world_to_map(start_pos))
        goal = tuple(self.map.world_to_map(goal_pos))


        open_set = [(self.heuristic(start, goal), 0.0, start, None, parent_dir)]
        visited = {}

        while open_set:
            _, g_cost, current, parent, vel_dir = heappop(open_set)

            if current in visited:
                continue
            visited[current] = (parent, vel_dir)

            if current == goal:
                path = [current]
                node = parent
                while node is not None:
                    path.append(node)
                    node, _ = visited[node]
                path.reverse()
                return [self.map.map_to_world(np.array(p)) for p in path]

            for d in self.directions:
                neighbor = tuple(np.add(current, d))
                occupancy_penalty = self.map.get_cost(neighbor)
                # if occupancy_penalty > 500:
                #     continue
                dir_unit = LinAlgTool.normalize(np.array(d))
                turn_penalty = 0.0
                if vel_dir is not None:
                    cos_angle = LinAlgTool.dot_safe(dir_unit, vel_dir)
                    cos_angle_2 = LinAlgTool.dot_safe(LinAlgTool.normalize(dir_unit[:2]), LinAlgTool.normalize(vel_dir[:2]))
                    # if cos_angle_2 < 0.5:
                    #     continue
                    turn_penalty = (1 - cos_angle) 

                new_vel_dir = dir_unit

                cost = 1.0 + beta * turn_penalty + gamma * occupancy_penalty
                new_g = g_cost + cost
                new_f = new_g + self.heuristic(neighbor, goal)

                heappush(open_set, (new_f, new_g, neighbor, current, new_vel_dir))

        return None
    
    # def plan_region(self,
    #          start_pos: NDArray[np.floating],
    #          goal_pos:NDArray[np.floating],
    #          goal_region_idx: Set[NDArray[np.integer]],
    #          parent_dir: NDArray[np.floating] = None,
    #          beta: np.floating = 50.0,
    #          gamma : np.floating = 50.0) -> List[NDArray[np.floating]]:

    #     start = tuple(self.map.world_to_map(start_pos))
    #     goal = tuple(self.map.world_to_map(goal_pos))


    #     open_set = [(self.heuristic(start, goal), 0.0, start, None, parent_dir)]
    #     visited = {}

    #     while open_set:
    #         _, g_cost, current, parent, vel_dir = heappop(open_set)

    #         if current in visited:
    #             continue
    #         visited[current] = (parent, vel_dir)

    #         if current in goal_region_idx:
    #             path = [current]
    #             node = parent
    #             while node is not None:
    #                 path.append(node)
    #                 node, _ = visited[node]
    #             path.reverse()
    #             return [self.map.map_to_world(np.array(p)) for p in path]

    #         for d in self.directions:
    #             neighbor = tuple(np.add(current, d))
    #             occupancy_penalty = self.map.get_cost(neighbor)
    #             dir_unit = LinAlgTool.normalize(np.array(d))
    #             turn_penalty = 0.0
    #             if vel_dir is not None:
    #                 cos_angle = LinAlgTool.dot_safe(dir_unit, vel_dir)
    #                 cos_angle_2 = LinAlgTool.dot_safe(dir_unit[:2], vel_dir[:2])
    #                 if cos_angle_2 < 0.2:
    #                     continue
    #                 turn_penalty = (1 - cos_angle_2) 

    #             new_vel_dir = dir_unit

    #             cost = 1.0 + beta * turn_penalty + gamma * occupancy_penalty
    #             new_g = g_cost + cost
    #             new_f = new_g + self.heuristic(neighbor, goal)

    #             heappush(open_set, (new_f, new_g, neighbor, current, new_vel_dir))

    #     return None


class AstarController(FresssackController):
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

        self.velocity = 0.5

        self.init_gates(obs = obs,
                         gate_inner_size = [0.3,0.4,0.2,0.4],
                         gate_outer_size = [1.0,0.8,0.8,0.8],
                         gate_safe_radius = [0.4,0.4,0.4,0.4],
                         entry_offset = [0.5,0.7,0.7,0.2],
                         exit_offset = [0.8,0.3,0.2,0.3],
                         thickness = [0.4, 0.4, 0.05, 0.05])
        self.init_obstacles(obs = obs,
                            obs_safe_radius = [0.1,0.15,0.1,0.15])
        self.init_states(obs = obs)

        self.compensate_gate_pos()
        self.compensate_obs_pos()
        # u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
        # self.gates[0].pos = self.gates[0].pos - u * 0.1

        # for gate in self.gates:
        #     u = LinAlgTool.normalize(np.cross(gate.norm_vec, np.array([0,0,1])))
        #     gate.pos = gate.pos - u * 0.1
        


        # Here starts A-star planner initilization
        self.occ_map_res = 0.1
        self.occ_map_xlim =  [-1.0, 1.5]
        self.occ_map_ylim =[-1.9, 1.6]
        self.occ_map_zlim = [0.1, 1.5]
        
        # Build the occupancy map
        self.occ_map = OccupancyMap3D(xlim = self.occ_map_xlim, ylim = self.occ_map_ylim, zlim = self.occ_map_zlim , resolution=self.occ_map_res)
        for gate in self.gates:
            self.occ_map.add_gate(center = gate.pos, inner_size = gate.inner_height,outer_size=gate.outer_height,  thickness = gate.thickness, norm_vec = gate.norm_vec)
        for idx, cylinder in enumerate(self.obstacles):
            self.occ_map.add_vertical_cylinder(center = cylinder.pos, radius = cylinder.safe_radius)
        
        # Visualize the map
        if LOCAL_MODE:
            self.fig = plt.figure(figsize=(12, 12))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.occ_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, adjust = True, new_window = False)
        
        # Plan a local path

        self.fore_see_gates = 1
        self.trajectory = None
        self.planner : AStarPlanner3D = AStarPlanner3D(occ_map = self.occ_map)


        self.path_segments = self.replan()
        if LOCAL_MODE:
            AstarController.visualize_A_star_path(fig = self.fig, ax = self.ax, path = self.path_segments)

        # Generate smooth path
        self.ds_step = 3
        self.path_raw = np.array(self.path_segments)
        path_length = self.path_raw.shape[0] * self.occ_map_res
        t_axis = np.linspace(self.current_t, self.current_t + path_length / self.velocity, self.path_raw.shape[0])
        self.trajectory = self.generate_smooth_path(t_axis = t_axis, ds_step = self.ds_step)
        self.track_t = self.trajectory.x[0]
        if LOCAL_MODE:
            FresssackController.visualize_trajectory(fig = self.fig, ax = self.ax,t_start = t_axis[0], t_end = t_axis[-1], trajectory = self.trajectory)
        
        # Enable a control tracker
        self.track_t = 0.0
    
    def compensate_gate_pos(self):
        # return
        u = LinAlgTool.normalize(np.cross(self.gates[0].norm_vec, np.array([0,0,1])))
        self.gates[0].pos = self.gates[0].pos - u * 0.1
        u = LinAlgTool.normalize(np.cross(self.gates[2].norm_vec, np.array([0,0,1])))
        self.gates[2].pos = self.gates[2].pos + u * 0.1
        u = LinAlgTool.normalize(np.cross(self.gates[2].norm_vec, np.array([0,0,1])))
        self.gates[2].pos = self.gates[2].pos - u * 0.1
    def compensate_obs_pos(self):
        # self.obstacles[0].pos = self.obstacles[0].pos + np.array([0, 0.05, 0])
        self.obstacles[1].pos = self.obstacles[1].pos + np.array([0.0, 0.05, 0])
        self.obstacles[2].pos = self.obstacles[2].pos + np.array([0, 0, 0])
        self.obstacles[3].pos = self.obstacles[3].pos + np.array([0, 0.15, 0])

    def replan(self, connect_end = False,  skip_this_entry = False, skip_the_gate = False) -> List[np.ndarray]:
       
        path_segments = []

        # if self.trajectory is not None:
        #     current_pos = self.trajectory(self.current_t)
        # else:
        #     current_pos = self.pos
        current_pos = self.pos
        if np.linalg.norm(self.vel) < 1e-3:
            parent_dir = None
        else:
            if self.trajectory is None:
                parent_dir = None
            else:
                parent_dir =  LinAlgTool.normalize(self.trajectory(self.current_t) - self.trajectory(self.current_t - 0.1))
            # parent_dir = LinAlgTool.normalize(self.vel)

        if connect_end:
            print("Connecting the exit of " + str(self.next_gate - 1))
            gate = self.gates[self.next_gate - 1]
            entry, exit = AstarController.gate_entry_exit(gate = gate, entry_offset = gate.entry_offset, exit_offset = gate.exit_offset)
            path_1 = self.planner.plan(start_pos = current_pos, goal_pos = exit, parent_dir = parent_dir)
            if path_1 is None:
                    return None
            if len(path_1) != 1:
                parent_dir = LinAlgTool.normalize(path_1[-1] - path_1[-2])
            path_segments += path_1[:-1]
            current_pos = path_1[-1]

        for i in range(self.next_gate, min(self.next_gate + self.fore_see_gates, len(self.gates))):
            gate = self.gates[i]
            # goal_region = gate.gate_goal_region_voxels(omap = self.occ_map, offset = -self.gate_offset)
            # path_1 = self.planner.plan_region(start_pos = current_pos, goal_pos = entry, goal_region_idx = goal_region, parent_dir = parent_dir)

            # entry, exit = AstarController.gate_entry_exit(gate = gate, offset = self.gate_offset)
            entry, exit = AstarController.gate_entry_exit(gate = gate, entry_offset = gate.entry_offset, exit_offset = gate.exit_offset)
            if i == self.next_gate and skip_this_entry:
                print("Skipping gate entry" + str(self.next_gate))
                path_1 = self.planner.plan(start_pos = current_pos, goal_pos = gate.pos, parent_dir = parent_dir)
                if path_1 is None:
                    return None
                if len(path_1) != 1:
                    parent_dir = LinAlgTool.normalize(path_1[-1] - path_1[-2])
                path_2 = self.planner.plan(start_pos = path_1[-1], goal_pos = exit, parent_dir = parent_dir)
                if path_2 is None:
                    return None
                if len(path_2) != 1:
                    parent_dir = LinAlgTool.normalize(path_2[-1] - path_2[-2])
            
                path_segments += path_1[:-1] + path_2[:-1]
                current_pos = path_2[-1]

            elif i == self.next_gate and skip_the_gate:
                print("Skipping gate" + str(self.next_gate))
                path_1 = self.planner.plan(start_pos = current_pos, goal_pos = exit, parent_dir = parent_dir)
                if path_1 is None:
                    return None
                if len(path_1) != 1:
                    parent_dir = LinAlgTool.normalize(path_1[-1] - path_1[-2])
                path_segments += path_1[:-1]
                current_pos = path_1[-1]
            
            else:
                path_1 = self.planner.plan(start_pos = current_pos, goal_pos = entry, parent_dir = parent_dir)
                if path_1 is None:
                    return None
                if len(path_1) != 1:
                    parent_dir = LinAlgTool.normalize(path_1[-1] - path_1[-2])
                # goal_region = gate.gate_goal_region_voxels(omap = self.occ_map, offset = self.gate_offset)
                # path_2 = self.planner.plan_region(start_pos = path_1[-1], goal_pos = exit, goal_region_idx = goal_region, parent_dir = parent_dir)
                path_2 = self.planner.plan(start_pos = path_1[-1], goal_pos = exit, parent_dir = parent_dir)

                if path_2 is None:
                    return None
                if len(path_2) != 1:
                    parent_dir = LinAlgTool.normalize(path_2[-1] - path_2[-2])
                path_segments += path_1[:-1] + path_2[:-1]
                current_pos = path_2[-1]

        return path_segments
    
    def generate_smooth_path(self, t_axis:np.ndarray, ds_step = 5) -> Union[CubicSpline, None]:
        self.path_raw_ds, t_ds = AstarController.downsample_path(self.path_raw, t_axis, step = ds_step)
        if len(t_ds) == 1:
            return None
        else:
            return CubicSpline(t_ds, self.path_raw_ds)

    def downsample_path(path: NDArray[np.floating],
                    t_axis: NDArray[np.floating],
                    step: int = 5) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        indices = list(range(0, len(path), step))
        if indices[-1] != len(path) - 1:
            indices.append(len(path) - 1)
        path_ds = path[indices]
        t_ds = t_axis[indices]
        return path_ds, t_ds
    
    def visualize_A_star_path(fig: figure.Figure, ax: axes.Axes, path: List[np.ndarray], color='blue', label = True) -> Tuple[figure.Figure, axes.Axes]:
        if LOCAL_MODE and path is not None and len(path) > 1:
            path = np.array(path)

            ax.plot(path[:, 0], path[:, 1], path[:, 2], c=color, linewidth=2, label='A* Path')

            # Mark start and end
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='green', s=50, label='Start')
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='red', s=50, label='Goal')

            ax.legend()
            plt.pause(0.001)
        return fig, ax

    def gate_entry_exit(gate: Gate, entry_offset: float, exit_offset:float = None):
        if exit_offset is None:
            exit_offset = entry_offset
        entry = gate.pos - gate.norm_vec * entry_offset
        exit  = gate.pos + gate.norm_vec * exit_offset
        return entry, exit
    
    def approaching_gate(self, threshold : np.float32 = 3.0):
        return np.linalg.norm(self.pos[:2] - self.gates[self.next_gate].pos[:2]) < self.gates[self.next_gate].entry_offset * threshold and (np.dot(self.vel[:2], self.gates[self.next_gate].norm_vec[:2]) > 0)
    
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        
        if LOCAL_MODE and self._tick % 10 == 0:
            FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = self.pos)
        need_gate_update, update_flags = self.check_gate_change(obs['gates_visited'])
        if need_gate_update:
            for index, flag in enumerate(update_flags):
                gate_pos = obs['gates_pos'][index]
                gate_quat = obs['gates_quat'][index]
                self.update_gates(gate_idx = index, gate_pos=gate_pos, gate_quat = gate_quat )
                    
            self.compensate_gate_pos()

        need_obs_update, update_flags = self.check_obs_change(obs['obstacles_visited'])
        if need_obs_update:
            for index, flag in enumerate(update_flags):
                obs_pos = obs['obstacles_pos'][index]
                self.update_obstacles(obst_idx=index, obst_pnt = obs_pos)
            self.compensate_obs_pos()

        if need_gate_update or need_obs_update:
            # Rebuild OC Map
            self.occ_map = OccupancyMap3D(xlim = self.occ_map_xlim, ylim = self.occ_map_ylim, zlim = self.occ_map_zlim , resolution=self.occ_map_res)
            for gate in self.gates:
                self.occ_map.add_gate(center = gate.pos, inner_size = gate.inner_height,outer_size=gate.outer_height,  thickness = 0.2, norm_vec = gate.norm_vec)
            for cylinder in self.obstacles:
                self.occ_map.add_vertical_cylinder(center = cylinder.pos, radius = cylinder.safe_radius)
            self.planner = AStarPlanner3D(occ_map = self.occ_map)

        next_gate = self.update_next_gate()
        if  next_gate or need_gate_update or need_obs_update:
            # Replan using A star
            if need_gate_update or need_obs_update:
                self.path_segments = self.replan(skip_this_entry = self.approaching_gate(threshold = 1.0), skip_the_gate = self.approaching_gate(threshold = 0.5))
            elif next_gate:
                self.path_segments = self.replan(skip_this_entry = False, skip_the_gate = False, connect_end = True)

        
            # Generate smooth path
            self.path_raw = np.array(self.path_segments)
            path_length = self.path_raw.shape[0] * self.occ_map_res
            t_axis = np.linspace(self.current_t, self.current_t + path_length / self.velocity, self.path_raw.shape[0])
            self.trajectory = self.generate_smooth_path(t_axis = t_axis, ds_step = self.ds_step)
            if self.trajectory is None:
                self.next_gate += 1
                print("Sigularity met at gate" + str(self.next_gate))
                self.path_segments = self.replan(skip_this_entry = False, skip_the_gate = False, connect_end = True)
                self.path_raw = np.array(self.path_segments)
                path_length = self.path_raw.shape[0] * self.occ_map_res
                t_axis = np.linspace(self.current_t, self.current_t + path_length / self.velocity, self.path_raw.shape[0])
                self.trajectory = self.generate_smooth_path(t_axis = t_axis, ds_step = self.ds_step)
            self.track_t = self.trajectory.x[0]
            if LOCAL_MODE:
                self.occ_map.visualize_occupancy_map(fig = self.fig, ax = self.ax, new_window = False)
                AstarController.visualize_A_star_path(fig = self.fig, ax = self.ax, path = self.path_segments)
                FresssackController.visualize_trajectory(fig = self.fig, ax = self.ax,t_start = t_axis[0], t_end = t_axis[-1], trajectory = self.trajectory)
        
        if self.current_t > 0.5:
            target_pos = self.get_trajectory_tracking_target(lookahead = 0.14, forward_window = 1.0, interval = 0.05)
        else:
            target_pos = self.get_trajectory_tracking_target(lookahead = 0.1, forward_window = 1.0, interval = 0.05)
        # target_pos = self.trajectory(self.current_t)
        
        return np.concatenate((target_pos, np.zeros(7)), dtype=np.float32)
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)


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

