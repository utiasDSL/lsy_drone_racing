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


class PrescriptedController(FresssackController):
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
        self.speeding_factor = 1.0
        
        self.init_gates(obs = obs,
                          gate_inner_size = [0.2,0.2,0.25,0.3],
                         gate_outer_size = [1.0,1.0,1.0,0.8],
                         gate_safe_radius = [0.4,0.4,0.4,0.4],
                         entry_offset = [0.3,0.3,0.5,0.1],
                         exit_offset = [0.4,0.4,0.2,0.3],
                        #  entry_offset = [0.3,0.7,0.3,0.2],
                        #  exit_offset = [0.5,0.1,0.1,0.3],
                         thickness = [0.1, 0.1, 0.05, 0.05],
                         vel_limit = [1.0, 1.0, 0.2, 1.0])
        # self.init_obstacles(obs = obs,
        #                     obs_safe_radius = [0.1,0.15,0.1,0.15])
        
        self.init_obstacles(obs = obs,
                            obs_safe_radius = [0.1,0.15,0.1,0.1])
        self.init_states(obs = obs)

        


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

        t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/test_run.csv")     
        initial_B_spline = UniformBSpline()
        initial_B_spline.parameter_2_bspline_uniform(pos,v_start = vel[0], v_end = vel[-1], dt = t[1] - t[0])
        if LOCAL_MODE:
            initial_B_spline.visualize_B_spline(self.fig, self.ax)
        
        self.trajectory = initial_B_spline.b_spline
        
        # input("Check the trajectories!")
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        # if LOCAL_MODE and self._tick % 10 == 0:
        #     FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = self.pos)

        return np.concatenate((self.trajectory(self.current_t * self.speeding_factor), np.zeros(10)), dtype=np.float32)
    
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