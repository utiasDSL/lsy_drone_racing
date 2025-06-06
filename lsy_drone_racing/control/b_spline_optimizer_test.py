from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional, Union
from scipy.interpolate import CubicSpline, BSpline

from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D
from lsy_drone_racing.control.kino_A_star_with_replan_controller import ReplanControllerConfig, KinoAStarWithReplanController, obstacle_observation_func, gate_observation_func
from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.planners.b_spline_optimizer import UniformBSpline, BsplineOptimizer

import time

LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False

class BSplineOptimizerTest(KinoAStarWithReplanController):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env = None):
        super().__init__(obs, info, config, env = env, individual_node = False)

        self.freq = config.env.freq
        self._tick = 0

        self.env = env

        t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv")     
        self.dt = t[1] - t[0]

        # Prepare for the occupancy map!
        self.occ_map_res = 0.05
        self.occ_map_xlim =  [-1.0, 1.5]
        self.occ_map_ylim =[-1.9, 1.6]
        self.occ_map_zlim = [0.1, 1.5]

        self.replan_controller_config = ReplanControllerConfig()

        self.replan_controller_config.fore_see_gates = 4
        self.replan_controller_config.b_spline_optimization = True

        self.replan_controller_config.obstacle_observation_function = obstacle_observation_func
        self.replan_controller_config.gate_observation_function = gate_observation_func

        self.replan_controller_config.use_tube_map = True
        self.replan_controller_config.generate_new_tube = False
        self.replan_controller_config.tube_map_dir = [r'lsy_drone_racing/saved_map/tube_maps_medium.pkl']


        self.init_tube_map()
        
        self.init_gate_obstacle_observation(obs = obs)

        # obs['gates_visited'][0] = True
        # obs['gates_pos'][0] = obs['gates_pos'][0] - self.gates[0].norm_vec * 0.3


        # obs['gates_visited'][2] = True
        # u = np.cross(self.gates[2].norm_vec, np.array([0,0,1]))
        # obs['gates_pos'][2] = obs['gates_pos'][2] + u * 0.5

        self.init_gate_obstacle_observation(obs = obs)


        self.rebuild_occ_map(use_tube_map = self.replan_controller_config.use_tube_map)
        
        
        grid, sdf_func =  OccupancyMap3D.compute_sdf(self.occ_map)

        trajectory = UniformBSpline()
        trajectory.parameter_2_bspline_uniform(
            pos,
            v_start = np.linalg.norm(vel[0]) * np.array([0,0,1]),
            v_end = vel[-1],
            dt = self.dt,
        )
        if LOCAL_MODE:
            trajectory.visualize_B_spline(self.fig, self.ax)
        start_time = time.time()
        optimizer = BsplineOptimizer(
            lam_smooth = 100.0,
            lam_vel = 100.0,
            lam_acc = 10000.0,
            lam_sdf = 100000.0,
            v_max = 5.0,
            a_max = 10.0,
            sdf_func = sdf_func,
            sdf_thres = 0.5,
            dt = self.dt,
            verbose=True
        )



        new_ctrl_pts = optimizer.optimize(
           trajectory.ctrl_pts
        )
        print(f'Elapsed time: {time.time() - start_time} sec.')
        optimized_bspline = UniformBSpline()
        optimized_bspline.k = trajectory.k
        optimized_bspline.t = t
        optimized_bspline.ctrl_pts = new_ctrl_pts
        optimized_bspline.b_spline = BSpline(t, new_ctrl_pts, trajectory.k)

        if LOCAL_MODE:
            optimized_bspline.visualize_B_spline(self.fig, self.ax, color = 'blue')
        input("stop here")

