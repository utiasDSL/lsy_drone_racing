"""Base class for controller implementations.

Your task is to implement your own controller. This class must be the parent class of your
implementation. You have to use the same function signatures as defined by the base class. Apart
from that, you are free to add any additional methods, attributes, or classes to your controller.

As an example, you could load the weights of a neural network in the constructor and use it to
compute the control commands in the :meth:`compute_control <.BaseController.compute_control>`
method. You could also use the :meth:`step_callback <.BaseController.step_callback>` method to
update the controller state at runtime.

Note:
    You can only define one controller class in a single file. Otherwise we will not be able to
    determine which class to use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from lsy_drone_racing.control.rrt import RRT
import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control import BaseController
import pybullet as p

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class RRT_Controller(BaseController):
    """Base class for controller implementations."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        Instructions:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        # Get intial position
        # print("initial_obs:", initial_obs, type(initial_obs))
        self.start = initial_obs['pos'][:3]
        print(self.start)
        # print("initial_info:", initial_obs)
        self.current_target_index = initial_obs["target_gate"]
        if self.current_target_index != -1:
            self.target_position = initial_obs["gates_pos"][self.current_target_index]
            print(initial_obs["gates_pos"])
        self.obs_list = [(*obstacle, 0.001) for obstacle in initial_obs["obstacles_pos"]]
        self.rrt = RRT(
                       start=self.start,
                       goal=initial_obs["gates_pos"][0],
                       rand_area=[0, 100],
                       obstacle_list= self.obs_list,
                       gates=(initial_obs["gates_pos"]).tolist(),
                       play_area=[-10,10,-10,10,0,10],
                       max_iter = 5000
                       )
        # print(initial_obs["obstacles_pos"])
        self.rrt_path = self.rrt.planning()

        if self.rrt_path is None:
            raise ValueError("RRT failed to find a path.")
        self.rrt_path = np.array(self.rrt_path)  # Convert to NumPy array

        # Create a cubic spline from the RRT path
        self.t_total = len(self.rrt_path)  # Total time proportional to the number of waypoints
        t = np.linspace(0, self.t_total, len(self.rrt_path))
        self.trajectory = CubicSpline(t, self.rrt_path)
        self._tick = 0
        self._freq = initial_info["env_freq"]
        
        try:
            t_vis = np.linspace(0, self.t_total - 1, 100)
            spline_points = self.trajectory(t_vis)
            for i in range(len(spline_points) - 1):
                p.addUserDebugLine(
                    spline_points[i],
                    spline_points[i + 1],
                    lineColorRGB=[1, 0, 0],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # Persistent line
                    physicsClientId=0,
                )
        except p.error:
            print("L")


    def compute_control(self, obs: NDArray[np.floating], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the control for the drone to follow the path through gates."""
        # If the current target index is not -1, move toward the next gate.
        target_pos = self.trajectory(min(self._tick / self._freq, self.t_total))

                
        self.trajectory = CubicSpline(np.linspace(0, self.t_total, len(self.rrt_path)), self.rrt_path)
                
        # print(self.current_target_index)
        return np.concatenate((target_pos, np.zeros(10)))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        self._tick += 1

    def episode_callback(self):
        """Callback function called once after each episode.

        You can use this function to reset your controller's internal state, save training data,
        train your models, compute additional statistics, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.
        """

    def reset(self):
        """Reset internal variables if necessary."""

    def episode_reset(self):
        """Reset the time step counter."""
        self._tick = 0