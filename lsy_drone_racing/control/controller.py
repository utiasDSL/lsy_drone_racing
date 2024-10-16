"""Base class for controller implementations.

Your task is to implement your own controller. This class must be the parent class of your
implementation. You have to use the same function signatures as defined by the base class. Apart
from that, you are free to add any additional methods, attributes, or classes to your controller.

As an example, you could load the weights of a neural network in the constructor and use it to
compute the control commands in the `compute_control` method. You could also use the `step_callback`
method to update the controller state at runtime.

Note:
    You can only define one controller class in a single file. Otherwise we will not be able to
    determine which class to use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BaseController(ABC):
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

    @abstractmethod
    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Instructions:
            Implement this method to return the target state to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] in absolute
            coordinates as a numpy array.
        """

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Callback function called once after the control step.

        You can use this function to update your controller's internal state, save training data,
        update your models, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.
        """

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
        """Reset the controller's internal state and models if necessary."""
