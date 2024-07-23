"""Base class for controller implementations.

Your task is to implement your own controller. This class must be the parent class of your
implementation. You have to use the same function signatures as defined by the base class. Apart
from that, you are free to add any additional methods, attributes, or classes to your controller.

As an example, you could load the weights of a neural network in the constructor and use it to
compute the control commands in the `compute_control` method. You could also use the `step_learn`
method to update the controller at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class BaseController(ABC):
    """Base class for controller implementations."""

    def __init__(
        self, initial_obs: npt.NDArray[np.floating], initial_info: dict, buffer_size: int = 100
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
            buffer_size: Size of the data buffers used in method `learn()`.
        """
        self._buffer_size = buffer_size  # Initialize data buffers for learning
        self.buffers = {
            k: deque([], maxlen=buffer_size)
            for k in ["action", "obs", "reward", "terminated", "truncated", "info"]
        }

    @abstractmethod
    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] in absolute coordinates as a numpy array.
        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    def step_learn(
        self,
        action: npt.NDArray[np.floating],
        obs: npt.NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the data buffers of actions, observations,
            rewards, terminated, truncated flags, and information dictionaries to learn, adapt,
            and/or re-plan.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.
        """
        self.buffers["action"].append(action)
        self.buffers["obs"].append(obs)
        self.buffers["reward"].append(reward)
        self.buffers["terminated"].append(terminated)
        self.buffers["truncated"].append(truncated)
        self.buffers["info"].append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the data buffers to learn, adapt, and/or
            re-plan.
        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters."""
        for buffer in self.buffers.values():
            buffer.clear()

    def episode_reset(self):
        """Reset the controller's internal state and models if necessary."""
        pass
