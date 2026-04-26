from controller import Controller

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

from enum import Enum


class KaFa1500_State(Enum):
    TAKEOFF = 0
    APPRCH_GATE = 1
    PASS_GATE = 2
    FINISH = 3


class KaFa1500(Controller):
    """State controller following a pre-defined trajectory."""

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

        self._state = KaFa1500_State.TAKEOFF
        self._tick = 0
        self._finished = False
        ...

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

        ...

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
        ...

    def episode_callback(self):
        """Reset the internal state."""

        ...

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""

        ...

    def state_machine(self):
        match self._state:
            case KaFa1500_State.TAKEOFF:
                ...
            case KaFa1500_State.APPRCH_GATE:
                ...
            case KaFa1500_State.PASS_GATE:
                ...
            case KaFa1500_State.FINISH:
                ...
            case _:
                raise ValueError(f"Invalid state: {self._state}")
