"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and an expanded observation space!
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box

from lsy_drone_racing.constants import Z_HIGH, Z_LOW

if TYPE_CHECKING:
    from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper

logger = logging.getLogger(__name__)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        self.env.unwrapped = None  # Add an (empty) unwrapped attribute
        self.env.render_mode = None
        # Gymnasium env required attributes
        # Action space:
        # [x, y, z, yaw]
        # 0-2) x, y, z are the desired position of the drone in the world frame.
        # 3)   yaw is the desired yaw angle.
        # All values are scaled to [-1, 1]. Transformed back, x, y, z values of 1 correspond to 5m.
        # The yaw value of 1 corresponds to pi radians.
        action_limits = np.ones(4)
        self.action_scale = np.array([5, 5, 5, np.pi])
        self.action_space = Box(-action_limits, action_limits, dtype=np.float32)
        # Observation space:
        # [x, y, z, yaw, gate_x, gate_y, gate_z, gate_yaw, gate_in_range]
        # 0-3)  x, y, z, yaw are the drone pose of the drone in the world frame. Position is in
        #       meters and yaw is in radians.
        # 4-7)  gate_x, gate_y, gate_z, gate_yaw are the pose of the current target gate. Position
        #       is in meters and yaw is in radians. The gate_z value is either 0.25 or 0.5.
        # 8)    gate_in_range is a boolean indicating if the drone is within the gate's range. If it
        #       is, the gate observation is more precise.

        obs_limits = np.ones(9)
        obs_limits[[0, 1, 2, 4, 5, 6]] = 5
        obs_limits[[3, 7]] = np.pi
        obs_limits[8] = 1
        self.observation_space = Box(-obs_limits, obs_limits, dtype=np.float32)

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config flags
        self.terminate_on_lap = terminate_on_lap
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)

    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()
        obs = expand_obs(obs, info).astype(np.float32)
        assert obs in self.observation_space, f"Invalid observation: {obs}"
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        if action not in self.action_space:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")
        action = self._action_tf(action)
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, action[3], zeros, self._sim_time)
        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, reward, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors
        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False
        if info["task_completed"] and info["current_target_gate_id"] != -1:
            truncated = True
        elif self.terminate_on_lap and info["current_target_gate_id"] == -1:
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        obs = expand_obs(obs, info).astype(np.float32)
        assert obs in self.observation_space, f"Invalid observation: {obs}"
        return obs, reward, terminated, truncated, info

    def _action_tf(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Scale the action from [-1, 1] to [-5, 5] for the position and [-pi, pi] for the yaw.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action_tf = np.zeros(14)
        scaled_action = action * self.action_scale
        action_tf[:3] = scaled_action[:3]
        action_tf[9] = scaled_action[3]
        return action_tf

    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"


def expand_obs(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """Expand the observation to include additional information.

    Args:
        obs: The observation to expand.
        info: Additional information to include in the observation.

    Returns:
        The expanded observation.
    """
    drone_pos = obs[0:6:2]
    drone_yaw = obs[8]
    # The initial info dict does not include the gate pose and range, but it does include the
    # nominal gate positions and types, which we can use as a fallback for the first step.
    initial = "nominal_gates_pos_and_type" in info
    if initial:
        gate_type = info["nominal_gates_pos_and_type"][0][-1]
        gate_pos = info["nominal_gates_pos_and_type"][0][:2]
        gate_yaw = info["nominal_gates_pos_and_type"][0][5]
    else:
        gate_type = info["current_target_gate_type"]
        gate_pos = info["current_target_gate_pos"][:2]
        gate_yaw = info["current_target_gate_pos"][5]
    gate_z = Z_LOW if gate_type == 1 else Z_HIGH
    gate_pose = np.array([*gate_pos, gate_z, gate_yaw])
    gate_in_range = info.get("current_target_gate_in_range", False)
    return np.concatenate([drone_pos, [drone_yaw], gate_pose, [gate_in_range]])
