from abc import abstractmethod

import numpy as np
from ml_collections import ConfigDict
from numpy.typing import NDArray

from lsy_drone_racing.ros import ROSConnector
from lsy_drone_racing.utils.utils import gate_passed


class DeployRaceCore:
    """Deployable version of the multi-agent drone racing environment."""

    def __init__(
        self, n_drones: int, freq: int, sensor_range: float = 0.5, track: ConfigDict | None = None
    ):
        """Initialize the deployable version of the multi-agent drone racing environment.

        Args:
            n_drones: Number of drones.
            freq: Environment step frequency.
            sensor_range: Sensor range.
        """
        self.freq = freq
        self.n_drones = n_drones
        self.sensor_range = sensor_range
        self.track = track
        self._ros_connector = ROSConnector()

    def _reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        # Jit compile gate_passed for numpy inputs to prevent jax jit compile on first step
        gate_passed(np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 4)), (0.45, 0.45))
        return self.obs(), self.info()

    def _step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment."""
        # Note: We do not send the action to the drone here.
        self._send_action(action)
        self._ros_connector.send_action(action)
        obs = self.obs()
        gate_pos = self.track.gates.pos[self.data.target_gate]
        gate_quat = self.track.gates.quat[self.data.target_gate]
        drone_pos = obs["pos"][self.drone_id]
        passed = gate_passed(drone_pos, self._last_drone_pos, gate_pos, gate_quat, (0.45, 0.45))
        self.data.target_gate += passed
        if self.data.target_gate >= len(gate_pos):
            self.data.target_gate = -1
        self._last_drone_pos = drone_pos
        return obs, self.reward(), self.terminated(), self.truncated(), self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the observation of the environment."""
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        gates_pos = self._ros_connector.gate_pos
        gates_quat = self._ros_connector.gate_quat
        obstacle_pos = self._ros_connector.obstacle_pos
        gates_pos = np.where(self.data.gates_visited, gates_pos, self.gates["nominal_pos"])
        gates_quat = np.where(self.data.gates_visited, gates_quat, self.gates["nominal_quat"])
        obstacles_pos = np.where(
            self.data.obstacles_visited, obstacle_pos, self.obstacles["nominal_pos"]
        )
        obs = {
            "pos": self._ros_connector.drone_pos,
            "quat": self._ros_connector.drone_quat,
            "vel": self._ros_connector.drone_vel,
            "ang_vel": self._ros_connector.drone_ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 * (self.data.target_gate == -1)  # Implicit float conversion

    def terminated(self) -> bool:
        """Check if the episode is terminated."""
        return self.data.target_gate == -1

    def truncated(self) -> bool:
        """Check if the episode is truncated."""
        return False

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    @abstractmethod
    def send_action(self, action: NDArray):
        """Send the action to the drone."""
