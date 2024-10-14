"""Example implementation of a controller using a pre-trained PPO model."""

from __future__ import annotations  # Python 3.10 type hints

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import PPO

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PPOController(BaseController):
    """Controller using a pre-trained PPO model."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)
        self.policy = PPO.load(Path(__file__).resolve().parents[2] / "models/ppo/model.zip")
        self._last_action = np.zeros(3)

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired position, velocity and orientation of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        obs_tf = self.obs_transform(obs, info, self._last_action)
        action, _ = self.policy.predict(obs_tf, deterministic=True)
        self._last_action[:] = action
        return np.concatenate([obs["pos"] + action, np.zeros(10)]).astype(np.float64)

    def obs_transform(
        self, obs: NDArray[np.floating], info: dict, action: NDArray[np.floating] | None
    ) -> NDArray[np.floating]:
        """Transform raw observations into the format expected by the PPO model.

        Args:
            obs: Raw observation from the environment.
            info: Additional information dictionary.
            action: Previous action taken, or None.

        Returns:
            Transformed observation as a numpy array.
        """
        gate_vec = info["gates.pos"][info["target_gate"]].copy()
        gate_vec /= np.linalg.norm(gate_vec)
        gate_angle = info["gates.rpy"][info["target_gate"], 2]
        gate_direction = np.array([np.cos(gate_angle), np.sin(gate_angle)])
        to_obstacles = info["obstacles.pos"] - obs["pos"]
        gate_id_onehot = np.zeros(info["gates.pos"].shape[0])
        gate_id_onehot[info["target_gate"]] = 1

        gates_pose = np.concatenate(
            [np.concatenate([p, [y]]) for p, y in zip(info["gates.pos"], info["gates.rpy"][:, 2])]
        )
        state = np.concatenate(
            [
                obs["pos"],
                obs["rpy"],
                obs["vel"],
                obs["ang_vel"],
                gates_pose,
                info["gates.in_range"],
                info["obstacles.pos"].flatten(),
                info["obstacles.in_range"],
            ]
        )
        obs = np.concatenate(
            [
                state,
                gate_id_onehot,
                (info["gates.pos"] - obs["pos"]).flatten(),
                to_obstacles.flatten(),
                gate_vec,
                gate_direction,
                action,
            ]
        )
        return obs.astype(np.float32)

    def episode_reset(self):
        """Reset internal variables at the start of a new episode."""
        self._last_action = np.zeros(3)
