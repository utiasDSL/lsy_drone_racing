"""Some classes to modify the environment during training."""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from lsy_drone_racing.constants import Z_HIGH, Z_LOW


class ObservationParser:
    """Class to parse the observation space of the firmware environment."""

    def __init__(
        self,
        drone_pos: np.ndarray,
        drone_yaw: float,
        gates_poses: np.ndarray,
        gates_in_range: np.ndarray,
        obstacles_pos: np.ndarray,
        obstacles_in_range: np.ndarray,
        gate_id: int,
    ):
        """Initialize the observation parser."""
        self.drone_pos = drone_pos
        self.drone_yaw = drone_yaw
        self.gates_poses = gates_poses
        self.gates_in_range = gates_in_range
        self.obstacles_pos = obstacles_pos
        self.obstacles_in_range = obstacles_in_range
        self.gate_id = gate_id

        # Hidden states that are not part of the observation space
        self.just_passed_gate: bool = False

    @classmethod
    def from_initial_observation(cls, obs: np.ndarray, info: dict[str, Any]):
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_yaw = obs[8]
        # The initial info dict does not include the gate pose and range, but it does include the
        # nominal gate positions and types, which we can use as a fallback for the first step.
        gate_type = info["gate_types"]
        gate_pos = info["nominal_gates_pos_and_type"][:, :2]
        gate_yaw = info["nominal_gates_pos_and_type"][:, 5]

        z = np.where(gate_type == 1, Z_LOW, Z_HIGH)  # Infer gate z position based on type
        gate_poses = np.concatenate([gate_pos, z[:, None], gate_yaw[:, None]], axis=1)
        obstacle_pos = info["obstacles_pos"][:, :3]
        obstacle_pos[:, 2] = 0.5
        gate_id = info["current_gate_id"]
        gates_in_range = info["gates_in_range"]
        obstacles_in_range = info["obstacles_in_range"]

        return cls(
            drone_pos=drone_pos,
            drone_yaw=drone_yaw,
            gates_poses=gate_poses,
            gates_in_range=gates_in_range,
            obstacles_pos=obstacle_pos,
            obstacles_in_range=obstacles_in_range,
            gate_id=gate_id,
        )

    def update(self, obs: np.ndarray, info: dict[str, Any]):
        """Update the observation parser with the new observation and info dict.

        Remark:
            We do not update the gate height here, the info dict does not contain this information.

        Args:
            obs: The new observation.
            info: The new info dict.
        """
        self.drone_pos = obs[0:6:2]
        self.drone_yaw = obs[8]
        gates_pos = info["gates_pos"][:, :2]
        gates_yaw = info["gates_pos"][:, 5]
        gates_z = np.where(info["gate_types"] == 1, Z_LOW, Z_HIGH)
        self.gates_poses = np.concatenate([gates_pos, gates_z[:, None], gates_yaw[:, None]], axis=1)
        self.gates_in_range = info["gates_in_range"]
        self.obstacles_pos = info["obstacles_pos"][:, :3]
        self.obstacles_in_range = info["obstacles_in_range"]
        self.just_passed_gate = self.gate_id != info["current_gate_id"]
        self.gate_id = info["current_gate_id"]

    def get_observation(self) -> np.ndarray:
        """Return the current observation.

        Returns:
            The current observation.
        """
        obs = np.concatenate(
            [
                self.drone_pos,
                [self.drone_yaw],
                self.gates_poses.flatten(),
                self.gates_in_range,
                self.obstacles_pos.flatten(),
                self.obstacles_in_range,
                [self.gate_id],
            ]
        )
        return obs


class Rewarder:
    """Class to allow custom rewards."""

    def __init__(
        self,
        collision: float = -1000.0,
        end_reached: float = 1000.0,
        dist_to_gate_mul: float = -1.0,
        dist_to_obstacle_mul: float = 0.2,
        gate_reached: float = 100.0,
    ):
        """Initialize the rewarder."""
        self.collision = collision
        self.end_reached = end_reached
        self.dist_to_gate_mul = dist_to_gate_mul
        self.dist_to_obstacle_mul = dist_to_obstacle_mul
        self.gate_reached = gate_reached

    @classmethod
    def from_yaml(cls, file_path: str) -> Rewarder:  # noqa: ANN102
        """Load the rewarder from a YAML file.

        Args:
            file_path: The path to the YAML file.

        Returns:
            The rewarder.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def get_custom_reward(self, obs: ObservationParser, info: dict) -> float:
        """Compute the custom reward.

        Args:
            reward: The reward from the firmware environment.
            obs: The current observation.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: The info dict from the firmware environment.

        Returns:
            The custom reward.
        """
        reward = 0.0

        if info["collision"][1]:
            return self.collision
        if info["task_completed"]:
            return self.end_reached

        # Reward for gating close to the next gate
        dist_to_gate = np.linalg.norm(obs.drone_pos - obs.gates_poses[obs.gate_id, :3])
        reward += dist_to_gate * self.dist_to_gate_mul

        # Reward for avoiding obstacles
        dist_to_obstacle = np.linalg.norm(obs.drone_pos - obs.obstacles_pos[obs.obstacles_in_range, :3])
        reward += dist_to_obstacle * self.dist_to_obstacle_mul

        # Reward for passing a gate
        if obs.just_passed_gate:
            reward += self.gate_reached

        return reward


def map_reward_to_color(reward: float) -> str:
    """Convert the reward to a color.

    We use a red-green color map, where red indicates a negative reward and green a positive reward.
    Args:
        reward: The reward.

    Returns:
        The color.
    """
    if reward < 0:
        return "red"
    return "green"

