"""Some classes to modify the environment during training."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import yaml
from gymnasium.spaces import Box

logger = logging.getLogger(__name__)


class ObservationParser(ABC):
    """Class to parse the observation space of the firmware environment."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
    ):
        """Initialize the observation parser."""
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles

        # Observable variables
        self.drone_pos = None
        self.drone_speed = None
        self.drone_rpy = None
        self.drone_angular_speed = None
        self.gates_pos = None
        self.gates_yaw = None
        self.gates_in_range = None
        self.obstacles_pos = None
        self.obstacles_in_range = None
        self.gate_id = None

        # Hidden states that are not part of the observation space
        self.just_passed_gate: bool = False
        self.gate_edge_size: float = None
        self.previous_action: np.array = np.zeros(4)
        self.previous_drone_pos: np.array = None
        self.reference_position: np.array = None

    def uninitialized(self) -> bool:
        """Check if the observation parser is uninitialized."""
        return self.drone_pos is None

    def out_of_bounds(self) -> bool:
        """Check if the drone is out of bounds."""
        return not self.observation_space.contains(self.get_observation())

    def update(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        initial: bool = False,
        action: np.ndarray = None,
    ):
        """Update the observation parser with the new observation and info dict.

        Remark:
            We do not update the gate height here, the info dict does not contain this information.

        Args:
            obs: The new observation.
            info: The new info dict.
            initial: True if this is the initial observation.
            action: The previous action.
        """
        self.previous_drone_pos = self.drone_pos if not initial else obs[0:6:2]
        if initial:
            self.gate_edge_size = info["gate_dimensions"]["tall"]["edge"]
            self.reference_position = info["x_reference"][0:6:2]
        if action is not None:
            self.previous_action = action
        self.drone_pos = obs[0:6:2]
        self.drone_speed = obs[1:6:2]
        self.drone_rpy = obs[6:9]
        self.drone_rpy_speed = obs[9:12]
        self.gates_pos = info["gates_pose"][:, :3]
        self.gates_yaw = info["gates_pose"][:, 5]
        self.gates_in_range = info["gates_in_range"]
        self.obstacles_pos = info["obstacles_pose"][:, :3]
        self.obstacles_in_range = info["obstacles_in_range"]
        self.just_passed_gate = self.gate_id != info["current_gate_id"]
        self.gate_id = info["current_gate_id"]

    @property
    def drone_yaw(self) -> float:
        """Return the yaw of the drone."""
        return self.drone_rpy[2]

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the current observation.

        Returns:
            The current observation.
        """
        raise NotImplementedError

    def get_relative_corners(self) -> np.ndarray:
        """Return the relative position of the corners of the gates with respect to the drone."""
        edge_cos = np.cos(self.gates_yaw)
        edge_sin = np.sin(self.gates_yaw)
        ones = np.ones_like(edge_cos)
        edge_vector_pos = self.gate_edge_size / 2 * np.array([edge_cos, edge_sin, ones]).T
        edge_vector_neg = self.gate_edge_size / 2 * np.array([-edge_cos, -edge_sin, ones]).T
        first_corners = self.gates_pos + edge_vector_pos - self.drone_pos
        second_corners = self.gates_pos + edge_vector_neg - self.drone_pos
        third_corners = self.gates_pos - edge_vector_pos - self.drone_pos
        fourth_corners = self.gates_pos - edge_vector_neg - self.drone_pos
        relative_distance_corners = np.concatenate(
            [first_corners, second_corners, third_corners, fourth_corners]
        )
        return relative_distance_corners

    def get_relative_obstacles(self) -> np.ndarray:
        """Return the relative position of the obstacles."""
        return self.obstacles_pos - self.drone_pos


class ScaramuzzaObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_yaw_limits: list = [np.pi],
        drone_speed_limits: list = [2] * 3,
        drone_yaw_speed_limits: list = [np.pi],
        gate_pos_limits: list = [5, 5, 5],
        obstacle_pos_limits: list = [5, 5, 5],
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles)
        n_corners = 4
        relative_corners_limits = gate_pos_limits * n_gates * n_corners
        obs_limits = (
            drone_speed_limits
            + drone_yaw_limits
            + drone_yaw_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)


    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners()
        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.drone_speed,
                [self.drone_yaw],
                [self.drone_yaw_speed],
                relative_corners.flatten(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class RelativeCornersObservationParser(ObservationParser):
    """Class to parse the observation space of the to use relative corners."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_speed_limits: list = [2] * 3,
        drone_yaw_limits: list = [np.pi],
        drone_yaw_speed_limits: list = [np.pi],
        gate_pos_limits: list = [5, 5, 5],
        gate_yaw_limits: list = [np.pi],
        gate_in_range_limits: list = [1],
        obstacle_pos_limits: list = [5, 5, 5],
        obstacle_in_range_limits: list = [1],
    ):
        """Initialize the relative corners observation parser."""
        super().__init__(n_gates, n_obstacles)

        n_corners = 4
        relative_corners_limits = gate_pos_limits * n_gates * n_corners

        obs_limits = (
            drone_pos_limits
            + drone_yaw_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners()
        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.drone_pos,
                [self.drone_yaw],
                relative_corners.flatten(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class ClassicObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided at the beginning of the competition."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_speed_limits: list = [2] * 3,
        drone_yaw_limits: list = [np.pi],
        drone_yaw_speed_limits: list = [np.pi],
        gate_pos_limits: list = [5, 5, 5],
        gate_yaw_limits: list = [np.pi],
        gate_in_range_limits: list = [1],
        obstacle_pos_limits: list = [5, 5, 5],
        obstacle_in_range_limits: list = [1],
    ):
        """Initialize the classic observation parser."""
        super().__init__(n_gates, n_obstacles)

        obs_limits = (
            drone_pos_limits
            + drone_yaw_limits
            + gate_pos_limits * n_gates
            + gate_yaw_limits * n_gates
            + gate_in_range_limits * n_gates
            + obstacle_pos_limits * n_obstacles
            + obstacle_in_range_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        obs = np.concatenate(
            [
                self.drone_pos,
                [self.drone_yaw],
                self.gates_pos.flatten(),
                self.gates_yaw,
                self.gates_in_range,
                self.obstacles_pos.flatten(),
                self.obstacles_in_range,
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


def make_observation_parser(
    n_gates: int,
    n_obstacles: int,
    observation_parser_type: str = "relative_corners",
    **kwargs: Any,
) -> ObservationParser:
    """Create an observation parser.

    Args:
        observation_parser_type: The type of the observation parser.
        n_gates: The number of gates.
        n_obstacles: The number of obstacles.
        kwargs: Additional arguments for the observation parser.

    Returns:
        The observation parser.
    """
    if observation_parser_type == "scaramuzza":
        return ScaramuzzaObservationParser(n_gates, n_obstacles, **kwargs)
    if observation_parser_type == "relative_corners":
        return RelativeCornersObservationParser(n_gates, n_obstacles, **kwargs)
    if observation_parser_type == "classic":
        return ClassicObservationParser(n_gates, n_obstacles, **kwargs)
    raise ValueError(f"Unknown observation parser type: {observation_parser_type}")


class Rewarder:
    """Class to allow custom rewards."""

    def __init__(
        self,
        collision: float = -1.0,
        out_of_bounds: float = -1.0,
        times_up: float = -1.0,
        dist_to_gate_mul: float = 1.0,
        end_reached: float = 10.0,
        gate_reached: float = 10.0,
        z_penalty: float = -3.0,
        z_penalty_threshold: float = 1.5,
        action_smoothness: float = 1e-4,
    ):
        """Initialize the rewarder."""
        self.collision = collision
        self.out_of_bounds = out_of_bounds
        self.end_reached = end_reached
        self.gate_reached = gate_reached
        self.times_up = times_up
        self.dist_to_gate_mul = dist_to_gate_mul
        self.z_penalty = z_penalty
        self.z_penalty_threshold = z_penalty_threshold
        self.action_smoothness = action_smoothness

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

    def get_custom_reward(
        self,
        obs_parser: ObservationParser,
        info: dict,
        terminated: bool = False,
    ) -> float:
        """Compute the custom reward.

        Args:
            reward: The reward from the firmware environment.
            obs_parser: The current observation.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: The info dict from the firmware environment.

        Returns:
            The custom reward.
        """
        reward = 0.0

        if info["collision"][1]:
            return self.collision

        if terminated and not info["task_completed"]:
            return self.times_up

        if obs_parser.out_of_bounds():
            return self.out_of_bounds

        if info["task_completed"]:
            logger.info("Task completed, hoorray!")
            return self.end_reached

        if obs_parser.gate_id == -1:
            # Reward for getting closer to the reference position
            dist_to_ref = np.linalg.norm(obs_parser.drone_pos - obs_parser.reference_position)
            previos_dist_to_ref = np.linalg.norm(
                obs_parser.previous_drone_pos - obs_parser.reference_position
            )
            reward += (previos_dist_to_ref - dist_to_ref) * self.dist_to_gate_mul
        else:
            dist_to_gate = np.linalg.norm(
                obs_parser.drone_pos - obs_parser.gates_pos[obs_parser.gate_id]
            )
            previos_dist_to_gate = np.linalg.norm(
                obs_parser.previous_drone_pos - obs_parser.gates_pos[obs_parser.gate_id]
            )
            reward += (previos_dist_to_gate - dist_to_gate) * self.dist_to_gate_mul

        if obs_parser.drone_pos[2] > self.z_penalty_threshold:
            reward += self.z_penalty

        # if action is not None:
        # reward += -np.linalg.norm(action - obs.previous_action) * self.action_smoothness

        if obs_parser.just_passed_gate:
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


def transform_action(
    raw_action: np.ndarray,
    transform_type: str = "relative",
    drone_pos: np.array = np.zeros(3),
    pos_scaling: np.array = [1.0, 1.0, 0.2],
    yaw_scaling: float = np.pi,
) -> np.ndarray:
    """Transform the raw action to the action space.

    Args:
        raw_action: The raw action from the model is in the range [-1, 1].
        observation_parser: The observation parser.
        transform_type: The type of transformation, either "relative" or "absolute".
        drone_pos: The current position of the drone.
        pos_scaling: The scaling of the position.
        yaw_scaling: The scaling of the angle

    Returns:
        The transformed action to control the drone.
    """
    if transform_type == "relative":
        action_transform = np.zeros(14)
        action_transform[:3] = drone_pos + raw_action[:3]
        action_transform[9] = yaw_scaling * raw_action[3]

    elif transform_type == "absolute":
        action_transform = np.zeros(14)
        scaled_action = raw_action * np.concatenate([pos_scaling, [yaw_scaling]])
        action_transform[:3] = scaled_action[:3]
        action_transform[9] = scaled_action[3]

    return action_transform
