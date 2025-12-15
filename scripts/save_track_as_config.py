"""Script for generating .toml configuration files from a real race track."""

import logging
from pathlib import Path

import fire
import numpy as np
import rclpy
import toml
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from ml_collections import ConfigDict
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config

logger = logging.getLogger("rosout." + __name__)


def main(config: str = "level2.toml", save_config_to: str = "real_track.toml"):
    """Check if the real race track conforms to the race configuration.

    Args:
        config: Path to the race configuration. Assumes the file is in `config/`.
        save_config_to: Path to save the track configuration if the check passes.
    """
    rclpy.init()
    config = load_config(Path(__file__).resolve().parents[1] / "config" / config)
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.obstacles)
    drone_names = [f"cf{drone['id']}" for drone in config.deploy.drones]
    gates, obstacles, drones = dict(), dict(), dict()
    gates["pos"], gates["quat"], obstacles["pos"] = query_track_poses(
        n_gates=n_gates, n_obstacles=n_obstacles
    )
    drones["pos"], drones["quat"] = query_drone_poses(drone_names=drone_names)

    config_output = update_level_config(
        config, gates=ConfigDict(gates), obstacles=ConfigDict(obstacles), drones=ConfigDict(drones)
    )
    output_path = Path(__file__).parents[1] / "config" / save_config_to
    if not output_path.suffix == ".toml":
        raise ValueError(f"Configuration file has to be a TOML file: {output_path}")
    with open(output_path, "w") as f:
        toml.dump(config_output.to_dict(), f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)


def query_track_poses(n_gates: int, n_obstacles: int) -> tuple[NDArray, NDArray, NDArray]:
    """Query the track poses from the motion capture system.

    Args:
        n_gates: Number of gates in the track.
        n_obstacles: Number of obstacles in the track.

    Returns:
        A tuple containing:
            - gate_pos: An (n_gates, 3) array of gate positions.
            - gate_quat: An (n_gates, 4) array of gate orientations as quaternions.
            - obstacle_pos: An (n_obstacles, 3) array of obstacle positions.
    """
    gate_pos = np.zeros((n_gates, 3), dtype=np.float32)
    gate_quat = np.zeros((n_gates, 4), dtype=np.float32)
    obstacle_pos = np.zeros((n_obstacles, 3), dtype=np.float32)

    tf_names = [f"gate{i}" for i in range(1, n_gates + 1)]
    tf_names += [f"obstacle{i}" for i in range(1, n_obstacles + 1)]
    try:
        ros_connector = ROSConnector(tf_names=tf_names, timeout=10.0)
        pos, quat = ros_connector.pos, ros_connector.quat
    finally:
        ros_connector.close()
    try:
        for i in range(n_gates):
            gate_pos[i, ...] = pos[f"gate{i + 1}"]
            gate_quat[i, ...] = quat[f"gate{i + 1}"]
        for i in range(n_obstacles):
            obstacle_pos[i, ...] = pos[f"obstacle{i + 1}"]
    except KeyError as e:
        raise ValueError(
            f"Could not find all track objects in the ROS TF tree: {e}. \
                        Have you enabled the track objects in Vicon/ \
                        started the motion capture tracking node?"
        ) from e
    return gate_pos, gate_quat, obstacle_pos


def query_drone_poses(drone_names: list[str]) -> tuple[NDArray, NDArray]:
    """Query the drone positions from the motion capture system and estimator node.

    Args:
        drone_names: List of drone estimator names.

    Returns:
        A tuple containing:
            - drone_pos: An (n_drones, 3) array of drone positions.
            - drone_quat: An (n_drones, 4) array of drone orientations as quaternions.
    """
    drone_pos = np.zeros((len(drone_names), 3), dtype=np.float32)
    drone_quat = np.zeros((len(drone_names), 4), dtype=np.float32)
    try:
        ros_connector = ROSConnector(estimator_names=drone_names, timeout=10.0)
        for i, drone_name in enumerate(drone_names):
            drone_pos[i, ...] = ros_connector.pos[drone_name]
            drone_quat[i, ...] = ros_connector.quat[drone_name]
    except KeyError as e:
        raise ValueError(
            f"Could not find drone in the ROS estimator messages: {e}. \
                        Have you enabled the drone in Vicon/ \
                        started the estimator node?"
        ) from e
    finally:
        ros_connector.close()
    return drone_pos, drone_quat


def update_level_config(
    config: ConfigDict, gates: ConfigDict, obstacles: ConfigDict, drones: ConfigDict
) -> ConfigDict:
    """Update the level config with the real track objects poses.

    Args:
        config: A ConfigDict storing the original level configuration.
        gates: A ConfigDict storing the updated gate positions and orientations.
        obstacles: A ConfigDict storing the updated obstacle positions.
        drones: A ConfigDict storing the updated drone starting positions and orientations.

    Returns:
        A ConfigDict object, with updated starting pose of gates, obstacles and drones
    """
    config = config.copy()
    # We store the real-world track layout, so randomization must be disabled
    config.env.track.randomize = False
    # Overwrite the original positions and orientations with the measured ones
    for i in range(gates.pos.shape[0]):
        config.env.track.gates.pos[i] = gates.pos[i].tolist()
        config.env.track.gates.rpy[i] = R.from_quat(gates.quat[i]).as_euler("xyz").tolist()
    for i in range(obstacles.pos.shape[0]):
        config.env.track.obstacles.pos[i] = obstacles.pos[i].tolist()
    for i in range(drones.pos.shape[0]):
        config.env.track.drones.pos[i] = drones.pos[i].tolist()
        config.env.track.drones.rpy[i] = R.from_quat(drones.quat[i]).as_euler("xyz").tolist()
    return config
