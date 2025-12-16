"""Script for checking the positioning of gates and obstacles."""

import logging
from pathlib import Path

import fire
import numpy as np
import rclpy
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.checks import check_race_track

logger = logging.getLogger("rosout." + __name__)


def main(config: str = "level2.toml"):
    """Check if the real race track conforms to the race configuration.

    Args:
        config: Path to the race configuration. Assumes the file is in `config/`.
    """
    rclpy.init()
    config = load_config(Path(__file__).resolve().parents[1] / "config" / config)
    nominal_gates_pos = [g["pos"] for g in config.env.track.gates]
    nominal_gates_quat = [R.from_euler("xyz", g["rpy"]).as_quat() for g in config.env.track.gates]
    nominal_obstacles_pos = [o["pos"] for o in config.env.track.obstacles]
    n_gates = len(nominal_gates_pos)
    n_obstacles = len(nominal_obstacles_pos)
    gates, obstacles = dict(), dict()
    gates["pos"], gates["quat"], obstacles["pos"] = query_track_poses(
        n_gates=n_gates, n_obstacles=n_obstacles
    )
    check_race_track(
        gates_pos=gates["pos"],
        nominal_gates_pos=nominal_gates_pos,
        gates_quat=gates["quat"],
        nominal_gates_quat=nominal_gates_quat,
        obstacles_pos=obstacles["pos"],
        nominal_obstacles_pos=nominal_obstacles_pos,
        rng_config=config.env.randomizations,
    )

    logger.info("Race track check passed")


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
