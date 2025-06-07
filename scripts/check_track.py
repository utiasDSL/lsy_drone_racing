"""Script for checking the positioning of gates and obstacles."""

import logging
from pathlib import Path

import fire
import rclpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.ros.ros_utils import check_race_track
from lsy_drone_racing.utils import load_config

logger = logging.getLogger("rosout." + __name__)


def main(config: str = "level2.toml"):
    """Check if the real race track conforms to the race configuration.

    Args:
        config: Path to the race configuration. Assumes the file is in `config/`.
    """
    rclpy.init()
    config = load_config(Path(__file__).resolve().parents[1] / "config" / config)
    gates_pos = [g["pos"] for g in config.env.track.gates]
    gates_quat = [R.from_euler("xyz", g["rpy"]).as_quat() for g in config.env.track.gates]
    obstacle_pos = [o["pos"] for o in config.env.track.obstacles]
    check_race_track(gates_pos, gates_quat, obstacle_pos, rng_config=config.env.randomizations)
    logger.info("Race track check passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
