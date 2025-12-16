"""Script for generating .toml configuration files from a real race track."""

import logging
from pathlib import Path

import fire
import rclpy
import toml
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.utils import query_drone_poses, query_track_poses
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
