"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_env import RealDroneRaceEnv

logger = logging.getLogger(__name__)


def main(config: str = "level2.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    rclpy.init()
    config = load_config(Path(__file__).parents[1] / "config" / config)
    env: RealDroneRaceEnv = gymnasium.make(
        "RealDroneRacing-v0",
        drones=config.deploy.drones,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    try:
        options = {
            "check_drone_start_pos": config.deploy.check_drone_start_pos,
            "check_race_track": config.deploy.check_race_track,
            "real_track_objects": config.deploy.real_track_objects,
        }
        obs, info = env.reset(options=options)
        next_obs = obs  # Set next_obs to avoid errors when the loop never enters

        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        controller_path = control_path / (controller or config.controller.file)
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = next_obs["target_gate"] == -1
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
