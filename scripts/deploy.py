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


def main(config: str = "level3.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    env: RealDroneRaceEnv = gymnasium.make(
        "RealDroneRacing-v0",
        drones=config.deploy.drones,
        rank=0,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    obs, info = env.reset()
    next_obs = obs  # Set next_obs to avoid errors when the loop never enters

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)
    controller = controller_cls(obs, info, config)
    try:
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            action = controller.compute_control(obs, info)
            t1 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(action)
            t2 = time.perf_counter()
            print(f"Step time: {t2 - t1:.3e}s")
            print(obs["pos"])
            controller.step_callback(action, next_obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        controller.episode_callback()
        logger.info(
            f"Track time: {ep_time:.3f}s" if next_obs["target_gate"] == -1 else "Task not completed"
        )
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
