#!/usr/bin/env python
"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.yaml>

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rospy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.drone_racing_deploy_env import DroneRacingDeployEnv

# rospy.init_node changes the default logging configuration of Python, which is bad practice at
# best. As a workaround, we can create loggers under the ROS root logger `rosout`.
# Also see https://github.com/ros/ros_comm/issues/1384
logger = logging.getLogger("rosout." + __name__)


def main(config: str = "level3.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    env: DroneRacingDeployEnv = gymnasium.make("DroneRacingDeploy-v0", config=config)
    obs, info = env.reset()

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)
    controller = controller_cls(obs, info)
    try:
        start_time = time.perf_counter()
        while not rospy.is_shutdown():
            t_loop = time.perf_counter()
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller.step_callback(action, next_obs, reward, terminated, truncated, info)
            obs = next_obs
            if terminated or truncated:
                break
            if dt := (time.perf_counter() - t_loop) < (1 / config.env.freq):
                time.sleep(config.env.freq - dt)  # Maintain the control loop frequency
            if time.perf_counter() - start_time > 9:
                break
        ep_time = time.perf_counter() - start_time
        controller.episode_callback()
        logger.info(
            f"Track time: {ep_time:.3f}s" if info["target_gate"] == -1 else "Task not completed"
        )
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
