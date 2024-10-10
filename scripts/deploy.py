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

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.drone_racing_deploy_env import DroneRacingDeployEnv

# rospy.init_node changes the default logging configuration of Python, which is bad practice at
# best. As a workaround, we can create loggers under the ROS root logger `rosout`.
# Also see https://github.com/ros/ros_comm/issues/1384
logger = logging.getLogger("rosout." + __name__)


def main(config: str = "config/level0.toml", controller: str = "examples/trajectory_controller.py"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration.
        controller: Path to the controller implementation.
    """
    config = load_config(Path(config))
    env: DroneRacingDeployEnv = gymnasium.make("DroneRacingDeploy-v0", config=config)
    obs, info = env.reset()

    controller_cls = load_controller(Path(controller))
    controller = controller_cls(obs, info)

    try:
        start_time = time.perf_counter()
        while True:
            t_loop = time.perf_counter()
            action = controller.compute_control(env.obs, env.info)  # Get the most recent obs/info
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller.step_learn(action, next_obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            if dt := (time.perf_counter() - t_loop) < config.env.freq:
                time.sleep(config.env.freq - dt)  # Maintain the control loop frequency
        ep_time = time.perf_counter() - start_time
        controller.episode_learn()
        logger.info(
            f"Track time: {ep_time:.3f}s" if next_obs["gate"] == -1 else "Task not completed"
        )
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
