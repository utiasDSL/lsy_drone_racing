#!/usr/bin/env python
"""Launch script for the real race with multiple drones.

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
import rospy

from lsy_drone_racing.controller_manager import ControllerManager
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.drone_racing_deploy_env import (
        DroneRacingAttitudeDeployEnv,
        DroneRacingDeployEnv,
    )

# rospy.init_node changes the default logging configuration of Python, which is bad practice at
# best. As a workaround, we can create loggers under the ROS root logger `rosout`.
# Also see https://github.com/ros/ros_comm/issues/1384
logger = logging.getLogger("rosout." + __name__)


def main(config: str = "multi_level3.toml"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    env_id = "DroneRacingAttitudeDeploy-v0" if "Thrust" in config.env.id else "DroneRacingDeploy-v0"
    env: DroneRacingDeployEnv | DroneRacingAttitudeDeployEnv = gymnasium.make(env_id, config=config)
    obs, info = env.reset()

    module_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_paths = [module_path / p if p.is_relative() else p for p in config.controller.files]
    controller_manager = ControllerManager([load_controller(p) for p in controller_paths])
    controller_manager.start(init_args=(obs, info))

    try:
        start_time = time.perf_counter()
        while not rospy.is_shutdown():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs, env.unwrapped.info
            # Compute the control action asynchronously. This limits delays and prevents slow
            # controllers from blocking the controllers for other drones.
            controller_manager.update_obs(obs, info)
            actions = controller_manager.latest_actions()
            next_obs, reward, terminated, truncated, info = env.step(actions)
            controller_manager.step_callback(actions, next_obs, reward, terminated, truncated, info)
            obs = next_obs
            if terminated or truncated:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        controller_manager.episode_callback()
        logger.info(
            f"Track time: {ep_time:.3f}s" if obs["target_gate"] == -1 else "Task not completed"
        )
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
