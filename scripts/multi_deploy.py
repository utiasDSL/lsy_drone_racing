#!/usr/bin/env python
"""Launch script for the real race with multiple drones.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from multiprocessing.synchronize import Barrier, Event

    from lsy_drone_racing.envs.drone_racing_deploy_env import (
        DroneRacingAttitudeDeployEnv,
        DroneRacingDeployEnv,
    )
# rospy.init_node changes the default logging configuration of Python, which is bad practice at
# best. As a workaround, we can create loggers under the ROS root logger `rosout`.
# Also see https://github.com/ros/ros_comm/issues/1384
logger = logging.getLogger("rosout." + __name__)


def control_loop(drone_id: int, config: dict, start_barrier: Barrier, shutdown: Event):
    """Control loop for the drone."""
    env_id = "DroneRacingAttitudeDeploy-v0" if "Thrust" in config.env.id else "DroneRacingDeploy-v0"
    env: DroneRacingDeployEnv | DroneRacingAttitudeDeployEnv = gymnasium.make(env_id, config=config)
    obs, info = env.reset()

    module_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = Path(config.controller[drone_id].file)
    if controller_path.is_relative():
        controller_path = module_path / controller_path
    Controller = load_controller(controller_path)
    controller = Controller(obs, info)

    start_barrier.wait(timeout=30.0)  # Wait for all drones to be ready at the same time

    try:
        start_time = time.perf_counter()
        while not shutdown.is_set():
            t_loop = time.perf_counter()
            obs, info = env.obs(), env.info()
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            controller.step_callback(action, next_obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = obs["target_gate"] == -1
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
        # TODO: Check if environment has been terminated because estimators died
        if not shutdown.is_set():  # Drone finished the track without emergency interrupt
            env.return_to_start()
    finally:
        env.close()


def main(config: str = "multi_level3.toml"):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    n_drones = len(config.controller)
    start_barrier = mp.Barrier(n_drones)
    shutdown = mp.Event()
    drone_processes = [
        mp.Process(target=control_loop, args=(i, config, start_barrier, shutdown[i]))
        for i in range(n_drones)
    ]
    for p in drone_processes:
        p.start()

    try:
        while any(p.is_alive() for p in drone_processes):
            time.sleep(0.2)
    except KeyboardInterrupt:
        shutdown.set()

    for p in drone_processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
