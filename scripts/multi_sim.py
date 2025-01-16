"""Simulate a multi-drone race.

Run as:

    $ python scripts/multi_sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from munch import Munch

    from lsy_drone_racing.control.controller import BaseController
    from lsy_drone_racing.envs.multi_drone_race import MultiDroneRacingEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "multi_level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: MultiDroneRacingEnv = gymnasium.make(
        config.env.id,
        n_envs=2,  # TODO: Remove this for single-world envs
        n_drones=config.env.n_drones,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        random_resets=config.env.random_resets,
        seed=config.env.seed,
    )

    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: BaseController = controller_cls(obs, info)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            action = np.array([action] * config.env.n_drones * 2)
            action[1, 0] += 0.2
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            # Update the controller internal state and models.
            controller.step_callback(action, obs, reward, terminated, truncated, info)
            # Add up reward, collisions

            # Synchronize the GUI.
            if config.sim.gui:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
            i += 1
            if done.all():
                break

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()

    # Close the environment
    env.close()


def log_episode_stats(obs: dict, info: dict, config: Munch, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    finished = gates_passed == -1
    logger.info((f"Flight time (s): {curr_time}\nDrones finished: {finished}\n"))


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
