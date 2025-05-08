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
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
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
    logger.warning(
        "The simulation currently only supports running with one controller type and one set of "
        "environment parameters (i.e. frequencies, control mode etc.). Only using the settings for "
        "the first drone."
    )
    # Load the controller module
    if controller is None:
        controller = config.controller[0]["file"]
    controller_path = Path(__file__).parents[1] / "lsy_drone_racing/control" / controller
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: MultiDroneRacingEnv = gymnasium.make(
        "MultiDroneRacing-v0",
        freq=config.env.kwargs[0]["freq"],
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.kwargs[0]["sensor_range"],
        control_mode=config.env.kwargs[0]["control_mode"],
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # We use the same example controllers for this script as for the single-drone case. These expect
    # the config to have env.freq set, so we copy it here. Actual multi-drone controllers should not
    # rely on this.
    config.env.freq = config.env.kwargs[0]["freq"]
    env = JaxToNumpy(env)
    n_drones, n_worlds = env.unwrapped.sim.n_drones, env.unwrapped.sim.n_worlds

    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            action = np.array([action] * n_drones * n_worlds, dtype=np.float32)
            action[1, 0] += 0.2
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            # Update the controller internal state and models.
            controller.step_callback(action, obs, reward, terminated, truncated, info)
            # Add up reward, collisions

            # Synchronize the GUI.
            if config.sim.gui:
                if ((i * fps) % config.env.freq) < fps:
                    try:
                        env.render()
                    # TODO: JaxToNumpy not working with None (returned by env.render()). Open issue
                    # in gymnasium and fix this.
                    except Exception as e:
                        if not e.args[0].startswith("No known conversion for Jax type"):
                            raise e
            i += 1
            if done:
                break

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()

    # Close the environment
    env.close()


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    finished = gates_passed == -1
    logger.info((f"Flight time (s): {curr_time}\nDrones finished: {finished}\n"))


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
