"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config config/level0.toml

Look for instructions in `README.md` and `edit_this.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import pybullet as p

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from munch import Munch


logger = logging.getLogger(__name__)


def simulate(
    config: str = "config/level0.toml",
    controller: str = "examples/controller.py",
    n_runs: int = 1,
    gui: bool = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The path to the controller module.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(config))
    config.sim.gui = gui
    # Load the controller module
    path = Path(__file__).parents[1] / controller
    ctrl_class = load_controller(path)  # This returns a class, not an instance
    # Create a statistics collection
    stats = {"ep_reward": 0, "collisions": 0, "violations": 0, "gates_passed": 0}
    ep_times = []

    # Create the racing environment
    env = gymnasium.make("DroneRacing-v0", config=config)

    for _ in range(n_runs):  # Run n_runs episodes with the controller
        ep_start = time.time()
        done = False
        obs, info = env.reset()
        info["ctrl_timestep"] = config.env.freq
        info["ctrl_freq"] = 1 / config.env.freq
        # obs = [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
        ctrl = ctrl_class(obs, info)
        gui_timer = p.addUserDebugText(
            "", textPosition=[0, 0, 1], physicsClientId=env.unwrapped.sim.pyb_client
        )
        i = 0
        while not done:
            curr_time = i / config.env.freq
            gui_timer = p.addUserDebugText(
                "Ep. time: {:.2f}s".format(curr_time),
                textPosition=[0, 0, 1.5],
                textColorRGB=[1, 0, 0],
                lifeTime=0,  # 3 / config.env.freq,
                textSize=1.5,
                parentObjectUniqueId=0,
                parentLinkIndex=-1,
                replaceItemUniqueId=gui_timer,
                physicsClientId=env.unwrapped.sim.pyb_client,
            )

            # Get the observation from the motion capture system
            # Compute control input.
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Update the controller internal state and models.
            ctrl.step_learn(action, obs, reward, terminated, truncated, info)
            # Add up reward, collisions, violations.
            stats["ep_reward"] += reward
            if info["collisions"]:
                stats["collisions"] += 1
            stats["violations"] += "constraint_violation" in info and info["constraint_violation"]

            # Synchronize the GUI.
            if config.sim.gui:
                if (elapsed := time.time() - ep_start) < i / config.env.freq:
                    time.sleep(i / config.env.freq - elapsed)
            i += 1

        # Learn after the episode if the controller supports it
        ctrl.episode_learn()  # Update the controller internal state and models.
        log_episode_stats(stats, info, config, curr_time)
        ctrl.episode_reset()
        # Reset the statistics
        stats["ep_reward"] = 0
        stats["collisions"] = 0
        stats["violations"] = 0
        ep_times.append(curr_time if info["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(stats: dict, info: dict, config: Munch, curr_time: float):
    """Log the statistics of a single episode."""
    stats["gates_passed"] = info["target_gate"]
    if stats["gates_passed"] == -1:  # The drone has passed the final gate
        stats["gates_passed"] = len(config.env.track.gates)
    if info["collisions"]:
        termination = "Collision"
    elif info["target_gate"] == -1:
        termination = "Task completed"
    else:
        termination = "Unknown"
    logger.info(
        (
            f"Flight time (s): {curr_time}\n"
            f"Reason for termination: {termination}\n"
            f"Gates passed: {stats['gates_passed']}\n"
            f"Total reward: {stats['ep_reward']}\n"
            f"Number of collisions: {stats['collisions']}\n"
            f"Number of constraint violations: {stats['violations']}\n"
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
