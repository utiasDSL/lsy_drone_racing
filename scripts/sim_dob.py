"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and `edit_this.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
import pybullet as p

from lsy_drone_racing.sim.noise import ExternalForceGrid
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils.disturbance_observer import FxTDO

if TYPE_CHECKING:
    from munch import Munch

    from lsy_drone_racing.control.controller import BaseController
    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level3.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = False,
    env_id: str | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.
        env_id: The id of the environment to use. If None, the environment specified in the config
            file is used.

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
    env: DroneRacingEnv = gymnasium.make(env_id or config.env.id, config=config)
    dist = ExternalForceGrid(3, [True, True, False], max_force=0.05, grid_size=1.0)
    env.sim.disturbances["dynamics"].append(dist) #WARN: env.sim to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.sim` for environment variables or `env.get_wrapper_attr('sim')` that will search the reminding wrappers.

    ep_times = []
    gui_timer = None
    predictions = [] # x, v, v_hat, f_dis, f_hat
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        done = False
        obs, info = env.reset()
        f_mass_z = (p.getDynamicsInfo(1, -1)[0] - 0.03454) * np.array([0, 0, -9.81])
        # print(f"f_mass_z={f_mass_z}")
        controller: BaseController = controller_cls(obs, info)
        observer = FxTDO(1 / config.env.freq)
        if gui:
            gui_timer = update_gui_timer(0.0, env.unwrapped.sim.pyb_client, gui_timer)
        i = 0

        while not done:
            t_start = time.time()
            curr_time = i / config.env.freq
            if gui:
                gui_timer = update_gui_timer(curr_time, env.unwrapped.sim.pyb_client, gui_timer) # this is slow!

            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            dist.update_pos(obs["pos"])
            done = terminated or truncated
            # Update the controller and observer internal state and models.
            controller.step_callback(action, obs, reward, terminated, truncated, info)

            # calculated like in physics DYN
            forces = np.array(env.sim.drone.rpm**2) * env.sim.drone.params.kf
            des_thrust = np.sum(forces)

            estimate = observer.step(obs, des_thrust)
            # Store observer predictions and real value                                       dist.force  
            predictions.append([np.array(obs["pos"]), np.array(obs["vel"]), estimate[:3]*1.0, env.sim.disturb_force + f_mass_z, estimate[3:]*1.0]) # the factor is to make a copy
            # Add up reward, collisions


            # Synchronize the GUI.
            if config.sim.gui:
                if (elapsed := time.time() - t_start) < 1 / config.env.freq:
                    time.sleep(1 / config.env.freq - elapsed)
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

        plot_predictions(predictions, 1 / config.env.freq)

    # Close the environment
    env.close()
    return ep_times


def update_gui_timer(t: float, client_id: int, g_id: int | None = None) -> int:
    """Update the timer in the GUI."""
    text = f"Ep. time: {t:.2f}s"
    if g_id is None:
        return p.addUserDebugText(text, textPosition=[0, 0, 1.5], physicsClientId=client_id)
    return p.addUserDebugText(
        text,
        textPosition=[0, 0, 1.5],
        textColorRGB=[1, 0, 0],
        lifeTime=0,
        textSize=1.5,
        parentObjectUniqueId=0,
        parentLinkIndex=-1,
        replaceItemUniqueId=g_id,
        physicsClientId=client_id,
    )


def log_episode_stats(obs: dict, info: dict, config: Munch, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    if info["collisions"]:
        termination = "Collision"
    elif obs["target_gate"] == -1:
        termination = "Task completed"
    else:
        termination = "Unknown"
    logger.info(
        (
            f"Flight time (s): {curr_time}\n"
            f"Reason for termination: {termination}\n"
            f"Gates passed: {gates_passed}\n"
        )
    )

def plot_predictions(predictions: list, dt:np.floating):
    """Plots the observations."""
    predictions = np.array(predictions) # x, v, v_hat, f_dis, f_hat; shape (..., 5, 3)
    steps = predictions.shape[0]
    times = np.arange(steps)*dt
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(times, predictions[:,0,0], label="pos_x", color="tab:blue")
    ax[0].plot(times, predictions[:,0,1], label="pos_y", color="tab:orange")
    ax[0].plot(times, predictions[:,0,2], label="pos_z", color="tab:green")
    ax[0].set_ylabel("Position in [m]")
    ax[0].legend()

    ax[1].plot(times, predictions[:,1,0], label="v_x", color="tab:blue")
    ax[1].plot(times, predictions[:,2,0], "--", label="v_x_hat", color="tab:blue")
    ax[1].plot(times, predictions[:,1,1], label="v_y", color="tab:orange")
    ax[1].plot(times, predictions[:,2,1], "--", label="v_y_hat", color="tab:orange")
    ax[1].plot(times, predictions[:,1,2], label="v_z", color="tab:green")
    ax[1].plot(times, predictions[:,2,2], "--", label="v_z_hat", color="tab:green")
    ax[1].set_ylabel("Velocity in [m/s]")
    ax[1].legend()

    ax[2].plot(times, predictions[:,3,0], label="f_d,x", color="tab:blue")
    ax[2].plot(times, predictions[:,4,0], "--", label="f_d,x_hat", color="tab:blue")
    ax[2].plot(times, predictions[:,3,1], label="f_d,y", color="tab:orange")
    ax[2].plot(times, predictions[:,4,1], "--", label="f_d,y_hat", color="tab:orange")
    ax[2].plot(times, predictions[:,3,2], label="f_d,z", color="tab:green")
    ax[2].plot(times, predictions[:,4,2], "--", label="f_d,z_hat", color="tab:green")
    ax[2].set_ylabel("External Force in [N]")
    ax[2].set_xlabel("Time in [s]")
    ax[2].legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
