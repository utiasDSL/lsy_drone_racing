"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

import torch

import logging
from functools import partial
from pathlib import Path

import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper
import datetime

from train_utils import save_observations, process_observation

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """
    Creates a drone racing environment.

    Args:
        config_path (Path): The path to the configuration file.
        gui (bool, optional): Flag indicating whether to enable GUI. Defaults to False.

    Returns:
        DroneRacingWrapper: The drone racing environment.

    Raises:
        AssertionError: If firmware must be used for the competition.
        AssertionError: If pyb_freq is not a multiple of firmware freq.
    """
    # Load configuration and check if firmware should be used.
    config = load_config(config_path)
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True)


def main(config: str = "config/getting_started.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""

    # Create a directory to save the trained model
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(__file__).resolve().parents[1] / f"trained_models/{current_datetime}/"
    save_path.mkdir(parents=True, exist_ok=True)
    #log_path = Path(__file__).resolve().parents[1] / (save_path / "logs/")
    #log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config
    env = create_race_env(config_path=config_path, gui=gui)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API

    epochs = 100
    model = PPO("MlpPolicy", 
                env, verbose=1,
                learning_rate=1e-5, 
                # tensorboard_log=log_path,
    )
    model.learn(total_timesteps=2048*epochs, progress_bar=True, log_interval=5)

    model.save(save_path / f"model_{current_datetime}.zip")

    # Get the observations from the environment
    obs_list = []
    vec_env = model.get_env()

    x = vec_env.reset()

    process_observation(x, True)

    done = False

    ret = 0.
    episode_length = 0
    while not done:
        action, *_ = model.predict(x)
        x, r, done ,info = vec_env.step(action)
        ret += r
        episode_length += 1
        obs_list.append(process_observation(x, False))

    save_observations(obs_list, save_path, current_datetime)

    print(save_path)
    
    


if __name__ == "__main__":
    fire.Fire(main)
