"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
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
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config
    env = create_race_env(config_path=config_path, gui=gui)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=4096)


if __name__ == "__main__":
    fire.Fire(main)
