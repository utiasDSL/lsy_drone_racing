"""Example training script using the stable-baselines3 library."""

from __future__ import annotations

import datetime
import logging
from functools import partial
from pathlib import Path

import fire
import yaml
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import DroneRacingWrapper

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))
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


def main(config: str = "../config/train0.yaml", gui: bool = False, log_level: int = logging.INFO):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=log_level)
    config_path = Path(__file__).resolve().parents[1] / config

    env = create_race_env(config_path=config_path, gui=False)
    eval_env = create_race_env(config_path=config_path, gui=gui)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=100_000,
        deterministic=True,
    )

    # Sanity check to ensure the environment conforms to the sb3 API
    # check_env(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=6e-4,
        verbose=1,
        tensorboard_log="logs",
    )  # Train the agent

    train_name = f"ppo_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model.learn(
        total_timesteps=100_000,
        progress_bar=True,
        log_interval=1,
        tb_log_name=train_name,
        callback=eval_callback,
    )
    model.save(f"models/{train_name}")

if __name__ == "__main__":
    fire.Fire(main)
