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


def create_race_env(
    level_path: Path,
    observation_parser_path: Path = "config/observation_parser/default.yaml",
    rewarder_path: Path = "config/rewarder/default.yaml",
    action_transformer_path: Path = "config/action_transformer/default.yaml",
    gui: bool = False,
    seed: int = 0,
) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    assert level_path.exists(), f"Configuration file not found: {level_path}"
    with open(level_path, "r") as file:
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
    return DroneRacingWrapper(
        firmware_env,
        terminate_on_lap=False,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
    )


def main(
    level: str = "config/level/train0.yaml",
    observation_parser: str = "config/observation_parser/default.yaml",
    rewarder: str = "config/rewarder/default.yaml",
    action_transformer: str = "config/action_transformer/default.yaml",
    gui: bool = False,
    gui_eval: bool = False,
    log_level: int = logging.INFO,
    seed: int = 0,
    num_timesteps: int = 500_000,
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=log_level)

    project_path = Path(__file__).resolve().parents[2]
    level_path = project_path / level
    observation_parser_path = project_path / observation_parser
    rewarder_path = project_path / rewarder
    action_transformer_path = project_path / action_transformer

    # Set level name and path
    level_name = level.split("/")[-1].split(".")[0]
    level_short_name = level_name[0] + level_name[-1]

    env = create_race_env(
        level_path=level_path,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
        gui=gui,
        seed=seed,
    )
    eval_env = create_race_env(
        level_path=level_path,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
        gui=gui_eval,
        seed=seed,
    )

    observation_parser_shortname = env.observation_parser.get_shortname()
    rewarder_shortname = env.rewarder.get_shortname()
    action_transformer_shortname = env.action_transformer.get_shortname()
    date_now = datetime.datetime.now().strftime("%m-%d-%H-%M")

    logger.info(
        f"Training {level_short_name} level "
        + f"with {observation_parser_shortname}"
        + f" observation parser, {rewarder_shortname} rewarder," 
        + f" {action_transformer_shortname} action transformer. "
        + f"Time: {date_now}"
    )
    
    train_name = "ppo_" + "_".join(
        [
            level_short_name,
            "obs",
            observation_parser_shortname,
            "rew",
            rewarder_shortname,
            "act",
            action_transformer_shortname,
            "num_timesteps",
            str(num_timesteps),
            "time",
            date_now,
        ]
    )

    best_model_save_path = f"models/{train_name}/best_model"

    # We save the params as a yaml file for reproducibility
    Path(f"models/{train_name}").mkdir(parents=True, exist_ok=True)
    with open(f"models/{train_name}/params.yaml", "w") as file:
        yaml.dump(
            {
                "level": level,
                "observation_parser": observation_parser,
                "rewarder": rewarder,
                "action_transformer": action_transformer,
                "gui": gui,
                "seed": seed,
                "num_timesteps": num_timesteps,
                "model_name": f"{best_model_save_path}/best_model"
            },
            file,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path="./logs/",
        eval_freq=100_000,
        deterministic=True,
    )

    # Sanity check to ensure the environment conforms to the sb3 API
    # check_env(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="logs",
    )  # Train the agent

    try:
        model.learn(
            total_timesteps=num_timesteps,
            progress_bar=True,
            tb_log_name=train_name,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving model.")

    model.save(f"models/{train_name}/{train_name}")


if __name__ == "__main__":
    fire.Fire(main)
