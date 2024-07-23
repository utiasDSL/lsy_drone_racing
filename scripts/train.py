"""SAC agent training script for drone racing."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import pip._vendor.tomli as tomllib
import torch
import wandb
from munch import Munch, munchify
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.noise import NormalActionNoise

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.sb3 import (
    NeuralStatsCallback,
    PlacticityCallback,
    RaceStatsCallback,
    WandbLogger,
)
from lsy_drone_racing.wrapper import (
    DroneRacingWrapper,
    MultiProcessingWrapper,
    ObsWrapper,
    RewardWrapper,
)

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

algos = {"sac": SAC, "td3": TD3, "ppo": PPO}


def create_race_env(config_path: Path, gui: bool = False) -> RewardWrapper:
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
    env = DroneRacingWrapper(firmware_env, terminate_on_lap=True)
    return ObsWrapper(RewardWrapper(MultiProcessingWrapper(env)))


def init_run(config: Path, init_wandb: bool = True) -> tuple[Run, Munch]:
    """Initialize the wandb run and load the configuration."""
    with open(config, "rb") as f:
        config = munchify(tomllib.load(f))
    if getattr(config.rng, "seed", None) is not None:
        torch.manual_seed(config.rng.seed)
    torch.backends.cudnn.benchmark = False  # Avoid selecting different algorithms on different runs
    run = None
    if init_wandb:
        save_path = Path(__file__).parents[1] / "saves"
        save_path.mkdir(exist_ok=True, parents=True)
        with open(Path(__file__).resolve().parents[1] / "secrets/wandb_api_key.secret", "r") as f:
            wandb_api_key = f.read()
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            config=config,
            dir=save_path,
        )
    return run, config


def model_kwargs(kwargs: dict) -> dict:
    new_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, dict)}
    if (action_noise := kwargs.get("action_noise")) is not None:
        if action_noise.type != "NormalActionNoise":
            raise NotImplementedError(f"Action noise {action_noise.type} not supported.")
        new_kwargs["action_noise"] = NormalActionNoise(
            mean=action_noise.kwargs.mean * torch.ones(kwargs["env"].action_space.shape[0]),
            sigma=action_noise.kwargs.sigma * torch.ones(kwargs["env"].action_space.shape[0]),
        )
    if (policy_kwargs := kwargs.get("policy_kwargs")) is not None:
        if (fn := policy_kwargs.get("activation_fn")) is not None:
            policy_kwargs["activation_fn"] = getattr(torch.nn, fn)
        new_kwargs["policy_kwargs"] = policy_kwargs
    return new_kwargs


def main(config: str = "config/learning.yaml", wandb: bool = True, algo: str = "sac"):
    """Train a drone racing agent."""
    algo = algo.lower()
    assert algo in algos, f"Algorithm {algo} not supported. Choose from {algos.keys()}."
    logging.basicConfig(level=logging.INFO)
    root_path = Path(__file__).resolve().parents[1]
    save_path = root_path / "saves" / algo
    save_path.mkdir(exist_ok=True, parents=True)
    config_path = root_path / config

    run, cfg = init_run(config=root_path / "config" / f"{algo}.toml", init_wandb=wandb)

    env = make_vec_env(
        lambda: create_race_env(config_path),
        cfg.env.n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = algos[algo]("MlpPolicy", env, **model_kwargs(cfg.model.toDict()), verbose=1)

    if run is not None:
        model.set_logger(Logger(folder=None, output_formats=[WandbLogger(verbose=1)]))

    eval_env = make_vec_env(
        lambda: create_race_env(config_path),
        cfg.eval.n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=save_path / "best_model",
        eval_freq=cfg.eval.freq,
        n_eval_episodes=cfg.eval.n_episodes,
        callback_on_new_best=None,
        verbose=1,
    )

    pkwargs = cfg.placticity.toDict() if cfg.get("placticity") else {}
    callbacks = [
        eval_callback,
        RaceStatsCallback(),
        PlacticityCallback(**pkwargs),
        NeuralStatsCallback(),
    ]
    model.learn(**cfg.learn.toDict(), callback=CallbackList(callbacks))
    model.save(save_path / "model.zip")
    # env.save(save_path / "env.pkl")
    logger.info("Training complete, model saved.")


if __name__ == "__main__":
    fire.Fire(main)
