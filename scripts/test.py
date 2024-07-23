"""SAC agent training script for drone racing."""
from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import DummyVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper, ObsWrapper, RewardWrapper

logger = logging.getLogger(__name__)


algos = {"sac": SAC, "td3": TD3, "ppo": PPO}


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
    return ObsWrapper(RewardWrapper(DroneRacingWrapper(firmware_env, terminate_on_lap=True)))


def main(
    config: str = "config/learning.yaml",
    gui: bool = True,
    n_tests: int = 1,
    delay: float = 0,
    algo: str = "SAC",
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    algo = algo.lower()
    assert algo in algos, f"Algorithm {algo} not supported. Choose from {algos.keys()}."
    logging.basicConfig(level=logging.INFO)
    root_path = Path(__file__).resolve().parents[1]
    save_path = root_path / "saves" / algo
    config_path = root_path / config

    env = make_vec_env(lambda: create_race_env(config_path, gui=gui), 1, vec_env_cls=DummyVecEnv)
    # env = VecNormalize.load(save_path / "env.pkl", env)
    assert algo in algos, f"Algorithm {algo} not supported."

    model = algos[algo].load(save_path / "model.zip")
    success = []
    steps = []
    rewards = []
    for i in range(n_tests):
        obs = env.reset()
        done = False
        steps.append(0)
        rewards.append(0)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(env.envs[0].sim_time)
            time.sleep(delay)
            steps[-1] += 1
            rewards[-1] += reward[0]
        print(f"Test {i + 1} reached gate {info[0]['current_gate_id']}.")
        success.append(info[0]["task_completed"])
    print(f"Success rate: {sum(success) / n_tests:.2f}")
    print(f"Avg. steps: {sum(steps) / n_tests:.2f}")
    print(f"Avg. reward: {sum(rewards) / n_tests:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
