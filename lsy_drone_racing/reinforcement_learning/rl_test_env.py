import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from datetime import datetime
import os
from pathlib import Path

from docs import conf
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RenderCallback

config = load_config(Path(__file__).parents[2] / "config" / "levelrl.toml")

env = RLDroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )

obs, info = env.reset()
i = 0
fps = 60

while True:
    curr_time = i / config.env.freq

    action = None
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
    # Synchronize the GUI.
    if config.sim.gui:
        if ((i * fps) % config.env.freq) < fps:
            env.render()
    i += 1

# Close the environment
env.close()