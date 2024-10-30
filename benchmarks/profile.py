from pathlib import Path

import gymnasium

import lsy_drone_racing  # noqa: F401, required for gymnasium.make
from lsy_drone_racing.utils import load_config


def main():
    config = load_config(Path(__file__).parents[1] / "config/level0.toml")
    env = gymnasium.make("DroneRacing-v0", config=config)
    env.reset()
    for _ in range(1_000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()


if __name__ == "__main__":
    main()
