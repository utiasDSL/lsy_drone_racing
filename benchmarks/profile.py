from pathlib import Path

import gymnasium

import lsy_drone_racing
from lsy_drone_racing.utils import load_config


def main():
    config = load_config(Path(__file__).parent / "config/test.toml")
    env = gymnasium.make("DroneRacing-v0", config=config)
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
    # 0.455


if __name__ == "__main__":
    main()
