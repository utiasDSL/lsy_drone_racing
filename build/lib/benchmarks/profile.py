from pathlib import Path

import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from pyinstrument import Profiler

import lsy_drone_racing  # noqa: F401, required for gymnasium.make
from lsy_drone_racing.utils import load_config


def main():
    config = load_config(Path(__file__).parents[1] / "config/level2.toml")
    env = gymnasium.make(
        "DroneRacing-v0",
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)
    # JIT compile the reset and step functions
    env.reset()
    env.step(env.action_space.sample())

    profiler = Profiler()
    profiler.start()

    for _ in range(1_000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    profiler.stop()
    profiler.print()
    profiler.open_in_browser()


if __name__ == "__main__":
    main()
