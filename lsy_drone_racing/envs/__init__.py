"""Register environments."""

from gymnasium import register

from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv

register(
    id="DroneRacing-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_env:DroneRacingEnv",
    max_episode_steps=900,  # 30 seconds * 30 Hz,
    disable_env_checker=True,  # Remove warnings about 2D observations
)

register(
    id="DroneRacingThrust-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_env:DroneRacingThrustEnv",
    max_episode_steps=900,
    disable_env_checker=True,
)

register(
    id="DroneRacingDeploy-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_deploy_env:DroneRacingDeployEnv",
    max_episode_steps=900,
    disable_env_checker=True,
)

__all__ = ["DroneRacingEnv"]
