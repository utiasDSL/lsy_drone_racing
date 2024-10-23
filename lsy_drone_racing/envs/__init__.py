"""This module contains the environments for the drone racing challenge.

Environments are split into simulation and real-world environments. The simulation environments use
the simulation module to provide a realistic simulation of the drone racing challenge for training,
testing and iterating on controller designs. The real-world environments mirror the interface of the
simulation environments, but use the Vicon motion capture system to track the drone and race track
elements in the lab, and sent the controller actions to the real drone.

Note:
    While the interfaces are the same and we try to keep the environments as similar as possible,
    the dynamics of the drone and all observations are subject to a sim2real gap. The transition
    between simulation and real-world may therefore require additional tuning of the controller
    design.
"""

from gymnasium import register

from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv

__all__ = ["DroneRacingEnv"]

register(
    id="DroneRacing-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_env:DroneRacingEnv",
    max_episode_steps=1800,  # 30 seconds * 60 Hz,
    disable_env_checker=True,  # Remove warnings about 2D observations
)

register(
    id="DroneRacingThrust-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_env:DroneRacingThrustEnv",
    max_episode_steps=1800,
    disable_env_checker=True,
)

register(
    id="DroneRacingDeploy-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_deploy_env:DroneRacingDeployEnv",
    max_episode_steps=1800,
    disable_env_checker=True,
)

register(
    id="DroneRacingThrustDeploy-v0",
    entry_point="lsy_drone_racing.envs.drone_racing_deploy_env:DroneRacingThrustDeployEnv",
    max_episode_steps=1800,
    disable_env_checker=True,
)
