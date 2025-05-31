from gymnasium import register

register(
    id="RLDroneRacing-v0",
    entry_point="lsy_drone_racing.reinforcement_learning.rl_drone_race:RLDroneRaceEnv",
    vector_entry_point="lsy_drone_racing.reinforcement_learning.rl_drone_race:VecRLDroneRaceEnv",
    max_episode_steps=1500,  # 30 seconds * 50 Hz,
    disable_env_checker=True,  # Remove warnings about 2D observations
)

register(
    id="RLDroneHover-v0",
    entry_point="lsy_drone_racing.reinforcement_learning.rl_drone_race:RLDroneHoverEnv",
    vector_entry_point="lsy_drone_racing.reinforcement_learning.rl_drone_race:VecRLDroneRaceEnv",
    max_episode_steps=1500,  # 30 seconds * 50 Hz,
    disable_env_checker=True,  # Remove warnings about 2D observations
)
