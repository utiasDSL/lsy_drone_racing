from pathlib import Path
from stable_baselines3 import PPO
from train import create_race_env
from train_utils import process_observation, save_observations
import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
import os


def play_trained_model(model_path: str, config_path: str, gui: bool = True):
    """Load a trained model and play it in the environment."""
    # Create environment
    env = DummyVecEnv([lambda: create_race_env(Path(config_path), gui=gui)])
    # Load the trained model
    model = PPO.load(model_path)
    # Set the model's environment
    model.set_env(env)
    # Play the model in the environment
    episodes = 100
    for i in range(episodes):
        obs_list = []
        x = env.reset()
        process_observation(x, False)
        done = False
        ret = 0.
        episode_length = 0
        while not done:
            action, *_ = model.predict(x)
            x, r, done, info = env.step(action)
            ret += r
            episode_length += 1
            obs_list.append(process_observation(x, False))
        # Save the observations
        save_path = episodes_path + "episodes"
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_observations(obs_list, save_path, i)    
    
    return ret, episode_length

if __name__ == '__main__':
    model_path = "trained_models/2024-06-02_21-10-26/best_model.zip"
    episodes_path = os.path.dirname(model_path) + "/"
    config_path = "config/getting_started.yaml"
    ret, episode_length = play_trained_model(model_path, config_path, episodes_path)
    print(f"Return: {ret}, Episode length: {episode_length}")