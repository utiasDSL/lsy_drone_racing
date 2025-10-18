"""Sweep it!"""
from pathlib import Path

import numpy as np
import torch
import wandb

from lsy_drone_racing.control.train_rl import Args, evaluate_ppo, train_ppo  # noqa: F401


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="ADR-PPO-sweep-stack") as run:
        args = Args.create(**dict(run.config))
        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        jax_device = args.jax_device
        sum_rewards_hist = train_ppo(args, model_path, device, jax_device, wandb_enabled=True)
        mean_rewards = np.asarray(sum_rewards_hist).mean()
        score = mean_rewards
        # _, rmse_pos, episode_rewards, _ = evaluate_ppo(args, eval, model_path, device)
        # score -= 10 * rmse_pos
        run.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "random",  # "random", "bayes", "grid"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-3},
        "gamma": {"min": 0.85, "max": 0.95},
        "gae_lambda": {"min": 0.9, "max": 0.99},
        "num_steps": {"values": [4, 8, 16]},
        "num_minibatches": {"values": [4, 8, 16]},
        # "clip_coef": {"min": 0.1, "max": 0.3},
        # "ent_coef": {"min": 0.0, "max": 1e-2},
        # "vf_coef": {"min": 0.3, "max": 1.0},
        # "max_grad_norm": {"min": 1.5, "max": 4.0},
        "n_obs": {"min": 1, "max": 4},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="ADR-PPO-sweep-stack", entity="fresssack")

wandb.agent(sweep_id, function=train, count=100)
