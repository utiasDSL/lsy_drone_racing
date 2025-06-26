"""Data logger utility for storing observations, actions, rewards and additional information of each step in JSONL files per episode."""

import json
import os
from typing import Optional


class DataLogger:
    """A simple data logger for storing the observations, actions, and rewards of each step in a JSONL file.

    Each episode is stored in a separate file named `episode_<index>.jsonl` in the specified log directory.
    The logger automatically finds the next available episode index to avoid overwriting existing files.
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize the DataLogger.

        Args:
            log_dir (str): The directory where the log files will be stored. Defaults to "logs".
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.episode_index = self._find_next_episode_index()
        self.step_log_file: Optional[open] = None

        # Open the first episode log file
        self._start_new_episode()

    def _find_next_episode_index(self) -> int:
        existing_files = os.listdir(self.log_dir)
        indices = []
        for name in existing_files:
            if name.startswith("episode_") and name.endswith(".jsonl"):
                try:
                    idx = int(name[len("episode_") : -len(".jsonl")])
                    indices.append(idx)
                except ValueError:
                    continue
        return max(indices, default=-1) + 1

    def _start_new_episode(self):
        if self.step_log_file:
            self.step_log_file.close()

        file_path = os.path.join(self.log_dir, f"episode_{self.episode_index}.jsonl")
        self.step_log_file = open(file_path, "w")

    def log_step(
        self, obs: dict, action: dict, reward: float, terminated: bool, truncated: bool, info: dict
    ):
        """Log a step in the current episode.

        Args:
            obs (dict): The current observation of the environment.
            action (dict): The action taken by the agent.
            reward (float): The reward received for the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information about the step.
        """
        step_data = {
            "action": action.tolist(),
            "obs": {k: v.tolist() for k, v in obs.items()},
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        self.step_log_file.write(json.dumps(step_data) + "\n")

    def store_episode(self):
        """Store the current episode and start a new one."""
        self.step_log_file.flush()
        self.step_log_file.close()
