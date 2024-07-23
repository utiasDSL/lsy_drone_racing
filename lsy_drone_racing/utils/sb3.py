from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, SeqWriter

logger = logging.getLogger(__name__)


class WandbLogger(KVWriter, SeqWriter):
    def __init__(self, verbose: int = 0):
        assert wandb.run is not None, "Wandb run must be initialized before using WandbLogger."
        self._log_to_stdout = verbose > 0

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0
    ):
        wandb.run.log(key_values, step=step)
        if self._log_to_stdout:
            logger.info("\n" + "\n".join(f"{k}: {v}" for k, v in key_values.items()) + "\n")

    def write_sequence(self, sequence: list[str]):
        pass


class RaceStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        # Check if the episode is done
        if any(self.locals["dones"]):
            infos = [self.locals["infos"][i] for i in np.where(self.locals["dones"])[0]]
            successes = [x["task_completed"] for x in infos]
            gate_ids = [x["current_gate_id"] for x in infos]
            # Taking the mean is slightly misleading. If two episodes are done, and one is a success
            # and one is a failure, the mean has more significance than if only one env was done. We
            # do not track this in WandB, so the stats are off. However, we expect this to average
            # out over time.
            drones_pos = np.array([x["drone_pose"][:3] for x in infos])
            gates_pos = infos[0]["gates_pose"][gate_ids, :3]
            distances = np.linalg.norm(drones_pos - gates_pos, axis=1)
            self.logger.record("rollout/gate_distance", distances.mean())
            self.logger.record("rollout/success", np.mean(successes))
            self.logger.record("rollout/gate_id", np.mean(gate_ids))
        return True


class NeuralStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        mag, n = 0, 0
        for p in self.model.policy.parameters():
            mag += p.data.abs().sum().item()
            n += p.data.numel()
        self.logger.record("train/param_mag", mag / n)


class PlacticityCallback(BaseCallback):
    def __init__(self, freq: int = 100, std: float = 1e-6, gamma: float = 0.998):
        super().__init__()
        self.freq = freq
        self.std = std
        self.gamma = gamma
        self._last_reset = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.model._n_updates - self._last_reset > self.freq:
            self._shrink_perturb()
            self._last_reset = self.model._n_updates

    def _shrink_perturb(self):
        for name, param in self.model.policy.named_parameters():
            # Only shrink weights, not biases
            if "weight" in name:
                param.data = self.gamma * param.data
            param.data = param.data + torch.randn_like(param.data) * self.std
