r"""Train an AIGP policy with a compressed curriculum (SB3 PPO).

This script is intentionally optional: it requires Stable-Baselines3 + torch.

Example:
    python scripts/train_aigp_curriculum.py train \
        --config aigp_stage0_single_gate.toml \
        --curriculum aigp_curriculum_10stage.toml \
        --out runs/aigp_debug \
        --num_envs 64
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fire
import numpy as np

from lsy_drone_racing.aigp.curriculum import CurriculumConfig, CurriculumManager
from lsy_drone_racing.aigp.eval import evaluate_vec_env, make_predict_policy
from lsy_drone_racing.aigp.sb3_vec_env import make_sb3_vec_env
from lsy_drone_racing.aigp.wrappers import (
    ActionLatencyWrapper,
    ImuBiasNoiseWrapper,
    ImuNoiseConfig,
    VioFailureConfig,
    VioFailureWrapper,
)
from lsy_drone_racing.envs.aigp_drone_race import VecAIGPDroneRaceEnv
from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def _require_sb3() -> Any:  # noqa: ANN401
    try:
        import stable_baselines3  # type: ignore[import-not-found]

        return stable_baselines3
    except Exception as exc:
        raise ImportError(
            "Stable-Baselines3 is required for this script. "
            "Install it (and torch) before running training."
        ) from exc


def _dr_specs_for_tier(
    tier: str, *, mult: float = 1.0
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (disturbances, randomizations) specs for a DR tier."""
    tier = str(tier).lower().strip()
    mult = float(mult)

    if tier in {"none", "", "off"}:
        return None, None

    # Conservative baselines (loosely inspired by `config/level1.toml`).
    # Specs are converted to JAX RNG functions via `rng_spec2fn()` in `RaceCoreEnv`.
    rand: dict[str, Any] = {
        "drone_pos": {
            "fn": "uniform",
            "kwargs": {
                "minval": [-0.10 * mult, -0.10 * mult, 0.0],
                "maxval": [0.10 * mult, 0.10 * mult, 0.02 * mult],
            },
        },
        "drone_rpy": {
            "fn": "uniform",
            "kwargs": {
                "minval": [-0.10 * mult, -0.10 * mult, -0.10 * mult],
                "maxval": [0.10 * mult, 0.10 * mult, 0.10 * mult],
            },
        },
        "drone_mass": {
            "fn": "uniform",
            "kwargs": {"minval": -0.005 * mult, "maxval": 0.005 * mult},
        },
        "drone_inertia": {
            "fn": "uniform",
            "kwargs": {
                "minval": [-1.0e-6 * mult, -1.0e-6 * mult, -1.0e-6 * mult],
                "maxval": [1.0e-6 * mult, 1.0e-6 * mult, 1.0e-6 * mult],
            },
        },
    }

    # Add propulsion randomizations for stronger tiers (first_principles only; no-ops otherwise).
    if tier in {"medium", "full", "aggressive"}:
        rand["drone_rpm2thrust"] = {
            "fn": "uniform",
            "kwargs": {
                "minval": [0.0, -1.0e-8 * mult, -1.0e-11 * mult],
                "maxval": [0.0, 1.0e-8 * mult, 1.0e-11 * mult],
            },
        }
        rand["drone_rotor_dyn_coef"] = {
            "fn": "uniform",
            "kwargs": {
                "minval": [-0.5 * mult, -1.0e-5 * mult, -0.2 * mult, -1.0e-5 * mult],
                "maxval": [0.5 * mult, 1.0e-5 * mult, 0.2 * mult, 1.0e-5 * mult],
            },
        }

    sigma = 0.0
    if tier == "low":
        sigma = 0.05
    elif tier == "medium":
        sigma = 0.10
    elif tier == "full":
        sigma = 0.20
    elif tier == "aggressive":
        sigma = 0.30

    disturbances: dict[str, Any] | None = None
    if sigma > 0.0:
        disturbances = {
            "dynamics": {
                "process": "ou",
                "theta": 0.8,
                "sigma": [sigma * mult, sigma * mult, sigma * mult],
                "mean": [0.0, 0.0, 0.0],
                "clip": 1.0,
            }
        }

    return disturbances, rand


def _wrap_sim2real(env: Any, *, tier: str, seed: int) -> Any:  # noqa: ANN401
    """Apply Python-level sim2real wrappers based on DR tier."""
    tier = str(tier).lower().strip()

    if tier in {"none", "", "off"}:
        return env

    # Mild latency for anything beyond "none".
    if tier in {"low", "medium", "full", "aggressive"}:
        env = ActionLatencyWrapper(
            env,
            latency_steps=(0, 1) if tier == "low" else (0, 2),
            seed=seed,
        )

    # Sensor noise / failures for stronger tiers.
    if tier in {"full", "aggressive"}:
        env = ImuBiasNoiseWrapper(
            env,
            ImuNoiseConfig(
                vel_bias_std=0.03 if tier == "full" else 0.06,
                ang_vel_bias_std=0.03 if tier == "full" else 0.06,
                vel_noise_std=0.01 if tier == "full" else 0.02,
                ang_vel_noise_std=0.01 if tier == "full" else 0.02,
                bias_drift_std=0.001 if tier == "full" else 0.002,
            ),
            seed=seed + 1,
        )
        env = VioFailureWrapper(
            env,
            VioFailureConfig(
                failure_prob=2.0e-4 if tier == "full" else 5.0e-4,
                max_hold_steps=10 if tier == "full" else 20,
                mode="hold",
            ),
            seed=seed + 2,
        )

    return env


def _save_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def train(
    *,
    config: str = "aigp_stage0_single_gate.toml",
    curriculum: str = "aigp_curriculum_10stage.toml",
    out: str = "runs/aigp",
    num_envs: int = 64,
    eval_envs: int = 32,
    eval_every_timesteps: int = 50_000,
    timesteps_per_stage: int = 300_000,
    seed: int = 0,
    device: str = "auto",
    net_arch: str = "256,128",
) -> None:
    """Train PPO with a curriculum.

    Args:
        config: Base env config file under `config/` (sim settings, control mode, freq, etc).
        curriculum: Curriculum config under `config/`.
        out: Output directory for checkpoints and logs.
        num_envs: Number of parallel training environments.
        eval_envs: Number of parallel evaluation environments.
        eval_every_timesteps: Train this many timesteps between evals.
        timesteps_per_stage: Maximum timesteps to spend in each stage before forcing advance.
        seed: Random seed.
        device: Torch device for SB3 ("cpu", "cuda", "mps", or "auto").
        net_arch: MLP architecture as comma-separated ints (e.g. "256,128").
    """
    sb3 = _require_sb3()
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-not-found]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "curriculum_log.jsonl"

    config_dir = Path(__file__).parents[1] / "config"
    base_cfg = load_config(config_dir / config)
    cur_cfg = CurriculumConfig.load(config_dir / curriculum)
    mgr = CurriculumManager(cur_cfg)

    arch = tuple(int(x) for x in net_arch.split(",") if x.strip())

    class _EpisodeCounter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = 0

        def _on_step(self) -> bool:  # noqa: ANN101
            dones = self.locals.get("dones")
            if dones is not None:
                self.episodes += int(np.sum(dones))
            return True

    model = None

    while True:
        stage = mgr.current_stage()
        stage_eval_eps = int(stage.eval_episodes or cur_cfg.eval_episodes)
        stage_max_steps = int(
            stage.max_episode_steps or base_cfg.env.get("max_episode_steps", 1500)
        )

        tracks, weights = mgr.build_stage_tracks(config_dir=config_dir)

        disturbances, randomizations = _dr_specs_for_tier(stage.dr_tier, mult=1.0)

        # Build training env.
        env = VecAIGPDroneRaceEnv(
            num_envs=int(num_envs),
            freq=int(base_cfg.env.freq),
            sim_config=base_cfg.sim,
            track=tracks[0],
            sensor_range=float(base_cfg.env.sensor_range),
            control_mode=str(base_cfg.env.control_mode),
            disturbances=disturbances,
            randomizations=randomizations,
            reward_config=str(stage.reward_preset),
            seed=int(seed),
            max_episode_steps=stage_max_steps,
            device=str(base_cfg.env.get("device", "cpu")),
        )
        if len(tracks) > 1 or weights is not None:
            env.set_track_pool(tracks, probs=weights)

        env = _wrap_sim2real(env, tier=stage.dr_tier, seed=seed)
        sb3_env = make_sb3_vec_env(env)

        if model is None:
            model = sb3.PPO(
                policy="MultiInputPolicy",
                env=sb3_env,
                verbose=1,
                tensorboard_log=str(out_dir / "tb"),
                device=device,
                policy_kwargs={"net_arch": list(arch)},
            )
        else:
            model.set_env(sb3_env)

        logger.info(
            "TRAIN_STAGE idx=%s name=%s dr_tier=%s reward=%s max_steps=%s",
            mgr.stage_idx,
            stage.name,
            stage.dr_tier,
            stage.reward_preset,
            stage_max_steps,
        )

        counter = _EpisodeCounter()
        stage_start_timesteps = int(model.num_timesteps)

        while True:
            # Train a chunk.
            model.learn(
                total_timesteps=int(eval_every_timesteps),
                reset_num_timesteps=False,
                callback=counter,
            )

            # Evaluate with a fresh env to avoid leaked state.
            eval_tracks, eval_weights = mgr.build_stage_tracks(config_dir=config_dir)
            eval_env = VecAIGPDroneRaceEnv(
                num_envs=int(eval_envs),
                freq=int(base_cfg.env.freq),
                sim_config=base_cfg.sim,
                track=eval_tracks[0],
                sensor_range=float(base_cfg.env.sensor_range),
                control_mode=str(base_cfg.env.control_mode),
                disturbances=disturbances,
                randomizations=randomizations,
                reward_config=str(stage.reward_preset),
                seed=int(seed) + 10_000,
                max_episode_steps=stage_max_steps,
                device=str(base_cfg.env.get("device", "cpu")),
            )
            if len(eval_tracks) > 1 or eval_weights is not None:
                eval_env.set_track_pool(eval_tracks, probs=eval_weights)
            eval_env = _wrap_sim2real(eval_env, tier=stage.dr_tier, seed=seed + 10_000)

            summary = evaluate_vec_env(
                eval_env,
                make_predict_policy(model, deterministic=True),
                n_episodes=stage_eval_eps,
                max_episode_steps=stage_max_steps,
            )
            eval_env.close()

            decision = mgr.update_after_eval(summary=summary, stage_episodes=int(counter.episodes))

            row = {
                "stage": {"idx": mgr.stage_idx, "name": stage.name, "dr_tier": stage.dr_tier},
                "timesteps": int(model.num_timesteps),
                "stage_episodes": int(counter.episodes),
                "eval": asdict(summary),
                "decision": {k: v for k, v in decision.items() if k != "adaptive_smoothed"},
            }
            _save_jsonl(log_path, row)

            ckpt = (
                out_dir
                / f"stage{mgr.stage_idx:02d}_{stage.name}"
                / f"step_{model.num_timesteps}.zip"
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt))

            if decision.get("rollback"):
                mgr.rollback()
                break

            if decision.get("advance"):
                mgr.advance()
                break

            if int(model.num_timesteps) - stage_start_timesteps >= int(timesteps_per_stage):
                logger.info("FORCE_ADVANCE stage=%s reason=timesteps_per_stage", stage.name)
                mgr.advance()
                break

        sb3_env.close()

        if mgr.stage_idx >= len(cur_cfg.stages) - 1:
            break


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire({"train": train}, serialize=lambda _: None)
