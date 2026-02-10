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

import copy
import json
import logging
from dataclasses import asdict
from importlib.util import find_spec
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


def _parse_net_arch(net_arch: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse `net_arch` from Fire.

    Fire will interpret comma-separated CLI values like `--net_arch 64,64` as a Python tuple.
    Accept both formats to avoid surprising runtime errors.
    """
    if isinstance(net_arch, str):
        return tuple(int(x) for x in net_arch.split(",") if x.strip())
    if isinstance(net_arch, (tuple, list)):
        return tuple(int(x) for x in net_arch)
    raise TypeError("net_arch must be a comma-separated string or a list/tuple of ints")


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


def _scale_gate_geometry(
    tracks: list[Any], *, gate_scale: float = 1.0, tolerance_mult: float = 1.0
) -> list[Any]:
    """Scale per-track gate sizes/tolerances (used for panic/recovery assist)."""
    scaled: list[Any] = []
    for t in tracks:
        t2 = copy.deepcopy(t)
        if "gate_size" not in t2:
            scaled.append(t2)
            continue

        gate_size = dict(t2.gate_size)
        if "width" in gate_size:
            gate_size["width"] = float(gate_size["width"]) * float(gate_scale)
        if "height" in gate_size:
            gate_size["height"] = float(gate_size["height"]) * float(gate_scale)
        if "tolerance" in gate_size:
            gate_size["tolerance"] = float(gate_size["tolerance"]) * float(tolerance_mult)
        t2["gate_size"] = gate_size
        scaled.append(t2)
    return scaled


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
    net_arch: str | tuple[int, ...] | list[int] = "256,128",
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
    tb_log = str(out_dir / "tb") if find_spec("tensorboard") is not None else None

    config_dir = Path(__file__).parents[1] / "config"
    base_cfg = load_config(config_dir / config)
    cur_cfg = CurriculumConfig.load(config_dir / curriculum)
    mgr = CurriculumManager(cur_cfg)
    final_stage_idx = len(cur_cfg.stages) - 1

    arch = _parse_net_arch(net_arch)

    class _EpisodeCounter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = 0

        def _on_step(self) -> bool:  # noqa: ANN101
            dones = self.locals.get("dones")
            if dones is not None:
                self.episodes += int(np.sum(dones))
            return True

    def _build_vec_env_for_stage(
        *,
        stage_idx: int,
        num_envs_local: int,
        seed_offset: int,
        dr_mult: float,
        gate_scale: float,
        tol_mult: float,
    ) -> Any:  # noqa: ANN401
        stage_local = mgr.stage_by_idx(stage_idx)
        stage_max_steps_local = int(
            stage_local.max_episode_steps or base_cfg.env.get("max_episode_steps", 1500)
        )

        tracks_local, weights_local = mgr.build_tracks_for_stage(
            stage_idx=stage_idx, config_dir=config_dir
        )
        tracks_local = _scale_gate_geometry(
            tracks_local, gate_scale=gate_scale, tolerance_mult=tol_mult
        )

        disturbances_local, randomizations_local = _dr_specs_for_tier(
            stage_local.dr_tier, mult=dr_mult
        )
        env_local = VecAIGPDroneRaceEnv(
            num_envs=int(num_envs_local),
            freq=int(base_cfg.env.freq),
            sim_config=base_cfg.sim,
            track=tracks_local[0],
            sensor_range=float(base_cfg.env.sensor_range),
            control_mode=str(base_cfg.env.control_mode),
            disturbances=disturbances_local,
            randomizations=randomizations_local,
            reward_config=str(stage_local.reward_preset),
            seed=int(seed) + int(seed_offset),
            max_episode_steps=stage_max_steps_local,
            device=str(base_cfg.env.get("device", "cpu")),
        )
        if len(tracks_local) > 1 or weights_local is not None:
            env_local.set_track_pool(tracks_local, probs=weights_local)
        env_local = _wrap_sim2real(env_local, tier=stage_local.dr_tier, seed=seed + seed_offset)
        return env_local

    model = None

    while True:
        stage_idx = int(mgr.stage_idx)
        stage = mgr.current_stage()
        stage_eval_eps = int(stage.eval_episodes or cur_cfg.eval_episodes)
        stage_max_steps = int(
            stage.max_episode_steps or base_cfg.env.get("max_episode_steps", 1500)
        )

        # Effective difficulty modifiers (updated after each eval).
        eff_dr_mult = 1.0
        eff_gate_scale = 1.0
        eff_tol_mult = 1.0

        env = _build_vec_env_for_stage(
            stage_idx=stage_idx,
            num_envs_local=int(num_envs),
            seed_offset=0,
            dr_mult=eff_dr_mult,
            gate_scale=eff_gate_scale,
            tol_mult=eff_tol_mult,
        )
        sb3_env = make_sb3_vec_env(env)

        if model is None:
            model = sb3.PPO(
                policy="MultiInputPolicy",
                env=sb3_env,
                verbose=1,
                tensorboard_log=tb_log,
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
        eval_round = 0

        while True:
            # Train a chunk.
            model.learn(
                total_timesteps=int(eval_every_timesteps),
                reset_num_timesteps=False,
                callback=counter,
            )
            eval_round += 1

            # Evaluate with a fresh env to avoid leaked state.
            eval_env = _build_vec_env_for_stage(
                stage_idx=stage_idx,
                num_envs_local=int(eval_envs),
                seed_offset=10_000,
                dr_mult=eff_dr_mult,
                gate_scale=eff_gate_scale,
                tol_mult=eff_tol_mult,
            )

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
                "effective": {
                    "dr_mult": float(eff_dr_mult),
                    "gate_scale": float(eff_gate_scale),
                    "tol_mult": float(eff_tol_mult),
                },
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

            # Forgetting detection: periodically re-evaluate the previous stage.
            if (
                mgr.forgetting.baseline_stage_idx is not None
                and mgr.forgetting.baseline_success_rate is not None
                and mgr.stage_idx > int(mgr.forgetting.baseline_stage_idx)
                and (eval_round % 3 == 0)
            ):
                prev_idx = int(mgr.forgetting.baseline_stage_idx)
                prev_stage = mgr.stage_by_idx(prev_idx)
                prev_eval_eps = int(
                    max(10, (prev_stage.eval_episodes or cur_cfg.eval_episodes) // 2)
                )
                prev_max_steps = int(
                    prev_stage.max_episode_steps or base_cfg.env.get("max_episode_steps", 1500)
                )
                prev_env = _build_vec_env_for_stage(
                    stage_idx=prev_idx,
                    num_envs_local=int(eval_envs),
                    seed_offset=20_000,
                    dr_mult=1.0,
                    gate_scale=1.0,
                    tol_mult=1.0,
                )
                prev_summary = evaluate_vec_env(
                    prev_env,
                    make_predict_policy(model, deterministic=True),
                    n_episodes=prev_eval_eps,
                    max_episode_steps=prev_max_steps,
                )
                prev_env.close()

                baseline = float(mgr.forgetting.baseline_success_rate)
                if prev_summary.success_rate < max(0.05, baseline * 0.5):
                    _save_jsonl(
                        log_path,
                        {
                            "event": "forgetting_rollback",
                            "timesteps": int(model.num_timesteps),
                            "baseline_sr": baseline,
                            "prev_stage_idx": prev_idx,
                            "prev_eval": asdict(prev_summary),
                        },
                    )
                    mgr.rollback()
                    break

            if decision.get("rollback"):
                mgr.rollback()
                break

            if decision.get("advance") and int(mgr.stage_idx) < final_stage_idx:
                mgr.advance()
                mgr.set_forgetting_baseline(success_rate=float(summary.success_rate))
                break

            # Update effective DR/assist multipliers for the next chunk.
            eff_dr_mult_new = float(decision.get("panic_dr_mult", 1.0)) * float(
                decision.get("adaptive_mult", 1.0)
            )
            assist_mult = float(decision.get("panic_assist_mult", 1.0))
            recovery_active = bool(decision.get("recovery_active", False))
            eff_gate_scale_new = assist_mult * (
                float(cur_cfg.recovery.gate_scale) if recovery_active else 1.0
            )
            eff_tol_mult_new = assist_mult * (
                float(cur_cfg.recovery.tolerance_mult) if recovery_active else 1.0
            )

            changed = (
                abs(eff_dr_mult_new - eff_dr_mult) > 1e-3
                or abs(eff_gate_scale_new - eff_gate_scale) > 1e-3
                or abs(eff_tol_mult_new - eff_tol_mult) > 1e-3
            )
            if changed:
                sb3_env.close()
                env = _build_vec_env_for_stage(
                    stage_idx=stage_idx,
                    num_envs_local=int(num_envs),
                    seed_offset=0,
                    dr_mult=eff_dr_mult_new,
                    gate_scale=eff_gate_scale_new,
                    tol_mult=eff_tol_mult_new,
                )
                sb3_env = make_sb3_vec_env(env)
                model.set_env(sb3_env)
                eff_dr_mult = eff_dr_mult_new
                eff_gate_scale = eff_gate_scale_new
                eff_tol_mult = eff_tol_mult_new

            if int(model.num_timesteps) - stage_start_timesteps >= int(timesteps_per_stage):
                logger.info("FORCE_ADVANCE stage=%s reason=timesteps_per_stage", stage.name)
                if int(mgr.stage_idx) < final_stage_idx:
                    mgr.advance()
                break

        sb3_env.close()

        # Stop once the final stage has been trained for its budget.
        if int(mgr.stage_idx) >= final_stage_idx and stage_idx >= final_stage_idx:
            break


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire({"train": train}, serialize=lambda _: None)
