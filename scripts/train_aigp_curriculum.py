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
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lsy_drone_racing.aigp.curriculum import CurriculumConfig, CurriculumManager, EvalSummary
from lsy_drone_racing.aigp.eval import (
    aggregate_track_eval_summaries,
    evaluate_sb3_vec_env,
    make_predict_policy,
)
from lsy_drone_racing.aigp.observation import CompetitionProxyObsWrapper
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

try:
    import fire
except Exception:  # pragma: no cover - optional dependency for offline runtimes
    fire = None

try:
    import wandb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]


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


def _parse_force_advance_mode(mode: str) -> str:
    """Parse and validate force-advance mode."""
    normalized = str(mode).strip().lower()
    valid = {"never", "if_passing", "always"}
    if normalized not in valid:
        raise ValueError(f"force_advance_mode must be one of {sorted(valid)}, got: {mode!r}")
    return normalized


def _parse_obs_mode(mode: str) -> str:
    """Parse and validate observation mode."""
    normalized = str(mode).strip().lower()
    valid = {"privileged", "competition_proxy"}
    if normalized not in valid:
        raise ValueError(f"obs_mode must be one of {sorted(valid)}, got: {mode!r}")
    return normalized


def _resolve_tournament_train_settings(
    *,
    tournament_mode: bool,
    obs_mode: str,
    qualifier_eval_profile: str | None,
    tournament_readiness_profile: str | None,
) -> tuple[str, str | None, str | None]:
    """Resolve tournament-specific defaults and enforce competition-only observations."""
    resolved_obs_mode = str(obs_mode)
    resolved_qualifier_profile = qualifier_eval_profile
    resolved_readiness_profile = tournament_readiness_profile

    if bool(tournament_mode):
        resolved_obs_mode = _parse_obs_mode(resolved_obs_mode)
        if resolved_obs_mode != "competition_proxy":
            raise ValueError("tournament_mode requires obs_mode='competition_proxy'")
        if resolved_qualifier_profile is None:
            resolved_qualifier_profile = "aigp_qualifier_eval_profile_default.toml"
        if resolved_readiness_profile is None:
            resolved_readiness_profile = "qualifier_strict"
    else:
        resolved_obs_mode = _parse_obs_mode(resolved_obs_mode)

    return resolved_obs_mode, resolved_qualifier_profile, resolved_readiness_profile


def _parse_wandb_mode(mode: str) -> str:
    """Parse and validate Weights & Biases mode."""
    normalized = str(mode).strip().lower()
    valid = {"online", "offline", "disabled"}
    if normalized not in valid:
        raise ValueError(f"wandb_mode must be one of {sorted(valid)}, got: {mode!r}")
    return normalized


def _parse_wandb_tags(tags: Any) -> list[str]:  # noqa: ANN401
    """Parse wandb tags from Fire-compatible inputs."""
    if tags is None:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, (tuple, list)):
        out: list[str] = []
        for tag in tags:
            t = str(tag).strip()
            if t:
                out.append(t)
        return out
    return [str(tags).strip()] if str(tags).strip() else []


def _normalize_wandb_run_path(value: Any) -> str | None:  # noqa: ANN401
    """Normalize wandb run path representation to a string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return "/".join(parts) if parts else None
    return str(value)


def _wandb_info_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return additive wandb metadata payload for JSONL rows."""
    return {
        "enabled": bool(state.get("enabled", False)),
        "mode": str(state.get("mode", "disabled")),
        "run_id": state.get("run_id"),
        "run_path": state.get("run_path"),
        "run_url": state.get("run_url"),
    }


def _build_run_meta(
    *,
    run_id: str,
    out_dir: Path,
    config_path: Path,
    curriculum_path: Path,
    config_hash: str,
    obs_mode: str,
    force_advance_mode: str,
    tournament_mode: bool = False,
    tournament_readiness_profile: str | None = None,
    wandb_info: dict[str, Any],
) -> dict[str, Any]:
    """Build run metadata persisted in the run output directory."""
    return {
        "schema_version": 2,
        "run_id": str(run_id),
        "created_at_unix_s": float(time.time()),
        "out_dir": str(out_dir),
        "config_path": str(config_path),
        "curriculum_path": str(curriculum_path),
        "config_hash": str(config_hash),
        "obs_mode": str(obs_mode),
        "force_advance_mode": str(force_advance_mode),
        "tournament_mode": bool(tournament_mode),
        "tournament_readiness_profile": (
            None if tournament_readiness_profile is None else str(tournament_readiness_profile)
        ),
        "wandb": dict(wandb_info),
    }


def _write_run_meta(path: Path, payload: dict[str, Any]) -> None:
    """Write run metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _init_wandb_run(
    *,
    wandb_enabled: bool,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_tags: list[str],
    wandb_run_name: str | None,
    out_dir: Path,
    run_id: str,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    """Initialize an optional W&B run and return runtime state."""
    state: dict[str, Any] = {
        "enabled": False,
        "mode": str(wandb_mode),
        "run": None,
        "run_id": None,
        "run_path": None,
        "run_url": None,
    }
    if not bool(wandb_enabled) or str(wandb_mode) == "disabled":
        return state
    if wandb is None:
        logger.warning("wandb not installed; continuing without W&B logging")
        return state

    os.environ.setdefault("WANDB_DIR", str(out_dir / "wandb"))
    run_name_local = (
        str(wandb_run_name).strip()
        if wandb_run_name is not None and str(wandb_run_name).strip()
        else f"{out_dir.name}-{run_id[:8]}"
    )
    try:
        entity_value = (
            None
            if wandb_entity is None or not str(wandb_entity).strip()
            else str(wandb_entity)
        )
        group_value = (
            None
            if wandb_group is None or not str(wandb_group).strip()
            else str(wandb_group)
        )
        wb_run = wandb.init(
            project=str(wandb_project),
            entity=entity_value,
            group=group_value,
            tags=(wandb_tags if wandb_tags else None),
            mode=str(wandb_mode),
            name=run_name_local,
            config=config_payload,
            dir=str(out_dir / "wandb"),
        )
    except Exception as exc:  # pragma: no cover - external service / auth dependent
        logger.warning("W&B initialization failed; continuing without W&B logging (%s)", exc)
        return state

    state["enabled"] = True
    state["run"] = wb_run
    state["run_id"] = getattr(wb_run, "id", None)
    state["run_path"] = _normalize_wandb_run_path(getattr(wb_run, "path", None))
    state["run_url"] = getattr(wb_run, "url", None)
    logger.info(
        "W&B enabled run_id=%s run_path=%s run_url=%s",
        state["run_id"],
        state["run_path"],
        state["run_url"],
    )
    return state


def _wandb_log(state: dict[str, Any], *, payload: dict[str, Any], step: int) -> None:
    """Best-effort W&B log helper."""
    if not bool(state.get("enabled")):
        return
    run = state.get("run")
    if run is None:
        return
    try:  # pragma: no cover - external service
        run.log(payload, step=int(step))
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("W&B log failed at step=%s (%s)", int(step), exc)


def _wandb_log_event(
    state: dict[str, Any], *, name: str, step: int, extra: dict[str, Any] | None = None
) -> None:
    """Best-effort event logger to W&B."""
    payload: dict[str, Any] = {f"event/{name}": 1}
    if extra:
        for k, v in extra.items():
            payload[f"event/{k}"] = v
    _wandb_log(state, payload=payload, step=int(step))


def _wandb_finish(state: dict[str, Any]) -> None:
    """Best-effort W&B run finalization."""
    run = state.get("run")
    if run is None:
        return
    try:  # pragma: no cover - external service
        run.finish()
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("W&B finish failed (%s)", exc)


_TRAIN_SNAPSHOT_MAP: dict[str, str] = {
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/entropy_loss": "entropy_loss",
    "train/explained_variance": "explained_variance",
    "train/loss": "loss",
    "train/policy_gradient_loss": "policy_gradient_loss",
    "train/value_loss": "value_loss",
    "train/std": "std",
    "train/n_updates": "n_updates",
    "time/fps": "fps",
}

_BLOCK_REASON_CODE: dict[str, int] = {
    "none": 0,
    "success_rate": 1,
    "min_episodes": 2,
    "stability": 3,
    "recovery": 4,
    "bossfight_tracks_not_covered": 5,
    "bossfight_min_track_fail": 6,
    "bossfight_bottomk_fail": 7,
    "bossfight_other": 8,
    "force_advance_blocked": 9,
    "stage1_zero_progress": 10,
    "stage1_consistency": 11,
    "other": 99,
}


def _block_reason_code(reason: str) -> int:
    """Encode block-reason strings as stable integers for dashboards."""
    return int(_BLOCK_REASON_CODE.get(str(reason), _BLOCK_REASON_CODE["other"]))


def _extract_train_snapshot_from_logger_values(values: dict[str, Any]) -> dict[str, float | int]:
    """Extract compact train metrics from SB3 logger values."""
    snapshot: dict[str, float | int] = {}
    for source_key, output_key in _TRAIN_SNAPSHOT_MAP.items():
        if source_key not in values:
            continue
        raw = values[source_key]
        if output_key == "n_updates":
            try:
                snapshot[output_key] = int(raw)
            except Exception:
                continue
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        snapshot[output_key] = value
    return snapshot


def _latest_train_snapshot_from_model(model: Any) -> dict[str, float | int]:  # noqa: ANN401
    """Read latest train metrics from the model logger."""
    if model is None:
        return {}
    logger_obj = getattr(model, "logger", None)
    values = getattr(logger_obj, "name_to_value", None)
    if not isinstance(values, dict):
        return {}
    return _extract_train_snapshot_from_logger_values(values)


def _build_wandb_train_payload(
    *,
    train_snapshot: dict[str, float | int],
    global_timesteps: int,
    include_rollout_metrics: bool,
    include_system_metrics: bool,
) -> dict[str, float | int]:
    """Build dense train payload for W&B using current snapshot."""
    payload: dict[str, float | int] = {"curriculum/global_timesteps": int(global_timesteps)}
    if include_rollout_metrics:
        for key in (
            "approx_kl",
            "clip_fraction",
            "entropy_loss",
            "explained_variance",
            "loss",
            "policy_gradient_loss",
            "value_loss",
            "std",
        ):
            if key in train_snapshot:
                payload[f"train/{key}"] = float(train_snapshot[key])
        if "n_updates" in train_snapshot:
            payload["train/n_updates"] = int(train_snapshot["n_updates"])
    if include_system_metrics and "fps" in train_snapshot:
        payload["runtime/fps"] = float(train_snapshot["fps"])
    return payload


def _wandb_define_metrics(
    *,
    state: dict[str, Any],
    include_rollout_metrics: bool,
    include_system_metrics: bool,
) -> None:
    """Define W&B metric schema with curriculum/global_timesteps as step metric."""
    run = state.get("run")
    if run is None:
        return
    try:  # pragma: no cover - external service
        run.define_metric("curriculum/global_timesteps")
        run.define_metric("curriculum/*", step_metric="curriculum/global_timesteps")
        run.define_metric("qualifier/*", step_metric="curriculum/global_timesteps")
        run.define_metric("bossfight/*", step_metric="curriculum/global_timesteps")
        run.define_metric("gate/*", step_metric="curriculum/global_timesteps")
        run.define_metric("event/*", step_metric="curriculum/global_timesteps")
        if include_rollout_metrics:
            run.define_metric("train/*", step_metric="curriculum/global_timesteps")
        if include_system_metrics:
            run.define_metric("runtime/*", step_metric="curriculum/global_timesteps")
    except Exception as exc:  # pragma: no cover - external service
        logger.warning("W&B metric schema setup failed (%s)", exc)


def _parse_int_list(values: Any, *, key: str) -> list[int]:  # noqa: ANN401
    if values is None:
        return []
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    if isinstance(values, str):
        cleaned = values.strip()
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    if isinstance(values, (tuple, list)):
        return [int(v) for v in values]
    raise TypeError(f"{key} must be an int, list/tuple of ints, or comma-separated string")


def _parse_float_list(values: Any, *, key: str) -> list[float]:  # noqa: ANN401
    if values is None:
        return []
    if isinstance(values, (float, int, np.floating, np.integer)):
        return [float(values)]
    if isinstance(values, str):
        cleaned = values.strip()
        if not cleaned:
            return []
        return [float(x.strip()) for x in cleaned.split(",") if x.strip()]
    if isinstance(values, (tuple, list)):
        return [float(v) for v in values]
    raise TypeError(f"{key} must be numeric, list/tuple of numerics, or comma-separated string")


def _load_qualifier_eval_profile(
    profile: str | None, *, config_dir: Path
) -> dict[str, Any] | None:
    """Load and normalize an optional qualifier-eval profile."""
    raw_profile = ("" if profile is None else str(profile)).strip()
    if raw_profile.lower() in {"", "none", "off", "false", "0"}:
        return None

    if raw_profile.lower() in {"default", "basic"}:
        return {
            "name": "default",
            "seed_offsets": [91_000, 91_097, 91_194],
            "track_indices": None,
            "dr_multipliers": [1.0, 1.1],
            "gate_scale_multipliers": [1.0],
            "tolerance_multipliers": [1.0],
            "eval_episodes": None,
            "every_evals": 1,
        }

    profile_path = Path(raw_profile)
    if not profile_path.is_absolute():
        profile_path = config_dir / raw_profile
    if not profile_path.exists():
        raise FileNotFoundError(f"qualifier_eval_profile not found: {profile_path}")

    if profile_path.suffix.lower() == ".json":
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    else:
        cfg = load_config(profile_path)
        if hasattr(cfg, "to_dict"):
            payload = cfg.to_dict()
        else:
            payload = dict(cfg)
    if "qualifier_eval" in payload and isinstance(payload["qualifier_eval"], dict):
        payload = payload["qualifier_eval"]

    perturbation_set = payload.get("perturbation_set", {})
    if not isinstance(perturbation_set, dict):
        perturbation_set = {}

    seed_offsets = _parse_int_list(payload.get("seed_offsets"), key="seed_offsets")
    if not seed_offsets:
        seed_offsets = [91_000, 91_097, 91_194]

    track_indices_raw = payload.get("track_indices")
    if track_indices_raw is None:
        track_indices = None
    else:
        track_indices = sorted(set(_parse_int_list(track_indices_raw, key="track_indices")))

    dr_multipliers = _parse_float_list(
        payload.get("dr_multipliers", perturbation_set.get("dr_multipliers")),
        key="dr_multipliers",
    )
    if not dr_multipliers:
        dr_multipliers = [1.0]

    gate_scale_multipliers = _parse_float_list(
        payload.get("gate_scale_multipliers", perturbation_set.get("gate_scale_multipliers")),
        key="gate_scale_multipliers",
    )
    if not gate_scale_multipliers:
        gate_scale_multipliers = [1.0]

    tolerance_multipliers = _parse_float_list(
        payload.get("tolerance_multipliers", perturbation_set.get("tolerance_multipliers")),
        key="tolerance_multipliers",
    )
    if not tolerance_multipliers:
        tolerance_multipliers = [1.0]

    eval_episodes_raw = payload.get("eval_episodes")
    eval_episodes = None if eval_episodes_raw is None else max(1, int(eval_episodes_raw))
    every_evals = max(1, int(payload.get("every_evals", 1)))
    name = str(payload.get("name", profile_path.stem)).strip() or profile_path.stem

    return {
        "name": name,
        "seed_offsets": [int(v) for v in seed_offsets],
        "track_indices": track_indices,
        "dr_multipliers": [float(v) for v in dr_multipliers],
        "gate_scale_multipliers": [float(v) for v in gate_scale_multipliers],
        "tolerance_multipliers": [float(v) for v in tolerance_multipliers],
        "eval_episodes": eval_episodes,
        "every_evals": every_evals,
    }


def _coverage_payload(covered: int, total: int) -> dict[str, float | int]:
    total_safe = max(int(total), 0)
    covered_safe = max(int(covered), 0)
    ratio = float(covered_safe) / float(total_safe) if total_safe > 0 else 0.0
    return {"covered": covered_safe, "total": total_safe, "ratio": ratio}


def _build_qualifier_eval_metrics(
    *,
    rows: list[tuple[int, int, EvalSummary]],
    tracks_total: int,
    seeds_total: int,
    profile_name: str,
) -> dict[str, Any]:
    """Build qualifier-oriented KPI aggregates from eval summary rows."""
    if not rows:
        return {
            "profile": profile_name,
            "course_completion_rate": 0.0,
            "lap_time_s_p50": None,
            "lap_time_s_p90": None,
            "robustness_std": 0.0,
            "track_coverage": _coverage_payload(covered=0, total=tracks_total),
            "seed_coverage": _coverage_payload(covered=0, total=seeds_total),
        }

    total_eps = max(1, int(sum(int(s.n_episodes) for _, _, s in rows)))
    completion = float(
        sum(float(s.success_rate) * int(s.n_episodes) for _, _, s in rows) / float(total_eps)
    )
    success_rates = [float(s.success_rate) for _, _, s in rows]
    robustness_std = float(np.std(success_rates)) if success_rates else 0.0

    lap_samples = [
        float(s.lap_time_s_median) for _, _, s in rows if s.lap_time_s_median is not None
    ]
    lap_p50 = float(np.percentile(lap_samples, 50)) if lap_samples else None
    lap_p90 = float(np.percentile(lap_samples, 90)) if lap_samples else None

    track_covered = len({int(track_idx) for track_idx, _, _ in rows})
    seed_covered = len({int(seed_offset) for _, seed_offset, _ in rows})

    return {
        "profile": profile_name,
        "course_completion_rate": completion,
        "lap_time_s_p50": lap_p50,
        "lap_time_s_p90": lap_p90,
        "robustness_std": robustness_std,
        "track_coverage": _coverage_payload(covered=track_covered, total=tracks_total),
        "seed_coverage": _coverage_payload(covered=seed_covered, total=seeds_total),
    }


def _apply_stage1_transition_guards(
    *,
    decision: dict[str, Any],
    summary: EvalSummary,
    stage_idx: int,
    stage_success_threshold: float,
    stage_elapsed_timesteps: int,
    stage1_nonzero_progress_seen: bool,
    stage1_success_streak: int,
    stage1_nonzero_progress_budget: int,
    stage1_required_streak: int,
) -> tuple[dict[str, Any], bool, int]:
    """Apply stage-bridge guardrails for stage1 -> stage2 promotion."""
    if int(stage_idx) != 1:
        return decision, bool(stage1_nonzero_progress_seen), int(stage1_success_streak)

    out = dict(decision)
    sr = float(summary.success_rate)

    nonzero_seen = bool(stage1_nonzero_progress_seen or sr > 0.0)
    if sr >= float(stage_success_threshold):
        success_streak = int(stage1_success_streak) + 1
    else:
        success_streak = 0

    out["stage1_nonzero_progress_seen"] = bool(nonzero_seen)
    out["stage1_success_streak"] = int(success_streak)
    out["stage1_required_streak"] = int(stage1_required_streak)
    out["stage1_nonzero_progress_budget"] = int(stage1_nonzero_progress_budget)
    out["stage1_elapsed_timesteps"] = int(stage_elapsed_timesteps)

    if int(stage_elapsed_timesteps) >= int(stage1_nonzero_progress_budget) and not nonzero_seen:
        out["advance"] = False
        out["rollback"] = True
        out["block_reason"] = "stage1_zero_progress"
        out["stage1_progress_deadline_reached"] = True
        return out, nonzero_seen, success_streak

    if bool(out.get("advance")) and int(success_streak) < int(stage1_required_streak):
        out["advance"] = False
        out["block_reason"] = "stage1_consistency"

    return out, nonzero_seen, success_streak


def _write_submission_inference_wrapper(path: Path) -> None:
    """Write a small standalone inference entrypoint for exported bundles."""
    text = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _zeros_like_space(space: Any) -> Any:  # noqa: ANN401
    from gymnasium import spaces

    if isinstance(space, spaces.Dict):
        return {k: _zeros_like_space(v) for k, v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return np.zeros((1, *space.shape), dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return np.zeros((1,), dtype=np.int32)
    raise TypeError(f"Unsupported observation space for dry-run: {space!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model_path = args.bundle_dir / "model_latest.zip"
    vecnorm_path = args.bundle_dir / "vecnormalize.pkl"
    model = PPO.load(str(model_path), device=args.device)

    payload = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path) if vecnorm_path.exists() else None,
    }
    if args.dry_run:
        obs = _zeros_like_space(model.observation_space)
        action, _ = model.predict(obs, deterministic=True)
        payload["action_shape"] = list(np.asarray(action).shape)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _export_submission_bundle(
    *,
    out_dir: Path,
    checkpoint_path: Path,
    vecnormalize_path: Path | None,
    run_id: str,
    eval_id: int,
    global_timesteps: int,
    obs_mode: str,
    config_name: str,
    curriculum_name: str,
    config_hash: str,
) -> Path:
    """Export a local submission bundle (model + stats + metadata + inference script)."""
    bundle_dir = out_dir / "submission_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_dst = bundle_dir / "model_latest.zip"
    shutil.copy2(checkpoint_path, model_dst)

    vecnorm_dst = None
    if vecnormalize_path is not None and vecnormalize_path.exists():
        vecnorm_dst = bundle_dir / "vecnormalize.pkl"
        shutil.copy2(vecnormalize_path, vecnorm_dst)

    wrapper_path = bundle_dir / "inference.py"
    _write_submission_inference_wrapper(wrapper_path)

    metadata = {
        "schema_version": 2,
        "bundle_version": 1,
        "run_id": str(run_id),
        "eval_id": int(eval_id),
        "global_timesteps": int(global_timesteps),
        "obs_mode": str(obs_mode),
        "config": str(config_name),
        "curriculum": str(curriculum_name),
        "config_hash": str(config_hash),
        "created_at_unix_s": float(time.time()),
        "artifacts": {
            "model": str(model_dst.name),
            "vecnormalize": None if vecnorm_dst is None else str(vecnorm_dst.name),
            "inference_entrypoint": str(wrapper_path.name),
        },
    }
    (bundle_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle_dir


def _hash_training_inputs(config_path: Path, curriculum_path: Path) -> str:
    """Hash config/curriculum inputs for export metadata traceability."""
    h = hashlib.sha256()
    h.update(config_path.read_bytes())
    h.update(b"\n")
    h.update(curriculum_path.read_bytes())
    return h.hexdigest()


def _capture_vecnormalize_state(vec_env: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Capture VecNormalize running stats from an env-like object."""
    if vec_env is None or not hasattr(vec_env, "obs_rms"):
        return None
    state: dict[str, Any] = {"obs_rms": copy.deepcopy(getattr(vec_env, "obs_rms"))}
    if hasattr(vec_env, "ret_rms"):
        state["ret_rms"] = copy.deepcopy(getattr(vec_env, "ret_rms"))
    return state


def _restore_vecnormalize_state(vec_env: Any, state: dict[str, Any] | None) -> None:  # noqa: ANN401
    """Restore VecNormalize running stats onto an env-like object."""
    if vec_env is None or not state:
        return
    if "obs_rms" in state and hasattr(vec_env, "obs_rms"):
        setattr(vec_env, "obs_rms", copy.deepcopy(state["obs_rms"]))
    if "ret_rms" in state and hasattr(vec_env, "ret_rms"):
        setattr(vec_env, "ret_rms", copy.deepcopy(state["ret_rms"]))


def _resolve_force_advance(
    *,
    stage_budget_reached: bool,
    stage_idx: int,
    final_stage_idx: int,
    decision_advance: bool,
    force_advance_mode: str,
) -> tuple[bool, bool]:
    """Resolve whether to force-advance and/or flag a force-advance block."""
    if not stage_budget_reached or stage_idx >= final_stage_idx or decision_advance:
        return False, False
    if force_advance_mode == "always":
        return True, False
    return False, True


def _vecnormalize_obs_keys(observation_space: Any) -> list[str] | None:  # noqa: ANN401
    """Return float Box dict-keys suitable for VecNormalize, if any."""
    if not isinstance(observation_space, gym.spaces.Dict):
        return None
    keys: list[str] = []
    for key, subspace in observation_space.spaces.items():
        if isinstance(subspace, gym.spaces.Box) and np.issubdtype(subspace.dtype, np.floating):
            keys.append(str(key))
    return keys if keys else None


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


def _aggregate_eval_summaries(summaries: list[EvalSummary]) -> EvalSummary:
    """Aggregate multiple evaluation summaries into one weighted summary."""
    if not summaries:
        raise ValueError("summaries must not be empty")
    if len(summaries) == 1:
        return summaries[0]

    n_total = int(sum(int(s.n_episodes) for s in summaries))
    if n_total <= 0:
        raise ValueError("total n_episodes must be > 0")

    success_rate = float(
        sum(float(s.success_rate) * int(s.n_episodes) for s in summaries) / float(n_total)
    )
    completion_mean = float(
        sum(float(s.completion_mean) * int(s.n_episodes) for s in summaries) / float(n_total)
    )
    completion_second_moment = float(
        sum(
            (float(s.completion_std) ** 2 + float(s.completion_mean) ** 2) * int(s.n_episodes)
            for s in summaries
        )
        / float(n_total)
    )
    completion_var = max(0.0, completion_second_moment - completion_mean**2)
    completion_std = float(np.sqrt(completion_var))

    lap_medians = [float(s.lap_time_s_median) for s in summaries if s.lap_time_s_median is not None]
    lap_time_s_median = float(np.median(lap_medians)) if lap_medians else None

    return EvalSummary(
        n_episodes=n_total,
        success_rate=success_rate,
        completion_mean=completion_mean,
        completion_std=completion_std,
        lap_time_s_median=lap_time_s_median,
    )


def _build_eval_scorecard(
    *, summary: EvalSummary, decision: dict[str, Any]
) -> dict[str, dict[str, Any] | bool]:
    """Build a compact stage-gating scorecard for logs and JSONL."""
    success_rate_value = float(summary.success_rate)
    success_rate_threshold = float(decision.get("gate_success_threshold", 0.0))
    min_episodes_value = int(decision.get("gate_stage_episodes", 0))
    min_episodes_required = int(decision.get("gate_min_episodes_required", 0))

    success_rate_ok = bool(decision.get("gate_success_ok", False))
    min_episodes_ok = bool(decision.get("gate_min_episodes_ok", False))
    stability_ok = bool(decision.get("gate_stability_ok", False))
    recovery_clear_ok = bool(decision.get("gate_recovery_clear", False))
    bossfight_ok = bool(decision.get("bossfight_ok", True))
    scorecard_pass = bool(
        success_rate_ok and min_episodes_ok and stability_ok and recovery_clear_ok and bossfight_ok
    )

    return {
        "pass": scorecard_pass,
        "checks": {
            "success_rate": {
                "ok": success_rate_ok,
                "value": success_rate_value,
                "threshold": success_rate_threshold,
            },
            "min_episodes": {
                "ok": min_episodes_ok,
                "value": min_episodes_value,
                "required": min_episodes_required,
            },
            "stability": {"ok": stability_ok},
            "recovery_clear": {"ok": recovery_clear_ok},
            "bossfight": {"ok": bossfight_ok},
        },
    }


def train(
    *,
    config: str = "aigp_stage0_single_gate.toml",
    curriculum: str = "aigp_curriculum_10stage.toml",
    out: str = "runs/aigp",
    num_envs: int = 64,
    eval_envs: int = 32,
    eval_every_timesteps: int = 50_000,
    timesteps_per_stage: int = 300_000,
    max_walltime_s: int | None = None,
    seed: int = 0,
    device: str = "auto",
    net_arch: str | tuple[int, ...] | list[int] = "256,128",
    ppo_n_steps: int = 2048,
    ppo_batch_size: int = 64,
    eval_repeats: int = 1,
    eval_seed_stride: int = 97,
    allow_append: bool = False,
    force_advance_mode: str = "if_passing",
    vecnorm_obs: bool = True,
    vecnorm_reward: bool = False,
    vecnorm_clip_obs: float = 10.0,
    policy_arch_pi: str | tuple[int, ...] | list[int] | None = None,
    policy_arch_vf: str | tuple[int, ...] | list[int] | None = None,
    obs_mode: str = "privileged",
    qualifier_eval_profile: str | None = None,
    tournament_mode: bool = False,
    tournament_readiness_profile: str | None = None,
    export_submission_bundle: bool = False,
    stage1_nonzero_progress_budget: int = 2_000_000,
    stage1_required_streak: int = 3,
    wandb_enabled: bool = False,
    wandb_project: str = "drone-racing",
    wandb_entity: str = "classimo",
    wandb_group: str | None = None,
    wandb_tags: str | tuple[str, ...] | list[str] | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    wandb_log_artifacts: bool = False,
    wandb_train_log_every_updates: int = 10,
    wandb_log_rollout_metrics: bool = True,
    wandb_log_system_metrics: bool = True,
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
        max_walltime_s: Optional walltime budget in seconds (stops after the next eval+checkpoint).
        seed: Random seed.
        device: Torch device for SB3 ("cpu", "cuda", "mps", or "auto").
        net_arch: MLP architecture as comma-separated ints (e.g. "256,128").
        ppo_n_steps: PPO rollout length per env (affects update cadence and min learn chunk size).
        ppo_batch_size: PPO minibatch size.
        eval_repeats: Number of independent eval sweeps to average per eval round.
        eval_seed_stride: Per-round seed offset stride for eval env construction.
        allow_append: Allow appending to an existing `curriculum_log.jsonl`.
        force_advance_mode: Force-advance policy when stage budget is reached:
            `never|if_passing|always`.
        vecnorm_obs: Enable VecNormalize observation normalization.
        vecnorm_reward: Enable VecNormalize reward normalization.
        vecnorm_clip_obs: VecNormalize observation clipping threshold.
        policy_arch_pi: Optional actor-only architecture override.
        policy_arch_vf: Optional critic-only architecture override.
        obs_mode: Observation mode (`privileged|competition_proxy`).
        qualifier_eval_profile: Optional profile preset/path for qualifier KPI evaluation.
        tournament_mode: Tournament run preset that enforces qualifier defaults.
        tournament_readiness_profile: Optional readiness profile stored in run metadata.
        export_submission_bundle: Export/update a local submission bundle after each eval.
        stage1_nonzero_progress_budget: Timesteps allowed in stage1 before requiring nonzero SR.
        stage1_required_streak: Consecutive stage1 threshold-passing evals required to advance.
        wandb_enabled: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team.
        wandb_group: Optional W&B run group.
        wandb_tags: Optional comma-separated tags for W&B run metadata.
        wandb_run_name: Optional explicit W&B run name.
        wandb_mode: W&B mode (`online|offline|disabled`).
        wandb_log_artifacts: Upload checkpoints as W&B artifacts on each eval checkpoint.
        wandb_train_log_every_updates: Dense W&B logging cadence in PPO update units.
        wandb_log_rollout_metrics: Enable dense `train/*` PPO metrics in W&B.
        wandb_log_system_metrics: Enable dense `runtime/*` metrics in W&B.
    """
    sb3 = _require_sb3()
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-not-found]
    from stable_baselines3.common.vec_env import VecNormalize  # type: ignore[import-not-found]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "curriculum_log.jsonl"
    if log_path.exists() and not bool(allow_append):
        raise FileExistsError(
            f"{log_path} already exists. Use --allow_append true to append explicitly."
        )
    vecnormalize_path = out_dir / "vecnormalize.pkl"
    tb_log = str(out_dir / "tb") if find_spec("tensorboard") is not None else None

    walltime_start = time.monotonic()
    walltime_deadline = (
        (walltime_start + float(max_walltime_s)) if max_walltime_s is not None else None
    )

    config_dir = Path(__file__).parents[1] / "config"
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = config_dir / config_path
    curriculum_path = Path(curriculum)
    if not curriculum_path.is_absolute():
        curriculum_path = config_dir / curriculum_path

    base_cfg = load_config(config_path)
    cur_cfg = CurriculumConfig.load(curriculum_path)
    mgr = CurriculumManager(cur_cfg)
    final_stage_idx = len(cur_cfg.stages) - 1

    obs_mode_local, qualifier_eval_profile, tournament_readiness_profile_local = (
        _resolve_tournament_train_settings(
            tournament_mode=bool(tournament_mode),
            obs_mode=str(obs_mode),
            qualifier_eval_profile=qualifier_eval_profile,
            tournament_readiness_profile=tournament_readiness_profile,
        )
    )
    force_advance_mode_local = None
    if bool(tournament_mode):
        force_advance_mode_local = "if_passing"
        logger.info("TOURNAMENT_PRESET active")
    wandb_mode_local = _parse_wandb_mode(wandb_mode)
    wandb_tags_local = _parse_wandb_tags(wandb_tags)
    wandb_enabled_local = bool(wandb_enabled) and (wandb_mode_local != "disabled")
    qualifier_profile_cfg = _load_qualifier_eval_profile(
        qualifier_eval_profile, config_dir=config_dir
    )
    qualifier_profile_name = (
        "inline_eval" if qualifier_profile_cfg is None else str(qualifier_profile_cfg["name"])
    )
    config_hash = _hash_training_inputs(config_path, curriculum_path)

    if force_advance_mode_local is None:
        force_advance_mode_local = _parse_force_advance_mode(force_advance_mode)
    if float(vecnorm_clip_obs) <= 0.0:
        raise ValueError("vecnorm_clip_obs must be > 0")
    if int(stage1_nonzero_progress_budget) <= 0:
        raise ValueError("stage1_nonzero_progress_budget must be > 0")
    if int(stage1_required_streak) <= 0:
        raise ValueError("stage1_required_streak must be > 0")
    if int(wandb_train_log_every_updates) <= 0:
        raise ValueError("wandb_train_log_every_updates must be > 0")
    use_vecnormalize = bool(vecnorm_obs or vecnorm_reward)
    vecnormalize_resume_pending = bool(
        use_vecnormalize and allow_append and vecnormalize_path.exists()
    )
    if vecnormalize_resume_pending:
        logger.info("VecNormalize resume stats found at %s", vecnormalize_path)

    arch = _parse_net_arch(net_arch)
    pi_arch = _parse_net_arch(policy_arch_pi) if policy_arch_pi is not None else arch
    vf_arch = _parse_net_arch(policy_arch_vf) if policy_arch_vf is not None else arch
    if policy_arch_pi is not None or policy_arch_vf is not None:
        policy_kwargs: dict[str, Any] = {"net_arch": {"pi": list(pi_arch), "vf": list(vf_arch)}}
    else:
        policy_kwargs = {"net_arch": list(arch)}

    eval_repeats_local = max(1, int(eval_repeats))
    eval_seed_stride_local = max(1, int(eval_seed_stride))
    run_id = str(uuid.uuid4())
    eval_id = 0
    last_eval_id = 0
    last_global_timesteps = -1
    logger.info(
        (
            "RUN_CONTEXT run_id=%s force_advance_mode=%s obs_mode=%s qualifier_profile=%s "
            "vecnorm_obs=%s vecnorm_reward=%s vecnorm_clip_obs=%.3f config_hash=%s "
            "wandb_enabled=%s wandb_mode=%s wandb_train_log_every_updates=%s "
            "wandb_log_rollout_metrics=%s wandb_log_system_metrics=%s"
        ),
        run_id,
        force_advance_mode_local,
        obs_mode_local,
        qualifier_profile_name,
        bool(vecnorm_obs),
        bool(vecnorm_reward),
        float(vecnorm_clip_obs),
        config_hash[:12],
        bool(wandb_enabled_local),
        wandb_mode_local,
        int(wandb_train_log_every_updates),
        int(bool(wandb_log_rollout_metrics)),
        int(bool(wandb_log_system_metrics)),
    )
    wandb_config_payload: dict[str, Any] = {
        "run_id": run_id,
        "config_path": str(config_path),
        "curriculum_path": str(curriculum_path),
        "config_hash": str(config_hash),
        "seed": int(seed),
        "num_envs": int(num_envs),
        "eval_envs": int(eval_envs),
        "eval_every_timesteps": int(eval_every_timesteps),
        "timesteps_per_stage": int(timesteps_per_stage),
        "force_advance_mode": str(force_advance_mode_local),
        "obs_mode": str(obs_mode_local),
        "qualifier_profile": str(qualifier_profile_name),
        "tournament_mode": bool(tournament_mode),
        "tournament_readiness_profile": (
            None
            if tournament_readiness_profile_local is None
            else str(tournament_readiness_profile_local)
        ),
        "vecnorm_obs": bool(vecnorm_obs),
        "vecnorm_reward": bool(vecnorm_reward),
        "vecnorm_clip_obs": float(vecnorm_clip_obs),
        "policy_arch_pi": list(pi_arch),
        "policy_arch_vf": list(vf_arch),
        "wandb_train_log_every_updates": int(wandb_train_log_every_updates),
        "wandb_log_rollout_metrics": bool(wandb_log_rollout_metrics),
        "wandb_log_system_metrics": bool(wandb_log_system_metrics),
    }
    wandb_state = _init_wandb_run(
        wandb_enabled=wandb_enabled_local,
        wandb_mode=wandb_mode_local,
        wandb_project=str(wandb_project),
        wandb_entity=(None if wandb_entity is None else str(wandb_entity)),
        wandb_group=wandb_group,
        wandb_tags=wandb_tags_local,
        wandb_run_name=wandb_run_name,
        out_dir=out_dir,
        run_id=run_id,
        config_payload=wandb_config_payload,
    )
    _wandb_define_metrics(
        state=wandb_state,
        include_rollout_metrics=bool(wandb_log_rollout_metrics),
        include_system_metrics=bool(wandb_log_system_metrics),
    )
    run_meta_path = out_dir / "run_meta.json"
    run_meta_payload = _build_run_meta(
        run_id=run_id,
        out_dir=out_dir,
        config_path=config_path,
        curriculum_path=curriculum_path,
        config_hash=config_hash,
        obs_mode=obs_mode_local,
        force_advance_mode=force_advance_mode_local,
        tournament_mode=bool(tournament_mode),
        tournament_readiness_profile=tournament_readiness_profile_local,
        wandb_info=_wandb_info_payload(wandb_state),
    )
    _write_run_meta(run_meta_path, run_meta_payload)

    class _EpisodeCounter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = 0

        def _on_step(self) -> bool:  # noqa: ANN101
            dones = self.locals.get("dones")
            if dones is not None:
                self.episodes += int(np.sum(dones))
            return True

    class _WalltimeLimit(BaseCallback):
        def __init__(self, *, deadline_s: float | None):
            super().__init__()
            self.deadline_s = deadline_s
            self.reached = False

        def _on_step(self) -> bool:  # noqa: ANN101
            if self.deadline_s is None:
                return True
            if time.monotonic() >= float(self.deadline_s):
                self.reached = True
                return False
            return True

    def _build_vec_env_for_stage(
        *,
        stage_idx: int,
        num_envs_local: int,
        seed_offset: int,
        dr_mult: float,
        gate_scale: float,
        tol_mult: float,
        track_indices: list[int] | None = None,
    ) -> Any:  # noqa: ANN401
        stage_local = mgr.stage_by_idx(stage_idx)
        stage_max_steps_local = int(
            stage_local.max_episode_steps or base_cfg.env.get("max_episode_steps", 1500)
        )

        tracks_local, weights_local = mgr.build_tracks_for_stage(
            stage_idx=stage_idx, config_dir=config_dir
        )
        if track_indices is not None:
            tracks_local = [tracks_local[int(i)] for i in track_indices]
            weights_local = None
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
        if obs_mode_local == "competition_proxy":
            env_local = CompetitionProxyObsWrapper(env_local)
        return env_local

    def _build_sb3_env_for_stage(
        *,
        stage_idx: int,
        num_envs_local: int,
        seed_offset: int,
        dr_mult: float,
        gate_scale: float,
        tol_mult: float,
        training: bool,
        vecnorm_state: dict[str, Any] | None = None,
        track_indices: list[int] | None = None,
    ) -> Any:  # noqa: ANN401
        nonlocal vecnormalize_resume_pending
        raw_env = _build_vec_env_for_stage(
            stage_idx=stage_idx,
            num_envs_local=num_envs_local,
            seed_offset=seed_offset,
            dr_mult=dr_mult,
            gate_scale=gate_scale,
            tol_mult=tol_mult,
            track_indices=track_indices,
        )
        sb3_env_local = make_sb3_vec_env(raw_env)
        if use_vecnormalize:
            if bool(training) and vecnormalize_resume_pending and vecnorm_state is None:
                sb3_env_local = VecNormalize.load(str(vecnormalize_path), sb3_env_local)
                vecnormalize_resume_pending = False
                logger.info("Loaded VecNormalize stats from %s", vecnormalize_path)
            else:
                norm_obs_keys = (
                    _vecnormalize_obs_keys(sb3_env_local.observation_space)
                    if bool(vecnorm_obs)
                    else None
                )
                sb3_env_local = VecNormalize(
                    sb3_env_local,
                    norm_obs=bool(vecnorm_obs),
                    norm_reward=bool(vecnorm_reward),
                    clip_obs=float(vecnorm_clip_obs),
                    training=bool(training),
                    norm_obs_keys=norm_obs_keys,
                )
                _restore_vecnormalize_state(sb3_env_local, vecnorm_state)
            sb3_env_local.training = bool(training)
        return sb3_env_local

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

        sb3_env = _build_sb3_env_for_stage(
            stage_idx=stage_idx,
            num_envs_local=int(num_envs),
            seed_offset=0,
            dr_mult=eff_dr_mult,
            gate_scale=eff_gate_scale,
            tol_mult=eff_tol_mult,
            training=True,
        )

        if model is None:
            model = sb3.PPO(
                policy="MultiInputPolicy",
                env=sb3_env,
                verbose=1,
                tensorboard_log=tb_log,
                device=device,
                policy_kwargs=policy_kwargs,
                n_steps=int(ppo_n_steps),
                batch_size=int(ppo_batch_size),
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
        chunk_round = 0
        stop_after_checkpoint = False
        stage1_nonzero_progress_seen = False
        stage1_success_streak = 0
        train_snapshot_latest: dict[str, float | int] = {}
        last_logged_n_updates: int | None = None
        last_eval_walltime_s = time.monotonic()

        while True:
            # Train a chunk.
            walltime_cb = _WalltimeLimit(deadline_s=walltime_deadline)
            model.learn(
                total_timesteps=int(eval_every_timesteps),
                reset_num_timesteps=False,
                callback=[counter, walltime_cb],
            )
            eval_round += 1
            chunk_round += 1
            if walltime_cb.reached:
                stop_after_checkpoint = True

            train_snapshot_latest = _latest_train_snapshot_from_model(model)
            global_timesteps = int(model.num_timesteps)
            should_log_dense_train = False
            n_updates_current = train_snapshot_latest.get("n_updates")
            if isinstance(n_updates_current, int):
                if (
                    last_logged_n_updates is None
                    or int(n_updates_current) - int(last_logged_n_updates)
                    >= int(wandb_train_log_every_updates)
                ):
                    should_log_dense_train = True
                    last_logged_n_updates = int(n_updates_current)
            elif (chunk_round % int(wandb_train_log_every_updates)) == 0:
                should_log_dense_train = True
            if should_log_dense_train:
                dense_payload = _build_wandb_train_payload(
                    train_snapshot=train_snapshot_latest,
                    global_timesteps=global_timesteps,
                    include_rollout_metrics=bool(wandb_log_rollout_metrics),
                    include_system_metrics=bool(wandb_log_system_metrics),
                )
                if len(dense_payload) > 1:
                    _wandb_log(wandb_state, payload=dense_payload, step=global_timesteps)

            # Evaluate with fresh envs and deterministic seeds.
            eval_seed_offsets: list[int] = []
            eval_tracks: dict[str, Any] | None = None
            train_vecnorm_state = _capture_vecnormalize_state(sb3_env)
            track_count = len(stage.tracks)
            eval_summaries: list[EvalSummary] = []
            qualifier_rows_inline: list[tuple[int, int, EvalSummary]] = []

            if track_count > 1:
                per_track: dict[int, list[EvalSummary]] = {}
                for eval_idx in range(eval_repeats_local):
                    eval_seed_base = (
                        10_000
                        + eval_round * eval_seed_stride_local
                        + eval_idx * 1_000
                    )
                    for track_idx in range(track_count):
                        eval_seed_offset = int(eval_seed_base + track_idx * 100_000)
                        eval_seed_offsets.append(eval_seed_offset)
                        eval_env = _build_sb3_env_for_stage(
                            stage_idx=stage_idx,
                            num_envs_local=int(eval_envs),
                            seed_offset=eval_seed_offset,
                            dr_mult=eff_dr_mult,
                            gate_scale=eff_gate_scale,
                            tol_mult=eff_tol_mult,
                            training=False,
                            vecnorm_state=train_vecnorm_state,
                            track_indices=[track_idx],
                        )
                        eval_summary = evaluate_sb3_vec_env(
                            eval_env,
                            make_predict_policy(model, deterministic=True),
                            n_episodes=stage_eval_eps,
                            max_episode_steps=stage_max_steps,
                        )
                        eval_env.close()
                        per_track.setdefault(track_idx, []).append(eval_summary)
                        qualifier_rows_inline.append(
                            (int(track_idx), int(eval_seed_offset), eval_summary)
                        )

                track_summary_rows: list[tuple[int, str, EvalSummary]] = []
                for track_idx, track_name in enumerate(stage.tracks):
                    summaries_for_track = per_track.get(track_idx, [])
                    if summaries_for_track:
                        track_summary = _aggregate_eval_summaries(summaries_for_track)
                    else:
                        track_summary = EvalSummary(
                            n_episodes=0,
                            success_rate=0.0,
                            completion_mean=0.0,
                            completion_std=0.0,
                            lap_time_s_median=None,
                        )
                    track_summary_rows.append((track_idx, track_name, track_summary))
                    if track_summary.n_episodes > 0:
                        eval_summaries.append(track_summary)

                if not eval_summaries:
                    raise RuntimeError("stratified evaluation produced no episodes")
                summary = _aggregate_eval_summaries(eval_summaries)
                eval_tracks = aggregate_track_eval_summaries(
                    track_summary_rows,
                    bottomk_fraction=float(stage.bossfight_bottomk_fraction),
                )
            else:
                for eval_idx in range(eval_repeats_local):
                    eval_seed_offset = (
                        10_000
                        + eval_round * eval_seed_stride_local
                        + eval_idx * 1_000
                    )
                    eval_seed_offsets.append(int(eval_seed_offset))
                    eval_env = _build_sb3_env_for_stage(
                        stage_idx=stage_idx,
                        num_envs_local=int(eval_envs),
                        seed_offset=int(eval_seed_offset),
                        dr_mult=eff_dr_mult,
                        gate_scale=eff_gate_scale,
                        tol_mult=eff_tol_mult,
                        training=False,
                        vecnorm_state=train_vecnorm_state,
                    )
                    eval_summary = evaluate_sb3_vec_env(
                        eval_env,
                        make_predict_policy(model, deterministic=True),
                        n_episodes=stage_eval_eps,
                        max_episode_steps=stage_max_steps,
                    )
                    eval_env.close()
                    eval_summaries.append(eval_summary)
                    qualifier_rows_inline.append((0, int(eval_seed_offset), eval_summary))
                summary = _aggregate_eval_summaries(eval_summaries)

            qualifier_eval = _build_qualifier_eval_metrics(
                rows=qualifier_rows_inline,
                tracks_total=max(1, int(track_count)),
                seeds_total=max(1, len(set(eval_seed_offsets))),
                profile_name="inline_eval",
            )
            if (
                qualifier_profile_cfg is not None
                and (eval_round % int(qualifier_profile_cfg["every_evals"])) == 0
            ):
                profile_seed_offsets = [int(v) for v in qualifier_profile_cfg["seed_offsets"]]
                profile_track_indices_raw = qualifier_profile_cfg.get("track_indices")
                if profile_track_indices_raw is None:
                    profile_track_indices = list(range(track_count))
                else:
                    profile_track_indices = [
                        int(i) for i in profile_track_indices_raw if 0 <= int(i) < track_count
                    ]
                if not profile_track_indices:
                    raise ValueError("qualifier_eval_profile resolved to an empty track set")

                profile_eval_eps = int(qualifier_profile_cfg.get("eval_episodes") or stage_eval_eps)
                profile_rows: list[tuple[int, int, EvalSummary]] = []
                for dr_idx, dr_scale in enumerate(qualifier_profile_cfg["dr_multipliers"]):
                    for gs_idx, gate_scale_mul in enumerate(
                        qualifier_profile_cfg["gate_scale_multipliers"]
                    ):
                        for tol_idx, tol_mul in enumerate(
                            qualifier_profile_cfg["tolerance_multipliers"]
                        ):
                            perturb_idx = (
                                dr_idx * len(qualifier_profile_cfg["gate_scale_multipliers"])
                                * len(qualifier_profile_cfg["tolerance_multipliers"])
                                + gs_idx * len(qualifier_profile_cfg["tolerance_multipliers"])
                                + tol_idx
                            )
                            for seed_base in profile_seed_offsets:
                                for track_idx in profile_track_indices:
                                    eval_seed_offset = (
                                        int(seed_base)
                                        + int(track_idx) * 100_000
                                        + int(perturb_idx) * 1_000_000
                                    )
                                    eval_env = _build_sb3_env_for_stage(
                                        stage_idx=stage_idx,
                                        num_envs_local=int(eval_envs),
                                        seed_offset=eval_seed_offset,
                                        dr_mult=float(eff_dr_mult) * float(dr_scale),
                                        gate_scale=float(eff_gate_scale) * float(gate_scale_mul),
                                        tol_mult=float(eff_tol_mult) * float(tol_mul),
                                        training=False,
                                        vecnorm_state=train_vecnorm_state,
                                        track_indices=[int(track_idx)],
                                    )
                                    eval_summary = evaluate_sb3_vec_env(
                                        eval_env,
                                        make_predict_policy(model, deterministic=True),
                                        n_episodes=profile_eval_eps,
                                        max_episode_steps=stage_max_steps,
                                    )
                                    eval_env.close()
                                    profile_rows.append(
                                        (int(track_idx), int(seed_base), eval_summary)
                                    )

                qualifier_eval = _build_qualifier_eval_metrics(
                    rows=profile_rows,
                    tracks_total=int(len(profile_track_indices)),
                    seeds_total=int(len(profile_seed_offsets)),
                    profile_name=str(qualifier_profile_cfg["name"]),
                )
                qualifier_eval["eval_episodes"] = int(profile_eval_eps)

            logger.info(
                (
                    "EVAL_QUALIFIER profile=%s ccr=%.3f lap_p50=%s lap_p90=%s robust_std=%.3f "
                    "track_cov=%s/%s seed_cov=%s/%s"
                ),
                str(qualifier_eval["profile"]),
                float(qualifier_eval["course_completion_rate"]),
                (
                    "None"
                    if qualifier_eval["lap_time_s_p50"] is None
                    else f"{float(qualifier_eval['lap_time_s_p50']):.3f}"
                ),
                (
                    "None"
                    if qualifier_eval["lap_time_s_p90"] is None
                    else f"{float(qualifier_eval['lap_time_s_p90']):.3f}"
                ),
                float(qualifier_eval["robustness_std"]),
                int(qualifier_eval["track_coverage"]["covered"]),
                int(qualifier_eval["track_coverage"]["total"]),
                int(qualifier_eval["seed_coverage"]["covered"]),
                int(qualifier_eval["seed_coverage"]["total"]),
            )

            decision = mgr.update_after_eval(
                summary=summary,
                stage_episodes=int(counter.episodes),
                eval_tracks=eval_tracks,
            )
            stage_elapsed_timesteps = int(model.num_timesteps) - stage_start_timesteps
            decision, stage1_nonzero_progress_seen, stage1_success_streak = (
                _apply_stage1_transition_guards(
                    decision=decision,
                    summary=summary,
                    stage_idx=stage_idx,
                    stage_success_threshold=float(stage.success_rate_threshold),
                    stage_elapsed_timesteps=stage_elapsed_timesteps,
                    stage1_nonzero_progress_seen=stage1_nonzero_progress_seen,
                    stage1_success_streak=stage1_success_streak,
                    stage1_nonzero_progress_budget=int(stage1_nonzero_progress_budget),
                    stage1_required_streak=int(stage1_required_streak),
                )
            )
            scorecard = _build_eval_scorecard(summary=summary, decision=decision)
            logger.info(
                (
                    "EVAL_SCORECARD stage_idx=%s stage=%s pass=%d sr=%.3f/%.3f "
                    "min_eps=%d/%d stable=%d recovery_clear=%d bossfight=%d block=%s "
                    "dr_mult=%.3f assist=%.3f"
                ),
                int(decision.get("stage_idx", mgr.stage_idx)),
                str(decision.get("stage_name", stage.name)),
                int(bool(scorecard["pass"])),
                float(scorecard["checks"]["success_rate"]["value"]),
                float(scorecard["checks"]["success_rate"]["threshold"]),
                int(scorecard["checks"]["min_episodes"]["value"]),
                int(scorecard["checks"]["min_episodes"]["required"]),
                int(bool(scorecard["checks"]["stability"]["ok"])),
                int(bool(scorecard["checks"]["recovery_clear"]["ok"])),
                int(bool(scorecard["checks"]["bossfight"]["ok"])),
                str(decision.get("block_reason", "none")),
                float(eff_dr_mult),
                float(decision.get("panic_assist_mult", 1.0)),
            )
            if stage_idx == 1:
                logger.info(
                    (
                        "EVAL_STAGE1_GUARD nonzero=%d streak=%d/%d elapsed=%d budget=%d block=%s"
                    ),
                    int(bool(decision.get("stage1_nonzero_progress_seen", False))),
                    int(decision.get("stage1_success_streak", 0)),
                    int(decision.get("stage1_required_streak", stage1_required_streak)),
                    int(decision.get("stage1_elapsed_timesteps", stage_elapsed_timesteps)),
                    int(
                        decision.get(
                            "stage1_nonzero_progress_budget", stage1_nonzero_progress_budget
                        )
                    ),
                    str(decision.get("block_reason", "none")),
                )
            if eval_tracks is not None:
                aggregate = eval_tracks.get("_aggregate", {})
                logger.info(
                    "EVAL_BOSSFIGHT stage_idx=%s covered=%s/%s sr_min=%s sr_b20=%s pass=%d",
                    int(decision.get("stage_idx", mgr.stage_idx)),
                    int(aggregate.get("tracks_covered", 0)),
                    int(aggregate.get("tracks_total", 0)),
                    (
                        "None"
                        if aggregate.get("success_rate_min") is None
                        else f"{float(aggregate.get('success_rate_min')):.3f}"
                    ),
                    (
                        "None"
                        if aggregate.get("success_rate_bottom20_mean") is None
                        else f"{float(aggregate.get('success_rate_bottom20_mean')):.3f}"
                    ),
                    int(bool(decision.get("bossfight_ok", True))),
                )

            stage_budget_reached = (
                int(model.num_timesteps) - stage_start_timesteps >= int(timesteps_per_stage)
            )
            forced_advance, force_advance_blocked = _resolve_force_advance(
                stage_budget_reached=stage_budget_reached,
                stage_idx=int(mgr.stage_idx),
                final_stage_idx=final_stage_idx,
                decision_advance=bool(decision.get("advance")),
                force_advance_mode=force_advance_mode_local,
            )

            row_block_reason = str(decision.get("block_reason", "none"))
            if force_advance_blocked:
                row_block_reason = "force_advance_blocked"
            row_block_reason_code = _block_reason_code(row_block_reason)
            eval_walltime_now = time.monotonic()
            eval_interval_s = float(max(0.0, eval_walltime_now - last_eval_walltime_s))
            last_eval_walltime_s = eval_walltime_now

            eval_id += 1
            global_timesteps = int(model.num_timesteps)
            if eval_id <= last_eval_id:
                raise RuntimeError(f"eval_id not strictly monotonic: {eval_id} <= {last_eval_id}")
            last_eval_id = eval_id
            if global_timesteps <= last_global_timesteps:
                raise RuntimeError(
                    "global_timesteps not strictly monotonic: "
                    f"{global_timesteps} <= {last_global_timesteps}"
                )
            last_global_timesteps = global_timesteps

            row = {
                "schema_version": 2,
                "run_id": run_id,
                "eval_id": int(eval_id),
                "global_timesteps": global_timesteps,
                "stage": {"idx": mgr.stage_idx, "name": stage.name, "dr_tier": stage.dr_tier},
                "timesteps": int(model.num_timesteps),
                "stage_episodes": int(counter.episodes),
                "effective": {
                    "dr_mult": float(eff_dr_mult),
                    "gate_scale": float(eff_gate_scale),
                    "tol_mult": float(eff_tol_mult),
                },
                "eval_repeats": int(eval_repeats_local),
                "eval_seed_offsets": eval_seed_offsets,
                "eval": asdict(summary),
                "obs_mode": obs_mode_local,
                "qualifier_eval": qualifier_eval,
                "wandb": _wandb_info_payload(wandb_state),
                "train_snapshot": train_snapshot_latest,
                "scorecard": scorecard,
                "block_reason": row_block_reason,
                "forced_advance": bool(forced_advance),
                "runtime": {
                    "eval_interval_s": eval_interval_s,
                    "walltime_s": float(eval_walltime_now - walltime_start),
                },
                "decision": {k: v for k, v in decision.items() if k != "adaptive_smoothed"},
            }
            if eval_tracks is not None:
                row["eval_tracks"] = eval_tracks
            _save_jsonl(log_path, row)
            wandb_eval_payload: dict[str, Any] = {
                "curriculum/stage_idx": int(mgr.stage_idx),
                "curriculum/stage_name": str(stage.name),
                "curriculum/dr_tier": str(stage.dr_tier),
                "curriculum/global_timesteps": int(global_timesteps),
                "curriculum/stage_episodes": int(counter.episodes),
                "curriculum/eval_id": int(eval_id),
                "curriculum/pass": int(bool(scorecard["pass"])),
                "curriculum/block_reason": str(row_block_reason),
                "curriculum/forced_advance": int(bool(forced_advance)),
                "curriculum/success_rate": float(summary.success_rate),
                "curriculum/completion_mean": float(summary.completion_mean),
                "curriculum/completion_std": float(summary.completion_std),
                "curriculum/gate_success_threshold": float(
                    scorecard["checks"]["success_rate"]["threshold"]
                ),
                "curriculum/gate_success_ok": int(
                    bool(scorecard["checks"]["success_rate"]["ok"])
                ),
                "curriculum/gate_min_episodes_ok": int(
                    bool(scorecard["checks"]["min_episodes"]["ok"])
                ),
                "curriculum/gate_stability_ok": int(bool(scorecard["checks"]["stability"]["ok"])),
                "curriculum/gate_recovery_ok": int(
                    bool(scorecard["checks"]["recovery_clear"]["ok"])
                ),
                "curriculum/bossfight_ok": int(bool(scorecard["checks"]["bossfight"]["ok"])),
                "curriculum/effective_dr_mult": float(eff_dr_mult),
                "curriculum/effective_gate_scale": float(eff_gate_scale),
                "curriculum/effective_tol_mult": float(eff_tol_mult),
                "gate/scorecard_pass": int(bool(scorecard["pass"])),
                "gate/stability_ok": int(bool(scorecard["checks"]["stability"]["ok"])),
                "gate/block_reason_code": int(row_block_reason_code),
                "qualifier/course_completion_rate": float(
                    qualifier_eval.get("course_completion_rate", 0.0)
                ),
                "qualifier/robustness_std": float(qualifier_eval.get("robustness_std", 0.0)),
                "qualifier/track_coverage_ratio": float(
                    qualifier_eval.get("track_coverage", {}).get("ratio", 0.0)
                ),
                "qualifier/seed_coverage_ratio": float(
                    qualifier_eval.get("seed_coverage", {}).get("ratio", 0.0)
                ),
                "runtime/eval_interval_s": float(eval_interval_s),
                "runtime/walltime_s": float(eval_walltime_now - walltime_start),
            }
            eval_train_payload = _build_wandb_train_payload(
                train_snapshot=train_snapshot_latest,
                global_timesteps=global_timesteps,
                include_rollout_metrics=bool(wandb_log_rollout_metrics),
                include_system_metrics=bool(wandb_log_system_metrics),
            )
            if len(eval_train_payload) > 1:
                wandb_eval_payload.update(eval_train_payload)
            lap_median = summary.lap_time_s_median
            if lap_median is not None:
                wandb_eval_payload["curriculum/lap_time_s_median"] = float(lap_median)
            lap_p50 = qualifier_eval.get("lap_time_s_p50")
            if lap_p50 is not None:
                wandb_eval_payload["qualifier/lap_time_s_p50"] = float(lap_p50)
            lap_p90 = qualifier_eval.get("lap_time_s_p90")
            if lap_p90 is not None:
                wandb_eval_payload["qualifier/lap_time_s_p90"] = float(lap_p90)
            if eval_tracks is not None:
                agg = eval_tracks.get("_aggregate", {})
                wandb_eval_payload["bossfight/tracks_total"] = int(agg.get("tracks_total", 0))
                wandb_eval_payload["bossfight/tracks_covered"] = int(
                    agg.get("tracks_covered", 0)
                )
                sr_min = agg.get("success_rate_min")
                if sr_min is not None:
                    wandb_eval_payload["bossfight/success_rate_min"] = float(sr_min)
                sr_b20 = agg.get("success_rate_bottom20_mean")
                if sr_b20 is not None:
                    wandb_eval_payload["bossfight/success_rate_bottom20_mean"] = float(sr_b20)
            _wandb_log(wandb_state, payload=wandb_eval_payload, step=int(global_timesteps))

            ckpt = (
                out_dir
                / f"stage{mgr.stage_idx:02d}_{stage.name}"
                / f"step_{model.num_timesteps}.zip"
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt))
            if use_vecnormalize and hasattr(sb3_env, "save"):
                sb3_env.save(str(vecnormalize_path))
            if bool(export_submission_bundle):
                bundle_dir = _export_submission_bundle(
                    out_dir=out_dir,
                    checkpoint_path=ckpt,
                    vecnormalize_path=(vecnormalize_path if use_vecnormalize else None),
                    run_id=run_id,
                    eval_id=int(eval_id),
                    global_timesteps=int(model.num_timesteps),
                    obs_mode=obs_mode_local,
                    config_name=str(config_path),
                    curriculum_name=str(curriculum_path),
                    config_hash=config_hash,
                )
                logger.info(
                    "SUBMISSION_BUNDLE_EXPORT bundle=%s checkpoint=%s",
                    bundle_dir,
                    ckpt,
                )
                _wandb_log_event(
                    wandb_state,
                    name="submission_bundle_export",
                    step=int(model.num_timesteps),
                    extra={"bundle_dir": str(bundle_dir), "checkpoint": str(ckpt)},
                )
                _save_jsonl(
                    log_path,
                    {
                        "schema_version": 2,
                        "run_id": run_id,
                        "event": "submission_bundle_export",
                        "wandb": _wandb_info_payload(wandb_state),
                        "global_timesteps": int(model.num_timesteps),
                        "timesteps": int(model.num_timesteps),
                        "eval_id": int(eval_id),
                        "bundle_dir": str(bundle_dir),
                        "checkpoint": str(ckpt),
                    },
                )
            if bool(wandb_log_artifacts) and bool(wandb_state.get("enabled")) and wandb is not None:
                run_obj = wandb_state.get("run")
                if run_obj is not None:
                    try:  # pragma: no cover - external service
                        artifact = wandb.Artifact(
                            name=f"{out_dir.name}-stage{int(mgr.stage_idx):02d}-eval{int(eval_id):04d}",
                            type="model",
                        )
                        artifact.add_file(str(ckpt), name=ckpt.name)
                        if use_vecnormalize and vecnormalize_path.exists():
                            artifact.add_file(str(vecnormalize_path), name=vecnormalize_path.name)
                        run_obj.log_artifact(artifact)
                    except Exception as exc:  # pragma: no cover - external service
                        logger.warning("W&B artifact upload failed (%s)", exc)

            if stop_after_checkpoint:
                elapsed_s = float(time.monotonic() - walltime_start)
                _wandb_log_event(
                    wandb_state,
                    name="walltime_exit",
                    step=int(model.num_timesteps),
                    extra={
                        "elapsed_s": elapsed_s,
                        "max_walltime_s": (
                            None if max_walltime_s is None else int(max_walltime_s)
                        ),
                    },
                )
                _save_jsonl(
                    log_path,
                    {
                        "schema_version": 2,
                        "run_id": run_id,
                        "event": "walltime_exit",
                        "wandb": _wandb_info_payload(wandb_state),
                        "global_timesteps": int(model.num_timesteps),
                        "timesteps": int(model.num_timesteps),
                        "elapsed_s": elapsed_s,
                        "max_walltime_s": (
                            int(max_walltime_s) if max_walltime_s is not None else None
                        ),
                        "checkpoint": str(ckpt),
                    },
                )
                break

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
                prev_env = _build_sb3_env_for_stage(
                    stage_idx=prev_idx,
                    num_envs_local=int(eval_envs),
                    seed_offset=20_000 + eval_round * eval_seed_stride_local,
                    dr_mult=1.0,
                    gate_scale=1.0,
                    tol_mult=1.0,
                    training=False,
                    vecnorm_state=_capture_vecnormalize_state(sb3_env),
                )
                prev_summary = evaluate_sb3_vec_env(
                    prev_env,
                    make_predict_policy(model, deterministic=True),
                    n_episodes=prev_eval_eps,
                    max_episode_steps=prev_max_steps,
                )
                prev_env.close()

                baseline = float(mgr.forgetting.baseline_success_rate)
                if prev_summary.success_rate < max(0.05, baseline * 0.5):
                    _wandb_log_event(
                        wandb_state,
                        name="forgetting_rollback",
                        step=int(model.num_timesteps),
                        extra={
                            "baseline_sr": baseline,
                            "prev_stage_idx": int(prev_idx),
                            "prev_stage_sr": float(prev_summary.success_rate),
                        },
                    )
                    _save_jsonl(
                        log_path,
                        {
                            "schema_version": 2,
                            "run_id": run_id,
                            "event": "forgetting_rollback",
                            "wandb": _wandb_info_payload(wandb_state),
                            "global_timesteps": int(model.num_timesteps),
                            "timesteps": int(model.num_timesteps),
                            "baseline_sr": baseline,
                            "prev_stage_idx": prev_idx,
                            "prev_eval": asdict(prev_summary),
                        },
                    )
                    mgr.rollback()
                    break

            if decision.get("rollback"):
                _wandb_log_event(
                    wandb_state,
                    name="rollback",
                    step=int(model.num_timesteps),
                    extra={
                        "stage_idx": int(mgr.stage_idx),
                        "block_reason": str(decision.get("block_reason", "none")),
                    },
                )
                mgr.rollback()
                break

            if decision.get("advance") and int(mgr.stage_idx) < final_stage_idx:
                _wandb_log_event(
                    wandb_state,
                    name="advance",
                    step=int(model.num_timesteps),
                    extra={
                        "stage_idx": int(mgr.stage_idx),
                        "success_rate": float(summary.success_rate),
                    },
                )
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
                vecnorm_state_for_rebuild = _capture_vecnormalize_state(sb3_env)
                sb3_env.close()
                sb3_env = _build_sb3_env_for_stage(
                    stage_idx=stage_idx,
                    num_envs_local=int(num_envs),
                    seed_offset=0,
                    dr_mult=eff_dr_mult_new,
                    gate_scale=eff_gate_scale_new,
                    tol_mult=eff_tol_mult_new,
                    training=True,
                    vecnorm_state=vecnorm_state_for_rebuild,
                )
                model.set_env(sb3_env)
                eff_dr_mult = eff_dr_mult_new
                eff_gate_scale = eff_gate_scale_new
                eff_tol_mult = eff_tol_mult_new

            if stage_budget_reached:
                if int(mgr.stage_idx) >= final_stage_idx:
                    logger.info(
                        "FINAL_STAGE_BUDGET_REACHED stage=%s timesteps=%s",
                        stage.name,
                        int(model.num_timesteps),
                    )
                    break
                if forced_advance:
                    logger.info(
                        "FORCE_ADVANCE stage=%s reason=timesteps_per_stage mode=%s",
                        stage.name,
                        force_advance_mode_local,
                    )
                    _wandb_log_event(
                        wandb_state,
                        name="force_advance",
                        step=int(model.num_timesteps),
                        extra={
                            "stage_idx": int(mgr.stage_idx),
                            "mode": str(force_advance_mode_local),
                        },
                    )
                    mgr.advance()
                    mgr.set_forgetting_baseline(success_rate=float(summary.success_rate))
                    break
                if not bool(decision.get("advance")):
                    logger.info(
                        "FORCE_ADVANCE_BLOCKED stage=%s mode=%s block_reason=%s",
                        stage.name,
                        force_advance_mode_local,
                        str(decision.get("block_reason", "none")),
                    )
                    _wandb_log_event(
                        wandb_state,
                        name="force_advance_blocked",
                        step=int(model.num_timesteps),
                        extra={
                            "stage_idx": int(mgr.stage_idx),
                            "mode": str(force_advance_mode_local),
                            "block_reason": str(decision.get("block_reason", "none")),
                        },
                    )

        sb3_env.close()

        if stop_after_checkpoint:
            break

        # Stop once the final stage has been trained for its budget.
        if int(mgr.stage_idx) >= final_stage_idx and stage_idx >= final_stage_idx:
            break

    run_meta_payload["finished_at_unix_s"] = float(time.time())
    run_meta_payload["last_global_timesteps"] = (
        None if model is None else int(getattr(model, "num_timesteps", 0))
    )
    _write_run_meta(run_meta_path, run_meta_payload)
    _wandb_finish(wandb_state)


def _coerce_fallback_cli_value(raw: str) -> Any:
    """Convert plain CLI values when Fire is unavailable."""
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _run_train_without_fire(argv: list[str]) -> None:
    """Fallback CLI path for environments without python-fire."""
    if not argv or argv[0] != "train":
        raise SystemExit("fire not installed; only `train` command is supported in fallback mode")

    kwargs: dict[str, Any] = {}
    idx = 1
    while idx < len(argv):
        token = str(argv[idx])
        if not token.startswith("--"):
            raise SystemExit(f"unexpected argument: {token!r}")
        key = token[2:].replace("-", "_")
        if idx + 1 >= len(argv) or str(argv[idx + 1]).startswith("--"):
            kwargs[key] = True
            idx += 1
            continue
        kwargs[key] = _coerce_fallback_cli_value(str(argv[idx + 1]))
        idx += 2

    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    if fire is not None:
        fire.Fire({"train": train}, serialize=lambda _: None)
    else:
        _run_train_without_fire(sys.argv[1:])
