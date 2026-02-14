#!/usr/bin/env python3
"""Launch AIGP curriculum training in Kaggle-friendly sessions.

This helper wraps the existing trainer with defaults intended for free notebook
runtimes where sessions can terminate unexpectedly. It keeps runs resumable via
`--allow_append true` when prior artifacts are present.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _can_resolve_host(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443)
        return True
    except OSError:
        return False


def _resolve_wandb_mode(
    *,
    wandb_enabled: bool,
    requested_mode: str,
    host_resolver: Callable[[str], bool] = _can_resolve_host,
) -> tuple[str, str | None]:
    mode = str(requested_mode).strip().lower()
    if not wandb_enabled or mode != "online":
        return mode, None
    if host_resolver("api.wandb.ai"):
        return mode, None
    return "offline", "dns_unresolved"


def _run_has_history(out_dir: Path) -> bool:
    """Return true when an output directory contains prior eval history."""
    log_path = out_dir / "curriculum_log.jsonl"
    return log_path.exists() and log_path.stat().st_size > 0


def _copy_resume_artifacts(resume_from: Path | None, out_dir: Path) -> None:
    """Seed out_dir with artifacts from a previous run if needed."""
    if resume_from is None:
        return
    if not resume_from.exists():
        raise FileNotFoundError(f"resume source does not exist: {resume_from}")
    if out_dir.exists() and any(out_dir.iterdir()):
        # Existing local artifacts take precedence.
        return
    shutil.copytree(resume_from, out_dir, dirs_exist_ok=True)


def _build_train_command(
    *,
    python_bin: str,
    config: str,
    curriculum: str,
    out: str,
    num_envs: int,
    timesteps_per_stage: int,
    seed: int,
    eval_repeats: int,
    eval_seed_stride: int,
    max_walltime_s: int,
    allow_append: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    obs_mode: str | None,
    qualifier_eval_profile: str | None,
    extra_train_args: Sequence[str],
) -> list[str]:
    """Assemble the trainer command with Kaggle-safe defaults."""
    resolved_obs_mode = None if obs_mode is None else str(obs_mode).strip()
    cmd = [
        str(python_bin),
        "scripts/train_aigp_curriculum.py",
        "train",
        "--config",
        str(config),
        "--curriculum",
        str(curriculum),
        "--num_envs",
        str(int(num_envs)),
        "--out",
        str(out),
        "--timesteps_per_stage",
        str(int(timesteps_per_stage)),
        "--seed",
        str(int(seed)),
        "--eval_repeats",
        str(int(eval_repeats)),
        "--eval_seed_stride",
        str(int(eval_seed_stride)),
        "--force_advance_mode",
        "if_passing",
        "--vecnorm_obs",
        "True",
        "--vecnorm_reward",
        "False",
        "--vecnorm_clip_obs",
        "10.0",
        "--policy_arch_pi",
        "256,256",
        "--policy_arch_vf",
        "256,256",
        "--wandb_enabled",
        _bool_text(wandb_enabled),
        "--wandb_project",
        str(wandb_project),
        "--wandb_entity",
        str(wandb_entity),
        "--wandb_mode",
        str(wandb_mode),
        "--max_walltime_s",
        str(int(max_walltime_s)),
        "--allow_append",
        _bool_text(allow_append),
    ]
    if resolved_obs_mode:
        cmd.extend(["--obs_mode", resolved_obs_mode])
    if qualifier_eval_profile is not None:
        cmd.extend(["--qualifier_eval_profile", str(qualifier_eval_profile)])
    cmd.extend(list(extra_train_args))
    return cmd


def _resolve_tournament_launch_settings(
    *,
    tournament_mode: bool,
    obs_mode: str | None,
    qualifier_eval_profile: str | None,
    extra_train_args: Sequence[str],
) -> tuple[str | None, str | None]:
    """Return tournament-resolved launch settings and validate explicit conflicts."""
    resolved_obs_mode = None if obs_mode is None else str(obs_mode).strip()
    resolved_profile = (
        None if qualifier_eval_profile is None else str(qualifier_eval_profile).strip()
    )

    if not tournament_mode:
        return resolved_obs_mode, resolved_profile

    if resolved_obs_mode and resolved_obs_mode.lower() != "competition_proxy":
        raise ValueError("tournament mode requires --obs-mode to be 'competition_proxy'")

    index = 0
    while index < len(extra_train_args):
        raw_token = str(extra_train_args[index]).strip()
        token = raw_token.split("=", 1)[0].lower()
        if token in {"--obs-mode", "--obs_mode"}:
            if "=" in raw_token:
                explicit = raw_token.split("=", 1)[1]
            else:
                if index + 1 >= len(extra_train_args):
                    raise ValueError(
                        "tournament mode requires --obs-mode/--obs_mode value"
                    )
                explicit = str(extra_train_args[index + 1])
                index += 1
            if str(explicit).strip().lower() != "competition_proxy":
                raise ValueError(
                    "tournament mode requires obs mode to be competition_proxy "
                    "in launch args"
                )
        index += 1

    if resolved_obs_mode is None:
        resolved_obs_mode = "competition_proxy"
    if resolved_profile is None:
        resolved_profile = "aigp_qualifier_eval_profile_default.toml"
    return resolved_obs_mode, resolved_profile


def _run_command(*, cmd: Sequence[str], cwd: Path, env: dict[str, str]) -> int:
    """Execute one command and return its process return code."""
    result = subprocess.run(list(cmd), cwd=str(cwd), env=env, check=False)
    return int(result.returncode)


def _build_preflight_command(
    *,
    python_bin: str,
    repo_root: Path,
    config: str,
    curriculum: str,
    pythonpath_mode: str,
    health_json: Path | None,
) -> list[str]:
    cmd = [
        str(python_bin),
        "scripts/kaggle_preflight.py",
        "--repo-root",
        str(repo_root),
        "--config",
        str(config),
        "--curriculum",
        str(curriculum),
        "--expected-python-bin",
        str(python_bin),
        "--pythonpath-mode",
        str(pythonpath_mode),
        "--json",
    ]
    if health_json is not None:
        cmd.extend(["--health-json", str(health_json)])
    return cmd


def _prepare_runtime_env(*, repo_root: Path, pythonpath_mode: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("SCIPY_ARRAY_API", "1")
    if pythonpath_mode == "repo-root":
        existing = env.get("PYTHONPATH", "")
        if existing:
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + existing
        else:
            env["PYTHONPATH"] = str(repo_root)
    return env


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Launch/restart AIGP training in Kaggle sessions")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", default="aigp_stage0_single_gate.toml")
    parser.add_argument("--curriculum", default="aigp_curriculum_10stage_tuned_v2.toml")
    parser.add_argument("--out", default="runs/aigp_kaggle_primary")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--timesteps-per-stage", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--eval-seed-stride", type=int, default=97)
    parser.add_argument("--max-walltime-s", type=int, default=42_000)
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--qualifier-eval-profile", default=None)
    parser.add_argument("--tournament-mode", type=_parse_bool, default=False)
    parser.add_argument("--readiness-profile", default=None)
    parser.add_argument("--wandb-enabled", type=_parse_bool, default=True)
    parser.add_argument("--wandb-project", default="drone-racing")
    parser.add_argument("--wandb-entity", default="classimo")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--preflight-only", type=_parse_bool, default=False)
    parser.add_argument("--strict-preflight", type=_parse_bool, default=True)
    parser.add_argument("--pythonpath-mode", choices=["repo-root", "none"], default="repo-root")
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--extra-train-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root does not exist: {repo_root}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    resume_from: Path | None = None
    if args.resume_from is not None:
        resume_from = args.resume_from.resolve()

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    _copy_resume_artifacts(resume_from, out_dir)
    allow_append = _run_has_history(out_dir)

    resolved_wandb_mode, wandb_fallback_reason = _resolve_wandb_mode(
        wandb_enabled=bool(args.wandb_enabled),
        requested_mode=str(args.wandb_mode),
    )
    if wandb_fallback_reason is not None:
        print(
            "WANDB_MODE_FALLBACK "
            f"requested={args.wandb_mode} resolved={resolved_wandb_mode} "
            f"reason={wandb_fallback_reason}"
        )

    tournament_mode = bool(args.tournament_mode)
    try:
        obs_mode, qualifier_eval_profile = _resolve_tournament_launch_settings(
            tournament_mode=tournament_mode,
            obs_mode=args.obs_mode,
            qualifier_eval_profile=args.qualifier_eval_profile,
            extra_train_args=list(args.extra_train_arg),
        )
    except ValueError as exc:
        parser.error(str(exc))
    readiness_profile = args.readiness_profile
    if readiness_profile is None:
        readiness_profile = "qualifier_strict" if tournament_mode else "default"

    train_cmd = _build_train_command(
        python_bin=str(args.python_bin),
        config=str(args.config),
        curriculum=str(args.curriculum),
        out=str(out_dir),
        num_envs=int(args.num_envs),
        timesteps_per_stage=int(args.timesteps_per_stage),
        seed=int(args.seed),
        eval_repeats=int(args.eval_repeats),
        eval_seed_stride=int(args.eval_seed_stride),
        max_walltime_s=int(args.max_walltime_s),
        allow_append=bool(allow_append),
        wandb_enabled=bool(args.wandb_enabled),
        wandb_project=str(args.wandb_project),
        wandb_entity=str(args.wandb_entity),
        wandb_mode=str(resolved_wandb_mode),
        obs_mode=obs_mode,
        qualifier_eval_profile=qualifier_eval_profile,
        extra_train_args=list(args.extra_train_arg),
    )
    if tournament_mode:
        train_cmd.extend(["--tournament_mode", "true"])

    readiness_cmd = [
        str(args.python_bin),
        "scripts/build_readiness_report.py",
        "--run-dir",
        str(out_dir),
        "--profile",
        str(readiness_profile),
        "--window-evals",
        "5",
        "--json",
    ]
    health_json = args.health_json
    if health_json is not None and not health_json.is_absolute():
        health_json = repo_root / health_json
    preflight_cmd = _build_preflight_command(
        python_bin=str(args.python_bin),
        repo_root=repo_root,
        config=str(args.config),
        curriculum=str(args.curriculum),
        pythonpath_mode=str(args.pythonpath_mode),
        health_json=health_json,
    )

    print(
        "KAGGLE_AIGP_SESSION "
        f"repo_root={repo_root} out_dir={out_dir} allow_append={int(allow_append)} "
        f"max_walltime_s={int(args.max_walltime_s)} "
        f"strict_preflight={int(bool(args.strict_preflight))} "
        f"preflight_only={int(bool(args.preflight_only))}"
    )
    print("PREFLIGHT_CMD", " ".join(preflight_cmd))
    print("TRAIN_CMD", " ".join(train_cmd))

    if args.dry_run:
        print("READINESS_CMD", " ".join(readiness_cmd))
        raise SystemExit(0)

    env = _prepare_runtime_env(repo_root=repo_root, pythonpath_mode=str(args.pythonpath_mode))

    preflight_rc = _run_command(cmd=preflight_cmd, cwd=repo_root, env=env)
    if bool(args.preflight_only):
        print(f"SESSION_RESULT preflight_rc={preflight_rc} train_rc=0 readiness_rc=0")
        raise SystemExit(int(preflight_rc))
    if bool(args.strict_preflight) and preflight_rc != 0:
        print("PRECHECK_FAILED strict_preflight=1 refusing to start trainer")
        print(f"SESSION_RESULT preflight_rc={preflight_rc} train_rc=0 readiness_rc=0")
        raise SystemExit(int(preflight_rc))

    train_rc = _run_command(cmd=train_cmd, cwd=repo_root, env=env)

    # Always attempt readiness report after training exits.
    readiness_rc = _run_command(cmd=readiness_cmd, cwd=repo_root, env=env)

    print(
        "SESSION_RESULT "
        f"preflight_rc={preflight_rc} train_rc={train_rc} readiness_rc={readiness_rc}"
    )
    raise SystemExit(train_rc)


if __name__ == "__main__":
    main()
