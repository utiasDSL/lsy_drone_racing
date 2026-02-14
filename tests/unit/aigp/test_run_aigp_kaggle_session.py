from pathlib import Path

import pytest

from scripts.run_aigp_kaggle_session import (
    _build_preflight_command,
    _build_train_command,
    _copy_resume_artifacts,
    _prepare_runtime_env,
    _resolve_tournament_launch_settings,
    _resolve_wandb_mode,
    _run_has_history,
)


@pytest.mark.unit
def test_run_has_history_detects_existing_curriculum_log(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True)
    assert _run_has_history(out_dir) is False

    log_path = out_dir / "curriculum_log.jsonl"
    log_path.write_text('{"eval_id":1}\n', encoding="utf-8")
    assert _run_has_history(out_dir) is True


@pytest.mark.unit
def test_copy_resume_artifacts_copies_when_out_dir_empty(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(parents=True)
    (src / "curriculum_log.jsonl").write_text('{"eval_id":1}\n', encoding="utf-8")
    (src / "vecnormalize.pkl").write_bytes(b"abc")

    _copy_resume_artifacts(src, dst)
    assert (dst / "curriculum_log.jsonl").exists()
    assert (dst / "vecnormalize.pkl").exists()


@pytest.mark.unit
def test_build_train_command_sets_allow_append_and_walltime() -> None:
    cmd = _build_train_command(
        python_bin="python",
        config="aigp_stage0_single_gate.toml",
        curriculum="aigp_curriculum_10stage_tuned_v2.toml",
        out="runs/aigp_kaggle_primary",
        num_envs=16,
        timesteps_per_stage=10_000_000,
        seed=4242,
        eval_repeats=2,
        eval_seed_stride=97,
        max_walltime_s=42_000,
        allow_append=True,
        wandb_enabled=True,
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_mode="online",
        obs_mode="competition_proxy",
        qualifier_eval_profile="aigp_qualifier_eval_profile_default.toml",
        extra_train_args=["--device", "auto"],
    )
    joined = " ".join(cmd)
    assert "--allow_append true" in joined
    assert "--max_walltime_s 42000" in joined
    assert "--force_advance_mode if_passing" in joined
    assert "--obs_mode competition_proxy" in joined
    assert "--qualifier_eval_profile aigp_qualifier_eval_profile_default.toml" in joined
    assert "--device auto" in joined


@pytest.mark.unit
def test_build_train_command_without_optional_obs_profile() -> None:
    cmd = _build_train_command(
        python_bin="python",
        config="aigp_stage0_single_gate.toml",
        curriculum="aigp_curriculum_10stage_tuned_v2.toml",
        out="runs/aigp_kaggle_primary",
        num_envs=16,
        timesteps_per_stage=10_000_000,
        seed=4242,
        eval_repeats=2,
        eval_seed_stride=97,
        max_walltime_s=42_000,
        allow_append=False,
        wandb_enabled=True,
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_mode="online",
        obs_mode=None,
        qualifier_eval_profile=None,
        extra_train_args=[],
    )
    joined = " ".join(cmd)
    assert "--obs_mode" not in joined
    assert "--qualifier_eval_profile" not in joined


@pytest.mark.unit
def test_copy_resume_artifacts_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _copy_resume_artifacts(tmp_path / "missing", tmp_path / "dst")


@pytest.mark.unit
def test_build_preflight_command_includes_health_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    health_json = tmp_path / "health" / "latest.json"
    cmd = _build_preflight_command(
        python_bin="python",
        repo_root=repo_root,
        config="aigp_stage0_single_gate.toml",
        curriculum="aigp_curriculum_10stage_tuned_v2.toml",
        pythonpath_mode="repo-root",
        health_json=health_json,
    )
    joined = " ".join(cmd)
    assert "scripts/kaggle_preflight.py" in joined
    assert "--pythonpath-mode repo-root" in joined
    assert f"--health-json {health_json}" in joined
    assert "--json" in joined


@pytest.mark.unit
def test_prepare_runtime_env_injects_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")
    env = _prepare_runtime_env(repo_root=tmp_path, pythonpath_mode="repo-root")
    assert env["JAX_ENABLE_COMPILATION_CACHE"] == "0"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_PLATFORM_NAME"] == "cpu"
    assert env["SCIPY_ARRAY_API"] == "1"
    assert env["PYTHONPATH"].startswith(str(tmp_path))
    assert env["PYTHONPATH"].endswith("/tmp/existing")


@pytest.mark.unit
def test_prepare_runtime_env_skips_pythonpath_when_disabled(tmp_path: Path) -> None:
    env = _prepare_runtime_env(repo_root=tmp_path, pythonpath_mode="none")
    assert env["JAX_ENABLE_COMPILATION_CACHE"] == "0"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_PLATFORM_NAME"] == "cpu"
    assert env["SCIPY_ARRAY_API"] == "1"


@pytest.mark.unit
def test_resolve_wandb_mode_falls_back_offline_when_dns_unavailable() -> None:
    mode, reason = _resolve_wandb_mode(
        wandb_enabled=True,
        requested_mode="online",
        host_resolver=lambda _: False,
    )
    assert mode == "offline"
    assert reason == "dns_unresolved"


@pytest.mark.unit
def test_resolve_wandb_mode_keeps_online_when_dns_available() -> None:
    mode, reason = _resolve_wandb_mode(
        wandb_enabled=True,
        requested_mode="online",
        host_resolver=lambda _: True,
    )
    assert mode == "online"
    assert reason is None


@pytest.mark.unit
def test_tournament_launch_resolves_defaults_to_competition_proxy() -> None:
    obs_mode, profile = _resolve_tournament_launch_settings(
        tournament_mode=True,
        obs_mode=None,
        qualifier_eval_profile=None,
        extra_train_args=[],
    )
    assert obs_mode == "competition_proxy"
    assert profile == "aigp_qualifier_eval_profile_default.toml"


@pytest.mark.unit
def test_tournament_launch_rejects_conflicting_obs_mode_arg() -> None:
    with pytest.raises(ValueError, match="competition_proxy"):
        _resolve_tournament_launch_settings(
            tournament_mode=True,
            obs_mode="competition_proxy",
            qualifier_eval_profile="aigp_qualifier_eval_profile_default.toml",
            extra_train_args=["--obs-mode", "privileged"],
        )
