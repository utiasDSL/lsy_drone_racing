import json
from pathlib import Path

import pytest

from lsy_drone_racing.aigp.curriculum import EvalSummary
from scripts.train_aigp_curriculum import (
    _apply_stage1_transition_guards,
    _build_qualifier_eval_metrics,
    _export_submission_bundle,
    _parse_obs_mode,
    _resolve_tournament_train_settings,
)


@pytest.mark.unit
def test_parse_obs_mode_validation():
    assert _parse_obs_mode("competition_proxy") == "competition_proxy"
    assert _parse_obs_mode("PRIVILEGED") == "privileged"
    with pytest.raises(ValueError):
        _parse_obs_mode("vision_only")


@pytest.mark.unit
def test_tournament_train_settings_enforce_competition_proxy() -> None:
    with pytest.raises(ValueError, match="competition_proxy"):
        _resolve_tournament_train_settings(
            tournament_mode=True,
            obs_mode="privileged",
            qualifier_eval_profile=None,
            tournament_readiness_profile=None,
        )


@pytest.mark.unit
def test_tournament_train_settings_default_profiles() -> None:
    obs_mode, qualifier_profile, readiness_profile = _resolve_tournament_train_settings(
        tournament_mode=True,
        obs_mode="competition_proxy",
        qualifier_eval_profile=None,
        tournament_readiness_profile=None,
    )
    assert obs_mode == "competition_proxy"
    assert qualifier_profile == "aigp_qualifier_eval_profile_default.toml"
    assert readiness_profile == "qualifier_strict"


@pytest.mark.unit
def test_stage1_guard_blocks_advance_without_required_streak():
    decision = {"advance": True, "rollback": False, "block_reason": "none"}
    summary = EvalSummary(
        n_episodes=20,
        success_rate=0.80,
        completion_mean=0.9,
        completion_std=0.05,
        lap_time_s_median=4.2,
    )
    updated, nonzero_seen, streak = _apply_stage1_transition_guards(
        decision=decision,
        summary=summary,
        stage_idx=1,
        stage_success_threshold=0.75,
        stage_elapsed_timesteps=500_000,
        stage1_nonzero_progress_seen=False,
        stage1_success_streak=0,
        stage1_nonzero_progress_budget=2_000_000,
        stage1_required_streak=3,
    )
    assert nonzero_seen
    assert streak == 1
    assert not updated["advance"]
    assert updated["block_reason"] == "stage1_consistency"


@pytest.mark.unit
def test_stage1_guard_rolls_back_when_no_progress_after_budget():
    decision = {"advance": False, "rollback": False, "block_reason": "success_rate"}
    summary = EvalSummary(
        n_episodes=20,
        success_rate=0.0,
        completion_mean=0.1,
        completion_std=0.01,
        lap_time_s_median=None,
    )
    updated, nonzero_seen, streak = _apply_stage1_transition_guards(
        decision=decision,
        summary=summary,
        stage_idx=1,
        stage_success_threshold=0.75,
        stage_elapsed_timesteps=2_100_000,
        stage1_nonzero_progress_seen=False,
        stage1_success_streak=0,
        stage1_nonzero_progress_budget=2_000_000,
        stage1_required_streak=3,
    )
    assert not nonzero_seen
    assert streak == 0
    assert not updated["advance"]
    assert updated["rollback"]
    assert updated["block_reason"] == "stage1_zero_progress"


@pytest.mark.unit
def test_build_qualifier_eval_metrics_coverage_and_percentiles():
    rows = [
        (0, 101, EvalSummary(10, 0.8, 0.9, 0.1, 4.0)),
        (1, 101, EvalSummary(10, 0.6, 0.7, 0.2, 5.0)),
        (0, 202, EvalSummary(10, 0.7, 0.8, 0.1, 6.0)),
    ]
    payload = _build_qualifier_eval_metrics(
        rows=rows, tracks_total=2, seeds_total=2, profile_name="test_profile"
    )
    assert payload["profile"] == "test_profile"
    assert payload["course_completion_rate"] == pytest.approx((0.8 + 0.6 + 0.7) / 3.0)
    assert payload["lap_time_s_p50"] == pytest.approx(5.0)
    assert payload["lap_time_s_p90"] == pytest.approx(5.8)
    assert payload["track_coverage"]["covered"] == 2
    assert payload["track_coverage"]["total"] == 2
    assert payload["seed_coverage"]["covered"] == 2
    assert payload["seed_coverage"]["total"] == 2


@pytest.mark.unit
def test_export_submission_bundle_contract(tmp_path: Path):
    ckpt = tmp_path / "stage00_step_100.zip"
    ckpt.write_bytes(b"fake-model")
    vecnorm = tmp_path / "vecnormalize.pkl"
    vecnorm.write_bytes(b"fake-vecnorm")

    bundle_dir = _export_submission_bundle(
        out_dir=tmp_path,
        checkpoint_path=ckpt,
        vecnormalize_path=vecnorm,
        run_id="run-123",
        eval_id=4,
        global_timesteps=123456,
        obs_mode="competition_proxy",
        config_name="cfg.toml",
        curriculum_name="cur.toml",
        config_hash="abc123",
    )

    assert (bundle_dir / "model_latest.zip").exists()
    assert (bundle_dir / "vecnormalize.pkl").exists()
    assert (bundle_dir / "inference.py").exists()
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["run_id"] == "run-123"
    assert metadata["obs_mode"] == "competition_proxy"
    assert metadata["artifacts"]["model"] == "model_latest.zip"
