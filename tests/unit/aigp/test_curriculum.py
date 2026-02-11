from dataclasses import replace
from pathlib import Path

import pytest

from lsy_drone_racing.aigp.curriculum import CurriculumConfig, CurriculumManager, EvalSummary


@pytest.mark.unit
def test_curriculum_config_load():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    assert cfg.name == "aigp_10stage"
    assert len(cfg.stages) == 10
    assert cfg.max_gates == 11


@pytest.mark.unit
def test_curriculum_track_overrides():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    mgr = CurriculumManager(cfg)

    config_dir = Path(__file__).parents[3] / "config"

    # Stage 0 uses the single-gate smoke config and overrides gate size.
    tracks, weights = mgr.build_stage_tracks(config_dir=config_dir)
    assert len(tracks) == 1
    assert weights is None or len(weights) == 1
    assert tracks[0].active_gate_count == 1
    assert float(tracks[0].gate_size.width) == pytest.approx(2.0)

    # Stage 1 should set active_gate_count=2 on the Swift track.
    mgr.advance()
    tracks, _ = mgr.build_stage_tracks(config_dir=config_dir)
    assert len(tracks) == 1
    assert int(tracks[0].active_gate_count) == 2


@pytest.mark.unit
def test_curriculum_gate_fields_fail_on_min_episodes():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    cfg = replace(cfg, stability_window=1, stability_threshold=0.0)
    mgr = CurriculumManager(cfg)

    summary = EvalSummary(n_episodes=10, success_rate=1.0, completion_mean=1.0, completion_std=0.0)
    decision = mgr.update_after_eval(summary=summary, stage_episodes=0)

    assert not decision["advance"]
    assert decision["gate_success_ok"]
    assert not decision["gate_min_episodes_ok"]
    assert decision["gate_stability_ok"]
    assert decision["gate_recovery_clear"]
    assert decision["gate_stage_episodes"] == 0
    assert decision["gate_min_episodes_required"] == mgr.current_stage().min_episodes
    assert decision["gate_success_rate"] == pytest.approx(1.0)
    assert decision["gate_success_threshold"] == pytest.approx(
        mgr.current_stage().success_rate_threshold
    )


@pytest.mark.unit
def test_curriculum_gate_fields_fail_on_success_rate_threshold():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    cfg = replace(cfg, stability_window=1, stability_threshold=0.0)
    mgr = CurriculumManager(cfg)

    threshold = float(mgr.current_stage().success_rate_threshold)
    summary = EvalSummary(
        n_episodes=10,
        success_rate=max(0.0, threshold - 0.2),
        completion_mean=1.0,
        completion_std=0.0,
    )
    stage_episodes = int(mgr.current_stage().min_episodes)
    decision = mgr.update_after_eval(summary=summary, stage_episodes=stage_episodes)

    assert not decision["advance"]
    assert not decision["gate_success_ok"]
    assert decision["gate_min_episodes_ok"]
    assert decision["gate_stability_ok"]
    assert decision["gate_recovery_clear"]
    assert decision["gate_stage_episodes"] == stage_episodes
    assert decision["gate_min_episodes_required"] == stage_episodes
    assert decision["gate_success_threshold"] == pytest.approx(threshold)


@pytest.mark.unit
def test_curriculum_gate_fields_pass_when_all_checks_pass():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    cfg = replace(cfg, stability_window=1, stability_threshold=0.0)
    mgr = CurriculumManager(cfg)

    summary = EvalSummary(n_episodes=10, success_rate=1.0, completion_mean=1.0, completion_std=0.0)
    stage_episodes = int(mgr.current_stage().min_episodes)
    decision = mgr.update_after_eval(summary=summary, stage_episodes=stage_episodes)

    assert decision["advance"]
    assert decision["gate_success_ok"]
    assert decision["gate_min_episodes_ok"]
    assert decision["gate_stability_ok"]
    assert decision["gate_recovery_clear"]
