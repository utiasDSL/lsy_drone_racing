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
def test_curriculum_advance_gate_requires_min_episodes_and_stability():
    cfg = CurriculumConfig.load(Path(__file__).parents[3] / "config/aigp_curriculum_10stage.toml")
    # Override stability to make the test deterministic/fast.
    cfg = replace(cfg, stability_window=1, stability_threshold=0.0)
    mgr = CurriculumManager(cfg)

    stage_episodes = 0
    summary = EvalSummary(n_episodes=10, success_rate=1.0, completion_mean=1.0, completion_std=0.0)
    decision = mgr.update_after_eval(summary=summary, stage_episodes=stage_episodes)
    assert not decision["advance"]

    stage_episodes = mgr.current_stage().min_episodes
    decision = mgr.update_after_eval(summary=summary, stage_episodes=stage_episodes)
    assert decision["advance"]
