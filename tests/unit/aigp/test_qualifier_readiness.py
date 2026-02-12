import json
from pathlib import Path

import pytest

from scripts.check_qualifier_readiness import _emit_report, evaluate_run, evaluate_run_with_profile


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.mark.unit
def test_evaluate_run_errors_without_eval_rows(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    code, payload = evaluate_run(
        run_dir=run_dir,
        window_evals=5,
        min_stage_idx=0,
        min_success_rate=0.7,
        min_course_completion_rate=0.7,
        max_robustness_std=0.2,
        min_track_coverage_ratio=1.0,
        min_seed_coverage_ratio=1.0,
        require_scorecard_pass=False,
        fail_on_forced_advance_without_pass=True,
    )
    assert code == 2
    assert payload["ready"] is False


@pytest.mark.unit
def test_evaluate_run_ready_when_window_metrics_pass(tmp_path: Path):
    run_dir = tmp_path / "run"
    rows = [
        {
            "schema_version": 2,
            "eval_id": 1,
            "global_timesteps": 65536,
            "stage": {"idx": 0, "name": "stage0_single_gate"},
            "eval": {"success_rate": 0.9},
            "qualifier_eval": {
                "course_completion_rate": 0.92,
                "robustness_std": 0.08,
                "track_coverage": {"ratio": 1.0},
                "seed_coverage": {"ratio": 1.0},
            },
            "scorecard": {"pass": True},
            "forced_advance": False,
        },
        {
            "schema_version": 2,
            "eval_id": 2,
            "global_timesteps": 131072,
            "stage": {"idx": 1, "name": "stage1_bridge"},
            "eval": {"success_rate": 0.85},
            "qualifier_eval": {
                "course_completion_rate": 0.88,
                "robustness_std": 0.10,
                "track_coverage": {"ratio": 1.0},
                "seed_coverage": {"ratio": 1.0},
            },
            "scorecard": {"pass": True},
            "forced_advance": False,
        },
    ]
    _write_rows(run_dir / "curriculum_log.jsonl", rows)

    code, payload = evaluate_run(
        run_dir=run_dir,
        window_evals=2,
        min_stage_idx=1,
        min_success_rate=0.7,
        min_course_completion_rate=0.7,
        max_robustness_std=0.2,
        min_track_coverage_ratio=1.0,
        min_seed_coverage_ratio=1.0,
        require_scorecard_pass=True,
        fail_on_forced_advance_without_pass=True,
    )
    assert code == 0
    assert payload["ready"] is True


@pytest.mark.unit
def test_evaluate_run_fails_when_forced_advance_without_pass(tmp_path: Path):
    run_dir = tmp_path / "run"
    rows = [
        {
            "schema_version": 2,
            "eval_id": 1,
            "global_timesteps": 65536,
            "stage": {"idx": 1, "name": "stage1_bridge"},
            "eval": {"success_rate": 0.8},
            "qualifier_eval": {
                "course_completion_rate": 0.8,
                "robustness_std": 0.05,
                "track_coverage": {"ratio": 1.0},
                "seed_coverage": {"ratio": 1.0},
            },
            "scorecard": {"pass": False},
            "forced_advance": True,
        }
    ]
    _write_rows(run_dir / "curriculum_log.jsonl", rows)

    code, payload = evaluate_run(
        run_dir=run_dir,
        window_evals=1,
        min_stage_idx=0,
        min_success_rate=0.7,
        min_course_completion_rate=0.7,
        max_robustness_std=0.2,
        min_track_coverage_ratio=1.0,
        min_seed_coverage_ratio=1.0,
        require_scorecard_pass=False,
        fail_on_forced_advance_without_pass=True,
    )
    assert code == 1
    assert payload["ready"] is False
    gate_by_name = {g["name"]: g for g in payload["gates"]}
    assert gate_by_name["forced_advance_without_pass"]["ok"] is False


@pytest.mark.unit
def test_profile_stage_gate_requires_latest_scorecard_pass(tmp_path: Path):
    run_dir = tmp_path / "run"
    rows = [
        {
            "schema_version": 2,
            "eval_id": 1,
            "global_timesteps": 65536,
            "stage": {"idx": 0, "name": "stage0_single_gate"},
            "eval": {"success_rate": 0.95},
            "qualifier_eval": {
                "course_completion_rate": 0.95,
                "robustness_std": 0.05,
                "track_coverage": {"ratio": 1.0},
                "seed_coverage": {"ratio": 1.0},
            },
            "scorecard": {"pass": False},
            "forced_advance": False,
        }
    ]
    _write_rows(run_dir / "curriculum_log.jsonl", rows)

    code, payload = evaluate_run_with_profile(
        run_dir=run_dir,
        profile="stage_gate",
        window_evals=1,
    )
    assert code == 1
    assert payload["ready"] is False
    assert "latest_scorecard_pass" in payload["failing_reasons"]
    assert payload["recommendation"] == "hold"


@pytest.mark.unit
def test_profile_qualifier_strict_uses_tighter_thresholds(tmp_path: Path):
    run_dir = tmp_path / "run"
    rows = [
        {
            "schema_version": 2,
            "eval_id": 1,
            "global_timesteps": 65536,
            "stage": {"idx": 0, "name": "stage0_single_gate"},
            "eval": {"success_rate": 0.75},
            "qualifier_eval": {
                "course_completion_rate": 0.75,
                "robustness_std": 0.05,
                "track_coverage": {"ratio": 1.0},
                "seed_coverage": {"ratio": 1.0},
            },
            "scorecard": {"pass": True},
            "forced_advance": False,
        }
    ]
    _write_rows(run_dir / "curriculum_log.jsonl", rows)

    code, payload = evaluate_run_with_profile(
        run_dir=run_dir,
        profile="qualifier_strict",
        window_evals=1,
    )
    assert code == 1
    assert payload["ready"] is False
    assert "mean_success_rate" in payload["failing_reasons"]
    assert "mean_course_completion_rate" in payload["failing_reasons"]


@pytest.mark.unit
def test_emit_report_writes_json_payload(tmp_path: Path):
    payload = {
        "ready": True,
        "profile": "default",
        "metrics": {"mean_success_rate": 0.9},
    }
    out = tmp_path / "report" / "latest.json"
    _emit_report(out, payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["ready"] is True
    assert loaded["profile"] == "default"
