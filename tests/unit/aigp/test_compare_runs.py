import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compare_aigp_runs import compare_runs


def _write_eval_rows(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "curriculum_log.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(
    *,
    eval_id: int,
    step: int,
    stage_idx: int,
    sr: float,
    ccr: float,
    robust: float,
    score_pass: bool,
    forced: bool = False,
) -> dict:
    return {
        "schema_version": 2,
        "eval_id": eval_id,
        "global_timesteps": step,
        "stage": {"idx": stage_idx, "name": f"stage{stage_idx}"},
        "eval": {"success_rate": sr},
        "qualifier_eval": {
            "course_completion_rate": ccr,
            "robustness_std": robust,
            "track_coverage": {"ratio": 1.0},
            "seed_coverage": {"ratio": 1.0},
        },
        "scorecard": {"pass": score_pass},
        "forced_advance": forced,
    }


@pytest.mark.unit
def test_compare_runs_ranks_ready_run_first(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    _write_eval_rows(
        run_a,
        [
            _row(
                eval_id=1,
                step=100,
                stage_idx=1,
                sr=0.9,
                ccr=0.9,
                robust=0.05,
                score_pass=True,
            )
        ],
    )
    _write_eval_rows(
        run_b,
        [
            _row(
                eval_id=1,
                step=100,
                stage_idx=1,
                sr=0.4,
                ccr=0.4,
                robust=0.05,
                score_pass=False,
                forced=True,
            )
        ],
    )

    payload = compare_runs(
        run_dirs=[run_a, run_b],
        profile="default",
        window_evals=1,
        wandb_run_paths=None,
    )

    ranking = payload["ranking"]
    assert ranking[0]["run_dir"] == str(run_a.resolve())
    assert ranking[0]["ready"] is True
    assert ranking[1]["ready"] is False
    assert ranking[1]["forced_advance_violations_total"] == 1


@pytest.mark.unit
def test_compare_runs_uses_robustness_tiebreaker(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    _write_eval_rows(
        run_a,
        [
            _row(
                eval_id=1,
                step=100,
                stage_idx=1,
                sr=0.85,
                ccr=0.85,
                robust=0.10,
                score_pass=True,
            )
        ],
    )
    _write_eval_rows(
        run_b,
        [
            _row(
                eval_id=1,
                step=120,
                stage_idx=1,
                sr=0.85,
                ccr=0.85,
                robust=0.03,
                score_pass=True,
            )
        ],
    )

    payload = compare_runs(
        run_dirs=[run_a, run_b],
        profile="default",
        window_evals=1,
        wandb_run_paths=None,
    )
    ranking = payload["ranking"]
    assert ranking[0]["run_dir"] == str(run_b.resolve())
    assert ranking[1]["run_dir"] == str(run_a.resolve())


@pytest.mark.unit
def test_compare_script_runs_without_pythonpath(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_eval_rows(
        run_a,
        [
            _row(
                eval_id=1,
                step=100,
                stage_idx=1,
                sr=0.9,
                ccr=0.9,
                robust=0.05,
                score_pass=True,
            )
        ],
    )
    _write_eval_rows(
        run_b,
        [
            _row(
                eval_id=1,
                step=100,
                stage_idx=1,
                sr=0.4,
                ccr=0.4,
                robust=0.05,
                score_pass=False,
                forced=True,
            )
        ],
    )
    out_prefix = tmp_path / "cmp_out"

    script = Path(__file__).resolve().parents[3] / "scripts" / "compare_aigp_runs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_a),
            "--run-dir",
            str(run_b),
            "--profile",
            "default",
            "--window-evals",
            "1",
            "--out-prefix",
            str(out_prefix),
            "--json",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ranking_count"] == 2
    assert payload["top_run"] in {str(run_a.resolve()), str(run_b.resolve())}
