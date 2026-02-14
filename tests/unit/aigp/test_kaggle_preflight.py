import json
import subprocess
import sys
from pathlib import Path

import pytest


def _script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "kaggle_preflight.py"


def _run_preflight(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_script_path()), *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


@pytest.mark.unit
def test_preflight_reports_path_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path / "missing_repo"
    repo_root.mkdir(parents=True)
    result = _run_preflight(
        [
            "--repo-root",
            str(repo_root),
            "--pythonpath-mode",
            "none",
            "--expected-python-bin",
            sys.executable,
            "--json",
        ]
    )
    assert result.returncode == 13
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 13
    assert payload["checks"]["paths"]["ok"] is False


@pytest.mark.unit
def test_preflight_reports_missing_asset_and_writes_health_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "scripts" / "train_aigp_curriculum.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "build_readiness_report.py").write_text("", encoding="utf-8")
    (repo_root / "config" / "aigp_stage0_single_gate.toml").write_text("", encoding="utf-8")
    (repo_root / "config" / "aigp_curriculum_10stage_tuned_v2.toml").write_text(
        "",
        encoding="utf-8",
    )
    health_json = tmp_path / "health" / "latest.json"
    result = _run_preflight(
        [
            "--repo-root",
            str(repo_root),
            "--pythonpath-mode",
            "none",
            "--expected-python-bin",
            sys.executable,
            "--health-json",
            str(health_json),
            "--json",
        ]
    )
    assert result.returncode == 12
    assert health_json.exists()
    payload = json.loads(health_json.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 12
    assert payload["checks"]["paths"]["ok"] is True
    assert payload["checks"]["assets"]["ok"] is False


@pytest.mark.unit
def test_preflight_reports_interpreter_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    result = _run_preflight(
        [
            "--repo-root",
            str(repo_root),
            "--pythonpath-mode",
            "none",
            "--expected-python-bin",
            "/usr/bin/does-not-exist-python",
            "--json",
        ]
    )
    assert result.returncode == 14
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 14
    assert payload["checks"]["interpreter"]["ok"] is False
