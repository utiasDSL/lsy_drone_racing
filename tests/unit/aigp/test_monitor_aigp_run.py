import json
from pathlib import Path

import pytest

from scripts.monitor_aigp_run import (
    _compute_alerts,
    _monitor_once,
    _read_recent_jsonl_rows,
    _resolve_wandb_run_path,
)


@pytest.mark.unit
def test_read_recent_jsonl_rows_ignores_invalid_lines(tmp_path: Path):
    path = tmp_path / "curriculum_log.jsonl"
    path.write_text('{"eval_id":1,"global_timesteps":100}\nnot-json\n{"eval_id":2,"global_timesteps":200}\n')
    rows = _read_recent_jsonl_rows(path)
    assert len(rows) == 2
    assert rows[-1]["eval_id"] == 2


@pytest.mark.unit
def test_resolve_wandb_run_path_precedence():
    out = _resolve_wandb_run_path(
        arg_run_path="arg/path/123",
        latest_row={"wandb": {"run_path": "row/path/1"}},
        run_meta={"wandb": {"run_path": "meta/path/1"}},
    )
    assert out == "arg/path/123"

    out = _resolve_wandb_run_path(
        arg_run_path=None,
        latest_row={"wandb": {"run_path": "row/path/1"}},
        run_meta={"wandb": {"run_path": "meta/path/1"}},
    )
    assert out == "meta/path/1"

    out = _resolve_wandb_run_path(
        arg_run_path=None,
        latest_row={"wandb": {"run_path": "row/path/1"}},
        run_meta=None,
    )
    assert out == "row/path/1"


@pytest.mark.unit
def test_compute_alerts_warns_on_stale_and_repeated_block():
    local_status = {
        "latest_eval": {"global_timesteps": 200, "block_reason": "stability"},
        "process_active": True,
        "recent_block_reasons": ["stability"] * 5,
        "markers": {"EVAL_STAGE1_GUARD": None},
    }
    alerts, prev, stale = _compute_alerts(
        local_status=local_status,
        prev_timesteps=200,
        stale_count=2,
        stale_limit=3,
    )
    assert prev == 200
    assert stale == 3
    msgs = " | ".join(a["message"] for a in alerts)
    assert "timesteps unchanged" in msgs
    assert "repeated block_reason='stability'" in msgs


@pytest.mark.unit
def test_compute_alerts_no_eval_rows_warns_while_process_active():
    local_status = {
        "latest_eval": None,
        "process_active": True,
        "recent_block_reasons": [],
        "markers": {"EVAL_STAGE1_GUARD": None},
    }
    alerts, prev, stale = _compute_alerts(
        local_status=local_status,
        prev_timesteps=None,
        stale_count=0,
        stale_limit=3,
    )
    assert prev is None
    assert stale == 0
    assert alerts == [
        {
            "severity": "warning",
            "message": "no eval rows yet; trainer appears active (waiting for first eval)",
        }
    ]


@pytest.mark.unit
def test_compute_alerts_no_eval_rows_errors_when_process_missing():
    local_status = {
        "latest_eval": None,
        "process_active": False,
        "recent_block_reasons": [],
        "markers": {"EVAL_STAGE1_GUARD": None},
    }
    alerts, prev, stale = _compute_alerts(
        local_status=local_status,
        prev_timesteps=None,
        stale_count=0,
        stale_limit=3,
    )
    assert prev is None
    assert stale == 0
    assert alerts == [
        {
            "severity": "error",
            "message": "no eval rows found in curriculum_log.jsonl",
        }
    ]


@pytest.mark.unit
def test_monitor_once_local_only_returns_zero_for_healthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "curriculum_log.jsonl").write_text(
        json.dumps(
            {
                "eval_id": 1,
                "global_timesteps": 1000,
                "stage": {"idx": 0, "name": "stage0_single_gate"},
                "block_reason": "none",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = Path(str(run_dir) + ".log")
    log_path.write_text(
        "INFO EVAL_SCORECARD stage_idx=0 pass=1\nINFO EVAL_QUALIFIER profile=inline_eval ccr=1.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.monitor_aigp_run._detect_process_active", lambda _run_dir: True)
    rc = _monitor_once(run_dir=run_dir, no_wandb=True, wandb_run_path=None, as_json=True)
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert rc == 0
    assert payload["local"]["latest_eval"]["global_timesteps"] == 1000
    assert payload["alerts"] == []
