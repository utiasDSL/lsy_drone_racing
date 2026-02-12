#!/usr/bin/env python3
"""Hybrid local + W&B monitor for AIGP curriculum runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


def _read_recent_jsonl_rows(path: Path, *, max_rows: int = 200) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return list(rows)


def _tail_markers(
    log_path: Path, *, markers: list[str], max_lines: int = 2000
) -> dict[str, str | None]:
    out: dict[str, str | None] = {m: None for m in markers}
    if not log_path.exists():
        return out
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        recent = deque(f, maxlen=max_lines)
    for line in reversed(recent):
        text = line.rstrip("\n")
        for marker in markers:
            if out[marker] is None and marker in text:
                out[marker] = text
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _detect_process_active(run_dir: Path) -> bool:
    run_name = run_dir.name
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "pid,command"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    candidates = [line for line in lines if "train_aigp_curriculum.py" in line]
    if not candidates:
        return False
    if any(run_name in line for line in candidates):
        return True
    # Fallback for truncated process arguments: if a trainer exists, treat as active.
    return True


def _resolve_wandb_run_path(
    *,
    arg_run_path: str | None,
    latest_row: dict[str, Any] | None,
    run_meta: dict[str, Any] | None,
) -> str | None:
    if arg_run_path and arg_run_path.strip():
        return arg_run_path.strip()
    if run_meta and isinstance(run_meta.get("wandb"), dict):
        run_path = run_meta["wandb"].get("run_path")
        if run_path:
            return str(run_path)
    if latest_row and isinstance(latest_row.get("wandb"), dict):
        run_path = latest_row["wandb"].get("run_path")
        if run_path:
            return str(run_path)
    return None


def _fetch_wandb_status(run_path: str | None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "run_path": run_path,
        "state": None,
        "url": None,
        "summary": {},
        "error": None,
    }
    if not run_path:
        return status
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception:
        status["error"] = "wandb import unavailable"
        return status

    try:
        api = wandb.Api()
        run = api.run(run_path)
    except Exception as exc:
        status["error"] = f"wandb API error: {exc}"
        return status

    summary = {}
    try:
        summary = dict(run.summary._json_dict)
    except Exception:
        summary = {}

    status["enabled"] = True
    status["available"] = True
    status["state"] = getattr(run, "state", None)
    status["url"] = getattr(run, "url", None)
    status["summary"] = {
        "curriculum/stage_idx": summary.get("curriculum/stage_idx"),
        "curriculum/stage_name": summary.get("curriculum/stage_name"),
        "curriculum/success_rate": summary.get("curriculum/success_rate"),
        "curriculum/global_timesteps": summary.get("curriculum/global_timesteps"),
        "curriculum/block_reason": summary.get("curriculum/block_reason"),
    }
    return status


def _build_local_status(run_dir: Path) -> dict[str, Any]:
    jsonl_path = run_dir / "curriculum_log.jsonl"
    run_meta_path = run_dir / "run_meta.json"
    log_path = Path(str(run_dir) + ".log")

    rows = _read_recent_jsonl_rows(jsonl_path, max_rows=300)
    eval_rows = [r for r in rows if "eval_id" in r and "global_timesteps" in r]
    latest_eval = eval_rows[-1] if eval_rows else None
    recent_block_reasons = [
        str(r.get("block_reason", "none")) for r in eval_rows[-10:] if "block_reason" in r
    ]

    markers = _tail_markers(
        log_path,
        markers=["EVAL_SCORECARD", "EVAL_BOSSFIGHT", "EVAL_QUALIFIER", "EVAL_STAGE1_GUARD"],
    )
    run_meta = _read_json(run_meta_path)

    return {
        "run_dir": str(run_dir),
        "jsonl_path": str(jsonl_path),
        "log_path": str(log_path),
        "run_meta_path": str(run_meta_path),
        "exists": run_dir.exists(),
        "process_active": _detect_process_active(run_dir),
        "latest_eval": latest_eval,
        "recent_block_reasons": recent_block_reasons,
        "markers": markers,
        "run_meta": run_meta,
    }


def _compute_alerts(
    *,
    local_status: dict[str, Any],
    prev_timesteps: int | None,
    stale_count: int,
    stale_limit: int,
) -> tuple[list[dict[str, str]], int | None, int]:
    alerts: list[dict[str, str]] = []
    latest_eval = local_status.get("latest_eval")
    current_step: int | None = None
    if isinstance(latest_eval, dict) and latest_eval.get("global_timesteps") is not None:
        current_step = int(latest_eval["global_timesteps"])

    if current_step is None:
        if bool(local_status.get("process_active")):
            alerts.append(
                {
                    "severity": "warning",
                    "message": "no eval rows yet; trainer appears active (waiting for first eval)",
                }
            )
        else:
            alerts.append(
                {
                    "severity": "error",
                    "message": "no eval rows found in curriculum_log.jsonl",
                }
            )
        return alerts, prev_timesteps, stale_count

    if prev_timesteps is not None and current_step <= int(prev_timesteps):
        stale_count += 1
    else:
        stale_count = 0
    if stale_count >= stale_limit:
        alerts.append(
            {
                "severity": "warning",
                "message": f"timesteps unchanged for {stale_count} monitor cycles",
            }
        )
    prev_timesteps = current_step

    if not bool(local_status.get("process_active")):
        alerts.append({"severity": "warning", "message": "trainer process not detected for run"})

    reasons = list(local_status.get("recent_block_reasons", []))
    if len(reasons) >= 5 and len(set(reasons[-5:])) == 1:
        reason = reasons[-1]
        if reason != "none":
            alerts.append(
                {
                    "severity": "warning",
                    "message": f"repeated block_reason='{reason}' across last 5 eval rows",
                }
            )

    stage1_marker = str(local_status.get("markers", {}).get("EVAL_STAGE1_GUARD") or "")
    if "block=stage1_zero_progress" in stage1_marker:
        alerts.append(
            {
                "severity": "error",
                "message": "stage1 zero-progress guard triggered (rollback expected)",
            }
        )

    return alerts, prev_timesteps, stale_count


def _render_human_report(
    *,
    local_status: dict[str, Any],
    wandb_status: dict[str, Any] | None,
    alerts: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    latest_eval = local_status.get("latest_eval") or {}
    stage = latest_eval.get("stage", {}) if isinstance(latest_eval, dict) else {}
    lines.append("=" * 72)
    lines.append(f"Run: {local_status.get('run_dir')}")
    lines.append(f"Process active: {int(bool(local_status.get('process_active')))}")
    lines.append(
        "Latest eval: "
        f"id={latest_eval.get('eval_id')} "
        f"step={latest_eval.get('global_timesteps')} "
        f"stage={stage.get('idx')}:{stage.get('name')} "
        f"block={latest_eval.get('block_reason')}"
    )
    markers = local_status.get("markers", {})
    lines.append(f"SCORECARD: {markers.get('EVAL_SCORECARD')}")
    lines.append(f"BOSSFIGHT: {markers.get('EVAL_BOSSFIGHT')}")
    lines.append(f"QUALIFIER: {markers.get('EVAL_QUALIFIER')}")
    lines.append(f"STAGE1_GUARD: {markers.get('EVAL_STAGE1_GUARD')}")

    if wandb_status is not None:
        lines.append("-" * 72)
        lines.append(
            f"W&B: available={int(bool(wandb_status.get('available')))} "
            f"path={wandb_status.get('run_path')} state={wandb_status.get('state')}"
        )
        if wandb_status.get("url"):
            lines.append(f"W&B URL: {wandb_status.get('url')}")
        if wandb_status.get("summary"):
            lines.append(f"W&B Summary: {json.dumps(wandb_status.get('summary'), sort_keys=True)}")
        if wandb_status.get("error"):
            lines.append(f"W&B error: {wandb_status.get('error')}")

    lines.append("-" * 72)
    if alerts:
        lines.append("Alerts:")
        for alert in alerts:
            lines.append(f"[{alert['severity'].upper()}] {alert['message']}")
    else:
        lines.append("Alerts: none")
    lines.append("=" * 72)
    return "\n".join(lines)


def _exit_code_from_alerts(alerts: list[dict[str, str]]) -> int:
    if any(a.get("severity") == "error" for a in alerts):
        return 2
    if any(a.get("severity") == "warning" for a in alerts):
        return 1
    return 0


def _monitor_once(
    *,
    run_dir: Path,
    no_wandb: bool,
    wandb_run_path: str | None,
    as_json: bool,
) -> int:
    local_status = _build_local_status(run_dir)
    alerts, _, _ = _compute_alerts(
        local_status=local_status,
        prev_timesteps=None,
        stale_count=0,
        stale_limit=3,
    )

    wb_status = None
    if not no_wandb:
        run_path = _resolve_wandb_run_path(
            arg_run_path=wandb_run_path,
            latest_row=local_status.get("latest_eval"),
            run_meta=local_status.get("run_meta"),
        )
        wb_status = _fetch_wandb_status(run_path)

    if as_json:
        payload = {"local": local_status, "wandb": wb_status, "alerts": alerts}
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(
            _render_human_report(
                local_status=local_status, wandb_status=wb_status, alerts=alerts
            )
        )
    return _exit_code_from_alerts(alerts)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Monitor an AIGP curriculum run")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--refresh-s", type=int, default=15, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run one report and exit")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--wandb-run-path",
        type=str,
        default=None,
        help="Optional entity/project/run_id",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B API lookup")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run directory does not exist: {run_dir}", file=sys.stderr)
        raise SystemExit(2)

    if args.once:
        raise SystemExit(
            _monitor_once(
                run_dir=run_dir,
                no_wandb=bool(args.no_wandb),
                wandb_run_path=args.wandb_run_path,
                as_json=bool(args.json),
            )
        )

    prev_timesteps: int | None = None
    stale_count = 0
    stale_limit = 3
    while True:
        local_status = _build_local_status(run_dir)
        alerts, prev_timesteps, stale_count = _compute_alerts(
            local_status=local_status,
            prev_timesteps=prev_timesteps,
            stale_count=stale_count,
            stale_limit=stale_limit,
        )
        wb_status = None
        if not args.no_wandb:
            run_path = _resolve_wandb_run_path(
                arg_run_path=args.wandb_run_path,
                latest_row=local_status.get("latest_eval"),
                run_meta=local_status.get("run_meta"),
            )
            wb_status = _fetch_wandb_status(run_path)

        if args.json:
            payload = {"local": local_status, "wandb": wb_status, "alerts": alerts}
            print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
        else:
            print(
                _render_human_report(
                    local_status=local_status, wandb_status=wb_status, alerts=alerts
                ),
                flush=True,
            )
        time.sleep(max(1, int(args.refresh_s)))


if __name__ == "__main__":
    main()
