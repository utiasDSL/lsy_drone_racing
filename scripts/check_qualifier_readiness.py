#!/usr/bin/env python3
"""Evaluate qualifier readiness from curriculum JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "min_stage_idx": 0,
        "min_success_rate": 0.70,
        "min_course_completion_rate": 0.70,
        "max_robustness_std": 0.20,
        "min_track_coverage_ratio": 1.0,
        "min_seed_coverage_ratio": 1.0,
        "require_scorecard_pass": True,
        "fail_on_forced_advance_without_pass": True,
    },
    "stage_gate": {
        "min_stage_idx": 0,
        "min_success_rate": 0.60,
        "min_course_completion_rate": 0.60,
        "max_robustness_std": 0.25,
        "min_track_coverage_ratio": 1.0,
        "min_seed_coverage_ratio": 1.0,
        "require_scorecard_pass": True,
        "fail_on_forced_advance_without_pass": True,
    },
    "qualifier_strict": {
        "min_stage_idx": 0,
        "min_success_rate": 0.80,
        "min_course_completion_rate": 0.80,
        "max_robustness_std": 0.15,
        "min_track_coverage_ratio": 1.0,
        "min_seed_coverage_ratio": 1.0,
        "require_scorecard_pass": True,
        "fail_on_forced_advance_without_pass": True,
    },
}


@dataclass(frozen=True)
class Gate:
    """Readiness gate result."""

    name: str
    ok: bool
    value: float | int | bool | None
    threshold: float | int | bool | None
    comparator: str


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [r for r in rows if "eval_id" in r and "global_timesteps" in r]
    return sorted(
        out,
        key=lambda r: (int(r.get("eval_id", 0)), int(r.get("global_timesteps", 0))),
    )


def read_eval_rows_from_run(run_dir: Path) -> list[dict[str, Any]]:
    """Read sorted eval rows from one run directory."""
    log_path = run_dir / "curriculum_log.jsonl"
    return _eval_rows(_read_jsonl_rows(log_path))


def _float(value: Any, default: float = 0.0) -> float:  # noqa: ANN401
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:  # noqa: ANN401
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _all_schema_v2(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all(_int(r.get("schema_version"), default=0) == 2 for r in rows)


def _gate_ge(name: str, value: float | int, threshold: float | int) -> Gate:
    return Gate(
        name=name,
        ok=float(value) >= float(threshold),
        value=value,
        threshold=threshold,
        comparator=">=",
    )


def _gate_le(name: str, value: float | int, threshold: float | int) -> Gate:
    return Gate(
        name=name,
        ok=float(value) <= float(threshold),
        value=value,
        threshold=threshold,
        comparator="<=",
    )


def _gate_bool(name: str, value: bool, expected: bool = True) -> Gate:
    return Gate(
        name=name,
        ok=bool(value) is bool(expected),
        value=bool(value),
        threshold=bool(expected),
        comparator="==",
    )


def _resolve_profile(name: str) -> dict[str, Any]:
    normalized = str(name).strip().lower()
    if normalized not in PROFILE_PRESETS:
        raise ValueError(
            f"unknown profile={name!r}; expected one of {sorted(PROFILE_PRESETS)}"
        )
    return dict(PROFILE_PRESETS[normalized])


def _profile_with_overrides(
    profile: dict[str, Any],
    *,
    min_stage_idx: int | None,
    min_success_rate: float | None,
    min_course_completion_rate: float | None,
    max_robustness_std: float | None,
    min_track_coverage_ratio: float | None,
    min_seed_coverage_ratio: float | None,
    require_scorecard_pass: bool | None,
    fail_on_forced_advance_without_pass: bool | None,
) -> dict[str, Any]:
    out = dict(profile)
    if min_stage_idx is not None:
        out["min_stage_idx"] = int(min_stage_idx)
    if min_success_rate is not None:
        out["min_success_rate"] = float(min_success_rate)
    if min_course_completion_rate is not None:
        out["min_course_completion_rate"] = float(min_course_completion_rate)
    if max_robustness_std is not None:
        out["max_robustness_std"] = float(max_robustness_std)
    if min_track_coverage_ratio is not None:
        out["min_track_coverage_ratio"] = float(min_track_coverage_ratio)
    if min_seed_coverage_ratio is not None:
        out["min_seed_coverage_ratio"] = float(min_seed_coverage_ratio)
    if require_scorecard_pass is not None:
        out["require_scorecard_pass"] = bool(require_scorecard_pass)
    if fail_on_forced_advance_without_pass is not None:
        out["fail_on_forced_advance_without_pass"] = bool(
            fail_on_forced_advance_without_pass
        )
    return out


def _build_gates(
    *,
    window_rows: list[dict[str, Any]],
    latest_row: dict[str, Any],
    min_stage_idx: int,
    min_success_rate: float,
    min_course_completion_rate: float,
    max_robustness_std: float,
    min_track_coverage_ratio: float,
    min_seed_coverage_ratio: float,
    require_scorecard_pass: bool,
    fail_on_forced_advance_without_pass: bool,
) -> tuple[list[Gate], dict[str, Any]]:
    success_rates = [_float(r.get("eval", {}).get("success_rate")) for r in window_rows]
    ccr_values = [
        _float(r.get("qualifier_eval", {}).get("course_completion_rate"))
        for r in window_rows
    ]
    robust_values = [
        _float(r.get("qualifier_eval", {}).get("robustness_std")) for r in window_rows
    ]
    track_cov_values = [
        _float(r.get("qualifier_eval", {}).get("track_coverage", {}).get("ratio"))
        for r in window_rows
    ]
    seed_cov_values = [
        _float(r.get("qualifier_eval", {}).get("seed_coverage", {}).get("ratio"))
        for r in window_rows
    ]

    mean_success = _mean(success_rates)
    mean_ccr = _mean(ccr_values)
    max_robust = max(robust_values) if robust_values else 0.0
    min_track_cov = min(track_cov_values) if track_cov_values else 0.0
    min_seed_cov = min(seed_cov_values) if seed_cov_values else 0.0

    latest_stage_idx = _int(latest_row.get("stage", {}).get("idx"), default=-1)
    latest_scorecard_pass = bool(latest_row.get("scorecard", {}).get("pass", False))

    forced_advance_violation_count = sum(
        1
        for r in window_rows
        if bool(r.get("forced_advance", False))
        and not bool(r.get("scorecard", {}).get("pass", False))
    )

    gates: list[Gate] = [
        _gate_bool("schema_v2_window", _all_schema_v2(window_rows)),
        _gate_ge("stage_idx", latest_stage_idx, min_stage_idx),
        _gate_ge("mean_success_rate", mean_success, min_success_rate),
        _gate_ge("mean_course_completion_rate", mean_ccr, min_course_completion_rate),
        _gate_le("max_robustness_std", max_robust, max_robustness_std),
        _gate_ge("min_track_coverage_ratio", min_track_cov, min_track_coverage_ratio),
        _gate_ge("min_seed_coverage_ratio", min_seed_cov, min_seed_coverage_ratio),
    ]
    if require_scorecard_pass:
        gates.append(_gate_bool("latest_scorecard_pass", latest_scorecard_pass))
    if fail_on_forced_advance_without_pass:
        gates.append(
            _gate_bool(
                "forced_advance_without_pass",
                bool(forced_advance_violation_count == 0),
            )
        )

    metrics = {
        "window_size": len(window_rows),
        "latest_eval_id": _int(latest_row.get("eval_id"), default=0),
        "latest_global_timesteps": _int(latest_row.get("global_timesteps"), default=0),
        "latest_stage_idx": latest_stage_idx,
        "latest_stage_name": str(latest_row.get("stage", {}).get("name")),
        "mean_success_rate": mean_success,
        "mean_course_completion_rate": mean_ccr,
        "max_robustness_std": max_robust,
        "min_track_coverage_ratio": min_track_cov,
        "min_seed_coverage_ratio": min_seed_cov,
        "latest_scorecard_pass": latest_scorecard_pass,
        "forced_advance_without_pass_count_window": int(forced_advance_violation_count),
    }
    return gates, metrics


def _recommendation(*, ready: bool, metrics: dict[str, Any], failing_reasons: list[str]) -> str:
    if bool(ready):
        return "promote"
    if int(metrics.get("forced_advance_without_pass_count_window", 0)) > 0:
        return "rollback_candidate"
    if "schema_v2_window" in set(failing_reasons):
        return "rollback_candidate"
    if int(metrics.get("latest_stage_idx", 0)) >= 1 and float(
        metrics.get("mean_success_rate", 0.0)
    ) < 0.10:
        return "rollback_candidate"
    return "hold"


def evaluate_run(
    *,
    run_dir: Path,
    window_evals: int,
    min_stage_idx: int,
    min_success_rate: float,
    min_course_completion_rate: float,
    max_robustness_std: float,
    min_track_coverage_ratio: float,
    min_seed_coverage_ratio: float,
    require_scorecard_pass: bool,
    fail_on_forced_advance_without_pass: bool,
    profile_name: str = "custom",
    wandb_run_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Evaluate one run directory against qualifier readiness gates."""
    log_path = run_dir / "curriculum_log.jsonl"
    rows = _read_jsonl_rows(log_path)
    eval_rows = _eval_rows(rows)
    if not eval_rows:
        return 2, {
            "ready": False,
            "error": f"no eval rows found in {log_path}",
            "run_dir": str(run_dir),
        }

    window = eval_rows[-max(1, int(window_evals)) :]
    latest = window[-1]

    gates, metrics = _build_gates(
        window_rows=window,
        latest_row=latest,
        min_stage_idx=min_stage_idx,
        min_success_rate=min_success_rate,
        min_course_completion_rate=min_course_completion_rate,
        max_robustness_std=max_robustness_std,
        min_track_coverage_ratio=min_track_coverage_ratio,
        min_seed_coverage_ratio=min_seed_coverage_ratio,
        require_scorecard_pass=require_scorecard_pass,
        fail_on_forced_advance_without_pass=fail_on_forced_advance_without_pass,
    )

    gate_rows = [
        {
            "name": g.name,
            "ok": g.ok,
            "value": g.value,
            "threshold": g.threshold,
            "comparator": g.comparator,
        }
        for g in gates
    ]
    failing_reasons = [str(g["name"]) for g in gate_rows if not bool(g["ok"])]
    ready = all(bool(g["ok"]) for g in gate_rows)
    gate_pass_ratio = (
        float(sum(1 for g in gate_rows if bool(g["ok"]))) / float(len(gate_rows))
        if gate_rows
        else 0.0
    )

    run_meta = _read_json(run_dir / "run_meta.json")
    resolved_wandb_run_path = wandb_run_path
    if resolved_wandb_run_path is None:
        if isinstance(run_meta, dict) and isinstance(run_meta.get("wandb"), dict):
            resolved_wandb_run_path = run_meta["wandb"].get("run_path")
        if resolved_wandb_run_path is None and isinstance(latest.get("wandb"), dict):
            resolved_wandb_run_path = latest["wandb"].get("run_path")

    payload = {
        "ready": bool(ready),
        "profile": str(profile_name),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "source": {"wandb_run_path": resolved_wandb_run_path},
        "run_meta": run_meta,
        "metrics": {**metrics, "gate_pass_ratio": gate_pass_ratio},
        "gates": gate_rows,
        "failing_reasons": failing_reasons,
        "recommendation": _recommendation(
            ready=bool(ready),
            metrics={**metrics, "gate_pass_ratio": gate_pass_ratio},
            failing_reasons=failing_reasons,
        ),
    }
    return (0 if ready else 1), payload


def evaluate_run_with_profile(
    *,
    run_dir: Path,
    profile: str = "default",
    window_evals: int = 5,
    wandb_run_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Evaluate one run using a named readiness profile."""
    profile_cfg = _resolve_profile(profile)
    if overrides:
        profile_cfg = _profile_with_overrides(profile_cfg, **overrides)

    return evaluate_run(
        run_dir=run_dir,
        window_evals=max(1, int(window_evals)),
        min_stage_idx=int(profile_cfg["min_stage_idx"]),
        min_success_rate=float(profile_cfg["min_success_rate"]),
        min_course_completion_rate=float(profile_cfg["min_course_completion_rate"]),
        max_robustness_std=float(profile_cfg["max_robustness_std"]),
        min_track_coverage_ratio=float(profile_cfg["min_track_coverage_ratio"]),
        min_seed_coverage_ratio=float(profile_cfg["min_seed_coverage_ratio"]),
        require_scorecard_pass=bool(profile_cfg["require_scorecard_pass"]),
        fail_on_forced_advance_without_pass=bool(
            profile_cfg["fail_on_forced_advance_without_pass"]
        ),
        profile_name=str(profile),
        wandb_run_path=wandb_run_path,
    )


def _emit_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Check AIGP qualifier readiness from run artifacts"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default="default")
    parser.add_argument(
        "--window-evals",
        type=int,
        default=5,
        help="Number of most recent eval rows",
    )
    parser.add_argument("--min-stage-idx", type=int, default=None)
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--min-course-completion-rate", type=float, default=None)
    parser.add_argument("--max-robustness-std", type=float, default=None)
    parser.add_argument("--min-track-coverage-ratio", type=float, default=None)
    parser.add_argument("--min-seed-coverage-ratio", type=float, default=None)
    parser.add_argument("--require-scorecard-pass", action="store_true")
    parser.add_argument("--allow-forced-advance-without-pass", action="store_true")
    parser.add_argument("--wandb-run-path", type=str, default=None)
    parser.add_argument("--emit-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run directory does not exist: {run_dir}")
        raise SystemExit(2)

    overrides = {
        "min_stage_idx": args.min_stage_idx,
        "min_success_rate": args.min_success_rate,
        "min_course_completion_rate": args.min_course_completion_rate,
        "max_robustness_std": args.max_robustness_std,
        "min_track_coverage_ratio": args.min_track_coverage_ratio,
        "min_seed_coverage_ratio": args.min_seed_coverage_ratio,
        "require_scorecard_pass": True if bool(args.require_scorecard_pass) else None,
        "fail_on_forced_advance_without_pass": (
            False if bool(args.allow_forced_advance_without_pass) else None
        ),
    }

    code, payload = evaluate_run_with_profile(
        run_dir=run_dir,
        profile=str(args.profile),
        window_evals=max(1, int(args.window_evals)),
        wandb_run_path=args.wandb_run_path,
        overrides=overrides,
    )

    if args.emit_file is not None:
        _emit_report(args.emit_file.resolve(), payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if code == 2:
            print(f"READINESS_ERROR {payload.get('error')}")
        else:
            state = "READY" if code == 0 else "NOT_READY"
            metrics = payload.get("metrics", {})
            print(
                "READINESS "
                f"status={state} "
                f"profile={payload.get('profile')} "
                f"eval_id={metrics.get('latest_eval_id')} "
                f"step={metrics.get('latest_global_timesteps')} "
                f"stage={metrics.get('latest_stage_idx')}:{metrics.get('latest_stage_name')} "
                f"sr_mean={float(metrics.get('mean_success_rate', 0.0)):.3f} "
                f"ccr_mean={float(metrics.get('mean_course_completion_rate', 0.0)):.3f} "
                f"robust_max={float(metrics.get('max_robustness_std', 0.0)):.3f} "
                f"gate_pass_ratio={float(metrics.get('gate_pass_ratio', 0.0)):.3f} "
                f"recommendation={payload.get('recommendation')}"
            )
            for gate in payload.get("gates", []):
                print(
                    "GATE "
                    f"name={gate['name']} "
                    f"ok={int(bool(gate['ok']))} "
                    f"value={gate['value']} "
                    f"cmp={gate['comparator']} "
                    f"threshold={gate['threshold']}"
                )

    raise SystemExit(code)


if __name__ == "__main__":
    main()
