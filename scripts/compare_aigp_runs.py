#!/usr/bin/env python3
"""Compare multiple AIGP runs using readiness-profile metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.check_qualifier_readiness import evaluate_run_with_profile, read_eval_rows_from_run


def _forced_advance_violations(rows: list[dict[str, Any]]) -> int:
    return int(
        sum(
            1
            for r in rows
            if bool(r.get("forced_advance", False))
            and not bool(r.get("scorecard", {}).get("pass", False))
        )
    )


def _as_float(value: Any) -> float:  # noqa: ANN401
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_int(value: Any) -> int:  # noqa: ANN401
    try:
        return int(value)
    except Exception:
        return 0


def _rank_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if bool(entry.get("ready", False)) else 1,
        -_as_float(entry.get("mean_course_completion_rate")),
        _as_float(entry.get("max_robustness_std")),
        -_as_int(entry.get("latest_stage_idx")),
        -_as_int(entry.get("latest_global_timesteps")),
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| rank | run | ready | rec | stage | step | ccr_mean | sr_mean | "
        "robust_max | gate_pass | forced_viol_total |\n"
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    body: list[str] = []
    for i, row in enumerate(rows, start=1):
        body.append(
            "| "
            f"{i} | {Path(str(row['run_dir'])).name} | {int(bool(row['ready']))} "
            f"| {row['recommendation']} | {int(row['latest_stage_idx'])} "
            f"| {int(row['latest_global_timesteps'])} | "
            f"{float(row['mean_course_completion_rate']):.3f} | "
            f"{float(row['mean_success_rate']):.3f} | "
            f"{float(row['max_robustness_std']):.3f} | "
            f"{float(row['gate_pass_ratio']):.3f} | "
            f"{int(row['forced_advance_violations_total'])} |"
        )
    return header + "\n" + "\n".join(body)


def compare_runs(
    *,
    run_dirs: list[Path],
    profile: str,
    window_evals: int,
    wandb_run_paths: list[str] | None,
) -> dict[str, Any]:
    """Compare runs and return ranked payload."""
    if wandb_run_paths and len(wandb_run_paths) not in {0, len(run_dirs)}:
        raise ValueError("--wandb-run-path count must be either 0 or match --run-dir count")

    rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs):
        resolved = run_dir.resolve()
        wb_path = None
        if wandb_run_paths and len(wandb_run_paths) == len(run_dirs):
            wb_path = wandb_run_paths[idx]

        code, report = evaluate_run_with_profile(
            run_dir=resolved,
            profile=str(profile),
            window_evals=int(window_evals),
            wandb_run_path=wb_path,
        )
        eval_rows = read_eval_rows_from_run(resolved)
        metrics = dict(report.get("metrics", {}))

        rows.append(
            {
                "run_dir": str(resolved),
                "profile": str(profile),
                "exit_code": int(code),
                "ready": bool(report.get("ready", False)),
                "recommendation": str(report.get("recommendation", "hold")),
                "latest_stage_idx": int(metrics.get("latest_stage_idx", -1)),
                "latest_stage_name": str(metrics.get("latest_stage_name", "")),
                "latest_global_timesteps": int(metrics.get("latest_global_timesteps", 0)),
                "mean_success_rate": float(metrics.get("mean_success_rate", 0.0)),
                "mean_course_completion_rate": float(
                    metrics.get("mean_course_completion_rate", 0.0)
                ),
                "max_robustness_std": float(metrics.get("max_robustness_std", 0.0)),
                "min_track_coverage_ratio": float(
                    metrics.get("min_track_coverage_ratio", 0.0)
                ),
                "min_seed_coverage_ratio": float(metrics.get("min_seed_coverage_ratio", 0.0)),
                "gate_pass_ratio": float(metrics.get("gate_pass_ratio", 0.0)),
                "forced_advance_violations_window": int(
                    metrics.get("forced_advance_without_pass_count_window", 0)
                ),
                "forced_advance_violations_total": _forced_advance_violations(eval_rows),
                "gates": report.get("gates", []),
                "failing_reasons": report.get("failing_reasons", []),
                "source": report.get("source", {}),
            }
        )

    ranking = sorted(rows, key=_rank_key)
    return {
        "generated_at_unix_s": float(time.time()),
        "profile": str(profile),
        "window_evals": int(window_evals),
        "runs": rows,
        "ranking": ranking,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare AIGP runs by readiness profile")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--window-evals", type=int, default=5)
    parser.add_argument("--wandb-run-path", action="append", default=[])
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_dirs = [Path(v) for v in args.run_dir]
    if len(run_dirs) < 2:
        raise SystemExit("provide at least two --run-dir values")

    payload = compare_runs(
        run_dirs=run_dirs,
        profile=str(args.profile),
        window_evals=max(1, int(args.window_evals)),
        wandb_run_paths=[str(v) for v in args.wandb_run_path],
    )

    if args.out_prefix is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_prefix = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "run_audit"
            / f"{ts}_comparison"
        )
    else:
        out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    ranked = list(payload.get("ranking", []))
    md_text = "# AIGP Run Comparison\n\n"
    md_text += f"- profile: `{payload.get('profile')}`\n"
    md_text += f"- window_evals: `{payload.get('window_evals')}`\n"
    md_text += f"- generated_at_unix_s: `{payload.get('generated_at_unix_s')}`\n\n"
    md_text += _markdown_table(ranked)
    md_text += "\n"
    out_md.write_text(md_text, encoding="utf-8")

    result = {
        "json_path": str(out_json),
        "markdown_path": str(out_md),
        "top_run": (ranked[0]["run_dir"] if ranked else None),
        "ranking_count": len(ranked),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "RUN_COMPARISON "
            f"json={out_json} md={out_md} "
            f"top_run={result['top_run']} count={len(ranked)}"
        )


if __name__ == "__main__":
    main()
