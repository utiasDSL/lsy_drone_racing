#!/usr/bin/env python3
"""Static repository invariant checks for the AIGP training stack."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    """Invariant finding record."""

    kind: str
    message: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"function {name!r} not found")


def _keyword_defaults(func: ast.FunctionDef) -> dict[str, Any]:
    names = [a.arg for a in func.args.kwonlyargs]
    defaults = list(func.args.kw_defaults)
    out: dict[str, Any] = {}
    for name, value in zip(names, defaults, strict=True):
        out[name] = ast.literal_eval(value) if value is not None else None
    return out


def _assert_substrings(
    *,
    haystack: str,
    snippets: list[str],
    findings: list[Finding],
    prefix: str,
) -> None:
    for snippet in snippets:
        if snippet not in haystack:
            findings.append(Finding("error", f"{prefix} missing snippet: {snippet}"))


def run_checks(repo_root: Path) -> list[Finding]:
    """Evaluate all repository invariants and return findings."""
    findings: list[Finding] = []

    trainer_path = repo_root / "scripts" / "train_aigp_curriculum.py"
    curriculum_path = repo_root / "lsy_drone_racing" / "aigp" / "curriculum.py"
    monitor_path = repo_root / "scripts" / "monitor_aigp_run.py"
    kaggle_session_path = repo_root / "scripts" / "run_aigp_kaggle_session.py"
    kaggle_push_path = repo_root / "scripts" / "push_aigp_kaggle_kernel.py"
    readiness_path = repo_root / "scripts" / "check_qualifier_readiness.py"
    readiness_builder_path = repo_root / "scripts" / "build_readiness_report.py"
    compare_runs_path = repo_root / "scripts" / "compare_aigp_runs.py"
    agents_path = repo_root / "AGENTS.md"
    docs_index_path = repo_root / "docs" / "aigp" / "index.md"

    required_paths = [
        trainer_path,
        curriculum_path,
        monitor_path,
        kaggle_session_path,
        kaggle_push_path,
        readiness_path,
        readiness_builder_path,
        compare_runs_path,
        agents_path,
        docs_index_path,
        repo_root / "tests" / "unit" / "aigp" / "test_curriculum.py",
        repo_root / "tests" / "unit" / "aigp" / "test_train_wandb.py",
        repo_root / "tests" / "unit" / "aigp" / "test_monitor_aigp_run.py",
        repo_root / "tests" / "unit" / "aigp" / "test_run_aigp_kaggle_session.py",
        repo_root / "tests" / "unit" / "aigp" / "test_push_aigp_kaggle_kernel.py",
    ]
    for path in required_paths:
        if not path.exists():
            findings.append(Finding("error", f"required file missing: {path}"))

    if not trainer_path.exists() or not curriculum_path.exists() or not monitor_path.exists():
        return findings

    trainer_text = _read_text(trainer_path)
    trainer_tree = ast.parse(trainer_text)
    train_fn = _find_function(trainer_tree, "train")
    defaults = _keyword_defaults(train_fn)

    expected_defaults = {
        "force_advance_mode": "if_passing",
        "obs_mode": "privileged",
        "wandb_enabled": False,
        "wandb_mode": "online",
        "allow_append": False,
    }
    for key, expected in expected_defaults.items():
        got = defaults.get(key)
        if got != expected:
            findings.append(
                Finding(
                    "error",
                    f"train() default for {key!r} is {got!r}, expected {expected!r}",
                )
            )

    _assert_substrings(
        haystack=trainer_text,
        snippets=[
            'if log_path.exists() and not bool(allow_append):',
            'raise FileExistsError(',
            'if eval_id <= last_eval_id:',
            'if global_timesteps <= last_global_timesteps:',
            '"schema_version": 2,',
            '"run_id": run_id,',
            '"eval_id": int(eval_id),',
            '"global_timesteps": global_timesteps,',
            '"scorecard": scorecard,',
            '"block_reason": row_block_reason,',
            '"forced_advance": bool(forced_advance),',
            '"qualifier_eval": qualifier_eval,',
            '"train_snapshot": train_snapshot_latest,',
            '"wandb": _wandb_info_payload(wandb_state),',
            '_parse_force_advance_mode(force_advance_mode)',
        ],
        findings=findings,
        prefix="trainer",
    )

    curriculum_text = _read_text(curriculum_path)
    _assert_substrings(
        haystack=curriculum_text,
        snippets=[
            '"gate_success_ok": bool(gate_success_ok),',
            '"gate_success_rate": gate_success_rate,',
            '"gate_success_threshold": gate_success_threshold,',
            '"gate_min_episodes_ok": bool(gate_min_episodes_ok),',
            '"gate_stage_episodes": gate_stage_episodes,',
            '"gate_min_episodes_required": gate_min_episodes_required,',
            '"gate_stability_ok": bool(gate_stability_ok),',
            '"gate_recovery_clear": bool(gate_recovery_clear),',
            '"block_reason": str(block_reason),',
            '"bossfight_ok": bool(bossfight_ok),',
            '"bossfight_tracks_total": bossfight_tracks_total,',
            '"bossfight_tracks_covered": bossfight_tracks_covered,',
            '"bossfight_success_rate_min": bossfight_success_rate_min,',
            '"bossfight_success_rate_bottom20_mean": bossfight_success_rate_bottom20_mean,',
        ],
        findings=findings,
        prefix="curriculum",
    )

    monitor_text = _read_text(monitor_path)
    _assert_substrings(
        haystack=monitor_text,
        snippets=[
            'markers=["EVAL_SCORECARD", "EVAL_BOSSFIGHT", "EVAL_QUALIFIER", "EVAL_STAGE1_GUARD"]',
            "def _exit_code_from_alerts",
            "return 2",
            "return 1",
            "return 0",
        ],
        findings=findings,
        prefix="monitor",
    )

    return findings


def main() -> None:
    """CLI entrypoint."""
    repo_root = Path(__file__).resolve().parents[1]
    findings = run_checks(repo_root)
    errors = [f for f in findings if f.kind == "error"]
    warnings = [f for f in findings if f.kind == "warning"]

    print("AIGP repository invariant check")
    print(f"- repo: {repo_root}")
    print(f"- errors: {len(errors)}")
    print(f"- warnings: {len(warnings)}")

    for finding in findings:
        label = "ERROR" if finding.kind == "error" else "WARN"
        print(f"[{label}] {finding.message}")

    if errors:
        raise SystemExit(1)

    print("Invariant check passed.")


if __name__ == "__main__":
    main()
