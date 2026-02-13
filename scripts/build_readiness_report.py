#!/usr/bin/env python3
"""Build and persist readiness reports for an AIGP run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__" and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_readiness_checker() -> Any:
    try:
        from scripts.check_qualifier_readiness import evaluate_run_with_profile
        return evaluate_run_with_profile
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.check_qualifier_readiness import evaluate_run_with_profile
        return evaluate_run_with_profile


evaluate_run_with_profile = _load_readiness_checker()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build readiness report files for a run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--window-evals", type=int, default=5)
    parser.add_argument("--wandb-run-path", type=str, default=None)
    parser.add_argument("--out-latest", type=Path, default=None)
    parser.add_argument("--out-history", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run directory does not exist: {run_dir}")
        raise SystemExit(2)

    code, payload = evaluate_run_with_profile(
        run_dir=run_dir,
        profile=str(args.profile),
        window_evals=max(1, int(args.window_evals)),
        wandb_run_path=args.wandb_run_path,
    )

    readiness_dir = run_dir / "readiness"
    latest_path = (
        args.out_latest.resolve()
        if args.out_latest is not None
        else readiness_dir / "latest.json"
    )
    history_path = (
        args.out_history.resolve()
        if args.out_history is not None
        else readiness_dir / "history.jsonl"
    )

    _write_json(latest_path, payload)
    _append_jsonl(
        history_path,
        {
            "generated_at_unix_s": float(time.time()),
            "profile": str(args.profile),
            "window_evals": int(args.window_evals),
            "exit_code": int(code),
            "ready": bool(payload.get("ready", False)),
            "recommendation": payload.get("recommendation"),
            "run_dir": str(run_dir),
            "report": payload,
        },
    )

    if args.json:
        print(
            json.dumps(
                {
                    "latest": str(latest_path),
                    "history": str(history_path),
                    "exit_code": int(code),
                    "report": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            "READINESS_REPORT "
            f"run_dir={run_dir} "
            f"ready={int(bool(payload.get('ready', False)))} "
            f"recommendation={payload.get('recommendation')} "
            f"latest={latest_path} "
            f"history={history_path}"
        )

    raise SystemExit(code)


if __name__ == "__main__":
    main()
