#!/usr/bin/env python3
"""Deterministic preflight checks for Kaggle AIGP runs."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_MISSING_IMPORT = 11
EXIT_MISSING_ASSETS = 12
EXIT_PATH_MISMATCH = 13
EXIT_INTERPRETER_MISMATCH = 14


def _path_exists(path: Path) -> bool:
    return path.exists()


def _build_default_asset_paths(repo_root: Path) -> list[Path]:
    primary_asset = (
        repo_root
        / "crazyflow"
        / "submodules"
        / "drone-models"
        / "drone_models"
        / "data"
        / "assets"
        / "cf21B"
        / "cf21B_prop-guards.stl"
    )
    sibling_asset = (
        repo_root.parent
        / "crazyflow"
        / "submodules"
        / "drone-models"
        / "drone_models"
        / "data"
        / "assets"
        / "cf21B"
        / "cf21B_prop-guards.stl"
    )
    if sibling_asset.exists():
        return [sibling_asset]
    return [primary_asset]


def _build_required_paths(repo_root: Path, *, config: str, curriculum: str) -> list[Path]:
    return [
        repo_root / "scripts" / "train_aigp_curriculum.py",
        repo_root / "scripts" / "build_readiness_report.py",
        repo_root / "config" / str(config),
        repo_root / "config" / str(curriculum),
    ]


def _check_interpreter(
    *,
    repo_root: Path,
    expected_python_bin: str | None,
    pythonpath_mode: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    if expected_python_bin is not None:
        expected = Path(expected_python_bin).resolve()
        actual = Path(sys.executable).resolve()
        if actual != expected:
            errors.append(f"python executable mismatch: actual={actual} expected={expected}")

    purelib = sysconfig.get_paths().get("purelib")
    if purelib and purelib not in sys.path:
        errors.append(f"site-packages path not present in sys.path: {purelib}")

    repo_root_in_sys_path = any(
        Path(entry).resolve() == repo_root.resolve() for entry in sys.path if entry
    )
    if pythonpath_mode == "repo-root" and not repo_root_in_sys_path:
        errors.append(f"repo root not present in sys.path: {repo_root}")

    details = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "purelib": purelib,
        "sys_path": list(sys.path),
        "repo_root_in_sys_path": repo_root_in_sys_path,
    }
    return (len(errors) == 0, errors, details)


def _check_required_paths(required_paths: list[Path]) -> tuple[bool, list[str], dict[str, Any]]:
    missing = [str(path) for path in required_paths if not _path_exists(path)]
    details = {"required_paths": [str(path) for path in required_paths], "missing": missing}
    return (len(missing) == 0, missing, details)


def _check_imports(modules: list[str]) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    resolved: dict[str, str] = {}
    for module_name in modules:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    module = importlib.import_module(module_name)
            module_path = getattr(module, "__file__", None)
            resolved[module_name] = str(module_path) if module_path is not None else "<builtin>"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    details = {"modules": modules, "resolved": resolved}
    return (len(errors) == 0, errors, details)


def _check_assets(asset_paths: list[Path]) -> tuple[bool, list[str], dict[str, Any]]:
    missing = [str(path) for path in asset_paths if not _path_exists(path)]
    details = {"asset_paths": [str(path) for path in asset_paths], "missing": missing}
    return (len(missing) == 0, missing, details)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run deterministic Kaggle preflight checks")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="aigp_stage0_single_gate.toml")
    parser.add_argument("--curriculum", default="aigp_curriculum_10stage_tuned_v2.toml")
    parser.add_argument("--asset-path", action="append", default=[])
    parser.add_argument("--expected-python-bin", default=None)
    parser.add_argument("--pythonpath-mode", choices=["repo-root", "none"], default="repo-root")
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checks: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    ok_interp, interp_errors, interp_details = _check_interpreter(
        repo_root=repo_root,
        expected_python_bin=args.expected_python_bin,
        pythonpath_mode=str(args.pythonpath_mode),
    )
    checks["interpreter"] = {"ok": bool(ok_interp), "details": interp_details}
    if not ok_interp:
        failures.append({"code": EXIT_INTERPRETER_MISMATCH, "errors": interp_errors})

    required_paths = _build_required_paths(
        repo_root, config=str(args.config), curriculum=str(args.curriculum)
    )
    ok_paths, path_errors, path_details = _check_required_paths(required_paths)
    checks["paths"] = {"ok": bool(ok_paths), "details": path_details}
    if not ok_paths:
        failures.append({"code": EXIT_PATH_MISMATCH, "errors": path_errors})

    ok_imports, import_errors, import_details = _check_imports(["lsy_drone_racing", "crazyflow"])
    checks["imports"] = {"ok": bool(ok_imports), "details": import_details}
    if not ok_imports:
        failures.append({"code": EXIT_MISSING_IMPORT, "errors": import_errors})

    asset_paths = [Path(path) for path in args.asset_path]
    if len(asset_paths) == 0:
        asset_paths = _build_default_asset_paths(repo_root)
    ok_assets, asset_errors, asset_details = _check_assets(asset_paths)
    checks["assets"] = {"ok": bool(ok_assets), "details": asset_details}
    if not ok_assets:
        failures.append({"code": EXIT_MISSING_ASSETS, "errors": asset_errors})

    if len(failures) == 0:
        exit_code = EXIT_OK
        errors: list[str] = []
    else:
        # deterministic priority for callers
        priority = [
            EXIT_INTERPRETER_MISMATCH,
            EXIT_PATH_MISMATCH,
            EXIT_MISSING_IMPORT,
            EXIT_MISSING_ASSETS,
        ]
        exit_code = EXIT_MISSING_IMPORT
        for candidate in priority:
            if any(item["code"] == candidate for item in failures):
                exit_code = candidate
                break
        errors = []
        for item in failures:
            errors.extend(list(item["errors"]))

    report = {
        "ok": exit_code == EXIT_OK,
        "exit_code": int(exit_code),
        "repo_root": str(repo_root),
        "checks": checks,
        "errors": errors,
    }

    if args.health_json is not None:
        args.health_json.parent.mkdir(parents=True, exist_ok=True)
        args.health_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if bool(args.json):
        print(json.dumps(report, indent=2, sort_keys=True))

    raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()
