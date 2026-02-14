#!/usr/bin/env python3
"""Build and optionally push a Kaggle kernel for AIGP curriculum training."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE_INCLUDE = (
    "lsy_drone_racing",
    "scripts",
    "config",
    "pyproject.toml",
    "README.md",
    "AIGP_CHECKLIST.md",
    "crazyflow/crazyflow",
    "crazyflow/pyproject.toml",
    "crazyflow/submodules/drone-models/drone_models",
    "crazyflow/submodules/drone-models/pyproject.toml",
    "crazyflow/submodules/drone-controllers/drone_controllers",
    "crazyflow/submodules/drone-controllers/pyproject.toml",
)

_SKIP_PARTS = {
    ".git",
    ".pixi",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "reports",
    "kaggle",
    "docs",
    "tests",
    "benchmark",
}
_SKIP_SUFFIXES = (".pyc", ".pyo", ".DS_Store")

CANONICAL_SOURCE_DATASET = "massimoraso/aigp-source-blob-fix6"
CANONICAL_WHEELHOUSE_DATASET = "massimoraso/aigp-wheelhouse"
CANONICAL_ARCHIVE_NAME = "lsy_drone_racing_src_fix6.bin"
KAGGLE_NOTEBOOK_SOURCE_LIMIT_BYTES = 1_000_000
KAGGLE_NOTEBOOK_SOURCE_SAFETY_MARGIN_BYTES = 75_000

_EXCLUDED_ASSET_FILENAMES = {
    "cf21B_full.stl",
    "cf21B_no-prop.stl",
    "cf21B_header.stl",
}
_EXCLUDED_ASSET_DIR_PREFIX = (
    "lsy_drone_racing/crazyflow/submodules/drone-models/drone_models/data/assets/cf2x/",
)

DEFAULT_FULL_KERNEL_ID = "massimoraso/aigp-kaggle-primary-trainer"
DEFAULT_FULL_TITLE = "AIGP Kaggle Primary Trainer"
DEFAULT_FULL_RUN_OUT = "runs/aigp_kaggle_primary"

DEFAULT_SMOKE_KERNEL_ID = "massimoraso/aigp-kaggle-smoke"
DEFAULT_SMOKE_TITLE = "AIGP Kaggle Smoke"
DEFAULT_SMOKE_RUN_OUT = "runs/aigp_kaggle_smoke"


def _json_bool(value: bool) -> bool:
    return bool(value)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _canonical_dataset_sources(values: list[str]) -> list[str]:
    merged: list[str] = []
    for item in values:
        if item not in merged:
            merged.append(item)
    for required in (CANONICAL_SOURCE_DATASET, CANONICAL_WHEELHOUSE_DATASET):
        if required not in merged:
            merged.append(required)
    return merged


def _build_notebook(
    *,
    repo_source_mode: str,
    repo_archive_name: str,
    repo_archive_b64: str,
    repo_url: str,
    branch: str,
    run_out: str,
    max_walltime_s: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    tournament_mode: bool = False,
    obs_mode: str | None = None,
    qualifier_eval_profile: str | None = None,
    readiness_profile: str | None = None,
    resume_from_input: str | None,
    extra_train_args: list[str],
    num_envs: int = 16,
    timesteps_per_stage: int = 10_000_000,
    eval_repeats: int = 2,
    eval_seed_stride: int = 97,
    preflight_only: bool = False,
    strict_preflight: bool = True,
) -> dict[str, Any]:
    """Build a deterministic notebook that launches AIGP training on Kaggle."""
    if str(repo_source_mode) != "bundle":
        raise ValueError("canonical Kaggle lane requires repo_source_mode='bundle'")

    optional_run_arg_lines: list[str] = []
    if bool(tournament_mode):
        optional_run_arg_lines.append("cmd += ['--tournament-mode', 'true']")
    if obs_mode:
        optional_run_arg_lines.append(f"cmd += ['--obs-mode', {obs_mode!r}]")
    if qualifier_eval_profile:
        optional_run_arg_lines.append(
            f"cmd += ['--qualifier-eval-profile', {qualifier_eval_profile!r}]"
        )
    if readiness_profile:
        optional_run_arg_lines.append(
            f"cmd += ['--readiness-profile', {readiness_profile!r}]"
        )
    optional_run_arg_code = "\n".join(optional_run_arg_lines) if optional_run_arg_lines else ""

    setup_cell = f"""import base64
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

REPO_ARCHIVE_NAME = {repo_archive_name!r}
REPO_ARCHIVE_B64 = {repo_archive_b64!r}
REPO_DIR = pathlib.Path('/kaggle/working/lsy_drone_racing')


def run_with_retries(cmd, attempts=5, sleep_s=15):
    last = None
    for idx in range(1, attempts + 1):
        try:
            print(f"[setup] run ({{idx}}/{{attempts}}):", " ".join(cmd))
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError as exc:
            last = exc
            print(f"[setup] command failed rc={{exc.returncode}}")
            if idx < attempts:
                time.sleep(sleep_s)
    raise last


def find_repo_archive(name):
    candidates = [pathlib.Path.cwd() / name, pathlib.Path('/kaggle/working') / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    kaggle_input = pathlib.Path('/kaggle/input')
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob(name):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f'archive not found: {{name}}')


def find_wheelhouse_dir():
    kaggle_input = pathlib.Path('/kaggle/input')
    if not kaggle_input.exists():
        return None
    for candidate in kaggle_input.iterdir():
        if not candidate.is_dir():
            continue
        if any(candidate.glob('*.whl')):
            return candidate
        for subdir in candidate.rglob('*'):
            if subdir.is_dir() and any(subdir.glob('*.whl')):
                return subdir
    return None


if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

if REPO_ARCHIVE_B64:
    archive_path = pathlib.Path('/kaggle/working') / REPO_ARCHIVE_NAME
    archive_path.write_bytes(base64.b64decode(REPO_ARCHIVE_B64))
    print('[setup] using embedded source archive:', archive_path)
else:
    archive_path = find_repo_archive(REPO_ARCHIVE_NAME)
    print('[setup] using input source archive:', archive_path)

with tarfile.open(archive_path, 'r:gz') as src:
    src.extractall(path='/kaggle/working')

if not (REPO_DIR / "pyproject.toml").exists():
    nested_repo = REPO_DIR / "lsy_drone_racing"
    if (nested_repo / "pyproject.toml").exists():
        print("[setup] using nested repo root:", nested_repo)
        REPO_DIR = nested_repo

print('[setup] resolved repo root:', REPO_DIR)
os.chdir(REPO_DIR)

wheelhouse_dir = find_wheelhouse_dir()
if wheelhouse_dir is None:
    raise RuntimeError('offline wheelhouse not found under /kaggle/input')
print('[setup] installing offline wheelhouse:', wheelhouse_dir)
run_with_retries(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-index',
        '--find-links',
        str(wheelhouse_dir),
        'mujoco',
        'mujoco-mjx',
        'casadi',
        'array-api-compat',
        'array-api-extra',
    ],
    attempts=2,
    sleep_s=10,
)

local_dep_paths = [
    REPO_DIR / 'crazyflow' / 'submodules' / 'drone-models',
    REPO_DIR / 'crazyflow' / 'submodules' / 'drone-controllers',
    REPO_DIR / 'crazyflow',
]
for dep_path in local_dep_paths:
    if not dep_path.exists():
        continue
    dep_install_cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '-e',
        str(dep_path),
        '--no-deps',
        '--no-build-isolation',
    ]
    print('[setup] installing local dep:', dep_path)
    run_with_retries(dep_install_cmd, attempts=2, sleep_s=10)

print('[setup] canonical offline install; skipping network package installs')
run_with_retries(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '-e',
        '.',
        '--no-deps',
        '--no-build-isolation',
    ],
    attempts=2,
    sleep_s=10,
)

print('Repo ready:', REPO_DIR)
    """

    run_cell = f"""import os
import pathlib
import subprocess
import sys

REPO_DIR = pathlib.Path('/kaggle/working/lsy_drone_racing')
if not (REPO_DIR / 'scripts' / 'run_aigp_kaggle_session.py').exists():
    nested_repo = REPO_DIR / 'lsy_drone_racing'
    if (nested_repo / 'scripts' / 'run_aigp_kaggle_session.py').exists():
        REPO_DIR = nested_repo
print('RESOLVED_REPO_DIR', REPO_DIR)
os.chdir(REPO_DIR)

cmd = [
    sys.executable,
    'scripts/run_aigp_kaggle_session.py',
    '--repo-root', str(REPO_DIR),
    '--out', {run_out!r},
    '--num-envs', {str(int(num_envs))!r},
    '--timesteps-per-stage', {str(int(timesteps_per_stage))!r},
    '--eval-repeats', {str(int(eval_repeats))!r},
    '--eval-seed-stride', {str(int(eval_seed_stride))!r},
    '--max-walltime-s', {str(int(max_walltime_s))!r},
    '--preflight-only', {('true' if bool(preflight_only) else 'false')!r},
    '--strict-preflight', {('true' if bool(strict_preflight) else 'false')!r},
    '--pythonpath-mode', 'repo-root',
    '--health-json', str(pathlib.Path({run_out!r}) / 'health' / 'preflight.json'),
    '--wandb-enabled', {('true' if bool(wandb_enabled) else 'false')!r},
    '--wandb-project', {wandb_project!r},
    '--wandb-entity', {wandb_entity!r},
    '--wandb-mode', {wandb_mode!r},
]
{optional_run_arg_code}

resume_from = {resume_from_input!r}
if resume_from:
    cmd += ['--resume-from', resume_from]

extra_args = {list(extra_train_args)!r}
for arg in extra_args:
    cmd += ['--extra-train-arg', str(arg)]

env = dict(os.environ)
env.setdefault('JAX_ENABLE_COMPILATION_CACHE', '0')
env.setdefault('SCIPY_ARRAY_API', '1')
existing_pythonpath = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = (
    str(REPO_DIR) if not existing_pythonpath else str(REPO_DIR) + os.pathsep + existing_pythonpath
)

print('RUN_CMD', ' '.join(cmd))
subprocess.check_call(cmd, env=env)
"""

    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in setup_cell.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in run_cell.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _build_kernel_metadata(
    *,
    kernel_id: str,
    title: str,
    code_file: str,
    is_private: bool,
    enable_gpu: bool,
    enable_tpu: bool,
    enable_internet: bool,
    dataset_sources: list[str],
) -> dict[str, Any]:
    """Build kernel-metadata.json payload."""
    return {
        "id": str(kernel_id),
        "title": str(title),
        "code_file": str(code_file),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": _json_bool(is_private),
        "enable_gpu": _json_bool(enable_gpu),
        "enable_tpu": _json_bool(enable_tpu),
        "enable_internet": _json_bool(enable_internet),
        "dataset_sources": list(dataset_sources),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _tar_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = set(Path(member.name).parts)
    name = str(member.name)
    if any(name.startswith(prefix) for prefix in _EXCLUDED_ASSET_DIR_PREFIX):
        return None
    if Path(member.name).name in _EXCLUDED_ASSET_FILENAMES:
        return None
    if parts.intersection(_SKIP_PARTS):
        return None
    if member.name.endswith(_SKIP_SUFFIXES):
        return None
    return member


def _write_repo_archive(
    *,
    repo_root: Path,
    bundle_dir: Path,
    archive_name: str,
    include_paths: list[str],
) -> Path:
    """Create a lightweight repo source tarball for Kaggle upload."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive_path = bundle_dir / archive_name
    top = "lsy_drone_racing"
    with tarfile.open(archive_path, "w:gz") as archive:
        for rel in include_paths:
            src = (repo_root / rel).resolve()
            # Canonical local layout keeps crazyflow as a sibling repo.
            if not src.exists() and Path(rel).parts[0] == "crazyflow":
                src = (repo_root.parent / rel).resolve()
            if not src.exists():
                continue
            arcname = f"{top}/{Path(rel)}"
            archive.add(src, arcname=arcname, filter=_tar_filter)
    return archive_path


def _write_bundle(*, bundle_dir: Path, metadata: dict[str, Any], notebook: dict[str, Any]) -> None:
    """Write kernel bundle to disk."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bundle_dir / metadata["code_file"]).write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build/push Kaggle kernel for AIGP training")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--kernel-id", default=DEFAULT_FULL_KERNEL_ID)
    parser.add_argument("--title", default=DEFAULT_FULL_TITLE)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("kaggle") / "aigp_kaggle_primary",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo-source-mode", choices=["bundle", "git"], default="bundle")
    parser.add_argument("--repo-archive-name", default=CANONICAL_ARCHIVE_NAME)
    parser.add_argument("--no-embed-archive", action="store_true", default=False)
    parser.add_argument("--bundle-include", action="append", default=[])
    parser.add_argument("--repo-url", default="https://github.com/Mrassimo/lsy_drone_racing.git")
    parser.add_argument("--branch", default="codex/aigp-port-merge")
    parser.add_argument("--run-out", default=DEFAULT_FULL_RUN_OUT)
    parser.add_argument("--resume-from-input", default=None)
    parser.add_argument("--max-walltime-s", type=int, default=42_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--timesteps-per-stage", type=int, default=10_000_000)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--eval-seed-stride", type=int, default=97)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--no-strict-preflight", action="store_true", default=False)
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true", default=True)
    parser.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-project", default="drone-racing")
    parser.add_argument("--wandb-entity", default="classimo")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--tournament-mode", type=_parse_bool, default=False)
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--qualifier-eval-profile", default=None)
    parser.add_argument("--readiness-profile", default=None)
    parser.add_argument("--enable-gpu", action="store_true", default=True)
    parser.add_argument("--disable-gpu", action="store_true", default=False)
    parser.add_argument("--enable-tpu", action="store_true", default=False)
    parser.add_argument("--disable-internet", action="store_true", default=False)
    parser.add_argument("--public", action="store_true", default=False)
    parser.add_argument("--dataset-source", action="append", default=[])
    parser.add_argument("--extra-train-arg", action="append", default=[])
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if str(args.repo_source_mode) != "bundle":
        raise ValueError(
            "canonical Kaggle runtime is offline-bundle only; set --repo-source-mode bundle"
        )

    kernel_id = str(args.kernel_id)
    title = str(args.title)
    run_out = str(args.run_out)
    num_envs = int(args.num_envs)
    timesteps_per_stage = int(args.timesteps_per_stage)
    eval_repeats = int(args.eval_repeats)
    eval_seed_stride = int(args.eval_seed_stride)
    max_walltime_s = int(args.max_walltime_s)

    if str(args.profile) == "smoke":
        if kernel_id == DEFAULT_FULL_KERNEL_ID:
            kernel_id = DEFAULT_SMOKE_KERNEL_ID
        if title == DEFAULT_FULL_TITLE:
            title = DEFAULT_SMOKE_TITLE
        if run_out == DEFAULT_FULL_RUN_OUT:
            run_out = DEFAULT_SMOKE_RUN_OUT
        num_envs = min(num_envs, 8)
        timesteps_per_stage = min(timesteps_per_stage, 131_072)
        eval_repeats = 1
        max_walltime_s = min(max_walltime_s, 7_200)

    enable_gpu = bool(args.enable_gpu) and not bool(args.disable_gpu)
    enable_internet = not bool(args.disable_internet)
    is_private = not bool(args.public)
    repo_root = args.repo_root.resolve()
    bundle_dir = args.bundle_dir.resolve()
    repo_archive_b64 = ""

    include_paths = [str(p) for p in args.bundle_include] or list(DEFAULT_BUNDLE_INCLUDE)
    archive_path = _write_repo_archive(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        archive_name=str(args.repo_archive_name),
        include_paths=include_paths,
    )
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    archive_size = archive_path.stat().st_size
    embed_allowed = (
        not bool(args.no_embed_archive)
        and archive_size
        <= (KAGGLE_NOTEBOOK_SOURCE_LIMIT_BYTES - KAGGLE_NOTEBOOK_SOURCE_SAFETY_MARGIN_BYTES)
    )
    if embed_allowed:
        repo_archive_b64 = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    else:
        if not bool(args.no_embed_archive):
            print(
                "KAGGLE_REPO_ARCHIVE payload too large for embedded notebook source; "
                "forcing --no-embed-archive"
            )
    print(
        f"KAGGLE_REPO_ARCHIVE path={archive_path} size_mb={size_mb:.2f} "
        f"include_count={len(include_paths)} embed={int(embed_allowed)} "
        f"raw_bytes={archive_size}"
    )

    dataset_sources = _canonical_dataset_sources([str(value) for value in args.dataset_source])
    notebook_name = "aigp_kaggle_train.ipynb"
    metadata = _build_kernel_metadata(
        kernel_id=kernel_id,
        title=title,
        code_file=notebook_name,
        is_private=bool(is_private),
        enable_gpu=bool(enable_gpu),
        enable_tpu=bool(args.enable_tpu),
        enable_internet=bool(enable_internet),
        dataset_sources=dataset_sources,
    )
    notebook = _build_notebook(
        repo_source_mode=str(args.repo_source_mode),
        repo_archive_name=str(args.repo_archive_name),
        repo_archive_b64=repo_archive_b64,
        repo_url=str(args.repo_url),
        branch=str(args.branch),
        run_out=run_out,
        max_walltime_s=max_walltime_s,
        num_envs=num_envs,
        timesteps_per_stage=timesteps_per_stage,
        eval_repeats=eval_repeats,
        eval_seed_stride=eval_seed_stride,
        wandb_enabled=bool(args.wandb_enabled),
        wandb_project=str(args.wandb_project),
        wandb_entity=str(args.wandb_entity),
        wandb_mode=str(args.wandb_mode),
        tournament_mode=bool(args.tournament_mode),
        obs_mode=str(args.obs_mode).strip() if args.obs_mode is not None else None,
        qualifier_eval_profile=(
            str(args.qualifier_eval_profile) if args.qualifier_eval_profile is not None else None
        ),
        readiness_profile=(
            str(args.readiness_profile) if args.readiness_profile is not None else None
        ),
        preflight_only=bool(args.preflight_only),
        strict_preflight=not bool(args.no_strict_preflight),
        resume_from_input=args.resume_from_input,
        extra_train_args=[str(v) for v in args.extra_train_arg],
    )
    _write_bundle(bundle_dir=bundle_dir, metadata=metadata, notebook=notebook)

    print(f"KAGGLE_KERNEL_BUNDLE path={bundle_dir} id={kernel_id} profile={args.profile}")

    if args.dry_run or not bool(args.push):
        raise SystemExit(0)

    kaggle_bin = os.environ.get("KAGGLE_BIN", "kaggle")
    cmd = [kaggle_bin, "kernels", "push", "-p", str(bundle_dir)]
    print("KAGGLE_PUSH_CMD", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
