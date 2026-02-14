import tarfile
from pathlib import Path

import pytest

from scripts.push_aigp_kaggle_kernel import (
    CANONICAL_SOURCE_DATASET,
    CANONICAL_WHEELHOUSE_DATASET,
    _build_kernel_metadata,
    _build_notebook,
    _canonical_dataset_sources,
    _write_repo_archive,
)


def test_build_kernel_metadata_defaults() -> None:
    payload = _build_kernel_metadata(
        kernel_id="massimoraso/aigp-kaggle-primary-trainer",
        title="AIGP Kaggle Primary Trainer",
        code_file="aigp_kaggle_train.ipynb",
        is_private=True,
        enable_gpu=True,
        enable_tpu=False,
        enable_internet=True,
        dataset_sources=[],
    )
    assert payload["kernel_type"] == "notebook"
    assert payload["is_private"] is True
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False
    assert payload["enable_internet"] is True


def test_build_notebook_includes_session_wrapper_command() -> None:
    notebook = _build_notebook(
        repo_source_mode="bundle",
        repo_archive_name="lsy_drone_racing_src.tar.gz",
        repo_archive_b64="YXJjaGl2ZQ==",
        repo_url="https://github.com/Mrassimo/lsy_drone_racing.git",
        branch="codex/aigp-port-merge",
        run_out="runs/aigp_kaggle_primary",
        max_walltime_s=42_000,
        wandb_enabled=True,
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_mode="offline",
        resume_from_input=None,
        extra_train_args=["--device", "auto"],
    )
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    text = "".join(cells[1]["source"])
    assert "scripts/run_aigp_kaggle_session.py" in text
    assert "--max-walltime-s" in text
    assert "'42000'" in text
    assert "--wandb-enabled" in text
    assert "--wandb-mode" in text
    assert "--preflight-only" in text
    assert "--strict-preflight" in text
    assert "--pythonpath-mode" in text
    assert "--health-json" in text
    setup_text = "".join(cells[0]["source"])
    assert "find_repo_archive" in setup_text
    assert "find_wheelhouse_dir" in setup_text
    assert "REPO_ARCHIVE_B64 = 'YXJjaGl2ZQ=='" in setup_text
    assert "using embedded source archive" in setup_text


def test_build_notebook_includes_tournament_overrides() -> None:
    notebook = _build_notebook(
        repo_source_mode="bundle",
        repo_archive_name="lsy_drone_racing_src.tar.gz",
        repo_archive_b64="YXJjaGl2ZQ==",
        repo_url="https://github.com/Mrassimo/lsy_drone_racing.git",
        branch="codex/aigp-port-merge",
        run_out="runs/aigp_kaggle_tournament",
        max_walltime_s=42_000,
        wandb_enabled=True,
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_mode="offline",
        tournament_mode=True,
        obs_mode="competition_proxy",
        qualifier_eval_profile="aigp_qualifier_eval_profile_default.toml",
        readiness_profile="qualifier_strict",
        resume_from_input=None,
        extra_train_args=[],
    )
    text = "".join(notebook["cells"][1]["source"])
    assert "cmd += ['--tournament-mode', 'true']" in text
    assert "cmd += ['--obs-mode', 'competition_proxy']" in text
    assert "cmd += ['--qualifier-eval-profile', 'aigp_qualifier_eval_profile_default.toml']" in text
    assert "cmd += ['--readiness-profile', 'qualifier_strict']" in text


def test_build_notebook_without_tournament_overrides() -> None:
    notebook = _build_notebook(
        repo_source_mode="bundle",
        repo_archive_name="lsy_drone_racing_src.tar.gz",
        repo_archive_b64="YXJjaGl2ZQ==",
        repo_url="https://github.com/Mrassimo/lsy_drone_racing.git",
        branch="codex/aigp-port-merge",
        run_out="runs/aigp_kaggle_primary",
        max_walltime_s=42_000,
        wandb_enabled=True,
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_mode="offline",
        resume_from_input=None,
        extra_train_args=[],
    )
    text = "".join(notebook["cells"][1]["source"])
    assert "cmd += ['--tournament-mode', 'true']" not in text
    assert "cmd += ['--obs-mode', 'competition_proxy']" not in text
    assert "cmd += ['--qualifier-eval-profile'" not in text
    assert "--readiness-profile" not in text


def test_build_notebook_rejects_non_bundle_mode() -> None:
    with pytest.raises(ValueError):
        _build_notebook(
            repo_source_mode="git",
            repo_archive_name="lsy_drone_racing_src.tar.gz",
            repo_archive_b64="",
            repo_url="https://github.com/Mrassimo/lsy_drone_racing.git",
            branch="codex/aigp-port-merge",
            run_out="runs/aigp_kaggle_primary",
            max_walltime_s=42_000,
            wandb_enabled=False,
            wandb_project="drone-racing",
            wandb_entity="classimo",
            wandb_mode="offline",
            resume_from_input=None,
            extra_train_args=[],
        )


def test_canonical_dataset_sources_always_include_required() -> None:
    sources = _canonical_dataset_sources(["massimoraso/other-dataset", CANONICAL_SOURCE_DATASET])
    assert CANONICAL_SOURCE_DATASET in sources
    assert CANONICAL_WHEELHOUSE_DATASET in sources


def test_write_repo_archive_stages_expected_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "lsy_drone_racing").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "lsy_drone_racing" / "__init__.py").write_text("# pkg\n", encoding="utf-8")
    (repo_root / "scripts" / "runner.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "config" / "cfg.toml").write_text("a=1\n", encoding="utf-8")
    (repo_root / "runs").mkdir(parents=True)
    (repo_root / "runs" / "skip.txt").write_text("skip\n", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    archive_path = _write_repo_archive(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        archive_name="repo.tar.gz",
        include_paths=["lsy_drone_racing", "scripts", "config", "runs"],
    )

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()

    assert "lsy_drone_racing/lsy_drone_racing/__init__.py" in names
    assert "lsy_drone_racing/scripts/runner.py" in names
    assert "lsy_drone_racing/config/cfg.toml" in names
    assert all("/runs/" not in name for name in names)


def test_write_repo_archive_includes_sibling_crazyflow(tmp_path: Path) -> None:
    repo_root = tmp_path / "lsy_drone_racing"
    repo_root.mkdir(parents=True)
    sibling_crazyflow = tmp_path / "crazyflow"
    (sibling_crazyflow / "submodules" / "drone-models").mkdir(parents=True)
    (sibling_crazyflow / "__init__.py").write_text("# cf\n", encoding="utf-8")
    (sibling_crazyflow / "submodules" / "drone-models" / "README.md").write_text(
        "ok\n", encoding="utf-8"
    )

    bundle_dir = tmp_path / "bundle"
    archive_path = _write_repo_archive(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        archive_name="repo.tar.gz",
        include_paths=["crazyflow"],
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()

    assert "lsy_drone_racing/crazyflow/__init__.py" in names
    assert "lsy_drone_racing/crazyflow/submodules/drone-models/README.md" in names


def test_write_repo_archive_skips_unused_crazyflow_assets(tmp_path: Path) -> None:
    repo_root = tmp_path / "lsy_drone_racing"
    repo_root.mkdir(parents=True)
    sibling_crazyflow = tmp_path / "crazyflow"
    assets_root = (
        sibling_crazyflow
        / "submodules"
        / "drone-models"
        / "drone_models"
        / "data"
        / "assets"
        / "cf21B"
    )
    assets_root.mkdir(parents=True)
    (assets_root / "cf21B_full.stl").write_text("skip-full", encoding="utf-8")
    (assets_root / "cf21B_no-prop.stl").write_text("skip-np", encoding="utf-8")
    (assets_root / "cf21B_header.stl").write_text("skip-header", encoding="utf-8")
    (assets_root / "cf21B_PropL.stl").write_text("keep", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    archive_path = _write_repo_archive(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        archive_name="repo.tar.gz",
        include_paths=["crazyflow/submodules/drone-models/drone_models"],
    )

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()

    assert all(
        "cf21B_full.stl" not in name and "cf21B_no-prop.stl" not in name
        and "cf21B_header.stl" not in name
        for name in names
    )
    assert "lsy_drone_racing/crazyflow/submodules/drone-models/drone_models/data/assets/cf21B/cf21B_PropL.stl" in names
