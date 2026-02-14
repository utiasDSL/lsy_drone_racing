from pathlib import Path

import pytest

from scripts import train_aigp_curriculum as train_mod
from scripts.train_aigp_curriculum import (
    _block_reason_code,
    _build_run_meta,
    _build_wandb_train_payload,
    _coerce_fallback_cli_value,
    _extract_train_snapshot_from_logger_values,
    _init_wandb_run,
    _parse_wandb_mode,
    _parse_wandb_tags,
    _run_train_without_fire,
    _wandb_define_metrics,
    _wandb_info_payload,
)


@pytest.mark.unit
def test_parse_wandb_mode_validation():
    assert _parse_wandb_mode("ONLINE") == "online"
    assert _parse_wandb_mode("offline") == "offline"
    assert _parse_wandb_mode("disabled") == "disabled"
    with pytest.raises(ValueError):
        _parse_wandb_mode("auto")


@pytest.mark.unit
def test_parse_wandb_tags_multiple_formats():
    assert _parse_wandb_tags(None) == []
    assert _parse_wandb_tags("a,b, c") == ["a", "b", "c"]
    assert _parse_wandb_tags(("x", "y")) == ["x", "y"]
    assert _parse_wandb_tags(["foo", " ", "bar"]) == ["foo", "bar"]


@pytest.mark.unit
def test_init_wandb_run_disabled_returns_minimal_state(tmp_path: Path):
    state = _init_wandb_run(
        wandb_enabled=False,
        wandb_mode="online",
        wandb_project="drone-racing",
        wandb_entity="classimo",
        wandb_group=None,
        wandb_tags=[],
        wandb_run_name=None,
        out_dir=tmp_path,
        run_id="abc123",
        config_payload={"x": 1},
    )
    assert not state["enabled"]
    assert state["mode"] == "online"
    assert state["run_id"] is None
    assert state["run_path"] is None
    assert state["run_url"] is None


@pytest.mark.unit
def test_wandb_info_payload_shape():
    payload = _wandb_info_payload(
        {
            "enabled": True,
            "mode": "online",
            "run_id": "run_1",
            "run_path": "entity/project/run_1",
            "run_url": "https://wandb.ai/entity/project/runs/run_1",
        }
    )
    assert payload["enabled"] is True
    assert payload["mode"] == "online"
    assert payload["run_id"] == "run_1"
    assert payload["run_path"] == "entity/project/run_1"
    assert payload["run_url"].startswith("https://")


@pytest.mark.unit
def test_build_run_meta_contains_wandb_and_paths(tmp_path: Path):
    meta = _build_run_meta(
        run_id="run-xyz",
        out_dir=tmp_path,
        config_path=Path("config.toml"),
        curriculum_path=Path("curriculum.toml"),
        config_hash="hash123",
        obs_mode="privileged",
        force_advance_mode="if_passing",
        wandb_info={
            "enabled": True,
            "mode": "online",
            "run_id": "w1",
            "run_path": "classimo/drone-racing/w1",
            "run_url": "https://wandb.ai/classimo/drone-racing/runs/w1",
        },
    )
    assert meta["schema_version"] == 2
    assert meta["run_id"] == "run-xyz"
    assert meta["obs_mode"] == "privileged"
    assert meta["force_advance_mode"] == "if_passing"
    assert meta["tournament_mode"] is False
    assert meta["tournament_readiness_profile"] is None
    assert meta["wandb"]["enabled"] is True
    assert meta["wandb"]["run_path"] == "classimo/drone-racing/w1"


@pytest.mark.unit
def test_extract_train_snapshot_from_logger_values_maps_expected_keys():
    snapshot = _extract_train_snapshot_from_logger_values(
        {
            "train/approx_kl": 0.01,
            "train/explained_variance": 0.42,
            "train/value_loss": 12.3,
            "train/n_updates": 30,
            "time/fps": 512,
            "train/unknown": 1.0,
        }
    )
    assert snapshot["approx_kl"] == pytest.approx(0.01)
    assert snapshot["explained_variance"] == pytest.approx(0.42)
    assert snapshot["value_loss"] == pytest.approx(12.3)
    assert snapshot["n_updates"] == 30
    assert snapshot["fps"] == pytest.approx(512.0)
    assert "unknown" not in snapshot


@pytest.mark.unit
def test_build_wandb_train_payload_respects_flags():
    snapshot = {
        "approx_kl": 0.02,
        "explained_variance": 0.5,
        "n_updates": 50,
        "fps": 440.0,
    }
    payload = _build_wandb_train_payload(
        train_snapshot=snapshot,
        global_timesteps=1234,
        include_rollout_metrics=True,
        include_system_metrics=False,
    )
    assert payload["curriculum/global_timesteps"] == 1234
    assert payload["train/approx_kl"] == pytest.approx(0.02)
    assert payload["train/explained_variance"] == pytest.approx(0.5)
    assert payload["train/n_updates"] == 50
    assert "runtime/fps" not in payload

    payload = _build_wandb_train_payload(
        train_snapshot=snapshot,
        global_timesteps=1234,
        include_rollout_metrics=False,
        include_system_metrics=True,
    )
    assert "train/approx_kl" not in payload
    assert payload["runtime/fps"] == pytest.approx(440.0)


@pytest.mark.unit
def test_block_reason_code_stable_mapping():
    assert _block_reason_code("none") == 0
    assert _block_reason_code("stability") == 3
    assert _block_reason_code("force_advance_blocked") == 9
    assert _block_reason_code("unknown_value") == 99


@pytest.mark.unit
def test_wandb_define_metrics_uses_step_metric():
    class _FakeRun:
        def __init__(self):
            self.calls = []

        def define_metric(self, name: str, step_metric: str | None = None):
            self.calls.append((name, step_metric))

    fake = _FakeRun()
    _wandb_define_metrics(
        state={"run": fake},
        include_rollout_metrics=True,
        include_system_metrics=True,
    )
    assert ("curriculum/global_timesteps", None) in fake.calls
    assert ("train/*", "curriculum/global_timesteps") in fake.calls
    assert ("runtime/*", "curriculum/global_timesteps") in fake.calls


@pytest.mark.unit
def test_coerce_fallback_cli_value() -> None:
    assert _coerce_fallback_cli_value("true") is True
    assert _coerce_fallback_cli_value("False") is False
    assert _coerce_fallback_cli_value("none") is None
    assert _coerce_fallback_cli_value("4242") == 4242
    assert _coerce_fallback_cli_value("0.25") == pytest.approx(0.25)
    assert _coerce_fallback_cli_value("256,256") == "256,256"


@pytest.mark.unit
def test_run_train_without_fire_parses_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_train(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(train_mod, "train", _fake_train)
    _run_train_without_fire(
        [
            "train",
            "--seed",
            "4242",
            "--wandb_enabled",
            "true",
            "--max_walltime_s",
            "42000",
            "--policy_arch_pi",
            "256,256",
        ]
    )
    assert captured["seed"] == 4242
    assert captured["wandb_enabled"] is True
    assert captured["max_walltime_s"] == 42000
    assert captured["policy_arch_pi"] == "256,256"


@pytest.mark.unit
def test_run_train_without_fire_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        _run_train_without_fire(["eval"])
