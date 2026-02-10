"""Convert DronePrix YAML track configs into lsy_drone_racing TOML configs.

This is a one-way conversion tool intended to preserve the "gold" competition tracks from the
DronePrix project while adopting the TOML config pattern used by lsy_drone_racing.

Notes:
    - The output TOML files are full environment configs (sim + env + track) so they can be loaded
      via :func:`lsy_drone_racing.utils.load_config`.
    - We keep obstacles empty (`obstacles = []`). The core env supports this (see envs/utils.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
import yaml


@dataclass(frozen=True)
class BaseSimConfig:
    """Default simulation config for converted track files."""

    physics: str = "first_principles"
    drone_model: str = "cf21B_500"
    camera_view: tuple[float, float, float, float, float, float] = (
        5.0,
        180.0,
        -25.0,
        0.0,
        0.0,
        0.0,
    )
    freq: int = 500
    attitude_freq: int = 500
    render: bool = False


@dataclass(frozen=True)
class BaseEnvConfig:
    """Default environment config for converted track files."""

    env_id: str = "AIGPDroneRacing-v0"
    seed: str = "random"
    freq: int = 50
    sensor_range: float = 100.0  # reveal everything by default for RL training
    control_mode: str = "attitude"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _convert_track(yaml_data: dict[str, Any]) -> dict[str, Any]:
    env = yaml_data.get("environment", {})
    drone = env.get("drone", {})
    bounds = env.get("bounds", {})
    gate_size = env.get("gate_size", {})

    gates = env.get("gates", [])
    yaws = env.get("gate_orientations", [0.0] * len(gates))
    if len(yaws) != len(gates):
        raise ValueError(f"gate_orientations length {len(yaws)} != gates length {len(gates)}")

    track_gates = [
        {"pos": [float(p[0]), float(p[1]), float(p[2])], "rpy": [0.0, 0.0, float(yaw)]}
        for p, yaw in zip(gates, yaws, strict=True)
    ]

    track = {
        "randomize": False,
        "gates": track_gates,
        "obstacles": [],  # no obstacles in DronePrix competition tracks
        "drones": [
            {
                "pos": [float(x) for x in drone.get("initial_position", [0.0, 0.0, 0.01])],
                "rpy": [float(x) for x in drone.get("initial_orientation", [0.0, 0.0, 0.0])],
                "vel": [0.0, 0.0, 0.0],
                "ang_vel": [0.0, 0.0, 0.0],
            }
        ],
        "safety_limits": {
            "pos_limit_low": [
                float(bounds.get("x", [-3.0, 3.0])[0]),
                float(bounds.get("y", [-3.0, 3.0])[0]),
                float(bounds.get("z", [-1e-3, 2.5])[0]),
            ],
            "pos_limit_high": [
                float(bounds.get("x", [-3.0, 3.0])[1]),
                float(bounds.get("y", [-3.0, 3.0])[1]),
                float(bounds.get("z", [-1e-3, 2.5])[1]),
            ],
        },
        "gate_size": {
            "width": float(gate_size.get("width", 0.45)),
            "height": float(gate_size.get("height", 0.45)),
            "tolerance": float(gate_size.get("tolerance", 0.0)),
        },
    }
    return track


def convert_file(src_yaml: Path, out_toml: Path, *, sim: BaseSimConfig, env: BaseEnvConfig) -> None:
    """Convert a single DronePrix YAML track file into a TOML environment config."""
    data = _load_yaml(src_yaml)
    track = _convert_track(data)

    cfg = {
        "controller": {"file": "state_controller.py"},
        "sim": {
            "physics": sim.physics,
            "drone_model": sim.drone_model,
            "camera_view": list(sim.camera_view),
            "freq": sim.freq,
            "attitude_freq": sim.attitude_freq,
            "render": sim.render,
        },
        "env": {
            "id": env.env_id,
            "seed": env.seed,
            "freq": env.freq,
            "sensor_range": env.sensor_range,
            "control_mode": env.control_mode,
            "track": track,
        },
    }

    out_toml.parent.mkdir(parents=True, exist_ok=True)
    with out_toml.open("w") as f:
        toml.dump(cfg, f)


def main() -> None:
    """CLI entrypoint."""
    repo_root = Path(__file__).resolve().parents[2]
    droneprix_tracks = repo_root / "droneprix_reference" / "configs" / "tracks"
    if not droneprix_tracks.exists():
        raise SystemExit(
            f"Expected DronePrix clone at {droneprix_tracks}. "
            "Clone it as ../droneprix_reference first."
        )

    out_dir = repo_root / "lsy_drone_racing" / "config" / "aigp_tracks"
    sim = BaseSimConfig()
    env = BaseEnvConfig()

    for src_yaml in sorted(droneprix_tracks.glob("competition_*.yaml")):
        out_toml = out_dir / (src_yaml.stem + ".toml")
        convert_file(src_yaml, out_toml, sim=sim, env=env)
        print(f"Wrote {out_toml}")

    for src_yaml in sorted(droneprix_tracks.glob("tii_real_track*.yaml")):
        out_toml = out_dir / (src_yaml.stem + ".toml")
        convert_file(src_yaml, out_toml, sim=sim, env=env)
        print(f"Wrote {out_toml}")


if __name__ == "__main__":
    main()
