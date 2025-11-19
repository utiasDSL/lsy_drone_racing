"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller, draw_line

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def _thin_polyline(points: np.ndarray, max_segments: int) -> np.ndarray:
    """Reduce a 3D polyline to at most max_segments segments (keeps endpoints)."""
    if points is None or len(points) <= 2:
        return points
    segs = len(points) - 1
    if segs <= max_segments:
        return points
    stride = math.ceil(segs / max_segments)
    thinned = points[::stride]
    if not np.allclose(thinned[-1], points[-1]):
        thinned = np.vstack([thinned, points[-1]])
    return thinned


def _sample_planned_spline(ctrl) -> np.ndarray | None:
    """
    Bevorzugt den im Controller vorhandenen CubicSpline (_des_pos_spline).
    Fallback: falls nur _current_waypoints existieren, lokal einen Spline erstellen.
    """
    # 1) echter Spline aus dem Controller
    if hasattr(ctrl, "_des_pos_spline") and hasattr(ctrl, "_t_total"):
        try:
            t_total = float(ctrl._t_total)
            # 200 gleichverteilte Stützstellen für eine glatte Linie
            ts = np.linspace(0.0, t_total, 200)
            pts = np.asarray(ctrl._des_pos_spline(ts), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2:
                return pts
        except Exception:
            pass

    # 2) Fallback: Waypoints → eigener Spline (gleichmäßige Parametrisierung)
    if hasattr(ctrl, "_current_waypoints"):
        from scipy.interpolate import CubicSpline  # lokal import, nur wenn benötigt
        wps = np.asarray(getattr(ctrl, "_current_waypoints"), dtype=float)
        if wps.ndim == 2 and wps.shape[1] == 3 and len(wps) >= 2:
            t_wp = np.linspace(0.0, 1.0, len(wps))
            try:
                cs = CubicSpline(t_wp, wps, axis=0)
                ts = np.linspace(0.0, 1.0, 200)
                pts = np.asarray(cs(ts), dtype=float)
                return pts
            except Exception:
                # Wenn Spline aus irgendwelchen Gründen fehlschlägt, einfach die Waypoints als Polyline zurückgeben
                return wps
    return None


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes."""
    # Load configuration and check if firmware should be used
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    # Create environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        # Nur geplante Trajektorie (Spline/Waypoints) vorbereiten
        planned_points = _sample_planned_spline(controller)

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)

            obs, reward, terminated, truncated, info = env.step(action)

            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            if terminated or truncated or controller_finished:
                break

            if config.sim.render:
                # Render nicht öfter als ~fps pro Sekunde
                if ((i * fps) % config.env.freq) < fps:
                    # Vor jedem Render die geplante Linie zeichnen
                    # Budget knapp halten (wir zeichnen nur eine Linie)
                    MAX_SEG_BUDGET = 900

                    # Neu samplen, falls Controller seine Trajektorie on-the-fly anpasst
                    planned_points = _sample_planned_spline(controller)

                    try:
                        if planned_points is not None and len(planned_points) >= 2:
                            planned_draw = _thin_polyline(
                                np.asarray(planned_points, float), MAX_SEG_BUDGET
                            )
                            draw_line(
                                env,
                                planned_draw,
                                rgba = np.array([0.0, 1.0, 0.0, 0.95]),   # grün
                                min_size=3.0,
                                max_size=3.0,
                            )
                    except RuntimeError:
                        # Falls Viewer/Marker knapp: Render trotzdem fortsetzen
                        pass

                    env.render()

            i += 1

        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs.get("target_gate", -2) == -1 else None)

    env.close()
    
    # ---- Add this block ----
    total_runs = n_runs
    completed_runs = sum(t is not None for t in ep_times)
    print(f"Completed: {completed_runs}/{total_runs}")
    # ------------------------

    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)