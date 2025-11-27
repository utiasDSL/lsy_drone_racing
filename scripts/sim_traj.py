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
    Sample trajectory from controller for visualization.
    Supports multiple controller types with different attribute names.
    """
    # ==========================================================================
    # 1) MPCC Controller: arc_trajectory (arc-length parameterized spline)
    # ==========================================================================
    if hasattr(ctrl, "arc_trajectory"):
        try:
            arc_spline = ctrl.arc_trajectory
            # arc_spline.x contains the arc-length parameter values
            s_max = float(arc_spline.x[-1])
            # Sample 200 points along the arc-length
            s_vals = np.linspace(0.0, s_max, 200)
            pts = np.asarray(arc_spline(s_vals), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2:
                return pts
        except Exception as e:
            logger.debug(f"Failed to sample arc_trajectory: {e}")
    
    # ==========================================================================
    # 2) MPCC Controller: trajectory (time-parameterized spline)
    # ==========================================================================
    if hasattr(ctrl, "trajectory"):
        try:
            traj_spline = ctrl.trajectory
            t_max = float(traj_spline.x[-1])
            ts = np.linspace(0.0, t_max, 200)
            pts = np.asarray(traj_spline(ts), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2:
                return pts
        except Exception as e:
            logger.debug(f"Failed to sample trajectory: {e}")
    
    # ==========================================================================
    # 3) MPCC Controller: waypoints (fallback)
    # ==========================================================================
    if hasattr(ctrl, "waypoints"):
        try:
            wps = np.asarray(ctrl.waypoints, dtype=float)
            if wps.ndim == 2 and wps.shape[1] == 3 and len(wps) >= 2:
                from scipy.interpolate import CubicSpline
                t_wp = np.linspace(0.0, 1.0, len(wps))
                cs = CubicSpline(t_wp, wps, axis=0)
                ts = np.linspace(0.0, 1.0, 200)
                pts = np.asarray(cs(ts), dtype=float)
                return pts
        except Exception as e:
            logger.debug(f"Failed to sample waypoints: {e}")
    
    # ==========================================================================
    # 4) Original controller: _des_pos_spline with _t_total
    # ==========================================================================
    if hasattr(ctrl, "_des_pos_spline") and hasattr(ctrl, "_t_total"):
        try:
            t_total = float(ctrl._t_total)
            ts = np.linspace(0.0, t_total, 200)
            pts = np.asarray(ctrl._des_pos_spline(ts), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2:
                return pts
        except Exception as e:
            logger.debug(f"Failed to sample _des_pos_spline: {e}")

    # ==========================================================================
    # 5) Original controller: _current_waypoints (fallback)
    # ==========================================================================
    if hasattr(ctrl, "_current_waypoints"):
        from scipy.interpolate import CubicSpline
        wps = np.asarray(getattr(ctrl, "_current_waypoints"), dtype=float)
        if wps.ndim == 2 and wps.shape[1] == 3 and len(wps) >= 2:
            t_wp = np.linspace(0.0, 1.0, len(wps))
            try:
                cs = CubicSpline(t_wp, wps, axis=0)
                ts = np.linspace(0.0, 1.0, 200)
                pts = np.asarray(cs(ts), dtype=float)
                return pts
            except Exception:
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

        # Sample initial planned trajectory
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
                # Render at ~fps per second
                if ((i * fps) % config.env.freq) < fps:
                    MAX_SEG_BUDGET = 900

                    # Re-sample trajectory (in case controller updates it)
                    planned_points = _sample_planned_spline(controller)

                    try:
                        if planned_points is not None and len(planned_points) >= 2:
                            planned_draw = _thin_polyline(
                                np.asarray(planned_points, float), MAX_SEG_BUDGET
                            )
                            draw_line(
                                env,
                                planned_draw,
                                rgba=np.array([0.0, 1.0, 0.0, 0.95]),  # green
                                min_size=3.0,
                                max_size=3.0,
                            )
                    except RuntimeError:
                        pass

                    env.render()

            i += 1

        controller.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs.get("target_gate", -2) == -1 else None)

    env.close()
    
    # Summary
    total_runs = n_runs
    completed_runs = sum(t is not None for t in ep_times)
    print(f"Completed: {completed_runs}/{total_runs}")

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
