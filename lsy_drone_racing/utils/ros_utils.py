"""Separate module for utility functions that require ROS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.vicon import Vicon

if TYPE_CHECKING:
    from munch import Munch

logger = logging.getLogger("rosout." + __name__)


def check_race_track(config: Munch):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        config: Race configuration.
    """
    gate_names = [f"gate{i}" for i in range(1, len(config.env.track.gates) + 1)]
    obstacle_names = [f"obstacle{i}" for i in range(1, len(config.env.track.obstacles) + 1)]
    vicon = Vicon(track_names=gate_names + obstacle_names, auto_track_drone=False, timeout=1.0)
    rng_info = config.env.get("randomization")
    if not rng_info:
        logger.error("Randomization info not found in the configuration.")
        raise RuntimeError("Randomization info not found in the configuration.")
    ang_tol = config.env.track.gates[0].rpy[2]  # Assume all gates have the same rotation
    assert rng_info.gate_pos.type == "uniform", "Race track checks expect uniform distributions"
    assert rng_info.obstacle_pos.type == "uniform", "Race track checks expect uniform distributions"
    for i, gate in enumerate(config.env.track.gates):
        name = f"gate{i+1}"
        gate_pos, gate_rot = vicon.pos[name], R.from_euler("xyz", vicon.rpy[name])
        check_bounds(name, gate_pos, gate.pos, rng_info.gate_pos.low, rng_info.gate_pos.high)
        check_rotation(name, gate_rot, R.from_euler("xyz", gate.rpy), ang_tol)

    for i, obstacle in enumerate(config.env.track.obstacles):
        name = f"obstacle{i+1}"
        low, high = rng_info.obstacle_pos.low, rng_info.obstacle_pos.high
        check_bounds(name, vicon.pos[name][:2], obstacle.pos[:2], low[:2], high[:2])

    vicon.close()


def check_bounds(
    name: str, actual: np.ndarray, desired: np.ndarray, low: np.ndarray, high: np.ndarray
):
    """Check if the actual value is within the specified bounds of the desired value."""
    if any(actual - desired < low):
        logger.error(f"Position is: {actual}, should be: {desired}")
        raise RuntimeError(f"{name} exceeds lower tolerances ({low})")
    if any(actual - desired > high):
        logger.error(f"Position is: {actual}, should be: {desired}")
        raise RuntimeError(f"{name} exceeds upper tolerances ({high})")


def check_rotation(name: str, actual_rot: R, desired_rot: R, ang_tol: float):
    """Check if the actual rotation is within the specified tolerance of the desired rotation."""
    if (actual_rot.inv() * desired_rot).magnitude() > ang_tol:
        actual, desired = actual_rot.as_euler("xyz"), desired_rot.as_euler("xyz")
        logger.error(f"Rotation is: {actual}, should be: {desired}")
        raise RuntimeError(f"{name} exceeds rotation tolerances ({ang_tol})")


def check_drone_start_pos(config: Munch):
    """Check if the real drone start position matches the settings.

    Args:
        config: Race configuration.
    """
    tol = 0.1
    vicon = Vicon(timeout=1.0)
    drone_pos = np.array(config.env.track.drone.pos)
    if (d := np.linalg.norm(drone_pos - vicon.pos[vicon.drone_name])) > tol:
        raise RuntimeError(
            (
                f"Distance between drone and starting position too great ({d:.2f}m). "
                f"Position is {vicon.pos[vicon.drone_name]}, should be {drone_pos}"
            )
        )
    vicon.close()
