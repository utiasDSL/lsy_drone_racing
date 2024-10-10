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
    gate_names = [f"gate{i}" for i in range(1, len(config.quadrotor_config.gates) + 1)]
    obstacle_names = [f"obstacle{i}" for i in range(1, len(config.quadrotor_config.obstacles) + 1)]
    vicon = Vicon(track_names=gate_names + obstacle_names, auto_track_drone=False, timeout=1.0)
    rng_info = config.quadrotor_config.gates_and_obstacles_randomization_info
    ang_tol = 0.3  # TODO: Adapt value based on experience in the lab
    assert rng_info.gates.distrib == "uniform"
    gate_poses = np.array(config.quadrotor_config.gates)
    for i, gate in enumerate(gate_poses):
        name = f"gate{i+1}"
        gate_pos, gate_rot = vicon.pos[name], R.from_euler("xyz", vicon.rpy[name])
        check_bounds(name, gate_pos, gate[:3], rng_info.gates.low, rng_info.gates.high)
        check_rotation(name, gate_rot, R.from_euler("xyz", gate[3:6]), ang_tol)

    obstacle_poses = np.array(config.quadrotor_config.obstacles)
    for i, obstacle in enumerate(obstacle_poses):
        name = f"obstacle{i+1}"
        low, high = rng_info.obstacles.low, rng_info.obstacles.high
        check_bounds(name, vicon.pos[name], obstacle, low, high)


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
    if actual_rot.inv() * desired_rot.magnitude() > ang_tol:
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
    init_state = config.quadrotor_config.init_state
    drone_pos = np.array([init_state[key] for key in ("init_x", "init_y", "init_z")])
    if (d := np.linalg.norm(drone_pos - vicon.pos[vicon.drone_name])) > tol:
        raise RuntimeError(
            (
                f"Distance between drone and starting position too great ({d:.2f}m)"
                f"Position is {vicon.pos['cf']}, should be {drone_pos}"
            )
        )
