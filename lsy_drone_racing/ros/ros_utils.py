"""Separate module for utility functions that require ROS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.ros.ros_connector import ROSConnector

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger("rosout." + __name__)


def check_race_track(gates: ConfigDict, obstacles: ConfigDict, rng_config: ConfigDict):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        gates: Gate configuration.
        obstacles: Obstacle configuration.
        rng_config: Environment randomization config.
    """
    assert rng_config.gate_pos.fn == "uniform", "Race track checks expect uniform distributions"
    assert rng_config.obstacle_pos.fn == "uniform", "Race track checks expect uniform distributions"

    n_gates, n_obstacles = len(gates.pos), len(obstacles.pos)
    gate_names = [f"gate{i}" for i in range(1, n_gates + 1)]
    obstacle_names = [f"obstacle{i}" for i in range(1, n_obstacles + 1)]
    ros_connector = ROSConnector(tf_names=gate_names + obstacle_names, timeout=5.0)
    try:
        ang_tol = rng_config.gate_rpy.kwargs.maxval[2]  # Only check yaw rotation
        for i in range(n_gates):
            name = f"gate{i + 1}"
            nominal_pos, nominal_rot = gates.pos[i, ...], R.from_quat(gates.quat[i, ...])
            gate_pos, gate_rot = ros_connector.pos[name], R.from_quat(ros_connector.quat[name])
            low, high = rng_config.gate_pos.kwargs.minval, rng_config.gate_pos.kwargs.maxval
            check_bounds(name, gate_pos, nominal_pos, low, high)
            check_rotation(name, gate_rot, nominal_rot, ang_tol)

        for i in range(n_obstacles):
            name = f"obstacle{i + 1}"
            nominal_pos = obstacles.pos[i, ...]
            low, high = rng_config.obstacle_pos.kwargs.minval, rng_config.obstacle_pos.kwargs.maxval
            check_bounds(name, ros_connector.pos[name][:2], nominal_pos[:2], low[:2], high[:2])
    finally:
        ros_connector.close()


def check_drone_start_pos(pos: NDArray, rng_config: ConfigDict, drone_name: str):
    """Check if the real drone start position matches the settings.

    Args:
        pos: Current drone position.
        rng_config: Environment randomization config.
        drone_name: Name of the drone (e.g. cf10).
    """
    assert rng_config.drone_pos.fn == "uniform", (
        "Drone start position check expects uniform distributions"
    )
    tol_min, tol_max = rng_config.drone_pos.kwargs.minval, rng_config.drone_pos.kwargs.maxval
    ros_connector = ROSConnector(estimator_names=[drone_name], timeout=5.0)
    try:
        real_pos = ros_connector.pos[drone_name]
    finally:
        ros_connector.close()
    check_bounds(drone_name, real_pos[:2], pos[:2], tol_min[:2], tol_max[:2])


def check_bounds(name: str, actual: NDArray, desired: NDArray, low: NDArray, high: NDArray):
    """Check if the actual value is within the specified bounds of the desired value."""
    if np.any(actual - desired < low):
        raise RuntimeError(
            f"{name} exceeds lower tolerances ({low}). Position is: {actual}, should be: {desired}"
        )
    if np.any(actual - desired > high):
        raise RuntimeError(
            f"{name} exceeds upper tolerances ({high}). Position is: {actual}, should be: {desired}"
        )


def check_rotation(name: str, actual_rot: R, desired_rot: R, ang_tol: float):
    """Check if the actual rotation is within the specified tolerance of the desired rotation."""
    if (actual_rot.inv() * desired_rot).magnitude() > ang_tol:
        actual, desired = actual_rot.as_euler("xyz"), desired_rot.as_euler("xyz")
        raise RuntimeError(
            f"{name} exceeds rotation tolerances ({ang_tol}).\n"
            f"Rotation is: {actual}, should be: {desired}"
        )
