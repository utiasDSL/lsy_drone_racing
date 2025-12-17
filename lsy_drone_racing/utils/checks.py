"""Separate module for all checks used in the environments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger("rosout." + __name__)


def check_race_track(
    gates_pos: NDArray,
    nominal_gates_pos: NDArray,
    gates_quat: NDArray,
    nominal_gates_quat: NDArray,
    obstacles_pos: NDArray,
    nominal_obstacles_pos: NDArray,
    rng_config: ConfigDict,
):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        gates_pos: The positions of the gates.
        nominal_gates_pos: The nominal positions of the gates.
        gates_quat: The orientations of the gates as quaternions.
        nominal_gates_quat: The nominal orientations of the gates as quaternions.
        obstacles_pos: The positions of the obstacles.
        nominal_obstacles_pos: The nominal positions of the obstacles.
        rng_config: Environment randomization config.
    """
    assert rng_config.gate_pos.fn == "uniform", "Race track checks expect uniform distributions"
    assert rng_config.obstacle_pos.fn == "uniform", "Race track checks expect uniform distributions"
    low, high = rng_config.gate_pos.kwargs.minval, rng_config.gate_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(gates_pos, nominal_gates_pos)):
        check_bounds(f"gate{i + 1}", pos, nominal_pos, np.array(low), np.array(high))

    high_tol = np.array(rng_config.gate_rpy.kwargs.maxval)
    low_tol = np.array(rng_config.gate_rpy.kwargs.minval)
    for i, (quat, nominal_quat) in enumerate(zip(gates_quat, nominal_gates_quat)):
        gate_rot = R.from_quat(quat)
        nominal_rot = R.from_quat(nominal_quat)
        check_rotation(f"gate{i + 1}", gate_rot, nominal_rot, low=low_tol, high=high_tol)

    low, high = rng_config.obstacle_pos.kwargs.minval, rng_config.obstacle_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(obstacles_pos, nominal_obstacles_pos)):
        check_bounds(
            f"obstacle{i + 1}", pos[:2], nominal_pos[:2], np.array(low[:2]), np.array(high[:2])
        )


def check_drone_start_pos(
    nominal_pos: NDArray, real_pos: NDArray, rng_config: ConfigDict, drone_name: str
):
    """Check if the real drone start position matches the settings.

    Args:
        nominal_pos: Nominal drone position.
        real_pos: Current drone position.
        rng_config: Environment randomization config.
        drone_name: Name of the drone (e.g. cf10).
    """
    assert rng_config.drone_pos.fn == "uniform", (
        "Drone start position check expects uniform distributions"
    )
    tol_min, tol_max = rng_config.drone_pos.kwargs.minval, rng_config.drone_pos.kwargs.maxval
    check_bounds(
        drone_name, real_pos[:2], nominal_pos[:2], np.array(tol_min[:2]), np.array(tol_max[:2])
    )


def check_bounds(name: str, actual: NDArray, desired: NDArray, low: NDArray, high: NDArray):
    """Check if the actual value is within the specified bounds of the desired value.

    Args:
        name: Name of the object being checked.
        actual: Values to check.
        desired: Reference values.
        low: Lower bound. Minimum permissible value of (actual - desired).
        high: Upper bound. Maximum value of (actual - desired).

    Raises:
            RuntimeError: The values are not in the permissible interval.
    """
    if np.any(actual - desired < low):
        raise RuntimeError(
            f"{name} exceeds lower tolerances ({low}). Position is: {actual}, should be: {desired}"
        )
    if np.any(actual - desired > high):
        raise RuntimeError(
            f"{name} exceeds upper tolerances ({high}). Position is: {actual}, should be: {desired}"
        )


def check_rotation(name: str, actual_rot: R, desired_rot: R, low: NDArray, high: NDArray):
    """Compare gate orientations in world-frame Euler xyz.

    Warning:
        Comparing Euler angles is tricky. While we try to sanitize the comparison as best as we
        can, edge cases may still cause failures.

    Todo:
        Switch to a more sane rotation check method.

    Args:
        name: Name of the object being checked.
        actual_rot: R object describing rotation of the real object.
        desired_rot:  R object describing rotation of the nominal object.
        low: Array designating the per axis rotation lower limit
        high: Array designating the per axis rotation higher limit

    """
    actual = actual_rot.as_euler("xyz", degrees=False)
    desired = desired_rot.as_euler("xyz", degrees=False)
    diff = (actual - desired + np.pi) % (2 * np.pi) - np.pi
    if np.any(diff < low):
        raise RuntimeError(
            f"{name} exceeds lower rotation tolerances ({low}).\n"
            f"Rotation is: {actual}, should be: {desired}"
        )
    elif np.any(diff > high):
        raise RuntimeError(
            f"{name} exceeds higher rotation tolerances ({high}).\n"
            f"Rotation is: {actual}, should be: {desired}"
        )
