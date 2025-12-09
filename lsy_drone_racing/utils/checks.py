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


def check_gates_in_bound(gates: ConfigDict, limits: ConfigDict, tolerance: float = 0.0):
    """Check if the gates layout is within the specified limits.

    Args:
        gates: A ConfigDict containing gate positions.
        limits: A ConfigDict containing min and max limits for gate positions.
        tolerance: A float value representing the tolerance for the position checks.
    """
    for i, pos in enumerate(gates.pos):
        if np.any(pos[:2] < np.array(limits.pos_limit_low) - tolerance):
            raise RuntimeError(
                f"gate{i + 1} position {pos[:2]} is below the predefined minimum limit {np.array(limits.pos_limit_low) - tolerance}"
            )
        if np.any(pos[:2] > np.array(limits.pos_limit_high) + tolerance):
            raise RuntimeError(
                f"gate{i + 1} position {pos[:2]} is above the predefined maximum limit {np.array(limits.pos_limit_high) + tolerance}"
            )


def check_race_track(gates: ConfigDict, obstacles: ConfigDict, rng_config: ConfigDict):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        gates: A ConfigDict containing gate positions and orientations, and their nominal values.
        obstacles: A ConfigDict containing obstacle positions, and their nominal values.
        rng_config: Environment randomization config.
    """
    assert rng_config.gate_pos.fn == "uniform", "Race track checks expect uniform distributions"
    assert rng_config.obstacle_pos.fn == "uniform", "Race track checks expect uniform distributions"

    low, high = rng_config.gate_pos.kwargs.minval, rng_config.gate_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(gates.pos, gates.nominal_pos)):
        check_bounds(f"gate{i + 1}", pos, nominal_pos, np.array(low), np.array(high))

    # TODO: Now the gate check should consider rotation in roll and pitch as well.
    high_tol = np.array(rng_config.gate_rpy.kwargs.maxval)
    low_tol = np.array(rng_config.gate_rpy.kwargs.minval)
    # ang_tol = rng_config.gate_rpy.kwargs.maxval[2]  # Only check yaw rotation
    for i, (quat, nominal_quat) in enumerate(zip(gates.quat, gates.nominal_quat)):
        gate_rot = R.from_quat(quat)
        nominal_rot = R.from_quat(nominal_quat)
        check_rotation(f"gate{i + 1}", gate_rot, nominal_rot, low=low_tol, high=high_tol)

    low, high = rng_config.obstacle_pos.kwargs.minval, rng_config.obstacle_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(obstacles.pos, obstacles.nominal_pos)):
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

    Raise error when the actual value exceeds the low or high boundaries.

    Args:
        name: Name of the object being checked.
        actual: NDArray of real value and is expected to be [N, ].
        desired: NDArray of nominal value and is expected to be [N, ].
        low: NDArray of the minimum value of (actual - desired) and is expected to be [N, ]
        high: NDArray of the maximum value of (actual - desired) and is expected to be [N, ]
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
    """Compare gate orientations in world-frame Euler XYZ.

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
