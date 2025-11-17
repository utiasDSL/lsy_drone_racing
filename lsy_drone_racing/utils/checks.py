"""Separate module for all checks used in the environments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger("rosout." + __name__)

def randomize_track(gates_pos: ArrayLike, gates_quat: ArrayLike, obstacles_pos: ArrayLike, rng_config: ConfigDict) -> tuple[NDArray, NDArray, NDArray]:
    assert rng_config.gate_pos.fn == "uniform", "Race track checks expect uniform distributions"
    assert rng_config.obstacle_pos.fn == "uniform", "Race track checks expect uniform distributions"
    gates_pos, gates_quat = np.array(gates_pos), np.array(gates_quat)
    gates_rot = R.from_quat(gates_quat).as_euler("xyz")
    obstacles_pos = np.array(obstacles_pos)
    n_gates, n_obstacles = len(gates_pos), len(obstacles_pos)

    gate_rpy_rand = rng_config.gate_rpy.kwargs
    gate_pos_rand = rng_config.gate_pos.kwargs
    sample_rpy = np.random.uniform(gate_rpy_rand.minval, gate_rpy_rand.maxval, size=(n_gates, 3))
    sample_pos = np.random.uniform(gate_pos_rand.minval, gate_pos_rand.maxval, size=(n_gates, 3))
    randomized_gate_pos = gates_pos + sample_pos
    randomized_gate_rot = gates_rot + sample_rpy
    randomized_gate_quat = R.from_euler("xyz", randomized_gate_rot).as_quat()

    obstacle_pos_rand = rng_config.obstacle_pos.kwargs
    sample_obs_pos = np.random.uniform(obstacle_pos_rand.minval, obstacle_pos_rand.maxval, size=(n_obstacles, 3))
    randomized_obstacle_pos = obstacles_pos + sample_obs_pos
    return randomized_gate_pos, randomized_gate_quat, randomized_obstacle_pos


def check_gates_layout(gates: ConfigDict, limits: ConfigDict):
    """Check if the gates layout is within the specified limits.

    Args:
        gates: A ConfigDict containing gate positions.
        limits: A ConfigDict containing min and max limits for gate positions.
    """   
    for i, pos in enumerate(gates.pos):
        if np.any(pos[:2] < limits.pos_limit_low):
            raise RuntimeError(
                f"gate{i + 1} position {pos[:2]} is below the minimum limit {limits.pos_limit_low}"
            )
        if np.any(pos[:2] > limits.pos_limit_high):
            raise RuntimeError(
                f"gate{i + 1} position {pos[:2]} is above the maximum limit {limits.pos_limit_high}"
            )


def check_race_track(
    gates: ConfigDict, obstacles: ConfigDict, rng_config: ConfigDict
):
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
        check_bounds(f"gate{i + 1}", np.array(pos), np.array(nominal_pos), np.array(low), np.array(high))
    
    # TODO: Now the gate check should consider rotation in roll and pitch as well.
    ang_tol = rng_config.gate_rpy.kwargs.maxval[2]  # Only check yaw rotation
    for i, (quat, nominal_quat) in enumerate(zip(gates.quat, gates.nominal_quat)):
        gate_rot = R.from_quat(quat)
        nominal_rot = R.from_quat(nominal_quat)
        check_rotation(f"gate{i + 1}", gate_rot, nominal_rot, ang_tol)

    low, high = rng_config.obstacle_pos.kwargs.minval, rng_config.obstacle_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(obstacles.pos, obstacles.nominal_pos)):
        check_bounds(f"obstacle{i + 1}", np.array(pos[:2]), np.array(nominal_pos[:2]), np.array(low[:2]), np.array(high[:2]))


def check_drone_start_pos(nominal_pos: NDArray, real_pos: NDArray, rng_config: ConfigDict, drone_name: str):
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
   
    check_bounds(drone_name, np.array(real_pos[:2]), np.array(nominal_pos[:2]), np.array(tol_min[:2]), np.array(tol_max[:2]))


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
