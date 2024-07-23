"""Separate module for utility functions that require ROS."""

from __future__ import annotations

from operator import gt, lt
from typing import TYPE_CHECKING, Callable

import numpy as np

from lsy_drone_racing.utils.rotations import map2pi
from lsy_drone_racing.vicon import Vicon

if TYPE_CHECKING:
    from munch import Munch


def check_race_track(config: Munch):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        config: Race configuration.
    """
    gate_names = [f"gate{i}" for i in range(1, len(config.quadrotor_config.gates) + 1)]
    obstacle_names = [f"obstacle{i}" for i in range(1, len(config.quadrotor_config.obstacles) + 1)]
    vicon = Vicon(track_names=gate_names + obstacle_names, auto_track_drone=False, timeout=1.0)
    rng_info = config.quadrotor_config.gates_and_obstacles_randomization_info
    # Gate checks
    ang_tol = 0.3  # TODO: Adapt value based on experience in the lab
    assert rng_info.gates.distrib == "uniform"
    gate_poses = np.array(config.quadrotor_config.gates)
    for i, gate in enumerate(gate_poses):
        name = f"gate{i+1}"
        _check_pos(vicon.pos[name][:2], gate[:2], rng_info.gates.low, lt, name)
        _check_pos(vicon.pos[name][:2], gate[:2], rng_info.gates.high, gt, name)
        _check_rot(vicon.rpy[name][2], gate[5], ang_tol, name)
    obstacle_poses = np.array(config.quadrotor_config.obstacles)
    for i, obstacle in enumerate(obstacle_poses):
        name = f"obstacle{i+1}"
        _check_pos(vicon.pos[name][:2], obstacle[:2], rng_info.obstacles.low, lt, name)
        _check_pos(vicon.pos[name][:2], obstacle[:2], rng_info.obstacles.high, gt, name)


def _check_pos(pos: np.ndarray, pos_des: np.ndarray, tol: float, comp_fn: Callable, name: str = ""):
    if any(comp_fn(pos - pos_des, tol)):
        # Print because ROS swallows the logger if rospy.init_node is called. Not all functions in
        # utils require ROS to run, so it would be inconsistent to set the logger to
        # (rosout. + __name__). This is on ROS (...), nothing we can do for now.
        print(f"Position is: {pos}, should be: {pos_des}")
        raise RuntimeError(f"{name} exceeds tolerances ({tol}, {comp_fn.__name__})")


def _check_rot(rot: float, rot_des: float, tol: float, name: str = ""):
    if np.abs(map2pi(rot - rot_des)) > tol:
        print(f"Rotation is: {rot:.3f}, should be: {rot_des:.3f}")
        raise RuntimeError(f"{name} exceeds rotation tolerances ({tol})")


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
