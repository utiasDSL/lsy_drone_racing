"""Utility module."""

from __future__ import annotations

import importlib.util
import logging
import sys
from operator import gt, lt
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Type

import numpy as np
import pybullet as p
import yaml
from munch import munchify

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.vicon import Vicon
from lsy_drone_racing.rotations import map2pi

if TYPE_CHECKING:
    from munch import Munch

logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[BaseController]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)
    assert hasattr(controller_module, "Controller")
    assert issubclass(controller_module.Controller, BaseController)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> Munch:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The munchified config dict.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    with open(path, "r") as file:
        return munchify(yaml.safe_load(file))


def check_gate_pass(
    gate_pose: np.ndarray, drone_pos: np.ndarray, last_drone_pos: np.ndarray
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the y-Axis (pointing from -y to +y). Therefore, we check if y
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the x
    and z box boundaries. First, we linearly interpolate to get the x and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        gate_pose: The pose of the gate in the world frame.
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
    """
    # Transform last and current drone position into current gate frame.
    cos_goal, sin_goal = np.cos(gate_pose[5]), np.sin(gate_pose[5])
    last_dpos = last_drone_pos - gate_pose[0:3]
    last_drone_pos_gate = np.array(
        [
            cos_goal * last_dpos[0] - sin_goal * last_dpos[1],
            sin_goal * last_dpos[0] + cos_goal * last_dpos[1],
            last_dpos[2],
        ]
    )
    dpos = drone_pos - gate_pose[0:3]
    drone_pos_gate = np.array(
        [cos_goal * dpos[0] - sin_goal * dpos[1], sin_goal * dpos[0] + cos_goal * dpos[1], dpos[2]]
    )
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    if last_drone_pos_gate[1] < 0 and drone_pos_gate[1] > 0:  # Drone has passed the goal plane
        alpha = -last_drone_pos_gate[1] / (drone_pos_gate[1] - last_drone_pos_gate[1])
        x_intersect = alpha * (drone_pos_gate[0]) + (1 - alpha) * last_drone_pos_gate[0]
        z_intersect = alpha * (drone_pos_gate[2]) + (1 - alpha) * last_drone_pos_gate[2]
        # TODO: Replace with autodetection of gate width
        if abs(x_intersect) < 0.45 and abs(z_intersect) < 0.45:
            return True
    return False


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


def draw_trajectory(
    initial_info: dict,
    waypoints: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_z: np.ndarray,
):
    """Draw a trajectory in PyBullet's GUI."""
    for point in waypoints:
        urdf_path = Path(initial_info["urdf_dir"]) / "sphere.urdf"
        p.loadURDF(
            str(urdf_path),
            [point[0], point[1], point[2]],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=initial_info["pyb_client"],
        )
    step = int(ref_x.shape[0] / 50)
    for i in range(step, ref_x.shape[0], step):
        p.addUserDebugLine(
            lineFromXYZ=[ref_x[i - step], ref_y[i - step], ref_z[i - step]],
            lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
            lineColorRGB=[1, 0, 0],
            physicsClientId=initial_info["pyb_client"],
        )
    p.addUserDebugLine(
        lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
        lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
        lineColorRGB=[1, 0, 0],
        physicsClientId=initial_info["pyb_client"],
    )
