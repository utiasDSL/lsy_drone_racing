"""Physics module for the LSY Drone Racing simulation.

This module provides various physics implementations for simulating drone dynamics in a racing
environment. It includes different physics modes, force and torque calculations, and effects such as
motor thrust, drag, ground effect, and downwash.

The module is designed to work with the PyBullet physics engine and supports both PyBullet-based and
custom dynamics implementations. It is used by the main simulation module (sim.py) to update the
drone state in the racing environment.

By exchanging the physics backend, we can easily support more complex physics models including
data-driven models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from lsy_drone_racing.sim.drone import Drone


GRAVITY: float = 9.81


ForceTorque = NamedTuple("ForceTorque", f=NDArray[np.floating], t=NDArray[np.floating])


class PhysicsMode(str, Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"  # Base PyBullet physics update.
    DEFAULT = PYB  # Default physics mode.
    DYN = "dyn"  # Update with an explicit model of the dynamics.
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect.
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag.
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash.
    # PyBullet physics update with ground effect, drag, and downwash.
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"
    SYS_ID = "sys_id"  # System identification dynamics model.


@dataclass
class SystemIdentificationParams:
    """Parameters for the system identification dynamics model."""

    acc_k1: float = 20.91
    acc_k2: float = 3.65
    roll_alpha: float = -3.96
    roll_beta: float = 4.08
    pitch_alpha: float = -6.00
    pitch_beta: float = 6.21
    yaw_alpha: float = 0.00
    yaw_beta: float = 0.00


SYS_ID_PARAMS = SystemIdentificationParams()


def force_torques(
    drone: Drone, rpms: NDArray[np.floating], mode: PhysicsMode, dt: float, pyb_client: int
) -> dict[int, list[str, NDArray[np.floating]]]:
    """Physics update function.

    We dynamically dispatch to the appropriate physics implementation based on the mode.

    Args:
        drone: The target drone to calculate the physics for.
        rpms: The rpms to apply to the drone rotors. Shape: (4,).
        mode: Physics mode that determines the physics implementation used for the dynamics.
        dt: The time step for the dynamics update in PhysicsMode.DYN.
        pyb_client: The PyBullet client id.
    """
    if mode == PhysicsMode.PYB:
        return motors(drone, rpms)
    elif mode == PhysicsMode.DYN:
        return dynamics(drone, rpms, dt)
    elif mode == PhysicsMode.PYB_GND:
        return motors(drone, rpms) + ground_effect(drone, rpms, pyb_client)
    elif mode == PhysicsMode.PYB_DRAG:
        return motors(drone, rpms) + drag(drone, rpms)
    elif mode == PhysicsMode.PYB_DW:
        return motors(drone, rpms) + downwash(drone, [])
    elif mode == PhysicsMode.PYB_GND_DRAG_DW:
        return motors(drone, rpms) + drag(drone, rpms) + downwash(drone, [])
    elif mode in PhysicsMode.SYS_ID:
        raise ValueError(f"Physics mode {mode} not supported for force/torque calculation.")
    raise NotImplementedError(f"Physics mode {mode} not implemented.")


def motors(drone: Drone, rpms: NDArray[np.floating]) -> list[tuple[int, ForceTorque]]:
    """Base physics implementation.

    Args:
        drone: The target drone to calculate the physics for.
        rpms: The rpms to apply to the drone rotors.

    Returns:
        A list of tuples containing the link id and a force/torque tuple.
    """
    ft = []
    forces = np.array(rpms**2) * drone.params.kf
    torques = np.array(rpms**2) * drone.params.km
    z_torque = torques[0] - torques[1] + torques[2] - torques[3]
    for i in range(4):
        ft.append((i, ForceTorque([0, 0, forces[i]], [0, 0, 0])))
    ft.append((4, ForceTorque([0, 0, 0], [0, 0, z_torque])))
    return ft


def dynamics(drone: Drone, rpms: NDArray[np.floating], dt: float) -> list[tuple[int, ForceTorque]]:
    """Explicit dynamics implementation.

    Based on code written at the Dynamic Systems Lab by James Xu.

    Args:
        drone: The target drone to calculate the physics for.
        rpms: The rpm to apply to the drone rotors.
        dt: The dynamics time step.
    """
    pos = drone.pos
    rpy = drone.rpy
    vel = drone.vel
    rpy_rates = R.from_euler("xyz", rpy).apply(drone.ang_vel, inverse=True)  # Now in body frame
    # Compute forces and torques.
    forces = np.array(rpms**2) * drone.params.kf
    thrust = np.array([0, 0, np.sum(forces)])
    thrust_world_frame = R.from_euler("xyz", rpy).apply(thrust)
    force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * drone.params.mass])
    z_torques = np.array(rpms**2) * drone.params.km
    z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
    L = drone.params.arm_len
    x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (L / np.sqrt(2))
    y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (L / np.sqrt(2))
    torques = np.array([x_torque, y_torque, z_torque])
    torques = torques - np.cross(rpy_rates, np.dot(drone.params.J, rpy_rates))
    rpy_rates_deriv = np.dot(drone.params.J_inv, torques)
    acc = force_world_frame / drone.params.mass
    # Update state.
    vel = vel + acc * dt
    rpy_rates = rpy_rates + rpy_rates_deriv * dt
    drone.pos[:] = pos + vel * dt
    drone.rpy[:] = rpy + rpy_rates * dt
    drone.vel[:] = vel
    drone.ang_vel[:] = R.from_euler("xyz", rpy).apply(rpy_rates)
    return []  # No forces/torques to apply. We set the drone state directly.


def sys_id_dynamics(
    drone: Drone, collective_thrust: float, attitude: NDArray[np.floating], dt: float
) -> list[tuple[int, ForceTorque]]:
    """Dynamics model identified from data collected on the real drone.

    Contrary to the other physics implementations, this function is not based on a physical model.
    Instead, we fit a linear model to the data collected on the real drone, and predict the next
    state based on the control inputs and the current state.

    Note:
        We do not explicitly simulate the onboard controller for this model. Instead, we assume that
        its dynamics are implicitly captured by the linear model.

    Args:
        drone: The target drone to calculate the physics for.
        collective_thrust: The summed thrust from all rotors.
        attitude: The desired drone orientation.
        dt: The dynamics time step.
    """
    rot = R.from_euler("xyz", drone.rpy)
    thrust = rot.apply(np.array([0, 0, collective_thrust]))
    drift = rot.apply(np.array([0, 0, 1]))
    acc = thrust * SYS_ID_PARAMS.acc_k1 + drift * SYS_ID_PARAMS.acc_k2 - np.array([0, 0, GRAVITY])
    roll_cmd, pitch_cmd, yaw_cmd = attitude
    roll_rate = SYS_ID_PARAMS.roll_alpha * drone.rpy[0] + SYS_ID_PARAMS.roll_beta * roll_cmd
    pitch_rate = SYS_ID_PARAMS.pitch_alpha * drone.rpy[1] + SYS_ID_PARAMS.pitch_beta * pitch_cmd
    yaw_rate = SYS_ID_PARAMS.yaw_alpha * drone.rpy[2] + SYS_ID_PARAMS.yaw_beta * yaw_cmd
    rpy_rates = np.array([roll_rate, pitch_rate, yaw_rate])
    drone.pos[:] = drone.pos + drone.vel * dt
    drone.rpy[:] = drone.rpy + rpy_rates * dt
    drone.vel[:] = drone.vel + acc * dt
    drone.ang_vel[:] = R.from_euler("xyz", drone.rpy).apply(rpy_rates)
    return []


def downwash(drone: Drone, other_drones: list[Drone]) -> list[tuple[int, ForceTorque]]:
    """Implementation of a downwash model.

    Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

    Args:
        drone: The target drone to calculate the downwash effect for.
        other_drones: List of other drones in the environment.

    Returns:
        A list of tuples containing the link id and a force/torque tuple.
    """
    ft = []
    for other_drone in other_drones:
        delta_z = drone.pos[2] - other_drone.pos[2]
        delta_xy = np.linalg.norm(drone.pos[:2] - other_drone.pos[:2])
        if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
            alpha = (
                other_drone.params.dw_coeff_1
                * (other_drone.params.prop_radius / (4 * delta_z)) ** 2
            )
            beta = other_drone.params.dw_coeff_2 * delta_z + other_drone.params.dw_coeff_3
            force = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
            ft.append((4, ForceTorque(force, [0, 0, 0])))
    return ft


def drag(drone: Drone, rpms: NDArray[np.floating]) -> list[tuple[int, ForceTorque]]:
    """Implementation of a drag model.

    Based on the the system identification in (Forster, 2015).

    Args:
        drone: The target drone to calculate the drag effect for.
        rpms: The rpms to apply to the drone rotors.

    Returns:
        A list of tuples containing the link id and a force/torque tuple.
    """
    # Simple draft model applied to the base/center of mass
    drag_factors = -1 * drone.params.drag_coeff * np.sum(np.array(2 * np.pi * rpms / 60))
    drag = R.from_euler("xyz", drone.rpy).apply(drag_factors * np.array(drone.vel))
    return [(4, ForceTorque(drag, [0, 0, 0]))]


def ground_effect(
    drone: Drone, rpms: NDArray[np.floating], pyb_client: int
) -> list[tuple[int, ForceTorque]]:
    """PyBullet implementation of a ground effect model.

    Inspired by the analytical model used for comparison in (Shi et al., 2019).

    Args:
        drone: The target drone to calculate the ground effect for.
        rpms: The rpms to apply to the drone rotors.
        pyb_client: The PyBullet client id.

    Returns:
        A list of tuples containing the link id and a force/torque tuple.
    """
    s = p.getLinkStates(
        drone.id,
        linkIndices=[0, 1, 2, 3],
        computeLinkVelocity=1,
        computeForwardKinematics=1,
        physicsClientId=pyb_client,
    )
    prop_heights = np.array([s[0][0][2], s[1][0][2], s[2][0][2], s[3][0][2]])
    prop_heights = np.clip(prop_heights, drone.params.gnd_eff_min_height_clip, np.inf)
    gnd_effects = (
        rpms**2
        * drone.params.kf
        * drone.params.gnd_eff_coeff
        * (drone.params.prop_radius / (4 * prop_heights)) ** 2
    )
    ft = []
    if np.abs(drone.rpy[:2]).max() < np.pi / 2:  # Ignore when not approximately level
        for i in range(4):
            ft.append((i, ForceTorque([0, 0, gnd_effects[i]], [0, 0, 0])))
    return ft


def apply_force_torques(
    pyb_client: int,
    drone: Drone,
    force_torques: list[tuple[int, ForceTorque]],
    external_force: NDArray[np.floating] | None = None,
):
    """Apply the calculated forces and torques in simulation.

    Args:
        pyb_client: The PyBullet client id.
        drone: The target drone to apply the forces and torques to.
        force_torques: A dictionary of forces and torques for each link of the drone body.
        external_force: An optional, external force to apply to the drone body.
    """
    for link_id, ft in force_torques:
        p.applyExternalForce(
            drone.id,
            link_id,
            forceObj=ft.f,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=pyb_client,
        )
        p.applyExternalTorque(
            drone.id, link_id, torqueObj=ft.t, flags=p.LINK_FRAME, physicsClientId=pyb_client
        )
    if external_force is not None:
        p.applyExternalForce(
            drone.id,
            linkIndex=4,  # Link attached to the quadrotor's center of mass.
            forceObj=external_force,
            posObj=drone.pos,
            flags=p.WORLD_FRAME,
            physicsClientId=pyb_client,
        )


def pybullet_step(pyb_client: int, drone: Drone, mode: PhysicsMode):
    """Step the PyBullet simulation.

    Args:
        pyb_client: The PyBullet client id.
        drone: The target drone to apply the forces and torques to.
        mode: The physics mode to use for the simulation step
    """
    if mode == PhysicsMode.DYN or mode == PhysicsMode.SYS_ID:
        p.resetBasePositionAndOrientation(
            drone.id, drone.pos, p.getQuaternionFromEuler(drone.rpy), physicsClientId=pyb_client
        )
        p.resetBaseVelocity(drone.id, drone.vel, drone.ang_vel, physicsClientId=pyb_client)
    else:
        p.stepSimulation(physicsClientId=pyb_client)
