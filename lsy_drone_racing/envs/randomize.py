"""Randomization functions for the simulation.

The functions in this module are inserted (compiled) into the reset function of the simulation for
efficiency. Because of this, they have to be functionally pure to work with JAX (see
https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).
"""

from typing import Callable

import jax
import jax.numpy as jp
from crazyflow.sim.structs import SimData
from crazyflow.utils import leaf_replace
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


def randomize_drone_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone position."""

    def randomize_drone_pos(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        drone_pos = data.states.pos + randomize_fn(subkey, shape=data.states.pos.shape)
        states = leaf_replace(data.states, mask, pos=drone_pos)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_pos


def randomize_drone_quat_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone quaternion."""

    def randomize_drone_quat(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        rpy = R.from_quat(data.states.quat).as_euler("xyz")
        quat = R.from_euler("xyz", rpy + randomize_fn(subkey, shape=rpy.shape)).as_quat()
        states = leaf_replace(data.states, mask, quat=quat)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_quat


def randomize_drone_mass_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone mass."""

    def randomize_drone_mass(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        mass = data.params.mass + randomize_fn(subkey, shape=data.params.mass.shape)
        params = leaf_replace(data.params, mask, mass=mass)
        return data.replace(core=data.core.replace(rng_key=key), params=params)

    return randomize_drone_mass


def randomize_drone_inertia_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone inertia."""

    def randomize_drone_inertia(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        J = data.params.J + randomize_fn(subkey, shape=data.params.J.shape)
        J_INV = jp.linalg.inv(J)
        params = leaf_replace(data.params, mask, J=J, J_INV=J_INV)
        return data.replace(core=data.core.replace(rng_key=key), params=params)

    return randomize_drone_inertia


def randomize_gate_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array], gate_ids: list[int]
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the gate position."""

    def randomize_gate_pos(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        gate_pos = data.mjx_data.mocap_pos[:, gate_ids, :]
        gate_pos = gate_pos + randomize_fn(subkey, shape=gate_pos.shape)
        mocap_pos = data.mjx_data.mocap_pos.at[:, gate_ids, :].set(gate_pos)
        mjx_data = leaf_replace(data.mjx_data, mask, mocap_pos=mocap_pos)
        return data.replace(core=data.core.replace(rng_key=key), mjx_data=mjx_data)

    return randomize_gate_pos


def randomize_gate_rpy_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array], gate_ids: list[int]
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the gate rotation."""

    def randomize_gate_rpy(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        gate_quat = data.mjx_data.mocap_quat[:, gate_ids, :][..., [1, 2, 3, 0]]
        gate_rpy = R.from_quat(gate_quat).as_euler("xyz")
        gate_rpy = gate_rpy + randomize_fn(subkey, shape=gate_rpy.shape)
        gate_quat = R.from_euler("xyz", gate_rpy).as_quat(scalar_first=True)
        mocap_quat = data.mjx_data.mocap_quat.at[:, gate_ids, :].set(gate_quat)
        mjx_data = leaf_replace(data.mjx_data, mask, mocap_quat=mocap_quat)
        return data.replace(core=data.core.replace(rng_key=key), mjx_data=mjx_data)

    return randomize_gate_rpy


def randomize_obstacle_pos_fn(
    randomize_fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array], obstacle_ids: list[int]
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the obstacle position."""

    def randomize_obstacle_pos(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        obstacle_pos = data.mjx_data.mocap_pos[:, obstacle_ids, :]
        obstacle_pos = obstacle_pos + randomize_fn(subkey, shape=obstacle_pos.shape)
        mocap_pos = data.mjx_data.mocap_pos.at[:, obstacle_ids, :].set(obstacle_pos)
        mjx_data = leaf_replace(data.mjx_data, mask, mocap_pos=mocap_pos)
        return data.replace(core=data.core.replace(rng_key=key), mjx_data=mjx_data)

    return randomize_obstacle_pos
