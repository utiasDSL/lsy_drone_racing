from typing import Callable

import jax
import jax.numpy as jp
from crazyflow.sim.structs import SimData
from crazyflow.utils import leaf_replace
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


def randomize_drone_pos_fn(
    rng: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone position."""

    def randomize_drone_pos(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        drone_pos = data.states.pos + rng(subkey, shape=data.states.pos.shape)
        states = leaf_replace(data.states, mask, pos=drone_pos)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_pos


def randomize_drone_quat_fn(
    rng: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone quaternion."""

    def randomize_drone_quat(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        rpy = R.from_quat(data.states.quat).as_euler("xyz")
        quat = R.from_euler("xyz", rpy + rng(subkey, shape=rpy.shape)).as_quat()
        states = leaf_replace(data.states, mask, quat=quat)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_quat


def randomize_drone_mass_fn(
    rng: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone mass."""

    def randomize_drone_mass(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        mass = data.states.mass + rng(subkey, shape=data.params.mass.shape)
        states = leaf_replace(data.states, mask, mass=mass)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_mass


def randomize_drone_inertia_fn(
    rng: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData, Array], SimData]:
    """Create a function that randomizes the drone inertia."""

    def randomize_drone_inertia(data: SimData, mask: Array) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        J = data.params.J + rng(subkey, shape=data.params.J.shape)
        J_inv = jp.linalg.inv(J)
        states = leaf_replace(data.states, mask, J=J, J_inv=J_inv)
        return data.replace(core=data.core.replace(rng_key=key), states=states)

    return randomize_drone_inertia
