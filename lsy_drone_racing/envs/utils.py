"""Utility functions for the drone racing environments."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as JR
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from jax import Array


def load_track(track: ConfigDict) -> tuple[ConfigDict, ConfigDict, ConfigDict]:
    """Load the track from a config dict.

    Gates and obstacles are loaded as a config dicts with keys `pos`, `quat`, `nominal_pos`, and
    `nominal_quat`. Drones are loaded as a config dicts with keys `pos`, `rpy`, `quat`, `vel` and
    `ang_vel`.

    Args:
        track: The track config dict.

    Returns:
        The gates, obstacles, and drones as config dicts.
    """
    assert "gates" in track, "Track must contain gates field."
    assert "obstacles" in track, "Track must contain obstacles field."
    assert "drones" in track, "Track must contain drones field."
    gate_pos = np.array([g["pos"] for g in track.gates], dtype=np.float32)
    gate_quat = (
        R.from_euler("xyz", np.array([g["rpy"] for g in track.gates])).as_quat().astype(np.float32)
    )
    gates = {
        "pos": gate_pos,
        "quat": gate_quat,
        "nominal_pos": gate_pos.copy(),
        "nominal_quat": gate_quat.copy(),
    }
    obstacle_pos = np.array([o["pos"] for o in track.obstacles], dtype=np.float32)
    obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos.copy()}
    drones = {
        k: np.array([drone.get(k) for drone in track.drones], dtype=np.float32)
        for k in track.drones[0].keys()
    }
    drones["quat"] = R.from_euler("xyz", drones["rpy"]).as_quat().astype(np.float32)
    return ConfigDict(gates), ConfigDict(obstacles), ConfigDict(drones)


@jax.jit
@partial(vectorize, signature="(3),(3),(3),(4)->()", excluded=[4])
def gate_passed(
    drone_pos: Array,
    last_drone_pos: Array,
    gate_pos: Array,
    gate_quat: Array,
    gate_size: tuple[float, float],
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the x-Axis (pointing from -x to +x). Therefore, we check if x
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the y
    and z box boundaries. First, we linearly interpolate to get the y and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
        gate_pos: The position of the gate in the world frame.
        gate_quat: The rotation of the gate as a wxyz quaternion.
        gate_size: The size of the gate box in meters.
    """
    # Transform last and current drone position into current gate frame.
    gate_rot = JR.from_quat(gate_quat)
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    passed_plane = (last_pos_local[0] < 0) & (pos_local[0] > 0)
    alpha = -last_pos_local[0] / (pos_local[0] - last_pos_local[0])
    y_intersect = alpha * (pos_local[1]) + (1 - alpha) * last_pos_local[1]
    z_intersect = alpha * (pos_local[2]) + (1 - alpha) * last_pos_local[2]
    # Divide gate size by 2 to get the distance from the center to the edges
    in_box = (abs(y_intersect) < gate_size[0] / 2) & (abs(z_intersect) < gate_size[1] / 2)
    return passed_plane & in_box


def generate_random_track(
    track: ConfigDict,
    key: jax.random.PRNGKey,
    border_safety_margin: float = 0.5,
    start_pos_min_r: float = 1.0,
    gates_min_r: float = 1.0,
    obstacle_min_r: float = 1.0,
    corridor_width_gates: float = 0.4,
    corridor_width_obstacles: float = 0.4,
    yaw_offset_randomization: float = 0.75,
    grid_size: tuple = (40, 40),
    jitter: bool = True,
) -> ConfigDict:
    """Fully JAX-jittable random track generator.

    Args:
        track: default track layout (n_gates, n_obs, start pos etc)
        key: for randomization
        border_safety_margin: min distance [m] of all objects fom the border
        start_pos_min_r: exclusion radius around inital drone position
        gates_min_r: exclusion radius around gates
        obstacle_min_r: minimum distance of obstacles from gates
        corridor_width_gates: width of corridor between gates
        corridor_width_obstacles: width of corridor between obstacles
        yaw_offset_randomization: amount of randomization for yaw
        grid_size: tuple(H, W) grid resolution
        jitter: whether to jitter gate inside grid cell

    Returns:
        New track layout with randomized tracks
    """
    # Get infos from track
    xmin, ymin = jnp.array(track.safety_limits["pos_limit_low"][:2]) + border_safety_margin
    xmax, ymax = jnp.array(track.safety_limits["pos_limit_high"][:2]) - border_safety_margin
    start_pos = jax.random.uniform(
        key,
        (2,),
        minval=jnp.array([xmin - border_safety_margin, ymin - border_safety_margin]),
        maxval=jnp.array([xmax + border_safety_margin, ymax + border_safety_margin]),
    )

    N_gates, N_obstacles = len(track.gates), len(track.obstacles)

    H, W = grid_size
    xs = jnp.linspace(xmin, xmax, W)
    ys = jnp.linspace(ymin, ymax, H)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    coords = jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
    coords_flat = coords.reshape(-1, 2)
    cell_w = (xmax - xmin) / W
    cell_h = (ymax - ymin) / H

    # Initial mask: everything allowed except around start_pos
    mask = jnp.ones((H, W), dtype=jnp.float32)
    dist2 = jnp.sum((coords - start_pos) ** 2, axis=-1)
    start_pos_mask = dist2 > (start_pos_min_r**2)
    mask = mask * start_pos_mask

    # Preallocate arrays
    assert N_gates == N_obstacles
    gates = jnp.full((N_gates, 3), jnp.nan, dtype=jnp.float32)
    obstacles = jnp.full((N_obstacles, 2), jnp.nan, dtype=jnp.float32)
    gate_distance_mask = jnp.ones((H, W), dtype=jnp.float32)
    gate_distance_mask_obstacles = jnp.ones((H, W), dtype=jnp.float32)
    obstacle_distance_mask = jnp.ones((H, W), dtype=jnp.float32)
    corridor_mask = jnp.ones((H, W), dtype=jnp.bool)

    # PRNG keys
    keys = jax.random.split(key, 2 * N_gates + 1)
    key_pos, key_yaw = keys[::2], keys[1::2]
    # --- Sample obstacles ---
    keys_obs = jax.random.split(keys[-1], N_obstacles)

    # --- Helper: yaw adjustment ---
    def adjust_yaw(i: int, yaw: jnp.floating, gates: Array, candidate: Array) -> jnp.floating:
        prev_pos = jax.lax.cond(
            i == 0, lambda _: start_pos, lambda _: gates[i - 1, :2], operand=None
        )
        travel_dir = candidate - prev_pos
        yaw += jnp.arctan2(travel_dir[1], travel_dir[0])
        return yaw % (2 * jnp.pi)

    # --- Scan body for gate placement ---
    def body(carry: tuple, i: int) -> tuple:
        (
            mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask_obstacles,
        ) = carry
        sub_pos, sub_yaw = key_pos[i], key_yaw[i]
        sub_obs = keys_obs[i]

        flat_mask = mask.reshape(-1)
        total = flat_mask.sum()
        p = jnp.where(total > 0, flat_mask / total, jnp.ones_like(flat_mask) / flat_mask.size)
        idx = jax.random.choice(sub_pos, flat_mask.shape[0], p=p)
        chosen_center = coords_flat[idx]

        # optional jitter
        if jitter:
            sub_pos, subk1, subk2 = jax.random.split(sub_pos, 3)
            off_x = (jax.random.uniform(subk1, ()) - 0.5) * cell_w
            off_y = (jax.random.uniform(subk2, ()) - 0.5) * cell_h
            candidate = chosen_center + jnp.array([off_x, off_y])
        else:
            candidate = chosen_center

        # sample yaw and adjust
        yaw = jax.random.uniform(
            sub_yaw, (), minval=-yaw_offset_randomization, maxval=yaw_offset_randomization
        )
        # yaw = adjust_yaw(i, yaw, gates, candidate)
        yaw = adjust_yaw(i, yaw, gates, candidate)
        # yaw = adjust_yaw(i, gates, candidate)

        gates = gates.at[i].set(jnp.array([candidate[0], candidate[1], yaw]))

        # mask out circular region around gate
        dist2 = jnp.sum((coords - candidate) ** 2, axis=-1)
        gate_distance_mask = gate_distance_mask * (dist2 > (gates_min_r**2))
        gate_distance_mask_obstacles = gate_distance_mask_obstacles * (dist2 > (obstacle_min_r**2))

        # mask out corridor from prev gate or start
        prev_pos = jax.lax.cond(i == 0, lambda _: start_pos, lambda _: gates[i - 1, :2], None)
        v = candidate - prev_pos
        v_norm = jnp.linalg.norm(v) + 1e-8
        u = v / v_norm
        p_to_line = coords - prev_pos
        proj = jnp.sum(p_to_line * u, axis=-1)

        proj_exp = proj[..., None]  # shape (H, W, 1)
        closest = prev_pos + proj_exp * u  # shape (H, W, 2)
        perp_dist = jnp.linalg.norm(coords - closest, axis=-1)  # shape (H, W)

        on_segment = (proj >= 0) & (proj <= v_norm)
        corridor_mask_gates = (perp_dist < corridor_width_gates) & on_segment
        corridor_mask_obstacles = (perp_dist < corridor_width_obstacles) & on_segment

        new_mask = mask * gate_distance_mask
        new_mask = new_mask * (1.0 - corridor_mask_gates)

        # mask_corridors = jnp.maximum(mask_corridors, corridor_mask.astype(jnp.float32))
        mask_corridors = (
            corridor_mask_obstacles
            * gate_distance_mask_obstacles
            * obstacle_distance_mask
            * start_pos_mask
        )

        # sample obstacle pos
        flat_mask = mask_corridors.reshape(-1)
        total = flat_mask.sum()
        p = jnp.where(total > 0, flat_mask / total, jnp.ones_like(flat_mask) / flat_mask.size)
        idx = jax.random.choice(sub_obs, flat_mask.shape[0], p=p)
        chosen_center = coords_flat[idx]

        candidate = chosen_center

        obstacles = obstacles.at[i].set(jnp.array([candidate[0], candidate[1]]))

        # mask out circular region around obstacle
        dist2 = jnp.sum((coords - candidate) ** 2, axis=-1)
        obstacle_distance_mask = obstacle_distance_mask * (dist2 > (obstacle_min_r**2))

        return (
            new_mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask_obstacles,
            # jnp.bool(mask_corridors),
        ), None

    (
        (
            mask_final,
            gates_final,
            obstacles_final,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask,
        ),
        _,
    ) = jax.lax.scan(
        body,
        (
            mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask,
        ),
        jnp.arange(N_gates),
    )

    # Write random track
    for i, d in enumerate(track.drones):
        d["pos"][:2] = start_pos
        # TODO multi drones?

    for i, g in enumerate(track.gates):
        g["pos"][:2] = gates_final[i, :2].tolist()
        g["rpy"][2] = gates_final[i, 2].tolist()

    for i, o in enumerate(track.obstacles):
        o["pos"][:2] = obstacles_final[i, :2].tolist()

    return track
