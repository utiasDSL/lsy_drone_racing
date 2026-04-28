"""Small math utilities for the attitude controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.types import Vec3


def normalize(vec: Vec3, fallback: Vec3) -> Vec3:
    """Normalize a vector, returning fallback for near-zero vectors."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def clip_norm(vec: NDArray[np.float32], max_norm: float) -> NDArray[np.float32]:
    """Clip a vector by Euclidean norm."""
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-8:
        return vec.astype(np.float32)
    return (vec * (max_norm / norm)).astype(np.float32)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def slew_angle(previous: float, target: float, max_step: float) -> float:
    """Rate-limit an angle command."""
    delta = np.clip(wrap_angle(target - previous), -max_step, max_step)
    return wrap_angle(previous + float(delta))


def dedupe(points: NDArray[np.float32], min_distance: float = 1e-4) -> NDArray[np.float32]:
    """Remove consecutive near-duplicate points."""
    if len(points) == 0:
        return points.astype(np.float32)
    out = [points[0].astype(np.float32)]
    for point in points[1:]:
        candidate = point.astype(np.float32)
        if float(np.linalg.norm(candidate - out[-1])) >= min_distance:
            out.append(candidate)
    return np.asarray(out, dtype=np.float32)


def cumulative_lengths(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute cumulative arc lengths for a sampled path."""
    if len(points) <= 1:
        return np.zeros(len(points), dtype=np.float32)
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    ).astype(np.float32)


def arc_params(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute strictly increasing spline parameters."""
    params = cumulative_lengths(points)
    for idx in range(1, len(params)):
        if params[idx] <= params[idx - 1] + 1e-3:
            params[idx] = params[idx - 1] + 1e-3
    return params.astype(np.float32)
