"""Adapter from Kaan cubic path output to the attitude follower path contract."""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.kafa1500_v2.types import CubicPath
from lsy_drone_racing.control.kafa1500_v2.utils import cumulative_lengths


class ReferenceAdapter:
    """Validate and translate path metadata for the attitude reference manager."""

    def __init__(self, gate_window_samples: int):
        """Store the gate window used by the downstream speed scheduler."""
        self._gate_window_samples = int(gate_window_samples)

    def adapt(self, path: CubicPath) -> CubicPath:
        """Return a path with the same geometry and attitude-compatible gate tags."""
        self._validate(path)
        return CubicPath(
            spline=path.spline,
            velocity_spline=path.velocity_spline,
            acceleration_spline=path.acceleration_spline,
            params=path.params.astype(np.float32),
            points=path.points.astype(np.float32),
            lengths=cumulative_lengths(path.points.astype(np.float32)),
            gate_indices=self._expand_sparse_gate_markers(path.gate_indices),
        )

    @staticmethod
    def _validate(path: CubicPath) -> None:
        if path.points.ndim != 2 or path.points.shape[1] != 3:
            raise ValueError("Path points must have shape (N, 3).")
        if len(path.params) != len(path.points):
            raise ValueError("Path params and points must have the same length.")
        if len(path.gate_indices) != len(path.points):
            raise ValueError("Path gate indices and points must have the same length.")
        if len(path.points) < 4:
            raise ValueError("Cubic path must contain at least four sampled points.")
        if not (
            np.all(np.isfinite(path.points))
            and np.all(np.isfinite(path.params))
            and np.all(np.isfinite(path.lengths))
        ):
            raise ValueError("Path contains non-finite values.")
        if bool(np.any(np.diff(path.params) <= 0.0)):
            raise ValueError("Path parameters must be strictly increasing.")

    def _expand_sparse_gate_markers(self, gate_indices: np.ndarray) -> np.ndarray:
        """Convert exact gate-center markers into local gate-region tags."""
        expanded = np.asarray(gate_indices, dtype=np.int32).copy()
        markers = np.flatnonzero(expanded >= 0)
        for marker in markers:
            gate_id = int(gate_indices[marker])
            start = max(0, marker - self._gate_window_samples)
            stop = min(len(expanded), marker + self._gate_window_samples + 1)
            expanded[start:stop] = gate_id
        return expanded.astype(np.int32)
