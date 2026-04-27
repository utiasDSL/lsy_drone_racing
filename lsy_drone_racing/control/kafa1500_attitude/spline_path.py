"""Strictly cubic spline construction and sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import make_interp_spline

from lsy_drone_racing.control.kafa1500_attitude.types import CubicPath
from lsy_drone_racing.control.kafa1500_attitude.utils import arc_params, cumulative_lengths, dedupe

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_attitude.config import PathConfig


class CubicPathBuilder:
    """Build and sample cubic interpolation splines."""

    def __init__(self, config: PathConfig):
        """Store path configuration."""
        self._config = config

    def build(self, controls: NDArray[np.float32], gate_ids: NDArray[np.int32]) -> CubicPath:
        """Build a strictly cubic path through the supplied controls."""
        controls = self._ensure_cubic_controls(controls)
        gate_ids = self._ensure_gate_ids(gate_ids, len(controls))
        params = arc_params(controls)
        spline = make_interp_spline(params, controls, k=3, axis=0)
        if spline.k != 3:
            raise RuntimeError("KaFa_1500_attitude requires cubic splines.")

        total = float(params[-1])
        n_samples = max(4, int(np.ceil(total / self._config.sample_spacing)) + 1)
        sample_params = np.linspace(0.0, total, n_samples, dtype=np.float32)
        points = np.asarray(spline(sample_params), dtype=np.float32)
        points[0] = controls[0]
        points[-1] = controls[-1]
        sample_gate_ids = self._sample_gate_ids(params, gate_ids, sample_params)
        return CubicPath(
            spline=spline,
            velocity_spline=spline.derivative(),
            acceleration_spline=spline.derivative(2),
            params=sample_params,
            points=points,
            lengths=cumulative_lengths(points),
            gate_indices=sample_gate_ids,
        )

    @staticmethod
    def _ensure_cubic_controls(points: NDArray[np.float32]) -> NDArray[np.float32]:
        points = dedupe(points.astype(np.float32), min_distance=0.035)
        if len(points) >= 4:
            return points
        if len(points) == 0:
            return np.zeros((4, 3), dtype=np.float32)
        if len(points) == 1:
            offsets = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.1], [0.2, 0.0, 0.1]],
                dtype=np.float32,
            )
            return (points[0] + offsets).astype(np.float32)
        if len(points) == 2:
            p0, p1 = points
            span = p1 - p0
            return np.vstack([p0, p0 + span / 3.0, p0 + 2.0 * span / 3.0, p1]).astype(
                np.float32
            )
        p0, p1, p2 = points
        insert = 0.5 * (p0 + p1)
        return np.vstack([p0, insert, p1, p2]).astype(np.float32)

    @staticmethod
    def _ensure_gate_ids(ids: NDArray[np.int32], length: int) -> NDArray[np.int32]:
        if len(ids) >= length:
            return ids[:length].astype(np.int32)
        if len(ids) == 0:
            return -np.ones(length, dtype=np.int32)
        tail = np.full(length - len(ids), int(ids[-1]), dtype=np.int32)
        return np.concatenate([ids.astype(np.int32), tail])

    @staticmethod
    def _sample_gate_ids(
        control_params: NDArray[np.float32],
        control_gate_ids: NDArray[np.int32],
        sample_params: NDArray[np.float32],
    ) -> NDArray[np.int32]:
        indices = np.searchsorted(control_params, sample_params, side="right") - 1
        indices = np.clip(indices, 0, len(control_gate_ids) - 1)
        return control_gate_ids[indices].astype(np.int32)

