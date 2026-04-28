"""Strictly cubic spline construction and sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import make_interp_spline

from lsy_drone_racing.control.kafa1500_v2.types import CubicPath
from lsy_drone_racing.control.kafa1500_v2.utils import arc_params, cumulative_lengths, dedupe

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.config import PathConfig


class CubicPathBuilder:
    """Build and sample cubic interpolation splines."""

    def __init__(self, config: PathConfig):
        """Store path configuration."""
        self._config = config

    def build(self, controls: NDArray[np.float32], gate_ids: NDArray[np.int32]) -> CubicPath:
        """Build and densely sample a spline from sparse path waypoints."""
        # Sparse controls are cleaned before fitting; dense samples are produced
        # afterward for reference tracking.  Spline parameters are cumulative arc
        # length, which avoids uneven behavior from raw waypoint indices.
        controls, gate_ids = self._prepare_controls(controls, gate_ids)
        controls = self._ensure_cubic_controls(controls)
        gate_ids = self._ensure_gate_ids(gate_ids, len(controls))
        params = arc_params(controls)
        spline = make_interp_spline(params, controls, k=3, axis=0)
        if spline.k != 3:
            raise RuntimeError("KaFa_1500_v2 requires cubic splines.")

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

    def _prepare_controls(
        self,
        controls: NDArray[np.float32],
        gate_ids: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Remove consecutive duplicates while preserving gate id alignment."""
        if len(controls) == 0:
            return controls.astype(np.float32), gate_ids.astype(np.int32)

        clean_points = [controls[0].astype(np.float32)]
        clean_ids = [int(gate_ids[0]) if len(gate_ids) else -1]
        for idx, point in enumerate(controls[1:], start=1):
            candidate = point.astype(np.float32)
            gate_id = int(gate_ids[idx]) if idx < len(gate_ids) else clean_ids[-1]
            if (
                gate_id < 0
                and
                float(np.linalg.norm(candidate - clean_points[-1]))
                < self._config.min_waypoint_spacing * 0.25
            ):
                continue
            clean_points.append(candidate)
            clean_ids.append(gate_id)
        return np.asarray(clean_points, dtype=np.float32), np.asarray(clean_ids, dtype=np.int32)

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
        sample_gate_ids = -np.ones(len(sample_params), dtype=np.int32)
        for control_param, gate_id in zip(control_params, control_gate_ids, strict=True):
            if int(gate_id) < 0:
                continue
            sample_idx = int(np.argmin(np.abs(sample_params - control_param)))
            sample_gate_ids[sample_idx] = int(gate_id)
        return sample_gate_ids
