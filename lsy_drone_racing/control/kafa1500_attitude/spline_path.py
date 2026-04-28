"""Strictly cubic spline construction and sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.kafa1500_attitude.types import CubicPath
from lsy_drone_racing.control.kafa1500_attitude.utils import arc_params, cumulative_lengths, dedupe

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_attitude.config import PathConfig


class CubicPathBuilder:
    """Build and sample local cubic interpolation splines."""

    def __init__(self, config: PathConfig):
        """Store path configuration."""
        self._config = config

    def build(self, controls: NDArray[np.float32], gate_ids: NDArray[np.int32]) -> CubicPath:
        """Build a local cubic path through the supplied controls."""
        controls = self._ensure_cubic_controls(controls)
        gate_ids = self._ensure_gate_ids(gate_ids, len(controls))
        params = arc_params(controls)
        tangents = self._limited_tangents(controls, params)

        total = float(params[-1])
        n_samples = max(4, int(np.ceil(total / self._config.sample_spacing)) + 1)
        sample_params = np.linspace(0.0, total, n_samples, dtype=np.float32)
        spline = CubicHermiteSpline(params, controls, tangents, axis=0)
        points = np.asarray(spline(sample_params), dtype=np.float32)
        for _ in range(5):
            if not self._has_xy_loop(points):
                break
            tangents *= 0.5
            spline = CubicHermiteSpline(params, controls, tangents, axis=0)
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
    def _limited_tangents(
        controls: NDArray[np.float32], params: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Use local Hermite tangents that cannot swing past waypoint chords."""
        spans = np.diff(controls, axis=0)
        dt = np.diff(params).reshape(-1, 1)
        slopes = spans / np.maximum(dt, 1e-6)
        tangents = np.zeros_like(controls, dtype=np.float32)

        for idx in range(1, len(controls) - 1):
            prev = slopes[idx - 1]
            nxt = slopes[idx]
            prev_norm = float(np.linalg.norm(prev))
            next_norm = float(np.linalg.norm(nxt))
            if prev_norm < 1e-6 or next_norm < 1e-6:
                continue

            prev_dir = prev / prev_norm
            next_dir = nxt / next_norm
            turn_cos = float(np.clip(prev_dir @ next_dir, -1.0, 1.0))
            if turn_cos <= 0.92:
                continue

            tangent = 0.5 * (prev + nxt)
            tangent_norm = float(np.linalg.norm(tangent))
            max_norm = 0.25 * min(prev_norm, next_norm) * turn_cos
            if tangent_norm > max_norm and tangent_norm > 1e-6:
                tangent *= max_norm / tangent_norm
            tangents[idx] = tangent.astype(np.float32)

        return tangents.astype(np.float32)

    @staticmethod
    def _has_xy_loop(points: NDArray[np.float32]) -> bool:
        """Detect sampled XY self-intersections caused by spline overshoot."""
        if len(points) < 5:
            return False
        xy = points[:, :2]
        for i in range(len(xy) - 1):
            a0, a1 = xy[i], xy[i + 1]
            if float(np.linalg.norm(a1 - a0)) < 1e-6:
                continue
            for j in range(i + 2, len(xy) - 1):
                if j == i + 1:
                    continue
                if i == 0 and j == len(xy) - 2:
                    continue
                b0, b1 = xy[j], xy[j + 1]
                if float(np.linalg.norm(b1 - b0)) < 1e-6:
                    continue
                if CubicPathBuilder._segments_intersect_xy(a0, a1, b0, b1):
                    return True
        return False

    @staticmethod
    def _segments_intersect_xy(
        a0: NDArray[np.float32],
        a1: NDArray[np.float32],
        b0: NDArray[np.float32],
        b1: NDArray[np.float32],
    ) -> bool:
        def orient(p: NDArray[np.float32], q: NDArray[np.float32], r: NDArray[np.float32]) -> float:
            return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

        o1 = orient(a0, a1, b0)
        o2 = orient(a0, a1, b1)
        o3 = orient(b0, b1, a0)
        o4 = orient(b0, b1, a1)
        eps = 1e-6
        return (o1 * o2 < -eps) and (o3 * o4 < -eps)

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
            return np.vstack([p0, p0 + span / 3.0, p0 + 2.0 * span / 3.0, p1]).astype(np.float32)
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
