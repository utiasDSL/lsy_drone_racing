"""Gate-aware safe target generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500_attitude.types import GateFrame
from lsy_drone_racing.control.kafa1500_attitude.utils import dedupe, normalize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_attitude.config import PathConfig
    from lsy_drone_racing.control.kafa1500_attitude.types import Observation, Vec3


class GateTargetPlanner:
    """Build cubic-spline control points through safe gate apertures."""

    def __init__(self, config: PathConfig):
        """Store path configuration."""
        self._config = config

    def build_control_points(self, obs: Observation, start_pos: Vec3) -> tuple[NDArray, NDArray]:
        """Return control points and the corresponding target gate index per point."""
        target_gate = max(0, int(obs["target_gate"]))
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        points: list[Vec3] = [start_pos.astype(np.float32)]
        protected: list[bool] = [True]
        gate_ids: list[int] = [-1]
        cursor = start_pos.astype(np.float32)

        if target_gate > 0:
            previous_gate = self.gate_frame(obs, target_gate - 1)
            escape = self._previous_gate_escape(cursor, previous_gate, obstacles)
            if float(np.linalg.norm(escape - cursor)) > 0.04:
                for point in self._route_segment(cursor, escape, obstacles, previous_gate.lateral):
                    points.append(point)
                    protected.append(False)
                    gate_ids.append(-1)
                cursor = escape.astype(np.float32)

        for gate_idx in range(target_gate, len(obs["gates_pos"])):
            gate = self.gate_frame(obs, gate_idx)
            pre_distance = self._per_gate(self._config.d_pre_per_gate, gate_idx, self._config.d_pre)
            post_distance = self._per_gate(
                self._config.d_post_per_gate, gate_idx, self._config.d_post
            )
            pre_gate = gate.traversal - pre_distance * gate.forward
            post_gate = gate.traversal + post_distance * gate.forward
            pass_gate = gate.traversal + self._config.d_pass * gate.forward

            approach = self._choose_approach(pre_gate, gate, obstacles)
            for point in self._route_segment(cursor, approach, obstacles, gate.lateral):
                points.append(point)
                protected.append(False)
                gate_ids.append(gate_idx)
            points.append(gate.traversal.astype(np.float32))
            protected.append(True)
            gate_ids.append(gate_idx)

            exit_cursor = gate.traversal.astype(np.float32)
            for exit_target in (post_gate, pass_gate):
                for point in self._route_segment(
                    exit_cursor, exit_target, obstacles, gate.lateral
                ):
                    points.append(point)
                    protected.append(False)
                    gate_ids.append(gate_idx)
                exit_cursor = exit_target.astype(np.float32)
            cursor = exit_cursor

        controls = dedupe(np.asarray(points, dtype=np.float32), min_distance=0.035)
        protected_array = self._align_protected_points(points, protected, controls)
        controls = self._smooth_controls(controls, obstacles, protected_array)
        gate_ids_array = self._align_gate_ids(points, gate_ids, controls)
        return controls, gate_ids_array

    def _previous_gate_escape(
        self, start: Vec3, gate: GateFrame, obstacles: NDArray[np.float32]
    ) -> Vec3:
        """Move clear of the just-passed gate before climbing toward the next gate."""
        rel = start - gate.traversal
        progress = float(rel @ gate.forward)
        if abs(progress) > self._config.previous_gate_escape_window:
            return start.astype(np.float32)
        target_progress = max(self._config.previous_gate_escape_distance, progress + 0.12)
        lateral = float(rel @ gate.lateral)
        vertical = float(rel @ gate.up)
        vertical = min(vertical, gate.safe_half_height)
        base = (
            gate.traversal
            + target_progress * gate.forward
            + lateral * gate.lateral
            + vertical * gate.up
        ).astype(np.float32)

        best = base
        best_score = np.inf
        for offset in self._config.previous_gate_escape_lateral_offsets:
            candidate = (base + float(offset) * gate.lateral).astype(np.float32)
            if not self._point_clear(candidate, obstacles):
                continue
            if self._segment_blocked(start, candidate, obstacles):
                continue
            score = abs(float(offset))
            if score < best_score:
                best = candidate
                best_score = score
        return best.astype(np.float32)

    def gate_frame(self, obs: Observation, gate_idx: int) -> GateFrame:
        """Extract one gate frame from observations."""
        position = obs["gates_pos"][gate_idx].astype(np.float32)
        rotation = R.from_quat(obs["gates_quat"][gate_idx]).as_matrix().astype(np.float32)
        forward = normalize(rotation[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        lateral = normalize(rotation[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        up = normalize(rotation[:, 2], np.array([0.0, 0.0, 1.0], dtype=np.float32))
        safe_half_width = max(
            0.035, 0.5 * self._config.gate_inner_width - self._config.gate_safety_margin
        )
        safe_half_height = max(
            0.035, 0.5 * self._config.gate_inner_height - self._config.gate_safety_margin
        )
        vertical_offset = float(
            np.clip(
                self._config.vertical_reference_offset,
                -safe_half_height + 0.025,
                safe_half_height - 0.025,
            )
        )
        traversal = (position + vertical_offset * up).astype(np.float32)
        return GateFrame(
            index=gate_idx,
            position=position,
            forward=forward,
            lateral=lateral,
            up=up,
            traversal=traversal,
            safe_half_width=float(safe_half_width),
            safe_half_height=float(safe_half_height),
        )

    def validate_gate_clearance(self, points: NDArray[np.float32], obs: Observation) -> bool:
        """Check that sampled path points near each gate plane remain inside the safe aperture."""
        target_gate = max(0, int(obs["target_gate"]))
        for gate_idx in range(target_gate, len(obs["gates_pos"])):
            gate = self.gate_frame(obs, gate_idx)
            local = self._to_gate_local(points, gate)
            near_plane = np.abs(local[:, 0]) <= 0.11
            if not bool(np.any(near_plane)):
                return False
            near = local[near_plane]
            inside = (np.abs(near[:, 1]) <= gate.safe_half_width) & (
                np.abs(near[:, 2]) <= gate.safe_half_height
            )
            if not bool(np.all(inside)):
                return False
        return True

    def _choose_approach(
        self, nominal: Vec3, gate: GateFrame, obstacles: NDArray[np.float32]
    ) -> Vec3:
        """Shift a pre-gate point laterally when an obstacle blocks the nominal corridor."""
        offsets = [0.0]
        for radius in np.linspace(self._config.obstacle_clearance, 0.42, 4):
            offsets.extend([float(radius), -float(radius)])

        best = nominal.astype(np.float32)
        best_score = np.inf
        for offset in offsets:
            candidate = (nominal + offset * gate.lateral).astype(np.float32)
            if self._segment_blocked(candidate, gate.traversal, obstacles):
                continue
            if not self._point_clear(candidate, obstacles):
                continue
            score = abs(offset)
            if score < best_score:
                best_score = score
                best = candidate
        return best

    def _route_segment(
        self, start: Vec3, end: Vec3, obstacles: NDArray[np.float32], preferred_lateral: Vec3
    ) -> list[Vec3]:
        """Add a few obstacle bypass control points; the final trajectory remains cubic."""
        cursor = start.astype(np.float32)
        route: list[Vec3] = []
        for _ in range(self._config.max_bypass_points):
            blocker = self._segment_blocker(cursor, end, obstacles)
            if blocker is None:
                break
            bypass = self._bypass(cursor, end, blocker, preferred_lateral)
            if float(np.linalg.norm(bypass - cursor)) < 0.05:
                break
            route.append(bypass)
            cursor = bypass
        route.append(end.astype(np.float32))
        return route

    def _bypass(
        self, start: Vec3, end: Vec3, blocker: tuple[Vec3, float], preferred_lateral: Vec3
    ) -> Vec3:
        """Create one bypass point around an obstacle pole."""
        obstacle, ratio = blocker
        segment = end[:2] - start[:2]
        direction = normalize(
            np.array([segment[0], segment[1], 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )[:2]
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
        preferred = normalize(preferred_lateral, np.array([0.0, 1.0, 0.0], dtype=np.float32))[:2]
        side = 1.0 if float(np.dot(perpendicular, preferred)) >= 0.0 else -1.0
        z = float(start[2] + ratio * (end[2] - start[2]))
        bypass_radius = self._config.obstacle_clearance + self._config.bypass_extra
        xy = obstacle[:2] + side * bypass_radius * perpendicular
        return np.array([xy[0], xy[1], z], dtype=np.float32)

    def _segment_blocker(
        self, start: Vec3, end: Vec3, obstacles: NDArray[np.float32]
    ) -> tuple[Vec3, float] | None:
        """Return the nearest obstacle blocking the XY segment."""
        if len(obstacles) == 0:
            return None
        segment = end[:2] - start[:2]
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return None
        best_distance = self._config.obstacle_clearance
        best: tuple[Vec3, float] | None = None
        for obstacle in obstacles:
            ratio = float(np.clip(np.dot(obstacle[:2] - start[:2], segment) / denom, 0.0, 1.0))
            if ratio <= 0.04 or ratio >= 0.96:
                continue
            closest = start[:2] + ratio * segment
            distance = float(np.linalg.norm(obstacle[:2] - closest))
            if distance < best_distance:
                best_distance = distance
                best = (obstacle.astype(np.float32), ratio)
        return best

    def _segment_blocked(
        self, start: Vec3, end: Vec3, obstacles: NDArray[np.float32]
    ) -> bool:
        return self._segment_blocker(start, end, obstacles) is not None

    def _point_clear(self, point: Vec3, obstacles: NDArray[np.float32]) -> bool:
        if len(obstacles) == 0:
            return True
        distances = np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)
        return bool(float(np.min(distances)) >= self._config.obstacle_clearance)

    def _smooth_controls(
        self,
        controls: NDArray[np.float32],
        obstacles: NDArray[np.float32],
        protected: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Lightly smooth intermediate controls while preserving gate traversal points."""
        if len(controls) < 4:
            return controls.astype(np.float32)
        smoothed = controls.copy()
        for _ in range(self._config.control_smoothing_passes):
            updated = smoothed.copy()
            for idx in range(1, len(smoothed) - 1):
                if bool(protected[idx]):
                    continue
                candidate = (
                    (1.0 - self._config.control_smoothing_weight) * smoothed[idx]
                    + self._config.control_smoothing_weight
                    * 0.5
                    * (smoothed[idx - 1] + smoothed[idx + 1])
                ).astype(np.float32)
                candidate[2] = smoothed[idx, 2]
                if self._point_clear(candidate, obstacles):
                    updated[idx] = candidate
            smoothed = updated
        return dedupe(smoothed, min_distance=0.035)

    @staticmethod
    def _to_gate_local(points: NDArray[np.float32], gate: GateFrame) -> NDArray[np.float32]:
        rel = points - gate.position
        return np.column_stack([rel @ gate.forward, rel @ gate.lateral, rel @ gate.up]).astype(
            np.float32
        )

    @staticmethod
    def _per_gate(values: tuple[float, ...], gate_idx: int, default: float) -> float:
        if gate_idx < len(values):
            return float(values[gate_idx])
        return default

    @staticmethod
    def _align_gate_ids(
        original_points: list[Vec3], original_ids: list[int], controls: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        ids: list[int] = []
        source = np.asarray(original_points, dtype=np.float32)
        for point in controls:
            nearest = int(np.argmin(np.linalg.norm(source - point, axis=1)))
            ids.append(original_ids[nearest])
        return np.asarray(ids, dtype=np.int32)

    @staticmethod
    def _align_protected_points(
        original_points: list[Vec3], original_protected: list[bool], controls: NDArray[np.float32]
    ) -> NDArray[np.bool_]:
        protected: list[bool] = []
        source = np.asarray(original_points, dtype=np.float32)
        for point in controls:
            nearest = int(np.argmin(np.linalg.norm(source - point, axis=1)))
            protected.append(original_protected[nearest])
        return np.asarray(protected, dtype=np.bool_)
