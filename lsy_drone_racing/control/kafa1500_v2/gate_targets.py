"""Gate-ordered sparse waypoint generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500_v2.types import GateFrame
from lsy_drone_racing.control.kafa1500_v2.utils import normalize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.config import PathConfig
    from lsy_drone_racing.control.kafa1500_v2.types import Observation, Vec3


@dataclass(frozen=True, slots=True)
class _Waypoint:
    point: Vec3
    gate_id: int
    target_gate: int
    protected: bool


@dataclass(frozen=True, slots=True)
class _Blocker:
    center: Vec3
    radius: float
    ratio: float
    is_obstacle: bool


class GateTargetPlanner:
    """Build sparse gate-ordered waypoints before cubic spline fitting."""

    def __init__(self, config: PathConfig):
        """Store path configuration."""
        self._config = config

    def build_control_points(self, obs: Observation, start_pos: Vec3) -> tuple[NDArray, NDArray]:
        """Return sparse controls and gate ids in strict race index order."""
        n_gates = len(obs["gates_pos"])
        target_gate = int(np.clip(int(obs["target_gate"]), 0, n_gates))
        gates = [self.gate_frame(obs, idx) for idx in range(n_gates)]
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float32)

        waypoints: list[_Waypoint] = [
            _Waypoint(start_pos.astype(np.float32), gate_id=-1, target_gate=target_gate, protected=True)
        ]

        # Gate local x-axis is the through-gate normal in the current project
        # convention.  Each gate contributes pre -> center -> post, so the spline
        # is constrained to cross the gate center approximately perpendicular to
        # the gate plane.
        for gate_idx in range(target_gate, n_gates):
            gate = gates[gate_idx]
            pre = gate.traversal - self._config.d_pre * gate.forward
            center = gate.traversal.astype(np.float32)
            post = gate.traversal + self._config.d_post * gate.forward
            waypoints.extend(
                [
                    _Waypoint(pre.astype(np.float32), -1, gate_idx, True),
                    _Waypoint(center, gate_idx, gate_idx, True),
                    _Waypoint(post.astype(np.float32), -1, gate_idx, True),
                ]
            )

        waypoints = self._insert_detours(waypoints, obstacles, gates)
        waypoints = self._cleanup_waypoints(waypoints)
        controls = np.asarray([waypoint.point for waypoint in waypoints], dtype=np.float32)
        gate_ids = np.asarray([waypoint.gate_id for waypoint in waypoints], dtype=np.int32)
        return controls, gate_ids

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
        return GateFrame(
            index=gate_idx,
            position=position,
            forward=forward,
            lateral=lateral,
            up=up,
            traversal=position.astype(np.float32),
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

    def _insert_detours(
        self,
        waypoints: list[_Waypoint],
        obstacles: NDArray[np.float32],
        gates: list[GateFrame],
    ) -> list[_Waypoint]:
        """Iteratively insert one detour for blocked waypoint segments."""
        routed = waypoints[:]
        for _ in range(self._config.max_detour_iterations):
            updated: list[_Waypoint] = [routed[0]]
            changed = False
            for start, end in zip(routed[:-1], routed[1:], strict=True):
                blocker = self._first_blocker(start, end, obstacles, gates)
                if blocker is not None:
                    for detour in self._detour_points(
                        start.point,
                        end.point,
                        blocker,
                        obstacles,
                        gates,
                        end.target_gate,
                    ):
                        updated.append(
                            _Waypoint(
                                detour,
                                gate_id=-1,
                                target_gate=end.target_gate,
                                protected=False,
                            )
                        )
                    changed = True
                updated.append(end)
            routed = updated
            if not changed:
                break
        return routed

    def _first_blocker(
        self,
        start: _Waypoint,
        end: _Waypoint,
        obstacles: NDArray[np.float32],
        gates: list[GateFrame],
    ) -> _Blocker | None:
        """Return the first obstacle or non-target gate too close to a segment."""
        if start.protected and end.protected and start.target_gate == end.target_gate:
            return None

        blockers: list[_Blocker] = []
        for obstacle in obstacles:
            blocker = self._segment_blocker(
                start.point,
                end.point,
                obstacle.astype(np.float32),
                self._config.saturation_radius,
                is_obstacle=True,
            )
            if blocker is not None:
                blockers.append(blocker)

        # While flying toward gate i, all other gate centers are forbidden XY
        # regions.  The current gate is excluded so pre_i -> center_i -> post_i
        # remains a valid crossing.
        for gate in gates:
            if gate.index == end.target_gate:
                continue
            blocker = self._segment_blocker(
                start.point,
                end.point,
                gate.position,
                self._config.gate_avoidance_radius,
                is_obstacle=False,
            )
            if blocker is not None:
                blockers.append(blocker)

        if not blockers:
            return None
        blockers.sort(key=lambda item: item.ratio)
        return blockers[0]

    @staticmethod
    def _segment_blocker(
        start: Vec3,
        end: Vec3,
        center: Vec3,
        radius: float,
        is_obstacle: bool,
    ) -> _Blocker | None:
        segment_xy = end[:2] - start[:2]
        denom = float(np.dot(segment_xy, segment_xy))
        if denom < 1e-9:
            return None
        ratio = float(np.clip(np.dot(center[:2] - start[:2], segment_xy) / denom, 0.0, 1.0))
        if ratio <= 0.02 or ratio >= 0.98:
            return None
        closest_xy = start[:2] + ratio * segment_xy
        if float(np.linalg.norm(center[:2] - closest_xy)) >= radius:
            return None
        return _Blocker(
            center=center.astype(np.float32),
            radius=float(radius),
            ratio=ratio,
            is_obstacle=is_obstacle,
        )

    def _detour_points(
        self,
        start: Vec3,
        end: Vec3,
        blocker: _Blocker,
        obstacles: NDArray[np.float32],
        gates: list[GateFrame],
        target_gate: int,
    ) -> list[Vec3]:
        """Choose a left/right XY detour around a cylindrical forbidden region."""
        segment_xy = end[:2] - start[:2]
        norm = float(np.linalg.norm(segment_xy))
        if norm < 1e-6:
            return [(0.5 * (start + end)).astype(np.float32)]
        direction = segment_xy / norm
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
        radius = blocker.radius + self._config.obstacle_detour_margin
        z = float(start[2] + blocker.ratio * (end[2] - start[2]))
        candidates = []
        for side in (1.0, -1.0):
            xy = blocker.center[:2] + side * radius * perpendicular
            if blocker.is_obstacle:
                # Obstacles are vertical cylinders.  Two bypass points in XY
                # keep the cubic spline from cutting through the cylinder
                # between sparse waypoints.
                span = 0.5 * blocker.radius
                ratios = (
                    float(np.clip(blocker.ratio - span / max(norm, 1e-6), 0.0, 1.0)),
                    float(np.clip(blocker.ratio + span / max(norm, 1e-6), 0.0, 1.0)),
                )
                points = [
                    np.array(
                        [
                            xy[0] - span * direction[0],
                            xy[1] - span * direction[1],
                            float(start[2] + ratios[0] * (end[2] - start[2])),
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            xy[0] + span * direction[0],
                            xy[1] + span * direction[1],
                            float(start[2] + ratios[1] * (end[2] - start[2])),
                        ],
                        dtype=np.float32,
                    ),
                ]
            else:
                points = [np.array([xy[0], xy[1], z], dtype=np.float32)]
            candidates.append(
                (self._detour_score(start, end, points, obstacles, gates, target_gate), points)
            )
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _detour_score(
        self,
        start: Vec3,
        end: Vec3,
        points: list[Vec3],
        obstacles: NDArray[np.float32],
        gates: list[GateFrame],
        target_gate: int,
    ) -> float:
        chain = [start, *points, end]
        turn_score = 0.0
        for prev, current, nxt in zip(chain[:-2], chain[1:-1], chain[2:], strict=True):
            before = current[:2] - prev[:2]
            after = nxt[:2] - current[:2]
            before_norm = float(np.linalg.norm(before))
            after_norm = float(np.linalg.norm(after))
            if before_norm < 1e-6 or after_norm < 1e-6:
                return np.inf
            before /= before_norm
            after /= after_norm
            turn_score += 1.0 - float(np.dot(before, after))
        clearance = min(
            self._minimum_clearance(point, obstacles, gates, target_gate) for point in points
        )
        if self._path_violates_obstacles(points, obstacles):
            return np.inf
        return turn_score - 0.05 * clearance

    def _minimum_clearance(
        self,
        point: Vec3,
        obstacles: NDArray[np.float32],
        gates: list[GateFrame],
        target_gate: int,
    ) -> float:
        clearances: list[float] = []
        for obstacle in obstacles:
            clearances.append(float(np.linalg.norm(point[:2] - obstacle[:2])))
        for gate in gates:
            if gate.index != target_gate:
                clearances.append(float(np.linalg.norm(point[:2] - gate.position[:2])))
        return min(clearances) if clearances else 0.0

    def _path_violates_obstacles(
        self,
        points: list[Vec3],
        obstacles: NDArray[np.float32],
    ) -> bool:
        for point in points:
            for obstacle in obstacles:
                if float(np.linalg.norm(point[:2] - obstacle[:2])) < self._config.saturation_radius:
                    return True
        return False

    def _cleanup_waypoints(self, waypoints: list[_Waypoint]) -> list[_Waypoint]:
        """Remove only duplicate or nearly collinear non-protected waypoints."""
        cleaned: list[_Waypoint] = []
        for waypoint in waypoints:
            if (
                cleaned
                and not waypoint.protected
                and float(np.linalg.norm(waypoint.point - cleaned[-1].point))
                < self._config.min_waypoint_spacing
            ):
                continue
            cleaned.append(waypoint)

        angle_threshold = np.deg2rad(self._config.collinear_angle_threshold_deg)
        idx = 1
        while idx < len(cleaned) - 1:
            waypoint = cleaned[idx]
            if waypoint.protected:
                idx += 1
                continue
            before = waypoint.point - cleaned[idx - 1].point
            after = cleaned[idx + 1].point - waypoint.point
            before_norm = float(np.linalg.norm(before))
            after_norm = float(np.linalg.norm(after))
            if before_norm < 1e-6 or after_norm < 1e-6:
                del cleaned[idx]
                continue
            angle = float(np.arccos(np.clip(np.dot(before, after) / (before_norm * after_norm), -1.0, 1.0)))
            if angle < angle_threshold:
                del cleaned[idx]
                continue
            idx += 1
        return cleaned

    @staticmethod
    def _to_gate_local(points: NDArray[np.float32], gate: GateFrame) -> NDArray[np.float32]:
        rel = points - gate.position
        return np.column_stack([rel @ gate.forward, rel @ gate.lateral, rel @ gate.up]).astype(
            np.float32
        )
