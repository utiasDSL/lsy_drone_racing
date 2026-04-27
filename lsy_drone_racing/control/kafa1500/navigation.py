"""Gate-local planning, obstacle routing, and path following."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500.settings import PlannerSettings
from lsy_drone_racing.control.kafa1500.types import (
    GateFrame,
    GatePlan,
    Observation,
    PathTarget,
    Vec3,
)


class GateNavigator:
    """Build and follow a local gate-centric path."""

    def __init__(self, settings: PlannerSettings):
        """Store the planner settings."""

        self._settings = settings

    def plan_gate(self, obs: Observation, gate_idx: int) -> GatePlan:
        """Create a fresh local plan for the current target gate."""

        pos = obs["pos"].astype(np.float32)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        gate = self._gate_frame(obs, gate_idx)

        align_target, approach_target = self._choose_gate_targets(gate, obstacles)
        pass_target = gate.position + self._settings.pass_distance * gate.forward

        route_points = self._plan_route(pos, align_target, approach_target, gate, obstacles)
        anchors = np.vstack([pos, *route_points]).astype(np.float32)
        control_points = self._build_smooth_path(anchors, obstacles)
        path_spline, path_spline_d1, path_params, path_points, path_lengths = (
            self._build_path_spline(control_points, obstacles)
        )
        route_line = np.vstack([path_points, gate.position, pass_target]).astype(np.float32)

        return GatePlan(
            gate_idx=gate_idx,
            gate_pos=gate.position,
            gate_x=gate.forward,
            pass_target=pass_target,
            path_spline=path_spline,
            path_spline_d1=path_spline_d1,
            path_params=path_params,
            path_points=path_points,
            path_lengths=path_lengths,
            route_line=route_line,
        )

    def _choose_gate_targets(
        self,
        gate: GateFrame,
        obstacles: NDArray[np.float32],
    ) -> tuple[Vec3, Vec3]:
        """Pick gate approach targets, shifting laterally if the nominal corridor is blocked."""

        base_align = gate.position - self._settings.align_distance * gate.forward
        base_approach = gate.position - self._settings.approach_distance * gate.forward

        candidate_offsets = [0.0]
        offset_radius = np.linspace(
            self._settings.obstacle_clearance + 0.06,
            0.65,
            8,
            dtype=np.float32,
        )
        for radius in offset_radius:
            candidate_offsets.extend([float(radius), -float(radius)])

        best_pair: tuple[Vec3, Vec3] | None = None
        best_score = np.inf
        for offset in candidate_offsets:
            align = (base_align + offset * gate.lateral).astype(np.float32)
            approach = (base_approach + offset * gate.lateral).astype(np.float32)
            if not self._point_is_clear(align, obstacles, self._settings.obstacle_clearance):
                continue
            if not self._point_is_clear(approach, obstacles, self._settings.obstacle_clearance):
                continue
            if self._segment_blocker(
                align,
                approach,
                obstacles,
                self._settings.obstacle_clearance,
            ):
                continue
            if self._segment_blocker(
                approach,
                gate.position,
                obstacles,
                self._settings.obstacle_clearance,
            ):
                continue

            score = abs(offset)
            if score < best_score:
                best_score = score
                best_pair = (align, approach)

        if best_pair is not None:
            return best_pair
        return base_align.astype(np.float32), base_approach.astype(np.float32)

    def follow_path(self, pos: Vec3, plan: GatePlan) -> PathTarget:
        """Project onto the current plan and sample a spline lookahead target."""

        projected = self._project_onto_path(
            pos,
            plan.path_points,
            plan.path_lengths,
            plan.progress,
        )
        plan.progress = max(plan.progress, projected)

        remaining = float(plan.path_lengths[-1] - plan.progress)
        lookahead = (
            self._settings.pre_gate_lookahead
            if remaining < 0.35
            else self._settings.path_lookahead
        )
        target_s = min(float(plan.path_lengths[-1]), plan.progress + lookahead)
        target_param = self._arc_length_to_param(target_s, plan.path_lengths, plan.path_params)
        target = np.asarray(plan.path_spline(target_param), dtype=np.float32)
        yaw_dir = np.asarray(plan.path_spline_d1(target_param), dtype=np.float32)
        yaw_norm = float(np.linalg.norm(yaw_dir))
        yaw_dir = plan.gate_x if yaw_norm < 1e-6 else yaw_dir / yaw_norm
        return PathTarget(
            target=target.astype(np.float32),
            yaw_dir=yaw_dir.astype(np.float32),
            remaining=remaining,
        )

    def segment_blocked(
        self,
        start: Vec3,
        end: Vec3,
        obstacles_pos: NDArray[np.floating],
        clearance: float | None = None,
    ) -> bool:
        """Check whether any obstacle blocks a straight segment."""

        obstacles = np.asarray(obstacles_pos, dtype=np.float32)
        segment_clearance = self._settings.obstacle_clearance if clearance is None else clearance
        blocker = self._segment_blocker(
            start.astype(np.float32),
            end.astype(np.float32),
            obstacles,
            segment_clearance,
        )
        return blocker is not None

    def _gate_frame(self, obs: Observation, gate_idx: int) -> GateFrame:
        """Return the gate center and its local forward/lateral axes."""

        gate_pos = obs["gates_pos"][gate_idx].astype(np.float32)
        gate_quat = obs["gates_quat"][gate_idx]
        rotation = R.from_quat(gate_quat).as_matrix().astype(np.float32)
        gate_x = rotation[:, 0]
        gate_y = rotation[:, 1]
        gate_x /= np.linalg.norm(gate_x)
        gate_y /= np.linalg.norm(gate_y)
        return GateFrame(
            position=gate_pos,
            forward=gate_x.astype(np.float32),
            lateral=gate_y.astype(np.float32),
        )

    def _plan_route(
        self,
        start_pos: Vec3,
        align_target: Vec3,
        approach_target: Vec3,
        gate: GateFrame,
        obstacles: NDArray[np.float32],
    ) -> list[Vec3]:
        """Build a short route that stays clear of nearby obstacle poles."""

        route_targets = [
            align_target.astype(np.float32),
            approach_target.astype(np.float32),
        ]

        cursor = start_pos.astype(np.float32)
        route_points: list[Vec3] = []
        for target in route_targets:
            segment_points = self._visibility_route(cursor, target, obstacles, gate.lateral)
            route_points.extend(segment_points)
            cursor = segment_points[-1]

        deduped: list[Vec3] = []
        for point in route_points:
            if not deduped or np.linalg.norm(point - deduped[-1]) > 0.05:
                deduped.append(point.astype(np.float32))

        simplified = self._shortcut_route(
            np.vstack([start_pos.astype(np.float32), *deduped]).astype(np.float32),
            obstacles,
        )
        relaxed = self._relax_control_points(simplified, obstacles)
        return [point.astype(np.float32) for point in relaxed[1:]]

    def _visibility_route(
        self,
        start: Vec3,
        end: Vec3,
        obstacles: NDArray[np.float32],
        preferred_lateral: Vec3,
    ) -> list[Vec3]:
        """Plan one obstacle-avoiding segment with a visibility graph."""

        planning_clearance = (
            self._settings.obstacle_clearance + self._settings.visibility_clearance_margin
        )
        if self._segment_blocker(start, end, obstacles, planning_clearance) is None:
            return [end.astype(np.float32)]

        nodes = [start.astype(np.float32), end.astype(np.float32)]
        for obstacle in obstacles:
            nodes.extend(
                self._obstacle_visibility_nodes(start, end, obstacle.astype(np.float32), obstacles)
            )

        node_count = len(nodes)
        adjacency: list[list[tuple[float, int]]] = [[] for _ in range(node_count)]
        for src_idx in range(node_count):
            for dst_idx in range(src_idx + 1, node_count):
                if self._segment_blocker(
                    nodes[src_idx],
                    nodes[dst_idx],
                    obstacles,
                    planning_clearance,
                ):
                    continue
                distance = float(np.linalg.norm(nodes[dst_idx] - nodes[src_idx]))
                if distance < 1e-4:
                    continue
                adjacency[src_idx].append((distance, dst_idx))
                adjacency[dst_idx].append((distance, src_idx))

        path = self._shortest_node_path(adjacency, nodes)
        if path is None:
            return self._route_segment(start, end, obstacles, preferred_lateral)

        simplified = self._shortcut_route(
            np.asarray(path, dtype=np.float32),
            obstacles,
            planning_clearance,
        )
        return [point.astype(np.float32) for point in simplified[1:]]

    def _obstacle_visibility_nodes(
        self,
        start: Vec3,
        end: Vec3,
        obstacle: Vec3,
        obstacles: NDArray[np.float32],
    ) -> list[Vec3]:
        """Generate candidate graph nodes around one obstacle."""

        radius = self._settings.obstacle_clearance + self._settings.visibility_clearance_margin
        segment = end[:2] - start[:2]
        base_angle = float(np.arctan2(segment[1], segment[0])) if np.linalg.norm(segment) > 1e-6 else 0.0
        angles = np.linspace(0.0, 2.0 * np.pi, self._settings.visibility_node_samples, endpoint=False)
        candidates: list[Vec3] = []

        for angle in base_angle + angles:
            xy = obstacle[:2] + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            ratio = self._project_ratio_xy(xy, start[:2], end[:2])
            z = float(start[2] + ratio * (end[2] - start[2]))
            candidate = np.array([xy[0], xy[1], z], dtype=np.float32)
            if self._point_is_clear(candidate, obstacles, radius - 1e-3):
                candidates.append(candidate.astype(np.float32))

        return candidates

    @staticmethod
    def _shortest_node_path(
        adjacency: list[list[tuple[float, int]]],
        nodes: list[Vec3],
    ) -> list[Vec3] | None:
        """Run Dijkstra on the visibility graph from node 0 to node 1."""

        goal_idx = 1
        distances = [np.inf] * len(nodes)
        previous = [-1] * len(nodes)
        distances[0] = 0.0
        heap: list[tuple[float, int]] = [(0.0, 0)]

        while heap:
            current_dist, current_idx = heappop(heap)
            if current_dist > distances[current_idx]:
                continue
            if current_idx == goal_idx:
                break
            for edge_cost, neighbor_idx in adjacency[current_idx]:
                next_dist = current_dist + edge_cost
                if next_dist >= distances[neighbor_idx]:
                    continue
                distances[neighbor_idx] = next_dist
                previous[neighbor_idx] = current_idx
                heappush(heap, (next_dist, neighbor_idx))

        if not np.isfinite(distances[goal_idx]):
            return None

        path_indices = []
        cursor = goal_idx
        while cursor != -1:
            path_indices.append(cursor)
            cursor = previous[cursor]
        path_indices.reverse()
        return [nodes[idx].astype(np.float32) for idx in path_indices]

    def _route_segment(
        self,
        start: Vec3,
        end: Vec3,
        obstacles: NDArray[np.float32],
        preferred_lateral: Vec3,
    ) -> list[Vec3]:
        """Expand a segment with bypass waypoints when an obstacle blocks it."""

        cursor = start.astype(np.float32)
        segment_points: list[Vec3] = []
        for _ in range(self._settings.max_route_points):
            blocker = self._segment_blocker(
                cursor,
                end,
                obstacles,
                self._settings.obstacle_clearance,
            )
            if blocker is None:
                break

            bypass = self._bypass_waypoint(cursor, end, blocker, obstacles, preferred_lateral)
            if np.linalg.norm(bypass - cursor) < 0.05 or np.linalg.norm(bypass - end) < 0.05:
                break

            segment_points.append(bypass.astype(np.float32))
            cursor = bypass.astype(np.float32)

        segment_points.append(end.astype(np.float32))
        return segment_points

    def _gate_swing_waypoint(
        self,
        gate: GateFrame,
        obstacles: NDArray[np.float32],
    ) -> Vec3 | None:
        """Insert a lateral swing waypoint if the gate corridor is blocked."""

        found_blocker = False
        best_lateral = 0.0
        best_along = 0.0
        min_abs_lateral = self._settings.obstacle_clearance + 0.04
        gate_x_xy = gate.forward[:2]
        gate_y_xy = gate.lateral[:2]

        for obstacle in obstacles:
            rel = obstacle[:2] - gate.position[:2]
            along = float(np.dot(rel, gate_x_xy))
            lateral = float(np.dot(rel, gate_y_xy))
            if (
                -(self._settings.align_distance + 0.12) < along < 0.10
                and abs(lateral) < min_abs_lateral
            ):
                found_blocker = True
                best_lateral = lateral
                best_along = along
                min_abs_lateral = abs(lateral)

        if not found_blocker:
            return None

        side = -1.0 if best_lateral > 0.0 else 1.0
        swing_xy = gate.position[:2] + (best_along - 0.12) * gate_x_xy
        swing_xy = swing_xy + side * (self._settings.obstacle_clearance + 0.18) * gate_y_xy
        return np.array([swing_xy[0], swing_xy[1], gate.position[2]], dtype=np.float32)

    def _segment_blocker(
        self,
        start: Vec3,
        end: Vec3,
        obstacles: NDArray[np.float32],
        clearance: float,
    ) -> tuple[Vec3, float, NDArray[np.float32]] | None:
        """Return the closest obstacle that violates the segment clearance."""

        segment = end[:2] - start[:2]
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return None

        best_distance = clearance
        best_blocker: tuple[Vec3, float, NDArray[np.float32]] | None = None
        for obstacle in obstacles:
            offset = obstacle[:2] - start[:2]
            t_clamped = float(np.clip(np.dot(offset, segment) / denom, 0.0, 1.0))
            if t_clamped <= 0.03 or t_clamped >= 0.97:
                continue

            closest_xy = start[:2] + t_clamped * segment
            distance = float(np.linalg.norm(obstacle[:2] - closest_xy))
            if distance < best_distance:
                best_distance = distance
                best_blocker = (
                    obstacle.astype(np.float32),
                    t_clamped,
                    closest_xy.astype(np.float32),
                )
        return best_blocker

    def _bypass_waypoint(
        self,
        start: Vec3,
        end: Vec3,
        blocker: tuple[Vec3, float, NDArray[np.float32]],
        obstacles: NDArray[np.float32],
        preferred_lateral: Vec3,
    ) -> Vec3:
        """Place a short bypass waypoint around one blocking obstacle."""

        _, t_clamped, closest_xy = blocker
        segment = end[:2] - start[:2]
        segment_norm = float(np.linalg.norm(segment))
        if segment_norm < 1e-6:
            return end.astype(np.float32)

        segment_u = segment / segment_norm
        perp = np.array([-segment_u[1], segment_u[0]], dtype=np.float32)
        preferred_xy = preferred_lateral[:2]
        preferred_norm = float(np.linalg.norm(preferred_xy))
        if preferred_norm > 1e-6:
            preferred_xy = preferred_xy / preferred_norm

        z_coord = float(start[2] + t_clamped * (end[2] - start[2]))
        candidates: list[tuple[float, Vec3]] = []

        for side in (-1.0, 1.0):
            xy = closest_xy + side * perp * (self._settings.obstacle_clearance + 0.14)
            xy = xy + segment_u * self._settings.route_progress
            candidate = np.array([xy[0], xy[1], z_coord], dtype=np.float32)

            score = float(np.linalg.norm(candidate[:2] - end[:2]))
            score += 0.25 * float(np.linalg.norm(candidate[:2] - start[:2]))
            if preferred_norm > 1e-6:
                score -= 0.20 * side * float(np.dot(perp, preferred_xy))

            for other in obstacles:
                distance = float(np.linalg.norm(candidate[:2] - other[:2]))
                if distance < self._settings.obstacle_clearance + 0.04:
                    score += 12.0 * (self._settings.obstacle_clearance + 0.04 - distance)

            candidates.append((score, candidate))

        _, best_candidate = min(candidates, key=lambda item: item[0])
        return best_candidate.astype(np.float32)

    def _build_smooth_path(
        self,
        anchors: NDArray[np.float32],
        obstacles: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Smooth the anchor polyline while preserving a straight final corridor."""

        anchors = self._dedupe_points(anchors.astype(np.float32))
        anchors = self._shortcut_route(anchors, obstacles)
        anchors = self._relax_control_points(anchors, obstacles)
        rounded = self._rounded_polyline(anchors)
        candidate = self._resample_polyline(rounded, self._settings.path_spacing)
        if self._path_is_clear(candidate, obstacles):
            path_points = candidate
        else:
            fallback = anchors.copy()
            for _ in range(self._settings.path_smoothing_iterations):
                fallback = self._chaikin(fallback)
                fallback = self._relax_control_points(fallback, obstacles)
            fallback = self._resample_polyline(fallback, self._settings.path_spacing)
            path_points = fallback if self._path_is_clear(fallback, obstacles) else anchors

        final_path = self._enforce_final_straight(path_points, anchors[-1], obstacles)
        return self._dedupe_points(final_path)

    def _shortcut_route(
        self,
        points: NDArray[np.float32],
        obstacles: NDArray[np.float32],
        clearance: float | None = None,
    ) -> NDArray[np.float32]:
        """Greedily remove intermediate waypoints when a direct hop is obstacle-free."""

        points = self._dedupe_points(points.astype(np.float32))
        if len(points) <= 2:
            return points

        segment_clearance = self._settings.obstacle_clearance if clearance is None else clearance
        simplified: list[Vec3] = [points[0].astype(np.float32)]
        anchor_idx = 0
        while anchor_idx < len(points) - 1:
            next_idx = anchor_idx + 1
            furthest_idx = next_idx
            while next_idx < len(points):
                if self._segment_blocker(
                    points[anchor_idx],
                    points[next_idx],
                    obstacles,
                    segment_clearance,
                ):
                    break
                furthest_idx = next_idx
                next_idx += 1
            simplified.append(points[furthest_idx].astype(np.float32))
            anchor_idx = furthest_idx

        return self._dedupe_points(np.asarray(simplified, dtype=np.float32))

    def _relax_control_points(
        self,
        points: NDArray[np.float32],
        obstacles: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Pull internal control points toward smooth arcs while preserving clearance."""

        points = self._dedupe_points(points.astype(np.float32))
        if len(points) <= 3:
            return points

        relaxed = points.copy()
        fixed_tail = 2 if len(points) > 3 else 1
        clearance = self._settings.obstacle_clearance
        weight = self._settings.control_relaxation_weight

        for _ in range(self._settings.control_relaxation_iterations):
            updated = relaxed.copy()
            for idx in range(1, len(relaxed) - fixed_tail):
                prev_point = updated[idx - 1]
                curr_point = relaxed[idx]
                next_point = relaxed[idx + 1]

                midpoint = 0.5 * (prev_point + next_point)
                candidate = (1.0 - weight) * curr_point + weight * midpoint
                candidate = candidate.astype(np.float32)
                candidate[2] = curr_point[2]

                if not self._point_is_clear(candidate, obstacles, clearance):
                    continue
                if self._segment_blocker(prev_point, candidate, obstacles, clearance):
                    continue
                if self._segment_blocker(candidate, next_point, obstacles, clearance):
                    continue

                updated[idx] = candidate
            relaxed = updated

        return self._dedupe_points(relaxed.astype(np.float32))

    def _build_path_spline(
        self,
        control_points: NDArray[np.float32],
        obstacles: NDArray[np.float32],
    ) -> tuple[
        BSpline,
        BSpline,
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Fit a smooth spline and sample it densely for rendering and projection."""

        control_points = self._dedupe_points(control_points.astype(np.float32))
        params = self._cumulative_lengths(control_points)
        total_param = float(params[-1])

        if total_param < 1e-6:
            sample_params = np.zeros(1, dtype=np.float32)
            path_points = control_points[:1].astype(np.float32)
            path_lengths = np.zeros(1, dtype=np.float32)
            path_spline = make_interp_spline(
                np.array([0.0, 1.0], dtype=np.float64),
                np.vstack([path_points[0], path_points[0]]).astype(np.float64),
                k=1,
                axis=0,
            )
            return (
                path_spline,
                path_spline.derivative(),
                sample_params,
                path_points,
                path_lengths,
            )

        linear_spline = self._make_interp_spline(params, control_points, degree=1)
        linear_params = self._spline_sample_params(total_param, params)
        linear_points = np.asarray(linear_spline(linear_params), dtype=np.float32)
        linear_points[0] = control_points[0]
        linear_points[-1] = control_points[-1]
        linear_params, linear_points = self._compress_curve_samples(linear_params, linear_points)
        linear_lengths = self._cumulative_lengths(linear_points)

        max_degree = min(3, len(control_points) - 1)
        for degree in range(max_degree, 0, -1):
            path_spline = self._make_interp_spline(params, control_points, degree=degree)
            sample_params = self._spline_sample_params(total_param, params)
            sampled_points = np.asarray(path_spline(sample_params), dtype=np.float32)
            sampled_points[0] = control_points[0]
            sampled_points[-1] = control_points[-1]
            sample_params, sampled_points = self._compress_curve_samples(
                sample_params,
                sampled_points,
            )
            sampled_lengths = self._cumulative_lengths(sampled_points)
            if self._path_is_clear(sampled_points, obstacles):
                return (
                    path_spline,
                    path_spline.derivative(),
                    sample_params,
                    sampled_points,
                    sampled_lengths,
                )

        if self._path_is_clear(linear_points, obstacles):
            return (
                linear_spline,
                linear_spline.derivative(),
                linear_params,
                linear_points,
                linear_lengths,
            )

        return (
            linear_spline,
            linear_spline.derivative(),
            params.astype(np.float32),
            control_points.astype(np.float32),
            self._cumulative_lengths(control_points),
        )

    def _path_is_clear(
        self,
        path_points: NDArray[np.float32],
        obstacles: NDArray[np.float32],
    ) -> bool:
        """Check point-wise and segment-wise obstacle clearance for a path."""

        if len(obstacles) == 0 or len(path_points) < 2:
            return True

        for point in path_points:
            if not self._point_is_clear(point, obstacles, self._settings.obstacle_clearance):
                return False

        for idx in range(len(path_points) - 1):
            blocker = self._segment_blocker(
                path_points[idx],
                path_points[idx + 1],
                obstacles,
                self._settings.obstacle_clearance,
            )
            if blocker is not None:
                return False
        return True

    @staticmethod
    def _point_is_clear(
        point: Vec3,
        obstacles: NDArray[np.float32],
        clearance: float,
    ) -> bool:
        """Check whether one point respects the obstacle clearance radius."""

        if len(obstacles) == 0:
            return True
        distances = np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)
        return bool(float(np.min(distances)) >= clearance)

    @staticmethod
    def _project_ratio_xy(
        point_xy: NDArray[np.float32],
        start_xy: NDArray[np.float32],
        end_xy: NDArray[np.float32],
    ) -> float:
        """Project an XY point onto an XY segment and return the clamped ratio."""

        segment_xy = end_xy - start_xy
        denom = float(np.dot(segment_xy, segment_xy))
        if denom < 1e-9:
            return 0.5
        return float(np.clip(np.dot(point_xy - start_xy, segment_xy) / denom, 0.0, 1.0))

    def _rounded_polyline(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Round waypoint corners with short Bezier blends."""

        points = self._dedupe_points(points.astype(np.float32))
        if len(points) <= 2:
            return points

        rounded: list[Vec3] = [points[0].astype(np.float32)]
        for idx in range(1, len(points) - 1):
            prev_point = points[idx - 1].astype(np.float32)
            corner = points[idx].astype(np.float32)
            next_point = points[idx + 1].astype(np.float32)

            incoming = corner - prev_point
            outgoing = next_point - corner
            incoming_norm = float(np.linalg.norm(incoming))
            outgoing_norm = float(np.linalg.norm(outgoing))
            if incoming_norm < 1e-6 or outgoing_norm < 1e-6:
                continue

            incoming_dir = incoming / incoming_norm
            outgoing_dir = outgoing / outgoing_norm
            turn_cos = float(np.dot(incoming_dir, outgoing_dir))
            if turn_cos > 0.985:
                rounded.append(corner)
                continue

            radius = min(
                self._settings.corner_radius,
                0.35 * incoming_norm,
                0.35 * outgoing_norm,
            )
            entry = corner - incoming_dir * radius
            exit = corner + outgoing_dir * radius
            if np.linalg.norm(entry - rounded[-1]) > 1e-4:
                rounded.append(entry.astype(np.float32))

            for sample_idx in range(1, self._settings.corner_samples):
                t = sample_idx / self._settings.corner_samples
                bezier_point = (
                    ((1.0 - t) ** 2) * entry
                    + (2.0 * (1.0 - t) * t) * corner
                    + (t**2) * exit
                )
                rounded.append(cast(Vec3, bezier_point.astype(np.float32)))

            rounded.append(exit.astype(np.float32))

        if np.linalg.norm(points[-1] - rounded[-1]) > 1e-4:
            rounded.append(points[-1].astype(np.float32))
        return self._dedupe_points(np.asarray(rounded, dtype=np.float32))

    def _enforce_final_straight(
        self,
        path_points: NDArray[np.float32],
        goal_point: Vec3,
        obstacles: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Keep a short straight tail into the gate approach point when safe."""

        path_points = self._dedupe_points(path_points.astype(np.float32))
        if len(path_points) <= 2:
            return path_points

        lengths = self._cumulative_lengths(path_points)
        total = float(lengths[-1])
        if total < 1e-6:
            return path_points

        tail_length = min(self._settings.final_straight_length, 0.45 * total)
        cut_s = max(0.0, total - tail_length)
        cut_idx = int(np.searchsorted(lengths, cut_s, side="left"))
        tail_start = self._sample_polyline(path_points, lengths, cut_s).astype(np.float32)

        prefix = path_points[:cut_idx].astype(np.float32)
        straight_tail = self._resample_polyline(
            np.vstack([tail_start, goal_point.astype(np.float32)]).astype(np.float32),
            self._settings.path_spacing,
        )
        candidate = self._dedupe_points(np.vstack([prefix, straight_tail]).astype(np.float32))
        return candidate if self._path_is_clear(candidate, obstacles) else path_points

    def _chaikin(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply one Chaikin corner-cutting pass while preserving endpoints."""

        if len(points) <= 2:
            return points.astype(np.float32)

        out: list[Vec3] = [points[0].astype(np.float32)]
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            out.extend([q.astype(np.float32), r.astype(np.float32)])
        out.append(points[-1].astype(np.float32))
        return np.asarray(out, dtype=np.float32)

    def _resample_polyline(
        self,
        points: NDArray[np.float32],
        spacing: float,
    ) -> NDArray[np.float32]:
        """Resample a polyline at approximately uniform spacing."""

        points = self._dedupe_points(points)
        if len(points) <= 1:
            return points.astype(np.float32)

        lengths = self._cumulative_lengths(points)
        total = float(lengths[-1])
        if total < 1e-6:
            return np.vstack([points[0], points[-1]]).astype(np.float32)

        samples = np.arange(0.0, total, spacing, dtype=np.float32)
        if total - samples[-1] > 1e-6:
            samples = np.append(samples, total)
        resampled = [self._sample_polyline(points, lengths, float(sample)) for sample in samples]
        return np.asarray(resampled, dtype=np.float32)

    def _spline_sample_params(
        self,
        total_param: float,
        control_params: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Create a dense parameter grid that always includes spline control parameters."""

        num_samples = max(
            128,
            int(np.ceil(total_param / self._settings.spline_sample_spacing)) + 1,
        )
        dense_params = np.linspace(0.0, total_param, num_samples, dtype=np.float32)
        return np.unique(np.concatenate([dense_params, control_params.astype(np.float32)])).astype(
            np.float32
        )

    @staticmethod
    def _make_interp_spline(
        params: NDArray[np.float32],
        control_points: NDArray[np.float32],
        degree: int,
    ) -> BSpline:
        """Create an interpolating spline with float64 internals for SciPy."""

        return make_interp_spline(
            params.astype(np.float64),
            control_points.astype(np.float64),
            k=degree,
            axis=0,
        )

    @staticmethod
    def _compress_curve_samples(
        sample_params: NDArray[np.float32],
        sample_points: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Drop duplicate neighboring spline samples while keeping endpoints."""

        if len(sample_points) <= 1:
            return sample_params.astype(np.float32), sample_points.astype(np.float32)

        keep_indices = [0]
        for idx in range(1, len(sample_points) - 1):
            if np.linalg.norm(sample_points[idx] - sample_points[keep_indices[-1]]) > 1e-4:
                keep_indices.append(idx)
        keep_indices.append(len(sample_points) - 1)

        unique_indices = np.asarray(keep_indices, dtype=np.int32)
        return (
            sample_params[unique_indices].astype(np.float32),
            sample_points[unique_indices].astype(np.float32),
        )

    @staticmethod
    def _arc_length_to_param(
        arc_length: float,
        sample_lengths: NDArray[np.float32],
        sample_params: NDArray[np.float32],
    ) -> float:
        """Map approximate arc length along the spline to the spline parameter."""

        if arc_length <= 0.0 or len(sample_params) == 1:
            return float(sample_params[0])
        if arc_length >= float(sample_lengths[-1]):
            return float(sample_params[-1])
        return float(np.interp(arc_length, sample_lengths, sample_params))

    def _project_onto_path(
        self,
        pos: Vec3,
        path_points: NDArray[np.float32],
        path_lengths: NDArray[np.float32],
        current_progress: float,
    ) -> float:
        """Project the current position onto the piecewise-linear path."""

        best_dist = np.inf
        best_s = current_progress
        for idx in range(len(path_points) - 1):
            start = path_points[idx]
            end = path_points[idx + 1]
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom < 1e-9:
                continue

            t = float(np.clip(np.dot(pos - start, segment) / denom, 0.0, 1.0))
            projection = start + t * segment
            distance = float(np.linalg.norm(pos - projection))
            if distance < best_dist:
                best_dist = distance
                segment_length = float(path_lengths[idx + 1] - path_lengths[idx])
                best_s = float(path_lengths[idx] + t * segment_length)
        return best_s

    @staticmethod
    def _sample_polyline(
        points: NDArray[np.float32],
        lengths: NDArray[np.float32],
        arc_length: float,
    ) -> Vec3:
        """Sample a point on a polyline using precomputed cumulative lengths."""

        if arc_length <= 0.0:
            return points[0]
        if arc_length >= float(lengths[-1]):
            return points[-1]

        idx = int(np.searchsorted(lengths, arc_length, side="right") - 1)
        idx = min(max(idx, 0), len(points) - 2)
        s0 = float(lengths[idx])
        s1 = float(lengths[idx + 1])
        if s1 - s0 < 1e-9:
            return points[idx + 1]

        ratio = (arc_length - s0) / (s1 - s0)
        return points[idx] + ratio * (points[idx + 1] - points[idx])

    @staticmethod
    def _cumulative_lengths(points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute cumulative arc lengths for a polyline."""

        if len(points) == 0:
            return np.zeros(0, dtype=np.float32)
        if len(points) == 1:
            return np.zeros(1, dtype=np.float32)

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1).astype(np.float32)
        return np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths)])

    @staticmethod
    def _dedupe_points(points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Drop near-duplicate consecutive points from a path."""

        if len(points) <= 1:
            return points.astype(np.float32)

        deduped: list[Vec3] = [points[0].astype(np.float32)]
        for point in points[1:]:
            if np.linalg.norm(point - deduped[-1]) > 1e-4:
                deduped.append(point.astype(np.float32))
        return np.asarray(deduped, dtype=np.float32)
