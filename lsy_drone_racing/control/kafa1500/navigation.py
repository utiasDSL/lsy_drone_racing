"""Gate-local planning, obstacle routing, and path following."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500.types import GateFrame, GatePlan, PathTarget

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import BSpline

    from lsy_drone_racing.control.kafa1500.settings import PlannerSettings
    from lsy_drone_racing.control.kafa1500.types import Observation, Vec3


class GateNavigator:
    """Build and follow a local gate-centric path."""

    def __init__(self, settings: PlannerSettings):
        """Store the planner settings."""
        self._settings = settings

    def plan_gate(
        self,
        obs: Observation,
        gate_idx: int,
        start_pos: Vec3 | None = None,
        start_vel: Vec3 | None = None,
    ) -> GatePlan:
        """Create a fresh local plan for the current target gate."""
        pos = obs["pos"].astype(np.float32) if start_pos is None else start_pos.astype(np.float32)
        vel = obs["vel"].astype(np.float32) if start_vel is None else start_vel.astype(np.float32)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        gate = self._gate_frame(obs, gate_idx)

        align_target, approach_target = self._choose_gate_targets(gate, obstacles)
        pass_target = gate.position + self._settings.pass_distance * gate.forward

        route_points = self._plan_route(pos, align_target, approach_target, gate, obstacles)
        (
            control_points,
            path_spline,
            path_spline_d1,
            path_spline_d2,
            path_params,
            path_points,
            path_lengths,
        ) = self._plan_gate_constrained_spline(
            start_pos=pos,
            start_vel=vel,
            route_points=route_points,
            gate=gate,
            pass_target=pass_target,
            obstacles=obstacles,
        )
        route_line = path_points.astype(np.float32)

        return GatePlan(
            gate_idx=gate_idx,
            gate_pos=gate.position,
            gate_traversal_pos=gate.traversal_point,
            gate_x=gate.forward,
            pass_target=pass_target,
            path_spline=path_spline,
            path_spline_d1=path_spline_d1,
            path_spline_d2=path_spline_d2,
            path_params=path_params,
            path_points=path_points,
            path_lengths=path_lengths,
            route_line=route_line,
        )

    def _plan_gate_constrained_spline(
        self,
        start_pos: Vec3,
        start_vel: Vec3,
        route_points: list[Vec3],
        gate: GateFrame,
        pass_target: Vec3,
        obstacles: NDArray[np.float32],
    ) -> tuple[
        NDArray[np.float32],
        BSpline,
        BSpline,
        BSpline,
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Build a cubic spline that must cross the gate centre and stay inside the aperture."""
        best: tuple[
            NDArray[np.float32],
            BSpline,
            BSpline,
            BSpline,
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.float32],
        ] | None = None

        for attempt in range(self._settings.gate_clearance_max_attempts):
            support_distance = (
                self._settings.gate_support_base_distance
                + attempt * self._settings.gate_support_step_distance
            )
            anchors, locked_indices = self._build_gate_constrained_anchors(
                start_pos=start_pos,
                start_vel=start_vel,
                route_points=route_points,
                gate=gate,
                pass_target=pass_target,
                support_distance=support_distance,
            )
            control_points = self._smooth_gate_constrained_anchors(
                anchors=anchors,
                locked_indices=locked_indices,
                obstacles=obstacles,
            )
            spline_pack = self._build_path_spline(control_points, obstacles)
            path_spline, path_spline_d1, path_spline_d2, path_params, path_points, path_lengths = (
                spline_pack
            )

            best = (
                control_points,
                path_spline,
                path_spline_d1,
                path_spline_d2,
                path_params,
                path_points,
                path_lengths,
            )

            path_clear = self._path_is_clear(path_points, obstacles)
            gate_clear = self._gate_frame_clearance_ok(path_points, gate)
            if path_clear and gate_clear:
                return best

        if best is not None:
            anchors, _ = self._build_gate_constrained_anchors(
                start_pos=start_pos,
                start_vel=start_vel,
                route_points=[],
                gate=gate,
                pass_target=pass_target,
                support_distance=(
                    self._settings.gate_support_base_distance
                    + self._settings.gate_clearance_max_attempts
                    * self._settings.gate_support_step_distance
                ),
            )
            (
                path_spline,
                path_spline_d1,
                path_spline_d2,
                path_params,
                path_points,
                path_lengths,
            ) = self._build_path_spline(anchors, obstacles)
            return (
                anchors,
                path_spline,
                path_spline_d1,
                path_spline_d2,
                path_params,
                path_points,
                path_lengths,
            )

        anchors, _ = self._build_gate_constrained_anchors(
            start_pos=start_pos,
            start_vel=start_vel,
            route_points=[],
            gate=gate,
            pass_target=pass_target,
            support_distance=self._settings.gate_support_base_distance,
        )
        path_spline, path_spline_d1, path_spline_d2, path_params, path_points, path_lengths = (
            self._build_path_spline(anchors, obstacles)
        )
        return (
            anchors,
            path_spline,
            path_spline_d1,
            path_spline_d2,
            path_params,
            path_points,
            path_lengths,
        )

    def _build_gate_constrained_anchors(
        self,
        start_pos: Vec3,
        start_vel: Vec3,
        route_points: list[Vec3],
        gate: GateFrame,
        pass_target: Vec3,
        support_distance: float,
    ) -> tuple[NDArray[np.float32], set[int]]:
        """Build anchor points with immutable pre/center/post gate support waypoints."""
        pre_gate = gate.traversal_point - support_distance * gate.forward
        post_gate_distance = max(support_distance, 0.65 * self._settings.pass_distance)
        post_gate = gate.traversal_point + post_gate_distance * gate.forward

        anchors: list[Vec3] = []
        locked_indices: set[int] = set()

        def append(point: Vec3, *, locked: bool = False) -> None:
            candidate = point.astype(np.float32)
            if anchors and np.linalg.norm(candidate - anchors[-1]) <= 1e-4:
                if locked:
                    locked_indices.add(len(anchors) - 1)
                return
            anchors.append(candidate)
            if locked:
                locked_indices.add(len(anchors) - 1)

        append(start_pos, locked=True)
        start_speed = float(np.linalg.norm(start_vel))
        if start_speed > 1e-3:
            heading = start_vel / start_speed
            lead_distance = float(
                np.clip(
                    start_speed * self._settings.continuity_start_dt,
                    self._settings.continuity_start_step_min,
                    self._settings.continuity_start_step_max,
                )
            )
            continuity_point = start_pos + lead_distance * heading
            append(continuity_point.astype(np.float32), locked=True)
        for point in route_points:
            append(point, locked=False)
        append(pre_gate.astype(np.float32), locked=True)
        append(gate.traversal_point.astype(np.float32), locked=True)
        append(post_gate.astype(np.float32), locked=True)
        append(pass_target.astype(np.float32), locked=True)

        return np.asarray(anchors, dtype=np.float32), locked_indices

    def _smooth_gate_constrained_anchors(
        self,
        anchors: NDArray[np.float32],
        locked_indices: set[int],
        obstacles: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Smooth route anchors while preserving immutable gate-centre support points."""
        if len(anchors) <= 3:
            return anchors.astype(np.float32)
        return self._relax_control_points(
            anchors.astype(np.float32),
            obstacles,
            locked_indices=locked_indices,
        )

    def _gate_frame_clearance_ok(
        self,
        path_points: NDArray[np.float32],
        gate: GateFrame,
    ) -> bool:
        """Validate that the spline crosses the safe gate center region and avoids the frame."""
        if len(path_points) == 0:
            return False

        center_distance = float(np.min(np.linalg.norm(path_points - gate.traversal_point, axis=1)))
        if center_distance > self._settings.gate_center_region_radius:
            return False

        local = self._points_in_gate_frame(path_points, gate)
        near_plane = np.abs(local[:, 0]) <= self._settings.gate_clearance_half_depth
        if not np.any(near_plane):
            return False

        safe_lateral = np.abs(local[near_plane, 1]) <= gate.safe_half_width
        safe_vertical = np.abs(local[near_plane, 2]) <= gate.safe_half_height
        if not bool(np.all(safe_lateral & safe_vertical)):
            return False

        intersections = self._gate_plane_intersections(local)
        if len(intersections) == 0:
            return False

        intersections = np.asarray(intersections, dtype=np.float32)
        in_aperture = (
            (np.abs(intersections[:, 0]) <= gate.safe_half_width)
            & (np.abs(intersections[:, 1]) <= gate.safe_half_height)
        )
        if not bool(np.all(in_aperture)):
            return False

        traversal_offset = gate.traversal_point - gate.position
        traversal_yz = np.array(
            [
                float(np.dot(traversal_offset, gate.lateral)),
                float(np.dot(traversal_offset, gate.up)),
            ],
            dtype=np.float32,
        )
        min_intersection_distance = float(
            np.min(np.linalg.norm(intersections - traversal_yz[None, :], axis=1))
        )
        return min_intersection_distance <= self._settings.gate_center_region_radius

    @staticmethod
    def _points_in_gate_frame(
        path_points: NDArray[np.float32],
        gate: GateFrame,
    ) -> NDArray[np.float32]:
        """Project world points into one gate's local forward/lateral/up frame."""
        rel = path_points - gate.position
        local_x = rel @ gate.forward
        local_y = rel @ gate.lateral
        local_z = rel @ gate.up
        return np.column_stack([local_x, local_y, local_z]).astype(np.float32)

    @staticmethod
    def _gate_plane_intersections(local_path: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """Collect intersections of consecutive samples with the gate plane x=0."""
        intersections: list[NDArray[np.float32]] = []
        for idx in range(len(local_path) - 1):
            p0 = local_path[idx]
            p1 = local_path[idx + 1]
            x0 = float(p0[0])
            x1 = float(p1[0])

            if abs(x0) <= 1e-6:
                intersections.append(p0[1:3].astype(np.float32))

            if x0 * x1 > 0.0:
                continue

            denom = x1 - x0
            if abs(denom) <= 1e-9:
                continue

            alpha = -x0 / denom
            if alpha < 0.0 or alpha > 1.0:
                continue
            yz = p0[1:3] + alpha * (p1[1:3] - p0[1:3])
            intersections.append(yz.astype(np.float32))

        return intersections

    def _choose_gate_targets(
        self,
        gate: GateFrame,
        obstacles: NDArray[np.float32],
    ) -> tuple[Vec3, Vec3]:
        """Pick gate approach targets, shifting laterally if the nominal corridor is blocked."""
        base_align = gate.traversal_point - self._settings.align_distance * gate.forward
        base_approach = gate.traversal_point - self._settings.approach_distance * gate.forward

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
                gate.traversal_point,
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

    def follow_path(
        self,
        pos: Vec3,
        plan: GatePlan,
        reference_speed: float,
    ) -> PathTarget:
        """Project onto the current plan and sample a spline-derived reference state."""
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
        d1 = np.asarray(plan.path_spline_d1(target_param), dtype=np.float32)
        d2 = np.asarray(plan.path_spline_d2(target_param), dtype=np.float32)

        tangent_norm = float(np.linalg.norm(d1))
        yaw_dir = plan.gate_x if tangent_norm < 1e-6 else d1 / tangent_norm

        clipped_speed = float(max(0.0, reference_speed))
        if clipped_speed < 1e-6 or tangent_norm < 1e-6:
            ref_vel = np.zeros(3, dtype=np.float32)
            ref_acc = np.zeros(3, dtype=np.float32)
        else:
            ds_dt = clipped_speed / tangent_norm
            ref_vel = (d1 * ds_dt).astype(np.float32)
            ref_acc = (d2 * (ds_dt**2)).astype(np.float32)

        return PathTarget(
            target=target.astype(np.float32),
            yaw_dir=yaw_dir.astype(np.float32),
            ref_vel=ref_vel.astype(np.float32),
            ref_acc=ref_acc.astype(np.float32),
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
        gate_z = rotation[:, 2]
        gate_x /= np.linalg.norm(gate_x)
        gate_y /= np.linalg.norm(gate_y)
        gate_z /= np.linalg.norm(gate_z)

        inner_half_width = 0.5 * self._settings.gate_inner_width
        inner_half_height = 0.5 * self._settings.gate_inner_height
        safe_half_width = max(0.02, inner_half_width - self._settings.gate_frame_safety_margin)
        safe_half_height = max(0.02, inner_half_height - self._settings.gate_frame_safety_margin)
        max_bias = max(
            0.0,
            safe_half_height - self._settings.gate_center_region_radius,
        )
        vertical_bias = float(np.clip(self._settings.gate_vertical_bias, -max_bias, max_bias))
        traversal_point = (gate_pos + vertical_bias * gate_z).astype(np.float32)

        return GateFrame(
            position=gate_pos,
            forward=gate_x.astype(np.float32),
            lateral=gate_y.astype(np.float32),
            up=gate_z.astype(np.float32),
            traversal_point=traversal_point,
            inner_half_width=float(inner_half_width),
            inner_half_height=float(inner_half_height),
            safe_half_width=float(safe_half_width),
            safe_half_height=float(safe_half_height),
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

        smoothed = self._smooth_obstacle_route(
            np.asarray(path, dtype=np.float32),
            obstacles,
            planning_clearance,
        )
        return [point.astype(np.float32) for point in smoothed[1:]]

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
        base_angle = (
            float(np.arctan2(segment[1], segment[0]))
            if np.linalg.norm(segment) > 1e-6
            else 0.0
        )
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self._settings.visibility_node_samples,
            endpoint=False,
        )
        candidates: list[Vec3] = []

        for angle in base_angle + angles:
            xy = obstacle[:2] + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            ratio = self._project_ratio_xy(xy, start[:2], end[:2])
            z = float(start[2] + ratio * (end[2] - start[2]))
            candidate = np.array([xy[0], xy[1], z], dtype=np.float32)
            if self._point_is_clear(candidate, obstacles, radius - 1e-3):
                candidates.append(candidate.astype(np.float32))

        return candidates

    def _shortest_node_path(
        self,
        adjacency: list[list[tuple[float, int]]],
        nodes: list[Vec3],
    ) -> list[Vec3] | None:
        """Run a turn-aware Dijkstra search from node 0 to node 1."""
        goal_idx = 1
        goal_dir = nodes[goal_idx][:2] - nodes[0][:2]
        goal_norm = float(np.linalg.norm(goal_dir))
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        start_state = (-1, 0)
        distances: dict[tuple[int, int], float] = {start_state: 0.0}
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start_state: None}
        best_goal_state: tuple[int, int] | None = None
        best_goal_cost = np.inf
        heap: list[tuple[float, int, int]] = [(0.0, -1, 0)]

        while heap:
            current_dist, prev_idx, current_idx = heappop(heap)
            state = (prev_idx, current_idx)
            if current_dist > distances.get(state, np.inf):
                continue
            if current_idx == goal_idx:
                best_goal_state = state
                best_goal_cost = current_dist
                break

            for edge_cost, neighbor_idx in adjacency[current_idx]:
                turn_penalty = self._turn_penalty(
                    nodes,
                    prev_idx,
                    current_idx,
                    neighbor_idx,
                    goal_dir.astype(np.float32),
                )
                next_state = (current_idx, neighbor_idx)
                next_dist = current_dist + edge_cost + turn_penalty
                if next_dist >= distances.get(next_state, np.inf):
                    continue
                distances[next_state] = next_dist
                previous[next_state] = state
                heappush(heap, (next_dist, current_idx, neighbor_idx))

        if best_goal_state is None or not np.isfinite(best_goal_cost):
            return None

        path_indices: list[int] = []
        cursor: tuple[int, int] | None = best_goal_state
        while cursor is not None:
            _, current_idx = cursor
            path_indices.append(current_idx)
            cursor = previous[cursor]
        path_indices.reverse()
        return [nodes[idx].astype(np.float32) for idx in path_indices]

    def _turn_penalty(
        self,
        nodes: list[Vec3],
        prev_idx: int,
        current_idx: int,
        next_idx: int,
        goal_dir: NDArray[np.float32],
    ) -> float:
        """Penalize heading changes so obstacle routes bend more gradually."""
        outgoing = nodes[next_idx][:2] - nodes[current_idx][:2]
        outgoing_norm = float(np.linalg.norm(outgoing))
        if outgoing_norm < 1e-6:
            return 0.0
        outgoing = outgoing / outgoing_norm

        if prev_idx == -1:
            if float(np.linalg.norm(goal_dir)) < 1e-6:
                return 0.0
            heading_cos = float(np.clip(np.dot(goal_dir, outgoing), -1.0, 1.0))
            return 0.5 * self._settings.visibility_turn_weight * (1.0 - heading_cos)

        incoming = nodes[current_idx][:2] - nodes[prev_idx][:2]
        incoming_norm = float(np.linalg.norm(incoming))
        if incoming_norm < 1e-6:
            return 0.0
        incoming = incoming / incoming_norm
        turn_cos = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        return self._settings.visibility_turn_weight * (1.0 - turn_cos)

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

    def _smooth_obstacle_route(
        self,
        points: NDArray[np.float32],
        obstacles: NDArray[np.float32],
        clearance: float,
    ) -> NDArray[np.float32]:
        """Relax a graph route into a wider, lower-curvature obstacle detour."""
        smoothed = self._shortcut_route(points.astype(np.float32), obstacles, clearance)
        for _ in range(self._settings.obstacle_route_relaxation_iterations):
            smoothed = self._relax_control_points(
                smoothed,
                obstacles,
                clearance=clearance,
                iterations=1,
                weight=min(self._settings.control_relaxation_weight + 0.10, 0.82),
                fixed_tail=1,
            )
            smoothed = self._shortcut_route(smoothed, obstacles, clearance)
        return self._dedupe_points(smoothed.astype(np.float32))

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
        clearance: float | None = None,
        iterations: int | None = None,
        weight: float | None = None,
        fixed_tail: int | None = None,
        locked_indices: set[int] | None = None,
    ) -> NDArray[np.float32]:
        """Pull internal control points toward smooth arcs while preserving clearance."""
        points = self._dedupe_points(points.astype(np.float32))
        if len(points) <= 3:
            return points

        relaxed = points.copy()
        tail = fixed_tail if fixed_tail is not None else (2 if len(points) > 3 else 1)
        segment_clearance = (
            self._settings.obstacle_clearance if clearance is None else clearance
        )
        blend_weight = (
            self._settings.control_relaxation_weight if weight is None else weight
        )
        num_iterations = (
            self._settings.control_relaxation_iterations
            if iterations is None
            else iterations
        )
        locked = set() if locked_indices is None else set(locked_indices)

        for _ in range(num_iterations):
            updated = relaxed.copy()
            for idx in range(1, len(relaxed) - tail):
                if idx in locked:
                    continue
                prev_point = updated[idx - 1]
                curr_point = relaxed[idx]
                next_point = relaxed[idx + 1]

                midpoint = 0.5 * (prev_point + next_point)
                candidate = (1.0 - blend_weight) * curr_point + blend_weight * midpoint
                candidate = candidate.astype(np.float32)
                candidate[2] = curr_point[2]

                if not self._point_is_clear(candidate, obstacles, segment_clearance):
                    continue
                if self._segment_blocker(prev_point, candidate, obstacles, segment_clearance):
                    continue
                if self._segment_blocker(candidate, next_point, obstacles, segment_clearance):
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
        BSpline,
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Fit and sample a strictly cubic spline trajectory for tracking."""
        control_points = self._dedupe_points(control_points.astype(np.float32))
        path_spline, sample_params, sampled_points, sampled_lengths = self._fit_and_sample_cubic(
            control_points
        )

        if not self._path_is_clear(sampled_points, obstacles):
            dense_spacing = max(0.015, 0.5 * self._settings.path_spacing)
            densified = self._resample_polyline(control_points, dense_spacing)
            dense_spline, dense_params, dense_points, dense_lengths = self._fit_and_sample_cubic(
                densified
            )
            if self._path_is_clear(dense_points, obstacles):
                path_spline = dense_spline
                sample_params = dense_params
                sampled_points = dense_points
                sampled_lengths = dense_lengths

        return (
            path_spline,
            path_spline.derivative(),
            path_spline.derivative(2),
            sample_params,
            sampled_points,
            sampled_lengths,
        )

    def _fit_and_sample_cubic(
        self,
        control_points: NDArray[np.float32],
    ) -> tuple[BSpline, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Fit one cubic spline candidate and sample it with arc-length parameterization."""
        cubic_controls = self._prepare_cubic_control_points(control_points)
        params = self._strictly_increasing_params(cubic_controls)
        total_param = float(params[-1])
        path_spline = self._make_interp_spline(params, cubic_controls, degree=3)

        if path_spline.k != 3:
            raise RuntimeError("Spline construction must remain cubic.")

        sample_params = self._spline_sample_params(total_param, params)
        sampled_points = np.asarray(path_spline(sample_params), dtype=np.float32)
        sampled_points[0] = cubic_controls[0]
        sampled_points[-1] = cubic_controls[-1]
        sample_params, sampled_points = self._compress_curve_samples(sample_params, sampled_points)
        sampled_lengths = self._cumulative_lengths(sampled_points)
        return path_spline, sample_params, sampled_points, sampled_lengths

    def _prepare_cubic_control_points(
        self,
        points: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Ensure at least four control points so cubic interpolation is always well-defined."""
        points = self._dedupe_points(points.astype(np.float32))

        if len(points) == 0:
            return np.zeros((4, 3), dtype=np.float32)

        if len(points) == 1:
            return np.repeat(points, 4, axis=0).astype(np.float32)

        if len(points) == 2:
            p0 = points[0]
            p1 = points[1]
            span = p1 - p0
            return np.vstack([
                p0,
                p0 + span / 3.0,
                p0 + (2.0 * span) / 3.0,
                p1,
            ]).astype(np.float32)

        if len(points) == 3:
            p0 = points[0]
            p1 = points[1]
            p2 = points[2]
            if np.linalg.norm(p1 - p0) >= np.linalg.norm(p2 - p1):
                inserted = 0.5 * (p0 + p1)
                return np.vstack([p0, inserted, p1, p2]).astype(np.float32)
            inserted = 0.5 * (p1 + p2)
            return np.vstack([p0, p1, inserted, p2]).astype(np.float32)

        return points.astype(np.float32)

    @staticmethod
    def _strictly_increasing_params(points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Build a cumulative parameter grid with a guaranteed positive step."""
        params = GateNavigator._cumulative_lengths(points).astype(np.float32)
        min_step = 1e-3
        for idx in range(1, len(params)):
            if params[idx] - params[idx - 1] < min_step:
                params[idx] = params[idx - 1] + min_step
        return params.astype(np.float32)

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
