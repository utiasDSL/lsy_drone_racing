"""
3D A* Pathfinding
-----------------
Finds a collision-free path between two 3D coordinates on a voxel grid,
avoiding a given set of obstacle voxels.
"""

import heapq
import math
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Coord3D = tuple[float, float, float]   # world-space coordinates
Voxel   = tuple[int,   int,   int]     # integer grid indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_voxel(coord: Coord3D, voxel_size: float) -> Voxel:
    """Snap a world-space coordinate to its voxel index."""
    return (
        int(math.floor(coord[0] / voxel_size)),
        int(math.floor(coord[1] / voxel_size)),
        int(math.floor(coord[2] / voxel_size)),
    )


def _to_world(voxel: Voxel, voxel_size: float) -> Coord3D:
    """Return the centre of a voxel in world space."""
    return (
        (voxel[0] + 0.5) * voxel_size,
        (voxel[1] + 0.5) * voxel_size,
        (voxel[2] + 0.5) * voxel_size,
    )


def _heuristic(a: Voxel, b: Voxel) -> float:
    """3-D Euclidean distance heuristic (admissible for diagonal moves)."""
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def _neighbors(voxel: Voxel) -> list[tuple[Voxel, float]]:
    """
    Return all 26-connected neighbours (face, edge, and corner) together
    with the exact Euclidean step cost.
    """
    x, y, z = voxel
    result: list[tuple[Voxel, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                result.append(((x + dx, y + dy, z + dz), cost))
    return result


def _expand_obstacles(
    blocked: set[Voxel],
    clearance_voxels: float,
) -> set[Voxel]:
    """
    Dilate each obstacle voxel by *clearance_voxels* in all directions,
    returning the full set of voxels that are considered impassable.

    Every voxel whose Euclidean distance (in voxel units) to the nearest
    obstacle is strictly less than *clearance_voxels* is added to the set.
    A value of 0 leaves the obstacle set unchanged.
    """
    if clearance_voxels <= 0:
        return blocked

    r = math.ceil(clearance_voxels)  # int radius that fully covers the float clearance
    offsets = [
        (dx, dy, dz)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dz in range(-r, r + 1)
        if math.sqrt(dx * dx + dy * dy + dz * dz) < clearance_voxels
    ]

    expanded: set[Voxel] = set()
    for (ox, oy, oz) in blocked:
        for (dx, dy, dz) in offsets:
            expanded.add((ox + dx, oy + dy, oz + dz))

    return expanded


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def astar_3d(
    start:               Coord3D,
    goal:                Coord3D,
    obstacles:           list[Coord3D],
    voxel_size:          float,
    obstacle_clearance:  float = 0.0,
    gate_normal:        Optional[tuple[Coord3D, Coord3D]] = None,
) -> Optional[list[Coord3D]]:
    """
    Find a path from *start* to *goal* in 3-D space, avoiding *obstacles*
    and keeping a minimum distance from them.

    Parameters
    ----------
    start               : (x, y, z) world-space start position.
    goal                : (x, y, z) world-space goal position.
    obstacles           : List of (x, y, z) world-space obstacle positions.
                          Each coordinate is snapped to its containing voxel.
    voxel_size          : Edge length of one voxel (same units as the
                          coordinates).
    obstacle_clearance  : Minimum world-space distance the path must keep
                          from every obstacle voxel centre. A value of 0.0
                          (default) means the path may pass directly adjacent
                          to obstacles. The clearance is converted to voxel
                          units internally, so it should be expressed in the
                          same units as *voxel_size* and the coordinates.

    Returns
    -------
    A list of (x, y, z) world-space waypoints from start to goal
    (voxel centres), or ``None`` if no path exists.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if obstacle_clearance < 0:
        raise ValueError("obstacle_clearance must be non-negative")

    # Build raw obstacle set in voxel space
    raw_blocked: set[Voxel] = {_to_voxel(obs, voxel_size) for obs in obstacles}

    # Expand obstacles according to the requested clearance
    clearance_voxels = obstacle_clearance / voxel_size
    blocked = _expand_obstacles(raw_blocked, clearance_voxels)

    start_v = _to_voxel(start, voxel_size)
    goal_v  = _to_voxel(goal,  voxel_size)

    # Trivial cases
    if start_v == goal_v:
        return [_to_world(start_v, voxel_size)]

    if start_v in blocked or goal_v in blocked:
        
        if start_v in blocked:
            start = np.array(start) - 0.2 * gate_normal[0] if gate_normal[0] is not None else np.array(start) - 0.2 * np.array([1, 0, 0])
            start_v = _to_voxel(start, voxel_size)
        if goal_v in blocked:
            goal = np.array(goal) + 0.2 * gate_normal[1]
            goal_v = _to_voxel(goal, voxel_size)    
        # print("A*: Start or goal is inside an obstacle (or clearance zone).")
        

    # Priority queue entries: (f_score, g_score, voxel)
    # g_score is included as a tiebreaker so equal f values are broken
    # deterministically without comparing Voxel tuples specially.
    open_heap: list[tuple[float, float, Voxel]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_v))

    came_from: dict[Voxel, Voxel] = {}
    g_score:   dict[Voxel, float] = {start_v: 0.0}
    in_open:   set[Voxel]         = {start_v}

    while open_heap:
        _, g_current, current = heapq.heappop(open_heap)
        in_open.discard(current)

        # Early exit
        if current == goal_v:
            return _reconstruct_path(came_from, current, voxel_size)

        # Skip stale heap entries
        if g_current > g_score.get(current, math.inf):
            continue

        for neighbor, step_cost in _neighbors(current):
            if neighbor in blocked:
                continue

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f = tentative_g + _heuristic(neighbor, goal_v)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
                in_open.add(neighbor)

    # print("A*: No path found.")
    return None  # No path found


def _reconstruct_path(
    came_from:  dict[Voxel, Voxel],
    current:    Voxel,
    voxel_size: float,
) -> list[Coord3D]:
    """Walk back through *came_from* and return world-space waypoints."""
    path: list[Voxel] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [_to_world(v, voxel_size) for v in path]
