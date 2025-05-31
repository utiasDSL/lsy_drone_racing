import numpy as np
from numpy.typing import NDArray
from typing import Set, Tuple, List
from lsy_drone_racing.tools.ext_tools import TransformTool
# from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D

LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    import matplotlib.collections
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False

class Obstacle:
    pos : NDArray[np.floating]
    safe_radius : np.float32

    def __init__(self, pos : NDArray[np.floating], safe_radius : np.float32):
        self.pos = pos
        self.safe_radius = safe_radius

class Gate:
    pos: NDArray[np.floating]
    norm_vec : NDArray[np.floating]
    inner_width : np.float32
    inner_height : np.float32
    outer_width : np.float32
    outer_height : np.float32
    safe_radius: np.float32
    entry_offset : np.float32
    exit_offset: np.float32
    thickness : np.float32
    _quat : NDArray[np.floating]
    vel_limit : np.floating

    def __init__(self, pos : NDArray[np.floating],
                  quat : NDArray[np.floating],
                    inner_width : np.float32 = 0.4,
                      inner_height : np.float32 = 0.4,
                        outer_width : np.float32 = 0.6,
                          outer_height : np.float32 = 0.6,
                            safe_radius : np.float32 = 0.1,
                            entry_offset : np.float32 = 0.1,
                            exit_offset : np.float32 = 0.1,
                            thickness : np.float32 = 0.1,
                            vel_limit : np.float32 = 0.2):
        self.pos = pos
        self._quat = quat
        self.norm_vec = TransformTool.quad_to_norm(quat, axis = 1)
        self.plane_vec = TransformTool.quad_to_norm(quat, axis = 0)
        self.inner_width = inner_width
        self.inner_height = inner_height
        self.outer_width = outer_width
        self.outer_height = outer_height
        self.safe_radius = safe_radius
        self.entry_offset = entry_offset
        self.exit_offset = exit_offset
        self.thickness = thickness
        self.vel_limit = vel_limit
    
    def update(self, pos : NDArray[np.floating], quat : NDArray[np.floating]) -> None:
        self.pos = pos
        self._quat = quat
        self.norm_vec = TransformTool.quad_to_norm(quat, axis = 1)
        self.plane_vec = TransformTool.quad_to_norm(quat, axis = 0)

    def in_gate_cylinder(self, pos : NDArray[np.floating], inner = False) -> bool:
        gate_distance = pos - self.pos
        perp_vec = gate_distance - np.dot(gate_distance, self.norm_vec) * self.norm_vec
        r = np.linalg.norm(perp_vec)
        return r < (self.outer_width / 2) if not inner else r < (self.inner_width / 2)

    # def gate_goal_region_voxels(self,
    #                          omap : OccupancyMap3D,
    #                          offset : np.float32 = 0,
    #                          voxel_margin: float = 0.05) -> Set[Tuple[int, int, int]]:
    #     norm_vec = self.norm_vec / np.linalg.norm(self.norm_vec)

    #     if np.allclose(norm_vec, [0, 0, 1]) or np.allclose(norm_vec, [0, 0, -1]):
    #         u = np.array([1.0, 0.0, 0.0])
    #     else:
    #         u = np.cross(norm_vec, [0, 0, 1])
    #         u /= np.linalg.norm(u)
    #     v = np.cross(norm_vec, u)

    #     half_w = self.inner_width / 2.0 - voxel_margin
    #     half_h = self.inner_height / 2.0 - voxel_margin

    #     d = omap.resolution
    #     us = np.arange(-half_w, half_w + d, d)
    #     vs = np.arange(-half_h, half_h + d, d)

    #     voxels = set()
    #     for u_ in us:
    #         for v_ in vs:
    #             pos = self.pos + u_ * u + v_ * v + offset * norm_vec
    #             idx = tuple(omap.world_to_map(pos))
    #             if not omap.out_of_range(idx):
    #                 voxels.add(idx)
    #     return voxels
    

class SceneSDF:
    def __init__(self):
        self.capsules: List[Tuple[NDArray, NDArray, float]] = []
        self._sdf_artists = []

    def from_gates_and_obstacles(self, gates: List[Gate], obstacles: List[Obstacle]):
        self.capsules.clear()
        for gate in gates:
            self._add_gate(gate)
        for obs in obstacles:
            self._add_obstacle(obs)

    def _add_gate(self, gate):
        pos = gate.pos
        u = gate.plane_vec
        v = np.cross(gate.norm_vec, u)
        w = gate.outer_width / 2.0
        h = gate.outer_height / 2.0
        r = gate.safe_radius

        segments = [
            (pos + w * u + h * v, pos + w * u - h * v), 
            (pos - w * u + h * v, pos - w * u - h * v), 
            (pos - w * u + h * v, pos + w * u + h * v), 
            (pos - w * u - h * v, pos + w * u - h * v),
        ]
        for a, b in segments:
            self.capsules.append((a, b, r))

    def _add_obstacle(self, obs):
        h = 1000.0
        a = obs.pos - np.array([0, 0, h])
        b = obs.pos + np.array([0, 0, h])
        self.capsules.append((a, b, obs.safe_radius))

    def capsule_sdf(p: NDArray, a: NDArray, b: NDArray, r: float) -> float:
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        closest = a + t * ab
        return np.linalg.norm(p - closest) - r
    
    def evaluate(self, p: NDArray) -> float:
        return min([SceneSDF.capsule_sdf(p, a, b, r) for a, b, r in self.capsules])

    def update(self, gates: List = None, obstacles: List = None):
        if gates is not None or obstacles is not None:
            self.from_gates_and_obstacles(gates or [], obstacles or [])

    def visualize_sdf_map(self, fig : figure.Figure, ax : axes.Axes) -> Tuple[figure.Figure, axes.Axes]:
        if not hasattr(self, "_sdf_artists") or self._sdf_artists is None:
            self._sdf_artists = []


        for artist in self._sdf_artists:
            artist.remove()
        self._sdf_artists.clear()

        for a, b, r in self.capsules:
            xs = [a[0], b[0]]
            ys = [a[1], b[1]]
            zs = [a[2], b[2]]
            line = ax.plot(xs, ys, zs, color='black', linewidth=2, alpha=0.6)[0]
            self._sdf_artists.append(line)

        return fig, ax
