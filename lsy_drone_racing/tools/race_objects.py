import numpy as np
from numpy.typing import NDArray
from typing import Set, Tuple
from lsy_drone_racing.tools.geometric_tools import TransformTool
from lsy_drone_racing.tools.occupancy_map import OccupancyMap3D
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

    def __init__(self, pos : NDArray[np.floating],
                  quat : NDArray[np.floating],
                    inner_width : np.float32 = 0.4,
                      inner_height : np.float32 = 0.4,
                        outer_width : np.float32 = 0.6,
                          outer_height : np.float32 = 0.6,
                            safe_radius : np.float32 = 0.1,
                            entry_offset : np.float32 = 0.1,
                            exit_offset : np.float32 = 0.1,
                            thickness : np.float32 = 0.1):
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
    
    def update(self, pos : NDArray[np.floating], quat : NDArray[np.floating]) -> None:
        self.pos = pos
        self._quat = quat
        self.norm_vec = TransformTool.quad_to_norm(quat, axis = 1)
        self.plane_vec = TransformTool.quad_to_norm(quat, axis = 0)

    def gate_goal_region_voxels(self,
                             omap : OccupancyMap3D,
                             offset : np.float32 = 0,
                             voxel_margin: float = 0.05) -> Set[Tuple[int, int, int]]:
        norm_vec = self.norm_vec / np.linalg.norm(self.norm_vec)

        if np.allclose(norm_vec, [0, 0, 1]) or np.allclose(norm_vec, [0, 0, -1]):
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.cross(norm_vec, [0, 0, 1])
            u /= np.linalg.norm(u)
        v = np.cross(norm_vec, u)

        half_w = self.inner_width / 2.0 - voxel_margin
        half_h = self.inner_height / 2.0 - voxel_margin

        d = omap.resolution
        us = np.arange(-half_w, half_w + d, d)
        vs = np.arange(-half_h, half_h + d, d)

        voxels = set()
        for u_ in us:
            for v_ in vs:
                pos = self.pos + u_ * u + v_ * v + offset * norm_vec
                idx = tuple(omap.world_to_map(pos))
                if not omap.out_of_range(idx):
                    voxels.add(idx)
        return voxels