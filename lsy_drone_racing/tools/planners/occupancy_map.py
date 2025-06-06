from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Optional, List, Callable
from scipy.ndimage import distance_transform_edt

from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, BSpline
from lsy_drone_racing.tools.ext_tools import GeometryTool

import itertools
from lsy_drone_racing.tools.race_objects import Gate

OC_MAP_LOCAL_MODE = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as axes
    import matplotlib.figure as figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PathCollection
    OC_MAP_LOCAL_MODE = True
except:
    OC_MAP_LOCAL_MODE = False


class OccupancyMap3D:
    resolution : np.float32
    limit: np.ndarray
    size: np.ndarray
    oc_map: np.ndarray

    def __init_plot_artists(self):
        self._occupied_artist: Optional[PathCollection] = None
        self._free_artist: Optional[PathCollection] = None

    def __init__(self, xlim : NDArray[np.floating], ylim : NDArray[np.floating], zlim: NDArray[np.floating], resolution : np.float32 = 0.1, init_val : int = 0):
        self.limit = np.zeros(shape = (3,2))
        self.limit[0] = np.array(xlim)
        self.limit[1] = np.array(ylim)
        self.limit[2] = np.array(zlim)

        self.resolution = resolution

        self.size = (np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]) / resolution).astype(dtype = np.int32) + 1
        self.oc_map = np.ones(self.size, dtype=np.uint8) * init_val

    def copy(self) -> OccupancyMap3D:
        result = OccupancyMap3D(xlim = self.limit[0], ylim = self.limit[1], zlim = self.limit[2], resolution = self.resolution)
        result.oc_map = self.oc_map.copy()
        return result
        
    def out_of_range(self, idx : NDArray[np.integer]) -> bool:
        return not (0 <= idx[0] < self.size[0] and 0 <= idx[1] < self.size[1] and 0 <= idx[2] < self.size[2])

    def world_to_map(self, pos) -> NDArray[np.integer]:
        return ((pos - self.limit[:,0])/self.resolution).astype(int)
        # return tuple(int(p / self.resolution) for p in pos)

    def map_to_world(self, idx : NDArray[np.integer], range_check = False) -> NDArray[np.floating]:
        ix, iy, iz = idx
        if range_check and self.out_of_range(idx = idx):
            return None
        else:
            return (self.limit[:,0] + self.resolution * idx)

    def set_obstacle_value(self, pos : NDArray[np.integer], obs_val:int = 1) -> bool:
        ix, iy, iz = self.world_to_map(pos)
        if not self.out_of_range((ix, iy, iz)):
            self.oc_map[ix, iy, iz] = obs_val
            return True
        else:
            return False
    
    def add_gate_object(self, gate : Gate, obs_val : int = 1, free_val : int = 0)-> None:
        assert gate.outer_height > gate.inner_height, "gate.outer_height must be larger than gate.inner_height!"
        assert gate.outer_width > gate.inner_width, "gate.outer_height must be larger than gate.inner_height!"


        norm_vec = gate.norm_vec / np.linalg.norm(gate.norm_vec)

        # Get two vectors u and v orthonormal to the norm_vec
        if np.allclose(norm_vec, [0, 0, 1]) or np.allclose(norm_vec, [0, 0, -1]):
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.cross(norm_vec, [0, 0, 1])
            u /= np.linalg.norm(u)
        v = np.cross(norm_vec, u)
        v /= np.linalg.norm(v)

        d = self.resolution / 2.0

        h_in = gate.inner_width / 2.0
        v_in = gate.inner_height / 2.0
        h_out = gate.outer_width / 2.0
        v_out = gate.outer_height / 2.0

        us = np.arange(-h_out, h_out + d, d)
        vs = np.arange(-v_out, v_out + d, d)
        uu, vv = np.meshgrid(us, vs, indexing='ij')
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()  

        for ui, vi in zip(uu_flat, vv_flat):
            # Outside inner square → part of frame
            if abs(ui) > h_in or abs(vi) > v_in:
                base_pt = gate.pos + ui * u + vi * v
                for s in np.arange(-gate.thickness / 2, gate.thickness / 2 + d, d):
                    pt = base_pt + s * norm_vec
                    self.set_obstacle_value(pt, obs_val)

        us = np.arange(-h_in, h_in + d, d)
        vs = np.arange(-v_in, v_in + d, d)
        uu, vv = np.meshgrid(us, vs, indexing='ij')
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()
        for ui, vi in zip(uu_flat, vv_flat):
            # Inner sqaure + entry space
            base_pt = gate.pos + ui * u + vi * v
            for s in np.arange(-gate.entry_offset, gate.exit_offset + d, d):
                pt = base_pt + s * norm_vec
                self.set_obstacle_value(pt, free_val)

    def add_trajectory_tube(self, spline: Union[BSpline, CubicSpline], radius: float = 0.3, free_val: int = 0) -> None:
        if isinstance(spline, CubicSpline):
            s_vals = np.linspace(spline.x[0], spline.x[-1], int((spline.x[-1] - spline.x[0]) / self.resolution * 2))
        else:
            s_vals = np.linspace(spline.t[0], spline.t[-1], int((spline.t[-1] - spline.t[0]) / self.resolution * 2))

        for s in s_vals:
            pt = spline(s)  # shape = (3,)
            # Generate a sphere region and fill in the empty value
            bound = int(radius / self.resolution) + 1
            center_idx = self.world_to_map(pt)
            for dx, dy, dz in itertools.product(range(-bound, bound+1),
                                                range(-bound, bound+1),
                                                range(-bound, bound+1)):
                offset = np.array([dx, dy, dz])
                idx = center_idx + offset
                if self.out_of_range(idx):
                    continue
                world_pos = self.limit[:, 0] + idx * self.resolution
                if np.linalg.norm(world_pos - pt) <= radius:
                    self.oc_map[tuple(idx)] = free_val

        return
    def add_sphere(self, pos : NDArray[np.floating], radius: float = 0.3, free_val: int = 0):
        center_idx = self.world_to_map(pos)
        bound = int(radius / self.resolution) + 1
        for dx, dy, dz in itertools.product(range(-bound, bound+1),
                                            range(-bound, bound+1),
                                            range(-bound, bound+1)):
            offset = np.array([dx, dy, dz])
            idx = center_idx + offset
            if self.out_of_range(idx):
                continue
            world_pos = self.limit[:, 0] + idx * self.resolution
            if np.linalg.norm(world_pos - pos) <= radius:
                self.oc_map[tuple(idx)] = free_val

    def add_gate(self,
                center: NDArray[np.floating],
                inner_size: np.floating,
                outer_size:np.floating,
                thickness: np.floating,
                norm_vec: NDArray[np.floating],
                obs_val: int = 1,
                empty_val : int = 0) -> None:
        """
        Add a sqaure gate in the occupancy map
        - center: World coordinate of the gate in array (3,)
        - size: World length of the gate
        - thickness: World thickness in the normal direction
        - norm_vec: Normal vector of the gate(must  be unit vector!)
        """
        assert outer_size > inner_size, "outer_size must be larger than inner_size!"
        norm_vec = norm_vec / np.linalg.norm(norm_vec)  # normalize

        # Get two vectors u and v orthonormal to the norm_vec
        if np.allclose(norm_vec, [0, 0, 1]) or np.allclose(norm_vec, [0, 0, -1]):
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.cross(norm_vec, [0, 0, 1])
            u /= np.linalg.norm(u)
        v = np.cross(norm_vec, u)
        v /= np.linalg.norm(v)

        h_in = inner_size / 2.0
        h_out = outer_size / 2.0

        d = self.resolution / 2.0

        us = np.arange(-h_out, h_out + d, d)
        vs = np.arange(-h_out, h_out + d, d)
        uu, vv = np.meshgrid(us, vs, indexing='ij')
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()  

        border_points = []

        for ui, vi in zip(uu_flat, vv_flat):
            # Outside inner square → part of frame
            if abs(ui) > h_in or abs(vi) > h_in:
                base_pt = center + ui * u + vi * v
                for s in np.arange(-thickness / 2, thickness / 2 + d, d):
                    pt = base_pt + s * norm_vec
                    self.set_obstacle_value(pt, obs_val)
        
        us = np.arange(-h_in, h_in + d, d)
        vs = np.arange(-h_in, h_in + d, d)
        uu, vv = np.meshgrid(us, vs, indexing='ij')
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()
        for ui, vi in zip(uu_flat, vv_flat):
            base_pt = center + ui * u + vi * v
            for s in np.arange(-thickness / 2, thickness / 2 + d, d):
                pt = base_pt + s * norm_vec
                self.set_obstacle_value(pt, empty_val)



    @classmethod
    def merge(cls, maps: List[OccupancyMap3D], mode: str = "intersection") -> OccupancyMap3D:
       
        assert len(maps) > 0, "merge need at least occupancy map"

        base = maps[0]
        for m in maps[1:]:
            assert np.allclose(m.limit, base.limit), "Different limit"
            assert m.resolution == base.resolution, "Different resolution"
            assert np.all(m.size == base.size), "Different size"

        if mode == "union":
            merged_oc_map = np.maximum.reduce([m.oc_map for m in maps])
        elif mode == "intersection":
            merged_oc_map = np.minimum.reduce([m.oc_map for m in maps])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        merged = cls(
            xlim=tuple(base.limit[0]),
            ylim=tuple(base.limit[1]),
            zlim=tuple(base.limit[2]),
            resolution=base.resolution
        )
        merged.oc_map = merged_oc_map
        return merged



    def add_vertical_cylinder(self, center: NDArray[np.floating], radius: float, obs_val: int = 1) -> None:
        """
        Add a vertical cylinder in the occupancy map that is infinite high
        - center: World coordinate of the cylinder(3,)
        - radius: World radius of the cylinder
        """
        cx, cy = center[:2]
        xlim = self.limit[0]
        ylim = self.limit[1]
        zlim = self.limit[2]

        # Temporary grid range
        x_vals = np.arange(max(xlim[0], cx - radius - self.resolution),
                       min(xlim[1], cx + radius + self.resolution),
                       self.resolution / 4.0)
        y_vals = np.arange(max(ylim[0], cy - radius - self.resolution),
                       min(ylim[1], cy + radius + self.resolution),
                       self.resolution / 4.0)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()

        # Check if it is inside the range of cylinder
        mask = (xx_flat - cx)**2 + (yy_flat - cy)**2 <= radius**2
        x_filtered = xx_flat[mask]
        y_filtered = yy_flat[mask]

        for xw, yw in zip(x_filtered, y_filtered):
            pos = np.array(self.world_to_map(pos = np.array([xw, yw, 0])))
            self.oc_map[pos[0], pos[1], :] = obs_val

    
    def get_cost(self, idx: Tuple[int, int, int]) -> float:
        """
        Return cost of stepping into this voxel:
        - 0.0 if free
        - high (e.g. 100.0) if occupied
        """
        if self.out_of_range(idx):
            return 1000.0
        elif self.oc_map[idx] == 1:
            return 50.0
        else:
            return 0.0

    def is_free(self, idx : NDArray[np.integer], range_check = True) -> bool:
        x, y, z = idx
        if range_check:
            return 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2] and self.oc_map[x, y, z] == 0
        else:
            return self.oc_map[x, y, z] == 0
        
    def clear_visualization(self) -> bool:
        if hasattr(self, "_occupied_artist"):
            if self._occupied_artist is not None:
                try:
                    self._occupied_artist.remove()
                except:
                    return False
            else:
                return False
        else:
            return False
        if hasattr(self, '_free_artist'):
            if self._free_artist is not None:
                try:
                    self._free_artist.remove()
                except:
                    return False
            else:
                return False
        else:
            return False
        return True
        
    def save_to_file(self, path: str) -> None:
        np.savez_compressed(
            path,
            resolution=self.resolution,
            limit=self.limit,
            size=self.size,
            oc_map=self.oc_map
        )


    def compute_sdf(oc_map: OccupancyMap3D) -> Tuple[np.ndarray, Callable[[np.ndarray], float]]:
        """
        From OccupancyMap3D construct a SDF grid and a trilinear query function.
        Returns:
            sdf_grid: np.ndarray of shape = oc_map.size, values in meters (distance to obstacle)
            sdf_query: function(pos_world: np.ndarray[3]) -> float (sdf at that world coordinate)
        """
        # Invert: occupied → 0, free → 1
        free_mask = (oc_map.oc_map == 0).astype(np.uint8)

        # Compute Euclidean Distance Transform
        sdf_voxel = distance_transform_edt(free_mask)

        # Convert voxel units to meters
        sdf_grid = sdf_voxel * oc_map.resolution

        def sdf_query(pos: np.ndarray) -> float:
            idx_f = (pos - oc_map.limit[:, 0]) / oc_map.resolution
            if np.any(idx_f < 0) or np.any(idx_f >= oc_map.size):
                return 0.0
            return GeometryTool.trilinear_interpolation(sdf_grid, idx_f)

        return sdf_grid, sdf_query

    @classmethod
    def from_file(cls, path: str) -> OccupancyMap3D:
        data = np.load(path)
        obj = cls(
            xlim=tuple(data['limit'][0]),
            ylim=tuple(data['limit'][1]),
            zlim=tuple(data['limit'][2]),
            resolution=data['resolution'].item()
        )
        obj.oc_map = data['oc_map']
        return obj

    def visualize_occupancy_map(self, fig : figure.Figure, ax : axes.Axes, adjust : bool = True, new_window : bool= True, occupied_color : str = 'black', occupied_alpha : np.floating = 0.8,  free_color : str = None, free_alpha :np.floating = 0.1) -> Tuple[figure.Figure, axes.Axes]:
        if OC_MAP_LOCAL_MODE:
            if not hasattr(self, "_occupied_artist"):
                self.__init_plot_artists()
            occupied = np.argwhere(self.oc_map == 1)
            free = np.argwhere(self.oc_map == 0)

            occupied = occupied *  self.resolution + self.limit[:,0]
            free = free * self.resolution + self.limit[:,0]

            if new_window or fig is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                self.__init_plot_artists()
            
            if self._occupied_artist is not None:
                self._occupied_artist.remove()
                self._occupied_artist = None

            if self._free_artist is not None:
                self._free_artist.remove()
                self._free_artist = None
            
            ax.cla()
            if occupied_color is not None and len(occupied) > 0:
                self._occupied_artist = ax.scatter(
                    occupied[:, 0], occupied[:, 1], occupied[:, 2],
                    c=occupied_color, alpha=occupied_alpha, s=10
                )            
            if free_color is not None and len(free) > 0:
                self._free_artist = ax.scatter(
                    free[:, 0], free[:, 1], free[:, 2],
                    c=free_color, alpha=free_alpha, s=1
                )
            
            if adjust:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Occupancy Grid')

                xlim, ylim, zlim = self.limit
                max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2.0

                mid_x = (xlim[1] + xlim[0]) / 2.0
                mid_y = (ylim[1] + ylim[0]) / 2.0
                mid_z = (zlim[1] + zlim[0]) / 2.0

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.view_init(elev=90, azim=-90)

                plt.pause(0.001)
            return fig, ax
        else:
            return None, None
            