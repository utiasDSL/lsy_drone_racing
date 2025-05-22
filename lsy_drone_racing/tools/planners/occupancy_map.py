import numpy as np
from typing import Tuple
from numpy.typing import NDArray
OC_MAP_LOCAL_MODE = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as axes
    import matplotlib.figure as figure
    from mpl_toolkits.mplot3d import Axes3D
    OC_MAP_LOCAL_MODE = True
except:
    OC_MAP_LOCAL_MODE = False

class OccupancyMap3D:
    resolution : np.float32
    limit: np.ndarray
    size: np.ndarray
    oc_map: np.ndarray


    def __init__(self, xlim : NDArray[np.floating], ylim : NDArray[np.floating], zlim: NDArray[np.floating], resolution : np.float32 = 0.1):
        self.limit = np.zeros(shape = (3,2))
        self.limit[0] = np.array(xlim)
        self.limit[1] = np.array(ylim)
        self.limit[2] = np.array(zlim)

        self.resolution = resolution

        self.size = (np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]) / resolution).astype(dtype = np.int32) + 1
        self.oc_map = np.zeros(self.size, dtype=np.uint8)

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

    def add_gate(self,
                center: NDArray[np.floating],
                inner_size: np.floating,
                outer_size:np.floating,
                thickness: np.floating,
                norm_vec: NDArray[np.floating],
                obs_val: int = 1) -> None:
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
        
    def visualize_occupancy_map(self, fig : figure.Figure, ax : axes.Axes, adjust : bool = True, new_window : bool= True) -> Tuple[figure.Figure, axes.Axes]:
        if OC_MAP_LOCAL_MODE:
            occupied = np.argwhere(self.oc_map == 1)
            free = np.argwhere(self.oc_map == 0)

            occupied = occupied *  self.resolution + self.limit[:,0]
            free = free * self.resolution + self.limit[:,0]

            if new_window or fig is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            
            # ax.scatter(free[:, 0], free[:, 1], free[:, 2], c='lightgray', alpha=0.1, s=1)
            ax.cla()
            ax.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2], c='black', s=10)
            
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
                plt.pause(0.001)
            return fig, ax
        else:
            return None, None
            