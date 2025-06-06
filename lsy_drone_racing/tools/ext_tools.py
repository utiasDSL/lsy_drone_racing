from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from typing import List
class TransformTool:
    def quad_to_norm(quad: NDArray[np.floating], axis : int= 1) -> NDArray[np.floating]:
        '''
        Return the normal vector of gates(x-axis, y-axis, z-axis)
        '''
        rotates = R.from_quat(quad)
        rot_matrices = np.array(rotates.as_matrix())
        if len(rot_matrices.shape) == 3:
            return np.array(rot_matrices[:,:,axis]) 
        elif len(rot_matrices.shape) == 2:
            return np.array(rot_matrices[:,axis])
        else:
            return None

class  LinAlgTool:
    def normalize(vec: NDArray[np.floating]) -> NDArray[np.floating]:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return vec
        return vec / norm
    
    def dot_safe(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

class PolynomialTool:
    def cubic_solve_real(a : np.floating, b : np.floating, c : np.floating, d: np.floating) -> List[np.float64]:
        roots = np.roots(np.array([a,b,c,d], dtype = np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0)]
    
    def quartic_solve_real(a : np.floating, b : np.floating, c : np.floating, d: np.floating, e : np.floating) -> List[np.float64]:
        roots = np.roots(np.array([a,b,c,d,e], dtype = np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0)]
    
class TrajectoryTool:

    def compute_3d_curvature_from_vector_spline(spline: CubicSpline, t_vals: np.ndarray, eps : np.ndarray = 1e-8, positive : bool= True) -> np.ndarray:
        r_dot = spline(t_vals, 1)   # shape: (len(t_vals), 3)
        r_ddot = spline(t_vals, 2)  # shape: (len(t_vals), 3)

        cross = np.cross(r_dot, r_ddot)
        cross_norm = np.linalg.norm(cross, axis=1)
        r_dot_norm = np.linalg.norm(r_dot, axis=1)

        curvature = cross_norm / (r_dot_norm ** 3 + eps)
        return np.abs(curvature) if positive else curvature
    
    def compute_3d_turning_radius_from_vector_spline(spline: CubicSpline, t_vals: np.ndarray, eps : np.ndarray = 1e-8, positive : bool = True) -> np.ndarray:
        r_dot = spline(t_vals, 1)   # shape: (len(t_vals), 3)
        r_ddot = spline(t_vals, 2)  # shape: (len(t_vals), 3)

        cross = np.cross(r_dot, r_ddot)
        cross_norm = np.linalg.norm(cross, axis=1)
        r_dot_norm = np.linalg.norm(r_dot, axis=1)

        radius = (r_dot_norm ** 3)/ (cross_norm + eps)
        return np.abs(radius) if positive else radius


    def calc_waypoints(
            self, drone_init_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_norm: NDArray[np.floating], distance: float = 0.5, num_int_pnts: int = 5,
    ) -> NDArray[np.floating]:
        """Compute waypoints interpolated between gates."""
        num_gates = gates_pos.shape[0]
        wp = np.concatenate([gates_pos - distance * gates_norm + i/(num_int_pnts-1) * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
        wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)
        return wp
    
    def trajectory_generate(
        self, t_total: float, waypoints: NDArray[np.floating],
    ) -> CubicSpline:
        """Generate a cubic spline trajectory from waypoints."""
        diffs = np.diff(waypoints, axis=0)
        segment_length = np.linalg.norm(diffs, axis=1)
        arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
        t = arc_cum_length / arc_cum_length[-1] * t_total
        return CubicSpline(t, waypoints)
    
    def arclength_reparameterize(
            self, trajectory: CubicSpline, arc_length:float = 0.05, epsilon:float = 1e-5
        ):
        """reparameterize trajectory by arc length
        return a CubicSpline object with parameter t in [0, total_length] and is uniform in arc_length

        Args:
            t_total: originally used total time
            trajectory: CubicSpline object (function)
        """
        # initialize total_length by t_total
        total_length = trajectory.x[-1]
        for _ in range(99):
            # sample total_length/0.1 waypoints
            t_sample = np.linspace(0, total_length, int(total_length / arc_length))
            wp_sample = trajectory(t_sample)
            # measure linear distances
            diffs = np.diff(wp_sample, axis=0)
            segment_length = np.linalg.norm(diffs, axis=1)
            arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
            t_reallocate = arc_cum_length
            total_length = arc_cum_length[-1]
            # regenerate spline function
            trajectory = CubicSpline(t_reallocate, wp_sample)
            # terminal condition
            if np.std(segment_length) <= epsilon:
                return trajectory
            
    def extend_trajectory(self, trajectory: CubicSpline, extend_length:float = 1):
        """takes an arbirary 3D trajectory and extend it in the direction of terminal derivative."""
        theta_original = trajectory.x
        delta_theta = min(theta_original[1] - theta_original[0], 0.2)
        # calc last derivative
        p_end = trajectory(theta_original[-1])
        dp_end = trajectory.derivative(1)(theta_original[-1] - 0.1)
        dp_end_normalized = dp_end / np.linalg.norm(dp_end)
        # calc extended theta list
        theta_extend = np.arange(theta_original[-1] + delta_theta, theta_original[-1] + extend_length, delta_theta)
        p_extend = np.array([p_end + dp_end_normalized * (s - theta_original[-1]) for s in theta_extend])
        # cat original traj and extended traj
        theta_new = np.concatenate([theta_original, theta_extend])
        p_new = np.vstack([trajectory(theta_original), p_extend])

        extended_trajectory = CubicSpline(theta_new, p_new, axis=0)
        return extended_trajectory

            
    def find_nearest_waypoint(
            self, trajectory: CubicSpline, pos: NDArray[np.floating], total_length: float = None, sample_interval:float = 0.05
            ):
        """find nearest waypoint to given position on a trajectory
        return index and 3D waypoint

        Args:
            total_length: total length of spline
            trajectory: CubicSpline object (function)
            pos: current drone position
        """
        if total_length is None:
            total_length = trajectory.x[-1]
        # sample waypoints
        # t_sample = np.linspace(0, total_length, int(total_length / sample_interval))
        t_sample = np.arange(0, total_length, sample_interval)
        wp_sample = trajectory(t_sample)
        # find nearest waypoint
        distances = np.linalg.norm(wp_sample - pos, axis=1)
        idx_nearest = np.argmin(distances)
        return idx_nearest * sample_interval, wp_sample[idx_nearest]
    
    def find_gate_waypoint(
            self, trajectory: CubicSpline, gates_pos: NDArray[np.floating], total_length: float = None
        ):
        """find waypoints of gates center, mainly corresponding indices

        Args:
            total_length: total length of spline
            trajectory: CubicSpline object (function)
            gates_pos: current gates position
        """
        if total_length is None:
            total_length = trajectory.x[-1]
        indices = []
        gates_wp = []
        for pos in gates_pos:
            idx, wp = self.find_nearest_waypoint(trajectory, pos, total_length)
            indices.append(idx)
            gates_wp.append(wp)
        return np.array(indices), np.array(gates_wp)

class GeometryTool:
    def trilinear_interpolation(grid: np.ndarray, idx_f: np.ndarray) -> float:
        x, y, z = idx_f
        x0, y0, z0 = np.floor([x, y, z]).astype(int)
        dx, dy, dz = x - x0, y - y0, z - z0

        def get(ix, iy, iz):
            if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1] and 0 <= iz < grid.shape[2]:
                return grid[ix, iy, iz]
            else:
                return 0.0

        c000 = get(x0, y0, z0)
        c001 = get(x0, y0, z0 + 1)
        c010 = get(x0, y0 + 1, z0)
        c011 = get(x0, y0 + 1, z0 + 1)
        c100 = get(x0 + 1, y0, z0)
        c101 = get(x0 + 1, y0, z0 + 1)
        c110 = get(x0 + 1, y0 + 1, z0)
        c111 = get(x0 + 1, y0 + 1, z0 + 1)

        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        return c0 * (1 - dz) + c1 * dz


    
if __name__ == "__main__":
    pass
