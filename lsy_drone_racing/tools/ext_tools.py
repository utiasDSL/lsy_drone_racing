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
            self, trajectory: CubicSpline, arc_length:float = 0.05
        ):
        """reparameterize trajectory by arc length
        return a CubicSpline object with parameter t in [0, total_length] and is uniform in arc_length

        Args:
            t_total: originally used total time
            trajectory: CubicSpline object (function)
        """
        epsilon = 1e-5
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
            
    def find_nearest_waypoint(
            self, total_length: float, trajectory: CubicSpline, pos: NDArray[np.floating]
            ):
        """find nearest waypoint to given position on a trajectory
        return index and 3D waypoint

        Args:
            total_length: total length of spline
            trajectory: CubicSpline object (function)
            pos: current drone position
        """
        # sample interval 0.05m
        t_sample = np.linspace(0, total_length, int(total_length / 0.05))
        wp_sample = trajectory(t_sample)
        # find nearest waypoint
        distances = np.linalg.norm(wp_sample - pos, axis=1)
        t_nearest = np.argmin(distances)
        return t_nearest, wp_sample[t_nearest]
    
    def find_gate_waypoint(
            self, total_length: float, trajectory: CubicSpline, gates_pos: NDArray[np.floating]
        ):
        """find waypoints of gates center, mainly corresponding indices

        Args:
            total_length: total length of spline
            trajectory: CubicSpline object (function)
            gates_pos: current gates position
        """
        indices = []
        gates_wp = []
        for pos in gates_pos:
            idx, wp = self.find_nearest_waypoint(total_length, trajectory, pos)
            indices.append(idx)
            gates_wp.append(wp)
        return np.array(indices), np.array(gates_wp)



    
if __name__ == "__main__":
    pass
