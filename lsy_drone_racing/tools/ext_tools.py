from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.typing import NDArray
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
    
if __name__ == "__main__":
    pass
