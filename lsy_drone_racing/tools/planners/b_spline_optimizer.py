import numpy as np
from scipy.interpolate import BSpline
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Callable

LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    import matplotlib.collections
    import matplotlib.lines as lines
    LOCAL_MODE = True
except ModuleNotFoundError:
    LOCAL_MODE = False

class UniformBSpline:

    t : NDArray[np.floating]
    k : int
    ctrl_pts : NDArray[np.floating]
    b_spline : BSpline
    _last_plot : lines.Line2D
    def __init__(self):
        self.t = None
        self.ctrl_pts = None
        self.b_spline = None
        self._last_plot = None

    def __call__(self, t: float | NDArray[np.floating]) -> NDArray[np.floating]:
        return self.b_spline(t)

    def derivative(self, nu: int = 1) -> Callable[[float | NDArray[np.floating]], NDArray[np.floating]]:
        return self.b_spline.derivative(nu)

    def parameter_2_bspline_uniform(
        self,
        waypoints: List[NDArray[np.floating]],
        v_start: NDArray[np.floating],
        v_end: NDArray[np.floating],
        dt: float = 0.1,
        offset : float = 0.0
    ) -> Tuple[BSpline, NDArray[np.floating], NDArray[np.floating]]:

        waypoints = np.array(waypoints)  # shape = (K, 3)
        K = len(waypoints)
        self.k = 3
        prow = np.array([1, 4, 1]) / 6.0
        vrow = np.array([-1, 0, 1]) / (2.0 * dt)

        A = np.zeros((K + 2, K + 2))
        Bx, By, Bz = np.zeros(K + 2), np.zeros(K + 2), np.zeros(K + 2)

        for i in range(K):
            A[i, i:i+3] = prow
            Bx[i], By[i], Bz[i] = waypoints[i]

        A[K, 0:3] = vrow
        A[K + 1, -3:] = vrow
        Bx[K], By[K], Bz[K] = v_start
        Bx[K + 1], By[K + 1], Bz[K + 1] = v_end

        
        Px = np.linalg.solve(A, Bx)
        Py = np.linalg.solve(A, By)
        Pz = np.linalg.solve(A, Bz)
        self.ctrl_pts = np.stack([Px, Py, Pz], axis=1)  # (K+2, 3)

       
        N = self.ctrl_pts.shape[0]
        self.t = np.concatenate([
            np.zeros(self.k),
            np.arange(N - self.k + 1),
            np.full(self.k, N - self.k)
        ]) * dt
        self.t += offset
        self.b_spline = BSpline(self.t, self.ctrl_pts, self.k)

        return self.b_spline, self.t, self.ctrl_pts
    
    def remove_plot(self) -> bool:
         if hasattr(self, "_last_plot") and self._last_plot is not None:
            try:
                self._last_plot.remove()
                return True
            except:
                return False
    def visualize_B_spline(self, fig : figure.Figure, ax : axes.Axes, color = 'red') -> Tuple[figure.Figure, axes.Axes]:
        if hasattr(self, "_last_plot") and self._last_plot is not None:
            self._last_plot.remove()

        t0 = self.t[self.k]
        t1 = self.t[-self.k - 1]
        ts = np.linspace(t0, t1, 200)

        pts = self.b_spline(ts)  # shape (N, 3)
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

        line = ax.plot(xs, ys, zs, color=color, linewidth=2, label='B-spline')[0]
        self._last_plot = line

        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax
    
    def sample(self, num: int = 200) -> Tuple[NDArray, NDArray, NDArray]:

        t0 = self.t[self.k]
        t1 = self.t[-self.k - 1]
        ts = np.linspace(t0, t1, num)
        pos = self.bspline(ts)
        vel = self.bspline.derivative(1)(ts)
        acc = self.bspline.derivative(2)(ts)
        return pos, vel, acc
    
    def reset(self):
        self.t = None
        self.ctrl_pts = None
        self.bspline = None
        if self._last_plot is not None:
            self._last_plot.remove()
            self._last_plot = None


from scipy.optimize import minimize

class BsplineOptimizer:
    def __init__(self,
                lam_smooth : np.floating = 1.0,
                lam_vel : np.floating = 1.0,
                lam_acc : np.floating = 1.0,
                lam_sdf : np.floating = 1.0,
                v_max : np.floating = 5.0, 
                a_max : np.floating = 10.0,
                sdf_func: Optional[Callable[[np.ndarray], float]] = None,
                sdf_thres : Optional[np.floating] = None, 
                dt: float = 0.1,
                verbose : bool = True):
        self.sdf_func = sdf_func
        self.dt = dt
        self.lam_smooth = lam_smooth
        self.lam_vel = lam_vel
        self.lam_acc = lam_acc
        self.lam_sdf = lam_sdf
        self.v_max = v_max
        self.a_max = a_max
        self.sdf_threshold = sdf_thres
        self.verbose = verbose

    
    

    def optimize(self, ctrl_pts_init: np.ndarray, fix_pnts : List[bool] = None) -> np.ndarray:
        

        N = ctrl_pts_init.shape[0]

        def cost(x_flat : NDArray[np.floating]):
            x = x_flat.reshape(N, 3)
            total_cost = 0.0

            for i in range(1, N - 1):
                d2 = x[i - 1] - 2 * x[i] + x[i + 1]
                total_cost += self.lam_smooth * np.sum(d2 ** 2)

            if self.lam_vel > 0.0:
                for i in range(N - 1):
                    v = (x[i + 1] - x[i]) / self.dt
                    v2 = np.sum(v ** 2)
                    if v2 > self.v_max ** 2:
                        total_cost += self.lam_vel * (v2 - self.v_max ** 2) ** 2

            if self.lam_acc > 0.0:
                for i in range(1, N - 1):
                    a = (x[i - 1] - 2 * x[i] + x[i + 1]) / (self.dt ** 2)
                    a2 = np.sum(a ** 2)
                    if a2 > self.a_max ** 2:
                        total_cost += self.lam_acc * (a2 - self.a_max ** 2) ** 2

            if self.sdf_func is not None and self.lam_sdf > 0.0:
                for i in range(3, N - 3):
                    dist = self.sdf_func(x[i])
                    if dist < self.sdf_threshold:
                        total_cost += self.lam_sdf * 1 / ((dist) ** 2 + 1e-6)

            return total_cost

        x0 = ctrl_pts_init.reshape(-1)
        result = minimize(cost, x0, method='L-BFGS-B', options={'disp': self.verbose})
        return result.x.reshape(N, 3)