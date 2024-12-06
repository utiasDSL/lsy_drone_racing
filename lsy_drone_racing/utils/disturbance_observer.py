"""Classes with external disturbance/noise observers/estimators with selectable dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.sim import GRAVITY

if TYPE_CHECKING:
    from numpy.typing import NDArray

g: NDArray[np.floating] = np.array([0,0,-GRAVITY])
MASS: float = 0.03454 # we can get that from Drone class (or the urdf file in sim/assets)

class DisturbanceObserver:
    """Base class for noise applied to inputs or dyanmics."""

    def __init__(self, dim: int, dt: np.floating):
        """Initialize basic parameters.

        Args:
            dim: The dimensionality of the noise.
            dt: Time step between callings.
        """
        self._dim = dim
        self._dt = dt
        self._state = np.zeros(self._dim)

    def reset(self):
        """Reset the noise to its initial state."""
        self._state = np.zeros(self._dim)

    def step(self):
        """Increment the noise step for time dependent noise classes."""
        return


class FxTDO(DisturbanceObserver):
    """Fixed time Disturbance Observer (FxTDO) as implemented by this publication.
    
    TODO.
    """
    def __init__(self, dt: np.floating):
        """Initialize basic parameters.

        Args:
            dim: The dimensionality of the observer.
            dt: Time step between callings.
        """
        super().__init__(6, dt)

        self._f_d_max = 0.1 # N
        self._f_d_dot_max = 1 # N/s
        self._v_max = 10 # m/s

        # Implementation as in 
        # "Fixed-time Disturbance Observer-Based MPC Robust Trajectory Tracking Control of Quadrotor" (2024)
        self._L1 = 0.2
        self._L2 = 2.0 # how fast the force converges (linear), but also how noisy it is
        self._k1 = np.array([1.0, 2.0, 2.55])
        self._k2 = np.array([0.01, 2.0, 3.0])
        self._d_inf = 0.9
        self._alpha1 = np.array([0.5, 1.0, 1/(1-self._d_inf)]) # from FxTDO MPC paper
        self._alpha2 = np.array([0.0, 1.0, (1+self._d_inf)/(1-self._d_inf)]) # from FxTDO MPC paper

        # Implementation as in 
        # "A fixed-time output feedback control scheme for double integrator systems" (2017)
        # self._L1 = 0.2
        # self._L2 = 5.0 # how fast the force converges (linear), but also how noisy it is
        # self._k1 = np.array([2.0, 0.0, 0.0])
        # self._k1[2] = self._k1[0]
        # self._k2 = np.array([0.06, 0.0, 0.0])
        # self._k2[1] = self._k2[0]*4
        # self._k2[2] = self._k2[0]*3 
        # self._alpha = 0.6 # in (0.5, 1)
        # self._alpha1 = np.array([self._alpha, 1.0, 2-self._alpha]) # from FTDO double integrator paper
        # self._alpha2 = np.array([2*self._alpha-1, 1.0, 3-2*self._alpha]) # from FTDO double integrator paper

        
    def set_parameters(self, 
                       f_d_max: np.floating, f_d_dot_max: np.floating, 
                       L1: np.floating, L2: np.floating, 
                       k1: NDArray[np.floating], k2: NDArray[np.floating], 
                       d_inf: np.floating):
        """Stores the parameters if valid."""
        if self.check_parameters(f_d_max, f_d_dot_max, L1, L2, k1, k2, d_inf):
            self._f_d_max = f_d_max
            self._f_d_dot_max = f_d_dot_max
            self._L1 = L1
            self._L2 = L2
            self._k1 = k1
            self._k2 = k2
            self._d_inf

    def check_parameters(self, f_d_max: np.floating, f_d_dot_max: np.floating, 
                       L1: np.floating, L2: np.floating, 
                       k1: NDArray[np.floating], k2: NDArray[np.floating], 
                       d_inf: np.floating) -> bool:
        """Checks ther parameters for validity. This is only needed to guarantee an upper bound on the estimation time.
         
        Returns:
            If the parameters are valid. 
        """
        # # first, simple checks
        # if L1 > 0 and L2 > 0 and k1.all > 0 and k2.all > 0 and 0 < d_inf and d_inf < 1:
        #     # now, more complicated checks
        #     if L2 > f_d_dot_max/k2[0]:
        #         return True
        # else: 
        #     return False
        return True
        
    def step(self, obs: NDArray[np.floating], thrust: np.floating) -> NDArray[np.floating]:
        """Steps the observer to calculate the next state and force estimate."""
        f_t = R.from_euler("xyz", obs["rpy"]).apply(np.array([0,0,thrust]))
        v = obs["vel"]
        v_hat = self._state[:3]
        f_hat = self._state[3:]
        e1 = v - v_hat

        # Calculate derivatives
        v_hat_dot = g + 1/MASS*f_t + 1/MASS*self._L1*self._phi1(e1) + 1/MASS*f_hat 
        f_hat_dot = self._L2*self._phi2(e1)

        # Integration step (forward Euler)
        v_hat = v_hat + np.clip(v_hat_dot, -5, 5) * self._dt
        v_hat = np.clip(v_hat, -self._v_max, self._v_max)
        f_hat = f_hat + np.clip(f_hat_dot, -self._f_d_dot_max, self._f_d_dot_max) * self._dt # Clipping if exceeding expectations
        f_hat = np.clip(f_hat, -self._f_d_max, self._f_d_max) # Clipping if exceeding expectations

        # Storing in the state
        self._state[:3] = v_hat
        self._state[3:] = f_hat

        return self._state
    
    def _phi1(self, e: np.floating) -> np.floating:
        s = 0
        for i in range(3):
            s = s + self._k1[i]*self._ud(e, self._alpha1[i])
        return s

    
    def _phi2(self, e: np.floating) -> np.floating:
        s = 0
        for i in range(3):
            s = s + self._k2[i]*self._ud(e, self._alpha2[i])
        return s

    def _ud(self, x: NDArray[np.floating], alpha: np.floating) -> NDArray[np.floating]:
        return np.sign(x) * (np.abs(x)**alpha)
