"""Classes with external disturbance/noise observers/estimators with selectable dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.sim import GRAVITY

if TYPE_CHECKING:
    from numpy.typing import NDArray

g: NDArray[np.floating] = np.array([0,0,-GRAVITY])
# The following constants can be taken from the drone object or directly from the .urdf file in sim/assets
# For now, we simply hard code them, assuming the same drone is always used. Should be done properly in the future
MASS: float = 0.03454 
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
J_inv = np.linalg.inv(J)
kf = 3.16e-10
km = 7.94e-12
L = 0.046

class DisturbanceObserver:
    """Base class for noise applied to inputs or dyanmics."""

    def __init__(self, state_dim: int, input_dim: int, obs_dim: int, dt: np.floating):
        """Initialize basic parameters.

        Args:
            state_dim: Dimensionality of the systems states, e.g., x of f(x,u)
            input_dim: Dimensionality of the input to the dynamics, e.g., u of f(x,u)
            obs_dim: Dimensionality of the observations, e.g., y
            dt: Time step between callings.
        """
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._input_dim = input_dim
        self._dt = dt
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    def reset(self):
        """Reset the noise to its initial state."""
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    def step(self):
        """Increment the noise step for time dependent noise classes."""
        raise NotImplementedError
    
    def set_input(self, u: NDArray[np.floating]):
        """Sets the input of the dynamical system. Assuming this class gets called multiple times between controller calls. We therefore store the input as a constant in the class.

        Args:
            u: Input to the dynamical system.
        """
        assert np.shape(u) == (self._input_dim,)
        self._input = u


class FxTDO(DisturbanceObserver):
    """Fixed time Disturbance Observer (FxTDO) as implemented by one of the two publications mentioned below."""
    def __init__(self, dt: np.floating):
        """Initialize basic parameters.

        Args:
            dim: The dimensionality of the observer.
            dt: Time step between callings.
        """
        super().__init__(6, 1, 12, dt)

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
        
    def step(self, obs: dict) -> NDArray[np.floating]:
        """Steps the observer to calculate the next state and force estimate."""
        f_t = R.from_euler("xyz", obs["rpy"]).apply(np.array([0,0,self._input[0]]))
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

class UKF(DisturbanceObserver):
    """TODO."""
    def __init__(self, dt: np.floating):
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
        """
        dim_x = 18
        super().__init__(dim_x, 4, 12, dt) # State = Observations = [pos, vel, rpy, ang_vel] + f + t 
        self._sigma_points = MerweScaledSigmaPoints(n=self._state_dim, alpha=1e-3, beta=2.0, kappa=0.0)
        self._UKF = UnscentedKalmanFilter(dim_x=self._state_dim, dim_z=self._obs_dim, 
                                          fx=self.f, hx=self.h, 
                                          dt=dt, points=self._sigma_points)
        
        self._state = self._UKF.x
        
        # Set process noise covariance (tunable)
        dim_euclidean = 3 #(x,y,z)
        self._UKF.Q = Q_discrete_white_noise(dim=dim_euclidean, dt=dt, var=0.001, block_size=dim_x//dim_euclidean, order_by_dim=False)
        self._UKF.Q[12:15, 12:15] = self._UKF.Q[12:15, 12:15]*0.1
        
        # Set measurement noise covariance (tunable)
        self._UKF.R = np.eye(self._obs_dim) * 0.001
        
        # # Initialize state and covariance
        # self._UKF.x = np.zeros(self._state_dim)
        # self._UKF.P = np.eye(self._state_dim) * 1.0

    def f(self, x: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
        """State transition function that maps states and inputs to next states.
        
        Args:
            x: State vector
            dt: Time step size

        Return:
            New state after one time step with dt
        """
        pos = x[0:3]
        rpy = x[3:6]
        vel = x[6:9]
        rpy_rates = R.from_euler("xyz", rpy).apply(x[9:12], inverse=True)  # Now in body frame
        f = x[12:15]
        t = x[15:18]
        # Compute forces and torques.
        forces = np.array(self._input**2) * kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = R.from_euler("xyz", rpy).apply(thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f
        z_torques = np.array(self._input**2) * km
        z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (L / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (L / np.sqrt(2))
        torques = np.array([x_torque, y_torque, z_torque]) + t
        torques = torques - np.cross(rpy_rates, np.dot(J, rpy_rates))
        rpy_rates_deriv = np.dot(J_inv, torques)
        acc = force_world_frame / MASS
        # Update state.
        vel = vel + acc * dt
        rpy_rates = rpy_rates + rpy_rates_deriv * dt
        x[0:3] = pos + vel * dt
        x[3:6] = rpy + rpy_rates * dt
        x[6:9] = vel
        x[9:12] = R.from_euler("xyz", rpy).apply(rpy_rates)
        return x
    
    def h(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Observation function that maps states to measurements."""
        return x[:self._obs_dim]
    
    def step(self, obs: dict) -> NDArray[np.floating]:
        """TODO."""
        # First, do prediction step
        self._UKF.predict()
        # Second, do correction step (=update)
        obs = np.array( [*obs["pos"], *obs["rpy"], *obs["vel"], *obs["ang_vel"]] )
        # obs = np.array( [*obs["pos"], *obs["rpy"]] )
        self._UKF.update(obs)
        
        return self._UKF.x #self._state
    
