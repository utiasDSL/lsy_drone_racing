"""Classes with external disturbance/noise observers/estimators with selectable dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from .drone_models import models
from .filterpy import MerweScaledSigmaPoints, Q_discrete_white_noise, UnscentedKalmanFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

# from filterpy.common import Q_discrete_white_noise
# from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter



class Estimator:
    """Base class for estimator implementations."""

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
        # if np.ndim(u) < 2:  # Make u batchable, if 1D
        #     u_ = np.array([u]) 
        assert np.shape(u)[0] == self._input_dim
        self._input = u


class FxTDO(Estimator):
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
        GRAVITY = 9.81
        g: NDArray[np.floating] = np.array([0,0,-GRAVITY])
        MASS: float = 0.03454 
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


class UKF(Estimator):
    """Unscented Kalman Filter class that wraps the filterPY toolbox."""
    def __init__(self, dt: np.floating, model_name: str = "analytical_mel_att", initial_obs: dict[str, NDArray[np.floating]] | None = None): # TODO give obs and info
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            model_name: The name of the model that is to be used. More information in drone_models.py
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
        """
        model = models(model_name)
        self.f_x = model[0] # TODO initialize this while creating UKF object
        self.f_x_continuous = model[1]

        dim_x = model[2] # (pos, rpy, vel, rpy rates, f_dis, t_dis)
        dim_u = model[3] # 4 times RPM
        dim_y = 6 # output (from VICON) 12 if velocity is available (which it isn't), 6 if only position 
        super().__init__(dim_x, dim_u, dim_y, dt) # State = Observations = [pos, vel, rpy, ang_vel] + f + t 
        self._sigma_points = MerweScaledSigmaPoints(n=self._state_dim, alpha=1e-3, beta=2.0, kappa=0.0)
        self._UKF = UnscentedKalmanFilter(dim_x=self._state_dim, dim_z=self._obs_dim, 
                                          fx=self.f, hx=self.h, 
                                          dt=dt, points=self._sigma_points)
        
        self._state = self._UKF.x
        
        # Set process noise covariance (tunable). Uncertainty in the dynamics. High Q -> less trust in model
        # Q_x = Q_discrete_white_noise(dim=2, dt=dt, var=1e-9, block_size=6, order_by_dim=False)
        # Q_f = np.eye(3)*1e-6 # Q_discrete_white_noise(dim=3, dt=dt, var=1e-6, block_size=1, order_by_dim=False)
        # Q_t = np.eye(3)*1e-6 # Q_discrete_white_noise(dim=3, dt=dt, var=1e-6, block_size=1, order_by_dim=False)
        # self._UKF.Q = block_diag(Q_x, Q_f, Q_t) # np.eye(18)*1e-6 # 
        # self._UKF.Q[12:15, 12:15] = self._UKF.Q[12:15, 12:15] * 1e1 # Force
        # self._UKF.Q[15:18, 15:18] = self._UKF.Q[15:18, 15:18] * 1e1 # Torque

        self._varQ = 1e-2
        self._varR = 1e-3

        # self._UKF.Q = Q_discrete_white_noise(dim=3, dt=self._dt, var=self._varQ, block_size=6, order_by_dim=False) # TODO manually setup matrix
        # self._UKF.Q[12:15, 12:15] = self._UKF.Q[12:15, 12:15] * 5e0 # Force
        # self._UKF.Q[15:18, 15:18] = self._UKF.Q[15:18, 15:18] * 5e0 # Torque
        # self._UKF.Q = np.eye(18)*1e-9
        # This way, pos, vel, and force (or rpy, angular vel, and torque) influence each other
        # print(self._UKF.Q.tolist()) 

        Q_p = np.eye(3)*self._varQ*1e-0
        Q_a = np.eye(3)*self._varQ*1e-0
        Q_v = np.eye(3)*self._varQ*1e-0
        Q_w = np.eye(3)*self._varQ*1e-0
        Q_f = np.eye(3)*self._varQ*1e-0
        Q_t = np.eye(3)*self._varQ*1e-0
        self._UKF.Q = block_diag(Q_p, Q_a, Q_v, Q_w, Q_f, Q_t)
        
        # Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        self._UKF.R = np.eye(self._obs_dim) * self._varR
        
        # Initialize state and covariance
        if initial_obs is not None:
            self._state[:6] = np.concatenate( (initial_obs["pos"], initial_obs["rpy"]) ) # only initialize position and angle
        self._UKF.x = self._state
        self._UKF.P = np.eye(self._state_dim) * 1e-6 # how certain are we initially? Basically 100%


    def set_parameters(self, varQ: np.floating, varR: np.floating):
        """Stores the parameters if valid. This function should not be called after step() is called for the first time."""
        self._varQ = varQ
        self._varR = varR
        self._UKF.Q = Q_discrete_white_noise(dim=3, dt=self._dt, var=self._varQ, block_size=6, order_by_dim=False)
        self._UKF.R = np.eye(self._obs_dim) * self._varR

    def f(self, x: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
        """State transition function that maps states and inputs to next states. Batched to increase performance.
        
        Args:
            x: State vector, batched shape: (batches, state_dim)
            dt: Time step size

        Return:
            New state after one time step with dt in dimension (batches, state_dim)
        """
        # Extend dimensions to be batchable
        if x.ndim == 1:
            x = np.array([x])
        if self._input.ndim < 2:
        # if x.shape[0] > self._input.shape[0]: # and self._input.ndim == 2:
            # Batched x but not batched u
            u = np.repeat([self._input], x.shape[0], axis=0)
        else:
            u = self._input

        # Calculate next value (either by integration or direct). Not calculating forces or torques
        if self.f_x_continuous: # Integrate. TODO Impement general integrator (so one can decide if Euler, RK4, ...)
            x[:, :12] = x[:, :12] + self.f_x(x, u, dt)[:, :12]*dt
        else: # Direct 
            x[:, :12] = self.f_x(x, u, dt)[:, :12]

        return x
    
    def h(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Observation function that maps states to measurements."""
        return x[:self._obs_dim]
    
    def step(self, obs: NDArray[np.floating], dt: np.floating | None = None, u: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Steps the UKF by one. Doing one prediction and correction step.
                
        Args:
            obs: Latest observation in the form of a dict with "pos" and "rpy"
            dt: Optional, time step size. If not specified, default time is used
            u: Optional, latest input to the system

        Return:
            New state prediction
        """
        # Update the input
        if u is not None:
            self.set_input(u)

        # Prediction step
        self._UKF.predict(dt=dt) # if dt is none is checked in function
        # Correction step (=update)
        self._UKF.update(obs) # np.concatenate( (obs["pos"], obs["rpy"]) ) #obs["vel"], obs["ang_vel"]

        self._state = self._UKF.x
        
        return self._UKF.x
    
