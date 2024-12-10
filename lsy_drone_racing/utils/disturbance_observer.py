"""Classes with external disturbance/noise observers/estimators with selectable dimensions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.linalg import block_diag
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
    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], dt: np.floating): # TODO give obs and info
        """Initialize basic parameters.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's observation space for details.
            dt: Time step between callings.
        """
        dim_x = 18
        dim_u = 4 # 4 times RPM
        dim_y = 6 # output (from VICON) 12 if velocity is available (which it isn't), 6 if only position 
        super().__init__(dim_x, dim_u, dim_y, dt) # State = Observations = [pos, vel, rpy, ang_vel] + f + t 
        self._sigma_points = MerweScaledSigmaPoints(n=self._state_dim, alpha=1e-3, beta=2.0, kappa=0.0)
        self._UKF = UnscentedKalmanFilter(dim_x=self._state_dim, dim_z=self._obs_dim, 
                                          fx=self.f_vec, hx=self.h, 
                                          dt=dt, points=self._sigma_points)
        
        self._state = self._UKF.x
        
        # Set process noise covariance (tunable). Uncertainty in the dynamics. High Q -> less trust in model
        # Q_x = Q_discrete_white_noise(dim=2, dt=dt, var=1e-9, block_size=6, order_by_dim=False)
        # Q_f = np.eye(3)*1e-6 # Q_discrete_white_noise(dim=3, dt=dt, var=1e-6, block_size=1, order_by_dim=False)
        # Q_t = np.eye(3)*1e-6 # Q_discrete_white_noise(dim=3, dt=dt, var=1e-6, block_size=1, order_by_dim=False)
        # self._UKF.Q = block_diag(Q_x, Q_f, Q_t) # np.eye(18)*1e-6 # 
        # self._UKF.Q[12:15, 12:15] = self._UKF.Q[12:15, 12:15] * 1e1 # Force
        # self._UKF.Q[15:18, 15:18] = self._UKF.Q[15:18, 15:18] * 1e1 # Torque

        self._varQ = 1e-3
        self._varR = 1e-4

        # self._UKF.Q = Q_discrete_white_noise(dim=3, dt=self._dt, var=self._varQ, block_size=6, order_by_dim=False) # TODO manually setup matrix
        # self._UKF.Q[12:15, 12:15] = self._UKF.Q[12:15, 12:15] * 5e0 # Force
        # self._UKF.Q[15:18, 15:18] = self._UKF.Q[15:18, 15:18] * 5e0 # Torque
        # self._UKF.Q = np.eye(18)*1e-9
        # This way, pos, vel, and force (or rpy, angular vel, and torque) influence each other
        # print(self._UKF.Q.tolist()) 

        Q_p = np.eye(3)*1e-9
        Q_a = np.eye(3)*1e-9
        Q_v = np.eye(3)*1e-9
        Q_w = np.eye(3)*1e-9
        Q_f = np.eye(3)*1e-6
        Q_t = np.eye(3)*1e-6
        self._UKF.Q = block_diag(Q_p, Q_a, Q_v, Q_w, Q_f, Q_t)
        
        # Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        self._UKF.R = np.eye(self._obs_dim) * self._varR
        
        # Initialize state and covariance
        self._state[:6] = [*initial_obs["pos"], *initial_obs["rpy"]] # only initialize position and angle
        self._UKF.x = self._state
        self._UKF.P = np.eye(self._state_dim) * 1e-6 # how certain are we initially? Basically 100%


    def set_parameters(self, varQ: np.floating, varR: np.floating):
        """Stores the parameters if valid. This function should not be called after step() is called for the first time."""
        self._varQ = varQ
        self._varR = varR
        self._UKF.Q = Q_discrete_white_noise(dim=3, dt=self._dt, var=self._varQ, block_size=6, order_by_dim=False)
        self._UKF.R = np.eye(self._obs_dim) * self._varR


    def f(self, x: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
        """State transition function that maps states and inputs to next states.
        
        Args:
            x: State vector
            dt: Time step size

        Return:
            New state after one time step with dt
        """
        # start = time.perf_counter()
        # u = args.pop("u")
        pos = x[0:3]
        rpy = x[3:6]
        vel = x[6:9]
        
        rot = R.from_euler("xyz", rpy) # Create rotation object once saves time!
        rpy_rates = rot.apply(x[9:12], inverse=True)  # Now in body frame
        f = x[12:15]
        t = x[15:18]
        # Compute forces and torques.
        forces = self._input**2 * kf
        # forces = u**2 * kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = rot.apply(thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f
        z_torques = self._input**2 * km
        # z_torques = u**2 * km
        z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * L
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * L
        torques = np.array([x_torque, y_torque, z_torque]) + t
        torques -= np.cross(rpy_rates, np.dot(J, rpy_rates))
        rpy_rates_deriv = np.dot(J_inv, torques)
        acc = force_world_frame / MASS
        # Update state.
        vel += acc * dt
        rpy_rates += rpy_rates_deriv * dt
        x[0:3] = pos + vel * dt #+ 0.5*acc*dt**2
        x[3:6] = rpy + rpy_rates * dt
        x[6:9] = vel
        x[9:12] = rot.apply(rpy_rates)
        # print(time.perf_counter() - start)
        return x
    
    def dxdt(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """TODO."""
        pos = x[:, 0:3]
        rpy = x[:, 3:6]
        vel = x[:, 6:9]
        
        rot = R.from_euler("xyz", rpy) # Create rotation object once saves time!
        rpy_rates = rot.apply(x[:, 9:12], inverse=True)  # Now in body frame
        f = x[:, 12:15]
        t = x[:, 15:18]
        # Compute forces and torques.
        forces = self._input**2 * kf
        # forces = u**2 * kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = rot.apply(thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f
        z_torques = self._input**2 * km
        # z_torques = u**2 * km
        z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * L
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * L
        torques = np.array([x_torque, y_torque, z_torque]) + t
        # torques -= np.cross(rpy_rates, np.dot(J, rpy_rates))
        torques -= np.cross(rpy_rates, rpy_rates @ J) # usually transposed, but skipped since J is symmetric
        # rpy_rates_deriv = np.dot(J_inv, torques)
        rpy_rates_deriv = torques @ J_inv # usually transposed, but skipped since J_inv is symmetric
        acc = force_world_frame / MASS
        # Update state.
        x[:, 0:3] = vel
        x[:, 3:6] = rpy_rates
        x[:, 6:9] = acc
        x[:, 9:12] = rot.apply(rpy_rates_deriv)

        return x

    def f_vec(self, x: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
        """State transition function that maps states and inputs to next states. Batched to increase performance.
        
        Args:
            x: State vector, batched shape: (batches, state_dim)
            dt: Time step size

        Return:
            New state after one time step with dt in dimension (batches, state_dim)
        """
        # x_ = x.copy()
        # start = time.perf_counter()
        # u = args.pop("u")
        pos = x[:, 0:3]
        rpy = x[:, 3:6]
        vel = x[:, 6:9]
        
        rot = R.from_euler("xyz", rpy) # Create rotation object once saves time!
        rpy_rates = rot.apply(x[:, 9:12], inverse=True)  # Now in body frame
        f = x[:, 12:15]
        t = x[:, 15:18]
        # Compute forces and torques.
        forces = self._input**2 * kf
        # forces = u**2 * kf
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = rot.apply(thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f
        z_torques = self._input**2 * km
        # z_torques = u**2 * km
        z_torque = z_torques[0] - z_torques[1] + z_torques[2] - z_torques[3]
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * L / np.sqrt(2)
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * L / np.sqrt(2)
        torques = np.array([x_torque, y_torque, z_torque]) + t
        # torques -= np.cross(rpy_rates, np.dot(J, rpy_rates))
        torques -= np.cross(rpy_rates, rpy_rates @ J) # usually transposed, but skipped since J is symmetric
        # rpy_rates_deriv = np.dot(J_inv, torques)
        rpy_rates_deriv = torques @ J_inv # usually transposed, but skipped since J_inv is symmetric
        acc = force_world_frame / MASS
        # Update state.
        vel += acc * dt
        rpy_rates += rpy_rates_deriv * dt

        
        # print(f"pos={pos[0]}, x={x[0, :3]}")
        # print(f"vel={vel[0]*dt}, x_dot={self.dxdt(x)[0, :3]*dt}") #+ +
        # print(f"pos_new={pos[0]+vel[0]*dt}, x_new={x[0, :3]+self.dxdt(x)[0, :3]*dt}")
        # x_[:, :3] = x[:, :3] + self.dxdt(x)[:, :3]*dt
        x[:, 0:3] = pos + vel * dt #+ 0.5*acc*dt**2
        x[:, 3:6] = rpy + rpy_rates * dt
        x[:, 6:9] = vel
        x[:, 9:12] = rot.apply(rpy_rates)
        # print(time.perf_counter() - start)

        
        return x
    
    def h(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Observation function that maps states to measurements."""
        return x[:self._obs_dim]
    
    def step(self, obs: dict, u: NDArray) -> NDArray[np.floating]:
        """TODO."""
        # First, do prediction step
        self._UKF.predict() # u=u
        # Second, do correction step (=update)
        # obs = np.array( [*obs["pos"], *obs["rpy"], *obs["vel"], *obs["ang_vel"]] )
        obs = np.concatenate( (obs["pos"], obs["rpy"]) )
        self._UKF.update(obs)

        # self._state = self._UKF.x
        
        # return self._state
        return self._UKF.x
    
