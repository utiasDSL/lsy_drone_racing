"""Classes with external disturbance/noise observers/estimators with selectable dimensions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

GRAVITY = 9.81
g: NDArray[np.floating] = np.array([0,0,-GRAVITY])
# The following constants can be taken from the drone object or directly from the .urdf file in sim/assets
# For now, we simply hard code them, assuming the same drone is always used. Should be done properly in the future
MASS: float = 0.03454 
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
J_inv = np.linalg.inv(J)
kf = 3.16e-10
km = 7.94e-12
L = 0.046
rpy_err_i = 0 # This way is ghetto -> put attitude2rpm into class
last_rpy = np.array([[0,0,0]])
last_rpy_rates = np.array([[0,0,0]])

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
    def __init__(self, dt: np.floating, initial_obs: dict[str, NDArray[np.floating]] | None = None): # TODO give obs and info
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
        """
        self.f_x = f_x_att_ctrl # TODO initialize this while creating UKF object
        self.f_x_continuous = True

        dim_x = 18 # (pos, rpy, vel, rpy rates, f_dis, t_dis)
        dim_u = 4 # 4 times RPM
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
        if self.f_x_continuous: # Integrate. TODO general integrator
            x[:, :12] = x[:, :12] + self.f_x(x, u, dt)[:, :12]*dt
        else: # Direct 
            x[:, :12] = self.f_x(x, u, dt)[:, :12]

        return x
    
    def h(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Observation function that maps states to measurements."""
        return x[:self._obs_dim]
    
    def step(self, obs: dict, dt: np.floating | None = None, u: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
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
        self._UKF.update(np.concatenate( (obs["pos"], obs["rpy"]) )) #obs["vel"], obs["ang_vel"]

        self._state = self._UKF.x
        
        return self._UKF.x
    
def f_x_dyn(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating | None = None) -> NDArray[np.floating]:
    """Example dynamics of the drone. Taken from drone_racing/sim/physics dyn model.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, rpms) of the 4 rotors.
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    x_dot = np.empty_like(x)
    
    rot = R.from_euler("xyz", x[:, 3:6]) # Create rotation object once saves time! rpy
    rpy_rates = rot.apply(x[:, 9:12], inverse=True)  # Now in body frame
    f = x[:, 12:15]
    t = x[:, 15:18]

    # Compute forces and torques.
    forces = u**2 * kf
    thrust = np.zeros((u.shape[0], 3))
    thrust[:, -1] = np.sum(forces, axis=1)
    thrust_world_frame = rot.apply(thrust)
    force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f
    x_torque = (forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]) * L
    y_torque = (-forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]) * L
    z_torques = u**2 * km
    z_torque = z_torques[:, 0] - z_torques[:, 1] + z_torques[:, 2] - z_torques[:, 3]
    torques = np.zeros((u.shape[0], 3))
    torques[:, 0] = x_torque
    torques[:, 1] = y_torque
    torques[:, 2] = z_torque
    torques += t
    torques -= np.cross(rpy_rates, rpy_rates @ J.T) # usually transposed, but skipped since J is symmetric
    rpy_rates_deriv = torques @ J_inv.T # usually transposed, but skipped since J_inv is symmetric

    # Set Derivatives
    x_dot[:, 0:3] = x[:, 6:9] # velocity
    x_dot[:, 3:6] = rpy_rates
    x_dot[:, 6:9] = force_world_frame / MASS # acceleration
    x_dot[:, 9:12] = rot.apply(rpy_rates_deriv)

    return x_dot

def f_x_att_ctrl(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Example dynamics of the drone. Extension of the above dyn model to attitude inputs for the Mellinger controller.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). [Thrust, R, P, Y]
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    u_RPM, _ = attitude_controller(x, u, dt)
    return f_x_dyn(x, u_RPM, dt)

def f_x_attitude_dyn(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Example dynamics of the drone. Extension of the above dyn model to attitude inputs.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). 
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    # From sim/drone collective_thrust_cmd()
    # np.rad2deg(rpy)
    global rpy_err_i
    global last_rpy
    global last_rpy_rates

    # From https://github.com/utiasDSL/crazyflow/blob/main/crazyflow/control/controller.py#L92
    P_T = np.array([70000.0, 70000.0, 60000.0])
    I_T = np.array([0.0, 0.0, 500.0])
    D_T = np.array([20000.0, 20000.0, 12000.0])
    MIX_MATRIX = np.array([[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]])
    MIN_PWM: float = 20000
    MAX_PWM: float = 65535
    
    # attitude2rpm
    rot = R.from_euler("xyz", x[:, 3:6])
    target_rot = R.from_euler("xyz", u[:, 1:])
    drot = (target_rot.inv() * rot).as_matrix() # delta rot

    # Extract the anti-symmetric part of the relative rotation matrix.
    rot_e = np.array([drot[:, 2, 1] - drot[:, 1, 2], drot[:, 0, 2] - drot[:, 2, 0], drot[:, 1, 0] - drot[:, 0, 1]]).T
    rpy_rates = rot.apply(x[:, 9:12], inverse=True)  # Now in body frame
    # rpy_rates_e = -(rot.as_euler("xyz") - last_rpy) / dt  # Assuming zero rpy_rates target
    rpy_rates_e =  (rpy_rates - last_rpy_rates)
    # last_rpy = rot.as_euler("xyz").copy()
    last_rpy_rates = rpy_rates.copy()
    # rpy_err_i = rpy_err_i - rot_e * dt
    # rpy_err_i = np.clip(rpy_err_i, -1500.0, 1500.0)
    # rpy_err_i[:, :2] = np.clip(rpy_err_i[:, :2], -1.0, 1.0)
    # PID target torques.
    target_torques = -P_T * rot_e + D_T * rpy_rates_e# + I_T * rpy_err_i
    target_torques = np.clip(target_torques, -3200, 3200)
    thrust_per_motor = u[0] / 4
    pwm = np.clip(thrust2pwm(thrust_per_motor) + target_torques @ MIX_MATRIX.T, MIN_PWM, MAX_PWM)
    rpms =  pwm2rpm(pwm)
    return f_x_dyn(x, rpms, dt)

def rot2frame(rot: R, yaw: np.floating) -> NDArray[np.floating]:
    """Converts a rotation object with a given yaw into a base frame."""
    z_axis_desired = rot.apply(np.array([0,0,1]), inverse=True) # desired z axis in body frame
    x_axis_desired = np.array([np.cos(yaw), np.sin(yaw), 0]) # temporary x axis from yaw angle
    y_axis_desired = np.cross(z_axis_desired, x_axis_desired) # desired y axis in body frame
    y_axis_desired = y_axis_desired/np.linalg.norm(y_axis_desired) # normalizing y_axis
    x_axis_desired = np.cross(y_axis_desired, z_axis_desired) # desired x axis in body frame


def thrust2pwm(thrust: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert the desired thrust into motor PWM.

    Args:
        thrust: The desired thrust *per motor*.

    Returns:
        The motors' PWMs to apply to the quadrotor.
    """
    KF: float = 3.16e-10
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM: float = 20000
    MAX_PWM: float = 65535
    MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
    MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
    MIN_THRUST: float = KF * MIN_RPM**2
    MAX_THRUST: float = KF * MAX_RPM**2
    thrust = np.clip(thrust, MIN_THRUST, MAX_THRUST)  # Protect against NaN values
    return np.clip((np.sqrt(thrust / KF) - PWM2RPM_CONST) / PWM2RPM_SCALE, MIN_PWM, MAX_PWM)

def pwms_to_thrust(pwms: NDArray[np.floating]) -> NDArray[np.floating]:
    pwm2rpm_scale = 0.2685
    pwm2rpm_const = 4070.3
    return kf * (pwm2rpm_scale * pwms + pwm2rpm_const) ** 2

def thrust_to_pwm(thrust):
    # from drone.py
    thrust_curve_a = -1.1264
    thrust_curve_b = 2.2541
    thrust_curve_c = 0.0209  # Thrust curve parameters for brushed motors
    max_pwm = 65535
    pwm = thrust_curve_a * thrust * thrust + thrust_curve_b * thrust + thrust_curve_c
    return np.clip(pwm, 0, 1) * max_pwm

def thrust_to_rpm(thrust: NDArray[np.floating]) -> NDArray[np.floating]:
    # From sim.py
    min_pwm = 20000
    max_pwm = 65535
    pwm2rpm_scale = 0.2685
    pwm2rpm_const = 4070.3
    min_rpm = pwm2rpm_scale * min_pwm + pwm2rpm_const
    max_rpm = pwm2rpm_scale * max_pwm + pwm2rpm_const
    min_thrust = kf * min_rpm**2
    max_thrust = kf * max_rpm**2

    thrust = np.clip(thrust, min_thrust, max_thrust)

    pwm = (np.sqrt(thrust / kf) - pwm2rpm_const) / pwm2rpm_scale
    pwm = np.clip(pwm, min_pwm, max_pwm)
    return pwm2rpm_const + pwm2rpm_scale * pwm


def pwm2rpm(pwm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert the motors' PWMs into RPMs."""
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    return PWM2RPM_CONST + PWM2RPM_SCALE * pwm

def batComp(pwm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compensate the battery voltage."""
    supply_voltage = 3.0 # assumed supply voltage. Taken from drone.py
    max_pwm = 65535
    pwm = pwm/max_pwm*60
    pwm_curve_a = -0.0006239  # PWM curve parameters for brushed motors
    pwm_curve_b = 0.088  # PWM curve parameters for brushed motors
    voltage = pwm_curve_a * pwm**2 + pwm_curve_b * pwm
    percentage = np.minimum(1, voltage/supply_voltage)
    return percentage * max_pwm

def f_x_fitted(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Fitted Dynamics of the drone from Haocheng.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). 
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    # 3D model parameters
    a = 16.2496
    b = 4.6013
    c = -99.94 # coefficient for roll
    d = -13.3 # coefficient for roll_dot
    e = 84.73 # coefficient for roll_cmd
    f = -130.3 # coefficient for pitch
    h = -16.33 # coefficient for pitch_dot
    l = 119.3 # coefficient for pitch_cmd
    m = 0 # coefficient for yaw
    n = 0 # coefficient for yaw_dot
    r = 0 # coefficient for yaw_cmd
    g = 9.81

    x_dot = np.empty_like(x)
    x_dot[:, 12:] = 0

    for i in range(x.shape[0]): #model doesnt support batches yet, so we iterate manually
        controls = u[i]
        pos = x[i, 0:3]
        rpy = x[i, 3:6]
        vel = x[i, 6:9]
        rot = R.from_euler("xyz", rpy)
        rpy_rates = rot.apply(x[i, 9:12], inverse=True) # TODO necessary???
        Fx = np.array([[1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WARNING: Order is different than regular state vector order!!!
                        [0, 1, 0, 0, 0, 0, dt * (a * controls[0] + b) * (- np.sin(rpy[0]) * np.sin(rpy[1]) * np.cos(rpy[2]) + np.cos(rpy[0]) * np.sin(rpy[2])), 0, dt * (a * controls[0] + b) * np.cos(rpy[0]) * np.cos(rpy[1]) * np.cos(rpy[2]), 0, dt * (a * controls[0] + b) * (- np.cos(rpy[0]) * np.sin(rpy[1]) * np.sin(rpy[2]) + np.sin(rpy[0]) * np.cos(rpy[2])), 0],
                        [0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, dt * (a * controls[0] + b) * (- np.sin(rpy[0]) * np.sin(rpy[1]) * np.sin(rpy[2]) - np.cos(rpy[0]) * np.cos(rpy[2])), 0, dt * (a * controls[0] + b) * (np.cos(rpy[0]) * np.cos(rpy[1]) * np.sin(rpy[2])), 0, dt * (a * controls[0] + b) * (np.cos(rpy[0]) * np.sin(rpy[1]) * np.cos(rpy[2]) + np.sin(rpy[0]) * np.sin(rpy[2])), 0],
                        [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, dt * -(a * controls[0] + b) * np.sin(rpy[0]) * np.cos(rpy[1]), 0, dt * -(a * controls[0] + b) * np.cos(rpy[0]) * np.sin(rpy[1]), 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, dt * c, 1 + dt * d, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, dt * f, 1 + dt * h, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dt * m, 1 + dt * n]])
        x_dot[i, :12] = Fx @ np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], 
                                  rpy[0], rpy_rates[0], rpy[1], rpy_rates[1], rpy[2], rpy_rates[2]])
        x_dot[i, 9:12] = rot.apply(x_dot[i, 9:12]) # TODO necessary?
        
    return x_dot

SYS_ID_PARAMS = {
    "acc_k1": 20.91,
    "acc_k2": 3.65,
    "roll_alpha": -3.96,
    "roll_beta": 4.08,
    "pitch_alpha": -6.00,
    "pitch_beta": 6.21,
    "yaw_alpha": 0.00,
    "yaw_beta": 0.00,
}

def f_x_sys_id_dynamics(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Dynamics model identified from data collected on the real drone.

    From sys_id_dynamics() in lsy_drone_racing/sim/physics.py

    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). 
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    x_dot = np.empty_like(x)
    x_dot[:, 12:] = 0

    for i in range(x.shape[0]):
        pos = x[i, 0:3]
        rpy = x[i, 3:6]
        vel = x[i, 6:9]
        thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd = u[i,:]
        rot = R.from_euler("xyz", rpy)
        thrust = rot.apply(np.array([0, 0, thrust_cmd]))
        drift = rot.apply(np.array([0, 0, 1]))
        acc = thrust * SYS_ID_PARAMS.acc_k1 + drift * SYS_ID_PARAMS.acc_k2 - np.array([0, 0, GRAVITY])
        roll_rate = SYS_ID_PARAMS.roll_alpha * rpy[0] + SYS_ID_PARAMS.roll_beta * roll_cmd
        pitch_rate = SYS_ID_PARAMS.pitch_alpha * rpy[1] + SYS_ID_PARAMS.pitch_beta * pitch_cmd
        yaw_rate = SYS_ID_PARAMS.yaw_alpha * rpy[2] + SYS_ID_PARAMS.yaw_beta * yaw_cmd
        rpy_rates = np.array([roll_rate, pitch_rate, yaw_rate])
        x_dot[i, 0:3] = vel
        x_dot[i, 3:6] = rpy_rates
        x_dot[i, 6:9] = acc
        x_dot[i, 9:12] = rot.apply(rpy_rates)/dt # model doesn't really have the derivative, so it's calculated manually


    return x_dot

def f_x_identified(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Dynamics model identified from data collected on the real drone.

    From https://github.com/utiasDSL/crazyflow/blob/main/crazyflow/sim/physics.py#L31

    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). 
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    x_dot = np.empty_like(x)
    x_dot[:, 12:] = 0

    for i in range(x.shape[0]):
        rpy = x[i, 3:6]
        vel = x[i, 6:9]
        collective_thrust, attitude = u[i, 0], u[i, 1:]
        rot = R.from_euler("xyz", rpy)

        thrust = rot.apply(np.array([0, 0, collective_thrust]))
        drift = rot.apply(np.array([0, 0, 1]))
        a1, a2 = SYS_ID_PARAMS["acc_k1"], SYS_ID_PARAMS["acc_k2"]
        acc = thrust * a1 + drift * a2 - np.array([0, 0, GRAVITY])
        roll_cmd, pitch_cmd, yaw_cmd = attitude
        rpy = rot.as_euler("xyz")
        roll_rate = SYS_ID_PARAMS["roll_alpha"] * rpy[0] + SYS_ID_PARAMS["roll_beta"] * roll_cmd
        pitch_rate = SYS_ID_PARAMS["pitch_alpha"] * rpy[1] + SYS_ID_PARAMS["pitch_beta"] * pitch_cmd
        yaw_rate = SYS_ID_PARAMS["yaw_alpha"] * rpy[2] + SYS_ID_PARAMS["yaw_beta"] * yaw_cmd
        rpy_rates = np.array([roll_rate, pitch_rate, yaw_rate])
        x_dot[i, 0:3] = vel
        x_dot[i, 3:6] = rpy_rates
        x_dot[i, 6:9] = acc
        x_dot[i, 9:12] = rot.apply(rpy_rates)/dt # model doesn't really have the derivative, so it's calculated manually

    return x_dot

def attitude_controller(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Simulates the attitude controller of the Mellinger controller.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). [Thrust, R, P, Y]
        dt: time step size

    Return:
        Commanded RPMs
    """
    # From firmware controller_mellinger 
    # l. 52
    mass_thrust = 132000
    # l. 66-79
    # Attitude
    kR_xy = 70000 # P
    kw_xy = 20000 # D
    ki_m_xy = 0.0 # I
    i_range_m_xy = 1.0

    # Yaw
    kR_z = 60000 # P
    kw_z = 12000 # D
    ki_m_z = 500 # I
    i_range_m_z  = 1500

    # roll and pitch angular velocity
    kd_omega_rp = 200 # D
    
    ### Calculate RPMs for first x (mean of UKF estimate) & u to keep RPM command constant 
    rpy = x[0, 3:6]
    angular_vel = x[0, 9:12]
    thrust_des = u[0, 0]
    rpy_des = u[0, 1:]

    ### From firmware controller_mellinger:
    # l. 220 ff
    rot = R.from_euler("xyz", rpy)
    rot_des = R.from_euler("xyz", rpy_des)
    R_act = rot.as_matrix()
    R_des = rot_des.as_matrix()
    eR = 0.5 * ( R_des.T@R_act - R_act.T@R_des )
    eR = np.array([eR[2,1], -eR[0,2], eR[1,0]]) # vee operator (SO3 to R3), the -y is to account for the frame of the crazyflie
    # l.256 ff
    angular_vel_des = np.zeros(3) # for now assuming angular_vel_des = 0 (would need to be given as input)
    ew = angular_vel - angular_vel_des
    ew[1] = -ew[1]
    # l. 259 ff
    prev_angular_vel_des = np.zeros(3) # zero for now (would need to be stored)
    prev_angular_vel = np.zeros(3) # for now zeros, would need to be stored outside
    err_d = ( (angular_vel_des - prev_angular_vel_des) - (angular_vel - prev_angular_vel) )/dt *0 # set to zeros cause not used
    err_d[1] = -err_d[1]
    # l. 268 ff
    i_error_m = np.zeros(3) # this part is usually longer, but we say the error is zeros for now
    # l. 279 ff
    Mx = -kR_xy * eR[0] + kw_xy * ew[0] + ki_m_xy * i_error_m[0] + kd_omega_rp * err_d[0]
    My = -kR_xy * eR[1] + kw_xy * ew[1] + ki_m_xy * i_error_m[1] + kd_omega_rp * err_d[1]
    Mz = -kR_z  * eR[2] + kw_z  * ew[2] + ki_m_z  * i_error_m[2]
    # l. 297 ff
    if thrust_des > 0:
        cmd_roll = np.clip(Mx, -32000, 32000)
        cmd_pitch = np.clip(My, -32000, 32000)
        cmd_yaw = np.clip(-Mz, -32000, 32000)
    else:
        cmd_roll = 0
        cmd_pitch = 0
        cmd_yaw = 0
    # From firmware powerDistributionLegacy() 
    # cmd_PWM = thrust2pwm(thrust_des/4) # /4 because we got total thrust, but function takes single thrust
    cmd_PWM = thrust_to_pwm(thrust_des)
    # cmd_PWM = mass_thrust*thrust_des
    cmd_roll = cmd_roll/2
    cmd_pitch = cmd_pitch/2
    cmd_PWM = cmd_PWM + np.array([-cmd_roll + cmd_pitch + cmd_yaw,
                                -cmd_roll - cmd_pitch - cmd_yaw,
                                +cmd_roll - cmd_pitch + cmd_yaw,
                                +cmd_roll + cmd_pitch - cmd_yaw])
    cmd_PWM = np.clip(cmd_PWM, 20000, 65535)

    # From firmware motors.c motorsSetRatio()
    # l. 236 ff
    cmd_PWM = batComp(cmd_PWM)

    # cmd_RPMs = pwm2rpm(cmd_PWM)
    cmd_RPMs = thrust_to_rpm((pwms_to_thrust(cmd_PWM)))

    ### Stack RPMs to fit
    u_RPM = np.zeros_like(u)
    u_RPM += cmd_RPMs
    return u_RPM, cmd_PWM