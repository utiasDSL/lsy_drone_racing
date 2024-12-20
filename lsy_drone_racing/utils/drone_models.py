"""All available drone models are in this file.

The model need to be listed in the dictionary in the beginning of this file.

Usage:
- Import the dictionary
- Use whatever entry you need
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from types import FunctionType

    from numpy.typing import NDArray

# models = {}

# @property
def models(model_name: str) -> tuple[FunctionType, bool, int, int]:
    """The models dictionary.
    
    Contains tupels with (function_pointer, is_continuous, dim_x, dim_u)
    """
    dict = {
        "analytical": (f_analytical, True, 18, 4), # inputs: [RPM, RPM, RPM, PRM]
        "analytical_mel_att": (f_analytical_mel_att, True, 18, 4), # inputs: [Thrust [N], roll [rad], pitch [rad], yaw [rad]]
    }
    return dict[model_name]

# Constants. TODO need to be set from outside somehow
# The following constants can be taken from the drone object or directly from the .urdf file in sim/assets
# For now, we simply hard code them, assuming the same drone is always used. Should be done properly in the future
GRAVITY = 9.81
g: NDArray[np.floating] = np.array([0,0,-GRAVITY])
MASS: float = 0.03454 
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
J_inv = np.linalg.inv(J)
L = 0.046
KF: float = 3.16e-10
KM: float = 7.94e-12
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM: float = 20000
MAX_PWM: float = 65535
MIN_RPM: float = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
MAX_RPM: float = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST
MIN_THRUST: float = KF * MIN_RPM**2
MAX_THRUST: float = KF * MAX_RPM**2
THRUST_CURVE_A: float = -1.1264
THRUST_CURVE_B: float = 2.2541
THRUST_CURVE_C: float = 0.0209  # Thrust curve parameters for brushed motors

rpy_err_i = 0 # This way is ghetto -> put attitude2rpm into class
last_rpy = np.array([[0,0,0]])
last_rpy_rates = np.array([[0,0,0]])


def f_analytical(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating | None = None) -> NDArray[np.floating]:
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
    f_dis = x[:, 12:15]
    t_dis = x[:, 15:18]

    # Compute forces and torques.
    forces = u**2 * KF
    thrust = np.zeros((u.shape[0], 3))
    thrust[:, -1] = np.sum(forces, axis=1)
    thrust_world_frame = rot.apply(thrust)
    force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY * MASS]) + f_dis
    x_torque = (forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]) * L
    y_torque = (-forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]) * L
    z_torques = u**2 * KM
    z_torque = z_torques[:, 0] - z_torques[:, 1] + z_torques[:, 2] - z_torques[:, 3]
    torques = np.zeros((u.shape[0], 3))
    torques[:, 0] = x_torque
    torques[:, 1] = y_torque
    torques[:, 2] = z_torque
    torques += t_dis
    torques -= np.cross(rpy_rates, rpy_rates @ J.T) # usually transposed, but skipped since J is symmetric
    rpy_rates_deriv = torques @ J_inv.T # usually transposed, but skipped since J_inv is symmetric

    # Set Derivatives
    x_dot[:, 0:3] = x[:, 6:9] # velocity
    x_dot[:, 3:6] = rpy_rates
    x_dot[:, 6:9] = force_world_frame / MASS # acceleration
    x_dot[:, 9:12] = rot.apply(rpy_rates_deriv)

    return x_dot

def f_analytical_mel_att(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
    """Example dynamics of the drone.
    
    Extension of the above analytical model to attitude inputs for the Mellinger controller.
    
    Args:
        x: State of the system, batched with shape (batches, states)
        u: input of the system, (batches, inputs). [Thrust, R, P, Y]
        dt: time step size

    Return:
        The derivative of x for given x and u
    """
    u_RPM, _ = mellinger_ctrl_att(x, u, dt)
    return f_analytical(x, u_RPM, dt)

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
    return f_analytical(x, rpms, dt)

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
    thrust = np.clip(thrust, MIN_THRUST, MAX_THRUST)  # Protect against NaN values
    return np.clip((np.sqrt(thrust / KF) - PWM2RPM_CONST) / PWM2RPM_SCALE, MIN_PWM, MAX_PWM)

def pwms_to_thrust(pwms: NDArray[np.floating]) -> NDArray[np.floating]:
    return KF * (PWM2RPM_SCALE * pwms + PWM2RPM_CONST) ** 2

def thrust_to_pwm(thrust):
    # from drone.py
    pwm = THRUST_CURVE_A * thrust * thrust + THRUST_CURVE_B * thrust + THRUST_CURVE_C
    return np.clip(pwm, 0, 1) * MAX_PWM

def thrust_to_rpm(thrust: NDArray[np.floating]) -> NDArray[np.floating]:
    # from drone.py
    thrust = np.clip(thrust, MIN_THRUST, MAX_THRUST)

    pwm = (np.sqrt(thrust / KF) - PWM2RPM_CONST) / PWM2RPM_SCALE
    pwm = np.clip(pwm, MIN_PWM, MAX_PWM)
    return PWM2RPM_SCALE * pwm + PWM2RPM_CONST


def pwm2rpm(pwm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert the motors' PWMs into RPMs."""
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    return PWM2RPM_CONST + PWM2RPM_SCALE * pwm

def batComp(pwm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compensate the battery voltage."""
    supply_voltage = 3.0 # assumed supply voltage. Taken from drone.py
    pwm = pwm/MAX_PWM*60
    pwm_curve_a = -0.0006239  # PWM curve parameters for brushed motors
    pwm_curve_b = 0.088  # PWM curve parameters for brushed motors
    voltage = pwm_curve_a * pwm**2 + pwm_curve_b * pwm
    percentage = np.minimum(1, voltage/supply_voltage)
    return percentage * MAX_PWM

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

def mellinger_ctrl_att(x: NDArray[np.floating], u: NDArray[np.floating], dt: np.floating) -> NDArray[np.floating]:
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