"""This module provides the Drone class, which represents a Crazyflie 2.1 quadrotor drone.

The Drone class serves as an abstraction layer for the pycffirmware module, implementing
controller logic for both Mellinger and PID controllers. It offers a convenient interface
for managing drone parameters and constants, which are read from a URDF file. The class
also maintains the state of the drone during simulation.

The Drone class is a core component of the simulation environment and is heavily utilized
in the sim.py module. It handles various aspects of drone behavior, including:

* Initialization of drone parameters and firmware
* Management of drone state and control inputs
* Implementation of controller logic
* Conversion between different units (e.g., thrust to RPM)
* Handling of sensor data and low-pass filtering

This module also includes the DroneParams dataclass, which encapsulates the physical
and inferred parameters of the Crazyflie 2.1 drone.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar
from xml.etree import ElementTree

import numpy as np
import pycffirmware
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

control_t = TypeVar("control_t")

logger = logging.getLogger(__name__)


class Drone:
    """Drone abstraction for the Crazyflie 2.1.

    The drone class is a wrapper around the pycffirmware module. It implements the controller logic
    for the mellinger and pid controllers. In addition, it provides a convenient interface for the
    parameters and constants of the Crazyflie 2.1 read from the URDF file, and a place to store the
    state of the drone during simulation.
    """

    def __init__(self, controller: Literal["pid", "mellinger"]):
        """Initialize the drone.

        Args:
            controller: The controller to use. Either "pid" or "mellinger".
        """
        self.params = DroneParams.from_urdf(Path(__file__).resolve().parent / "assets/cf2x.urdf")
        self.firmware_freq = self.params.firmware_freq
        self.nominal_params = copy.deepcopy(self.params)  # Store parameters without disturbances
        # Initialize firmware states
        self._state = pycffirmware.state_t()
        self._control = pycffirmware.control_t()
        self._setpoint = pycffirmware.setpoint_t()
        self._sensor_data = pycffirmware.sensorData_t()
        self._acc_lpf = [pycffirmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [pycffirmware.lpf2pData() for _ in range(3)]

        assert controller in ["pid", "mellinger"], f"Invalid controller {controller}."
        self._controller = controller
        # Helper variables for the controller
        self.desired_thrust = np.zeros(4)  # Desired thrust for each motor
        self._pwms = np.zeros(4)  # PWM signals for each motor
        self.rpm = np.zeros(4)  # RPM for each motor
        self._tick = 0  # Current controller tick
        self._n_tumble = 0  # Number of consecutive steps the drone is tumbling
        self._last_att_ctrl_call = 0  # Last time attitude controller was called
        self._last_pos_ctrl_call = 0  # Last time position controller was called
        self._last_vel = np.zeros(3)
        self._last_rpy = np.zeros(3)
        self._fullstate_cmd = True  # Disables high level commander if True

        self.init_pos, self.init_rpy = np.zeros(3), np.zeros(3)
        self.init_vel, self.init_ang_vel = np.zeros(3), np.zeros(3)
        self.pos = np.zeros(3)
        """Current position of the drone in the world frame."""
        self.rpy = np.zeros(3)
        """Current roll, pitch, yaw of the drone."""
        self.vel = np.zeros(3)
        """Current velocity of the drone in the world frame."""
        self.ang_vel = np.zeros(3)
        """Current angular velocity of the drone in the world frame."""
        self.id = -1

    def reset(
        self,
        pos: NDArray[np.floating] | None = None,
        rpy: NDArray[np.floating] | None = None,
        vel: NDArray[np.floating] | None = None,
    ):
        """Reset the drone state and controllers.

        Args:
            pos: Initial position of the drone. If None, uses the values from `init_state`. Shape:
                (3,).
            rpy: Initial roll, pitch, yaw of the drone. If None, uses the values from `init_state`.
                Shape: (3,).
            vel: Initial velocity of the drone. If None, uses the values from `init_state`. Shape:
                (3,).
        """
        self.params = copy.deepcopy(self.nominal_params)
        self._reset_firmware_states()
        self._reset_low_pass_filters()
        self._reset_helper_variables()
        self._reset_controller()
        # Initilaize high level commander
        pycffirmware.crtpCommanderHighLevelInit()
        pos = self.init_pos if pos is None else pos.copy()
        rpy = self.init_rpy if rpy is None else rpy.copy()
        vel = self.init_vel if vel is None else vel.copy()
        self._update_state(0, pos, np.rad2deg(rpy), vel, np.array([0, 0, 1.0]))
        self._last_vel[...], self._last_rpy[...] = vel, rpy
        pycffirmware.crtpCommanderHighLevelTellState(self._state)

    def step_controller(
        self, pos: NDArray[np.floating], rpy: NDArray[np.floating], vel: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Take a drone controller step.

        Each step, we 1.) update the drone state, 2.) update the sensor data, 3.) update the
        setpoint, and then 4.) step the controller.

        Args:
            pos: Current position of the drone. Shape: (3,).
            rpy: Current roll, pitch, yaw of the drone. Shape: (3,).
            vel: Current velocity of the drone. Shape: (3,).
        """
        pos, rpy, vel = pos.copy(), rpy.copy(), vel.copy()
        acc = (vel - self._last_vel) * self.params.firmware_freq / 9.81 + np.array([0.0, 0.0, 1.0])
        self._last_vel = vel
        # Update state
        timestamp = int(self._tick / self.params.firmware_freq * 1e3)
        self._update_state(timestamp, pos, np.rad2deg(rpy), vel, acc)
        # Update sensor data
        sensor_timestamp = int(self._tick / self.params.firmware_freq * 1e6)
        body_acc = R.from_euler("xyz", rpy).apply(acc, inverse=True)
        body_ang_vel = R.from_euler("xyz", rpy).apply(self.ang_vel, inverse=True)
        self._update_sensor_data(sensor_timestamp, body_acc, np.rad2deg(body_ang_vel))
        # Update setpoint
        self._update_setpoint(self._tick / self.params.firmware_freq)
        # Step controller
        success = self._step_controller()
        self._tick += 1
        if not success:
            self._pwms[...] = 0
            return np.zeros(4)
        return self._pwms_to_thrust(self._pwms)

    @property
    def tick(self) -> int:
        """Current controller tick.

        Each tick represents a step in the controller loop. Ticks are set to 0 after a reset.
        """
        return self._tick

    def _update_state(
        self,
        timestamp: float,
        pos: NDArray[np.floating],
        rpy: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
    ):
        self._state.timestamp = timestamp
        # Legacy cf coordinate system uses inverted pitch
        self._state.roll, self._state.pitch, self._state.yaw = rpy * np.array([1, -1, 1])
        if self._controller == "mellinger":  # Requires quaternion
            quat = R.from_euler("xyz", rpy, degrees=True).as_quat()
            quat_state = self._state.attitudeQuaternion
            quat_state.x, quat_state.y, quat_state.z, quat_state.w = quat
        self._state.position.x, self._state.position.y, self._state.position.z = pos
        self._state.velocity.x, self._state.velocity.y, self._state.velocity.z = vel
        self._state.acc.x, self._state.acc.y, self._state.acc.z = acc

    def full_state_cmd(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
        yaw: float,
        rpy_rate: NDArray[np.floating],
    ):
        """Send a full state command to the controller.

        Notes:
            Overrides any high level commands being processed.

        Args:
            pos: [x, y, z] position of the CF (m)
            vel: [x, y, z] velocity of the CF (m/s)
            acc: [x, y, z] acceleration of the CF (m/s^2)
            yaw: yaw of the CF (rad)
            rpy_rate: roll, pitch, yaw rates (rad/s)
        """
        timestep = self._tick / self.params.firmware_freq
        pycffirmware.crtpCommanderHighLevelStop()  # Resets planner object
        pycffirmware.crtpCommanderHighLevelUpdateTime(timestep)

        for name, x in zip(("pos", "vel", "acc", "rpy_rate"), (pos, vel, acc, rpy_rate)):
            assert isinstance(x, np.ndarray), f"{name} must be a numpy array."
            assert len(x) == 3, f"{name} must have length 3."
        self._setpoint.position.x, self._setpoint.position.y, self._setpoint.position.z = pos
        self._setpoint.velocity.x, self._setpoint.velocity.y, self._setpoint.velocity.z = vel
        s_acc = self._setpoint.acceleration
        s_acc.x, s_acc.y, s_acc.z = acc
        s_a_rate = self._setpoint.attitudeRate
        s_a_rate.roll, s_a_rate.pitch, s_a_rate.yaw = np.rad2deg(rpy_rate)
        s_quat = self._setpoint.attitudeQuaternion
        s_quat.x, s_quat.y, s_quat.z, s_quat.w = R.from_euler("xyz", [0, 0, yaw]).as_quat()
        # initilize setpoint modes to match cmdFullState
        mode = self._setpoint.mode
        mode_abs, mode_disable = pycffirmware.modeAbs, pycffirmware.modeDisable
        mode.x, mode.y, mode.z = mode_abs, mode_abs, mode_abs
        mode.quat = mode_abs
        mode.roll, mode.pitch, mode.yaw = mode_disable, mode_disable, mode_disable
        # This may end up skipping control loops
        self._setpoint.timestamp = int(timestep * 1000)
        self._fullstate_cmd = True

    def collective_thrust_cmd(self, thrust: float, rpy: NDArray[np.floating]):
        """Send a collective thrust command to the controller.

        Notes:
            Overrides any high level commands being processed.

        Args:
            thrust: Thrust of rotors in Newtons.
            rpy: [roll, pitch, yaw] orientation of the Crazyflie in radians.
        """
        # All choices were made with respect to src/mod/src/crtp_commander_rpyt.c in the firmware.
        pwms = self._thrust_to_pwms(thrust)
        timestep = self._tick / self.params.firmware_freq
        pycffirmware.crtpCommanderHighLevelStop()  # Resets planner object
        pycffirmware.crtpCommanderHighLevelUpdateTime(timestep)

        # Set Setpoints to be passed to the firmware.
        self._setpoint.position.x, self._setpoint.position.y, self._setpoint.position.z = np.zeros(
            3, dtype=np.float64
        )
        s_vel = self._setpoint.velocity
        s_vel.x, s_vel.y, s_vel.z = np.array(
            [0.0, 0.0, (pwms - 32767.0) / 32767.0], dtype=np.float64
        )

        self._setpoint.thrust = pwms

        s_acc = self._setpoint.acceleration
        s_acc.x, s_acc.y, s_acc.z = np.zeros(3, dtype=np.float64)

        s_a_rate = self._setpoint.attitudeRate
        s_a_rate.roll, s_a_rate.pitch, s_a_rate.yaw = np.zeros(3, dtype=np.float64)

        s_a = self._setpoint.attitude
        s_a.roll, s_a.pitch, s_a.yaw = np.rad2deg(rpy)

        s_quat = self._setpoint.attitudeQuaternion
        s_quat.x, s_quat.y, s_quat.z, s_quat.w = R.from_euler("xyz", rpy).as_quat()

        # initilize setpoint modes to match thrust interface.
        mode = self._setpoint.mode
        mode_abs, mode_disable, mode_velocity = (
            pycffirmware.modeAbs,
            pycffirmware.modeDisable,
            pycffirmware.modeVelocity,
        )
        mode.x, mode.y, mode.z = mode_disable, mode_disable, mode_velocity
        mode.quat = mode_disable
        mode.roll, mode.pitch, mode.yaw = mode_abs, mode_abs, mode_abs

        # This may end up skipping control loops
        self._setpoint.timestamp = int(timestep * 1000)
        self._fullstate_cmd = True

    def takeoff_cmd(self, height: float, duration: float, yaw: float | None = None) -> None:
        """Send a takeoff command to the controller.

        Args:
            height: Target takeoff height (m)
            duration: Length of manuever (s)
            yaw: Target yaw (rad). If None, yaw is not set.
        """
        self._fullstate_cmd = False
        if yaw is None:
            return pycffirmware.crtpCommanderHighLevelTakeoff(height, duration)
        pycffirmware.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)

    def takeoff_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a takeoff vel command to the controller.

        Args:
            height: Target takeoff height (m)
            vel: Target takeoff velocity (m/s)
            relative: Flag if takeoff height is relative to CF's current position
        """
        pycffirmware.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self._fullstate_cmd = False

    def land_cmd(self, height: float, duration: float, yaw: float | None = None) -> None:
        """Send a land command to the controller.

        Args:
            height: Target landing height (m)
            duration:: Length of manuever (s)
            yaw: Target yaw (rad). If None, yaw is not set.
        """
        self._fullstate_cmd = False
        if yaw is None:
            return pycffirmware.crtpCommanderHighLevelLand(height, duration)
        pycffirmware.crtpCommanderHighLevelLandYaw(height, duration, yaw)

    def land_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a land vel command to the controller.

        Args:
            height: Target landing height (m)
            vel: Target landing velocity (m/s)
            relative: Flag if landing height is relative to CF's current position
        """
        pycffirmware.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self._fullstate_cmd = False

    def stop_cmd(self):
        """Send a stop command to the controller."""
        pycffirmware.crtpCommanderHighLevelStop()
        self._fullstate_cmd = False

    def go_to_cmd(self, pos: NDArray[np.floating], yaw: float, duration: float, relative: bool):
        """Send a go to command to the controller.

        Args:
            pos: [x, y, z] target position (m)
            yaw: Target yaw (rad)
            duration: Length of manuever
            relative: Flag if setpoint is relative to CF's current position
        """
        pycffirmware.crtpCommanderHighLevelGoTo(*pos, yaw, duration, relative)
        self._fullstate_cmd = False

    def notify_setpoint_stop(self):
        """Send a notify setpoint stop cmd to the controller."""
        pycffirmware.crtpCommanderHighLevelTellState(self._state)
        self._fullstate_cmd = False

    def _reset_firmware_states(self):
        self._state = pycffirmware.state_t()
        self._control = pycffirmware.control_t()
        self._setpoint = pycffirmware.setpoint_t()
        self._sensor_data = pycffirmware.sensorData_t()
        self._tick = 0
        self._pwms[...] = 0

    def _reset_low_pass_filters(self):
        freq = self.params.firmware_freq
        self._acc_lpf = [pycffirmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [pycffirmware.lpf2pData() for _ in range(3)]
        for i in range(3):
            pycffirmware.lpf2pInit(self._acc_lpf[i], freq, self.params.acc_lpf_cutoff)
            pycffirmware.lpf2pInit(self._gyro_lpf[i], freq, self.params.gyro_lpf_cutoff)

    def _reset_helper_variables(self):
        self._n_tumble = 0
        self._last_att_ctrl_call = 0
        self._last_pos_ctrl_call = 0
        self._last_vel = np.zeros(3)
        self._last_rpy = np.zeros(3)
        self.rpm[...] = 0
        self.desired_thrust[...] = 0

    def _reset_controller(self):
        if self._controller == "pid":
            pycffirmware.controllerPidInit()
        else:
            pycffirmware.controllerMellingerInit()

    def _step_controller(self) -> bool:
        # Check if the drone is tumblig. If yes, set the control signal to zero.
        if self._state.acc.z < self.params.tumble_threshold:
            self._n_tumble += 1
        else:
            self._n_tumble = 0
        if self._n_tumble > self.params.tumble_duration:
            logger.debug("CrazyFlie is tumbling. Killing motors to simulate damage prevention.")
            self._pwms[...] = 0
            return False  # Skip controller step
        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was
        # designed for
        tick = self._determine_controller_tick()
        if self._controller == "pid":
            ctrl = pycffirmware.controllerPid
        else:
            ctrl = pycffirmware.controllerMellinger
        ctrl(self._control, self._setpoint, self._sensor_data, self._state, tick)
        self._update_pwms(self._control)
        return True

    def _determine_controller_tick(self) -> Literal[0, 1, 2]:
        """Determine which controller to run based on time passed.

        This allows us to run the PID controller slower than the 1000Hz it was designed for.

        Returns:
            0: Run position and attitude controller.
            1: Run neither controller.
            2: Run only attitude controller.
        """
        time = self._tick / self.params.firmware_freq
        if time - self._last_att_ctrl_call > 0.002 and time - self._last_pos_ctrl_call > 0.01:
            self._last_att_ctrl_call = time
            self._last_pos_ctrl_call = time
            return 0
        if time - self._last_att_ctrl_call > 0.002:
            self._last_att_ctrl_call = time
            return 2
        return 1

    def _update_pwms(self, control: control_t):
        """Update the motor PWMs based on the control input.

        Args:
            control: Control signal.
        """
        r = control.roll / 2
        p = control.pitch / 2
        y = control.yaw
        pwm = control.thrust
        pwms = np.array([pwm - r + p + y, pwm - r - p - y, pwm + r - p + y, pwm + r + p - y])
        np.clip(pwms, 0, self.params.max_pwm, out=pwms)  # Limit pwms to motor range
        pwms = self._scale_pwms(pwms)
        np.clip(pwms, self.params.min_pwm, self.params.max_pwm, out=pwms)
        self._pwms = pwms

    def _scale_pwms(self, pwms: NDArray[np.floating]) -> NDArray[np.floating]:
        """Scale the control PWM signal to the actual PWM signal.

        Assumes brushed motors.

        Args:
            pwms: An array of PWM values.

        Returns:
            The properly scaled PWM signal.
        """
        pwms = pwms / self.params.max_pwm * 60
        volts = self.params.pwm_curve_a * pwms**2 + self.params.pwm_curve_b * pwms
        percentage = np.minimum(1, volts / self.params.supply_voltage)
        return percentage * self.params.max_pwm

    def _pwms_to_thrust(self, pwms: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.params.kf * (self.params.pwm2rpm_scale * pwms + self.params.pwm2rpm_const) ** 2

    def _thrust_to_pwms(self, thrust: float) -> NDArray[np.floating]:
        """Convert thrust to pwm using a quadratic thrust curve.

        Returns:
            The desired pwms in the range of [0.0, pwm_max].
        """
        pwm = (
            self.params.thrust_curve_a * thrust * thrust
            + self.params.thrust_curve_b * thrust
            + self.params.thrust_curve_c
        )
        return np.clip(pwm, 0, 1) * self.params.max_pwm

    def _update_sensor_data(
        self, timestamp: float, acc: NDArray[np.floating], gyro: NDArray[np.floating]
    ):
        """Update the onboard sensors with low-pass filtered values.

        Args:
            timestamp: Sensor reading time in microseconds.
            acc: Acceleration values in Gs.
            gyro: Gyro values in deg/s.
        """
        for name, i, val in zip(("x", "y", "z"), range(3), acc):
            setattr(self._sensor_data.acc, name, pycffirmware.lpf2pApply(self._acc_lpf[i], val))
        for name, i, val in zip(("x", "y", "z"), range(3), gyro):
            setattr(self._sensor_data.gyro, name, pycffirmware.lpf2pApply(self._gyro_lpf[i], val))
        self._sensor_data.interruptTimestamp = timestamp

    def _update_setpoint(self, timestep: float):
        if not self._fullstate_cmd:
            pycffirmware.crtpCommanderHighLevelTellState(self._state)
            pycffirmware.crtpCommanderHighLevelUpdateTime(timestep)
            pycffirmware.crtpCommanderHighLevelGetSetpoint(self._setpoint, self._state)


@dataclass
class DroneParams:
    """A collection of physical and inferred parameters of the Crazyflie 2.1.

    The preferred way to create a `DroneParams` object is to read the parameters from the
    corresponding drone URDF.
    """

    mass: float
    arm_len: float
    prop_radius: float
    collision_r: float
    collision_h: float
    collision_z_offset: float

    J: NDArray[np.floating]

    max_speed_kmh: float
    thrust2weight_ratio: float

    gnd_eff_coeff: float
    drag_coeff: NDArray[np.floating]
    dw_coeff_1: float
    dw_coeff_2: float
    dw_coeff_3: float

    kf: float
    km: float

    pwm2rpm_scale: float
    pwm2rpm_const: float
    min_pwm: float
    max_pwm: float

    # Defaults are calculated in __post_init__ according to the other parameters
    J_inv: NDArray[np.floating] = field(default_factory=lambda: np.zeros((3, 3)))

    gnd_eff_min_height_clip: float = 0.0
    firmware_freq: int = 500  # Firmware frequency in Hz
    supply_voltage: float = 3.0  # Power supply voltage

    min_rpm: float = 0.0
    max_rpm: float = 0.0
    pwm_curve_a: float = -0.0006239  # PWM curve parameters for brushed motors
    pwm_curve_b: float = 0.088  # PWM curve parameters for brushed motors

    min_thrust: float = 0.0
    max_thrust: float = 0.0
    thrust_curve_a: float = -1.1264
    thrust_curve_b: float = 2.2541
    thrust_curve_c: float = 0.0209  # Thrust curve parameters for brushed motors

    tumble_threshold: float = -0.5  # Vertical acceleration threshold for tumbling detection
    tumble_duration: int = 30  # Number of consecutive steps before tumbling is detected

    acc_lpf_cutoff: int = 80  # Low-pass filter cutoff freq
    gyro_lpf_cutoff: int = 30  # Low-pass filter cutoff freq

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.J_inv = np.linalg.inv(self.J)
        self.min_rpm = self.pwm2rpm_scale * self.min_pwm + self.pwm2rpm_const
        self.max_rpm = self.pwm2rpm_scale * self.max_pwm + self.pwm2rpm_const
        self.min_thrust = self.kf * self.min_rpm**2
        self.max_thrust = self.kf * self.max_rpm**2
        self.gnd_eff_min_height_clip = (
            0.25
            * self.prop_radius
            * np.sqrt((15 * self.max_rpm**2 * self.kf * self.gnd_eff_coeff) / self.max_thrust)
        )

    @staticmethod
    def from_urdf(path: Path) -> DroneParams:
        """Load the drone parameters from the URDF file in `assets/` with a custom XML parser."""
        urdf = ElementTree.parse(path).getroot()
        params = DroneParams(
            mass=float(urdf[1][0][1].attrib["value"]),
            arm_len=float(urdf[0].attrib["arm"]),
            thrust2weight_ratio=float(urdf[0].attrib["thrust2weight"]),
            J=np.diag([float(urdf[1][0][2].attrib[c]) for c in ("ixx", "iyy", "izz")]),
            kf=float(urdf[0].attrib["kf"]),
            km=float(urdf[0].attrib["km"]),
            collision_h=float(urdf[1][2][1][0].attrib["length"]),
            collision_r=float(urdf[1][2][1][0].attrib["radius"]),
            collision_z_offset=[float(s) for s in urdf[1][2][0].attrib["xyz"].split(" ")][2],
            max_speed_kmh=float(urdf[0].attrib["max_speed_kmh"]),
            gnd_eff_coeff=float(urdf[0].attrib["gnd_eff_coeff"]),
            prop_radius=float(urdf[0].attrib["prop_radius"]),
            drag_coeff=np.array(
                [float(urdf[0].attrib["drag_coeff_" + c]) for c in ("xy", "xy", "z")]
            ),
            dw_coeff_1=float(urdf[0].attrib["dw_coeff_1"]),
            dw_coeff_2=float(urdf[0].attrib["dw_coeff_2"]),
            dw_coeff_3=float(urdf[0].attrib["dw_coeff_3"]),
            pwm2rpm_scale=float(urdf[0].attrib["pwm2rpm_scale"]),
            pwm2rpm_const=float(urdf[0].attrib["pwm2rpm_const"]),
            min_pwm=float(urdf[0].attrib["pwm_min"]),
            max_pwm=float(urdf[0].attrib["pwm_max"]),
        )
        return params
