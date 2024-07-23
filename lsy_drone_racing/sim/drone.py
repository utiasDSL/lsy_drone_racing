from __future__ import annotations

import copy
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from types import ModuleType

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
        self.firmware = self._load_firmware()
        self.firmware_freq = self.params.firmware_freq
        self.nominal_params = copy.deepcopy(self.params)  # Store parameters without disturbances
        # Initialize firmware states
        self._state = self.firmware.state_t()
        self._control = self.firmware.control_t()
        self._setpoint = self.firmware.setpoint_t()
        self._sensor_data = self.firmware.sensorData_t()
        self._acc_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [self.firmware.lpf2pData() for _ in range(3)]

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
        self.pos, self.rpy = np.zeros(3), np.zeros(3)
        self.vel, self.ang_vel = np.zeros(3), np.zeros(3)
        self.id = -1

    def reset(
        self,
        pos: npt.NDArray[np.float64] | None = None,
        rpy: npt.NDArray[np.float64] | None = None,
        vel: npt.NDArray[np.float64] | None = None,
    ):
        """Reset the drone state and controllers.

        Args:
            pos: Initial position of the drone. If None, uses the values from `init_state`.
            rpy: Initial roll, pitch, yaw of the drone. If None, uses the values from `init_state`.
            vel: Initial velocity of the drone. If None, uses the values from `init_state`.
        """
        self.params = copy.deepcopy(self.nominal_params)
        self._reset_firmware_states()
        self._reset_low_pass_filters()
        self._reset_helper_variables()
        self._reset_controller()
        # Initilaize high level commander
        self.firmware.crtpCommanderHighLevelInit()
        pos = self.init_pos if pos is None else pos
        rpy = self.init_rpy if rpy is None else rpy
        vel = self.init_vel if vel is None else vel
        self._update_state(0, pos, np.rad2deg(rpy), vel, np.array([0, 0, 1.0]))
        self._last_vel[...], self._last_rpy[...] = vel, rpy
        self.firmware.crtpCommanderHighLevelTellState(self._state)

    def step_controller(
        self,
        pos: npt.NDArray[np.float64],
        rpy: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Take a drone controller step.

        Args:
            pos: Current position of the drone.
            rpy: Current roll, pitch, yaw of the drone.
            vel: Current velocity of the drone.
        """
        body_rot = R.from_euler("XYZ", rpy).inv()
        # Estimate rates
        rotation_rates = (rpy - self._last_rpy) * self.params.firmware_freq  # body coord, rad/s
        self._last_rpy = rpy
        acc = (vel - self._last_vel) * self.params.firmware_freq / 9.81 + np.array([0, 0, 1])
        self._last_vel = vel
        # Update state
        timestamp = int(self._tick / self.params.firmware_freq * 1e3)
        self._update_state(timestamp, pos, np.rad2deg(rpy), vel, acc)
        # Update sensor data
        sensor_timestamp = int(self._tick / self.params.firmware_freq * 1e6)
        self._update_sensor_data(sensor_timestamp, body_rot.apply(acc), np.rad2deg(rotation_rates))
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
        pos: npt.NDArray[np.float64],
        rpy: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        acc: npt.NDArray[np.float64],
    ):
        self._state.timestamp = timestamp
        # Legacy cf coordinate system uses inverted pitch
        self._state.roll, self._state.pitch, self._state.yaw = rpy * np.array([1, -1, 1])
        if self._controller == "mellinger":  # Requires quaternion
            quat = R.from_euler("XYZ", rpy, degrees=True).as_quat()
            quat_state = self._state.attitudeQuaternion
            quat_state.x, quat_state.y, quat_state.z, quat_state.w = quat
        self._state.position.x, self._state.position.y, self._state.position.z = pos
        self._state.velocity.x, self._state.velocity.y, self._state.velocity.z = vel
        self._state.acc.x, self._state.acc.y, self._state.acc.z = acc

    def _pwms_to_thrust(self, pwms: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.params.kf * (self.params.pwm2rpm_scale * pwms + self.params.pwm2rpm_const) ** 2

    def full_state_cmd(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        acc: npt.NDArray[np.float64],
        yaw: float,
        rpy_rate: npt.NDArray[np.float64],
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
        self.firmware.crtpCommanderHighLevelStop()  # Resets planner object
        self.firmware.crtpCommanderHighLevelUpdateTime(timestep)

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
        s_quat.x, s_quat.y, s_quat.z, s_quat.w = R.from_euler("XYZ", [0, 0, yaw]).as_quat()
        # initilize setpoint modes to match cmdFullState
        mode = self._setpoint.mode
        mode_abs, mode_disable = self.firmware.modeAbs, self.firmware.modeDisable
        mode.x, mode.y, mode.z = mode_abs, mode_abs, mode_abs
        mode.quat = mode_abs
        mode.roll, mode.pitch, mode.yaw = mode_disable, mode_disable, mode_disable
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
            return self.firmware.crtpCommanderHighLevelTakeoff(height, duration)
        self.firmware.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)

    def takeoff_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a takeoff vel command to the controller.

        Args:
            height: Target takeoff height (m)
            vel: Target takeoff velocity (m/s)
            relative: Flag if takeoff height is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
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
            return self.firmware.crtpCommanderHighLevelLand(height, duration)
        self.firmware.crtpCommanderHighLevelLandYaw(height, duration, yaw)

    def land_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a land vel command to the controller.

        Args:
            height: Target landing height (m)
            vel: Target landing velocity (m/s)
            relative: Flag if landing height is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self._fullstate_cmd = False

    def stop_cmd(self):
        """Send a stop command to the controller."""
        self.firmware.crtpCommanderHighLevelStop()
        self._fullstate_cmd = False

    def go_to_cmd(self, pos: npt.NDArray[np.float64], yaw: float, duration: float, relative: bool):
        """Send a go to command to the controller.

        Args:
            pos: [x, y, z] target position (m)
            yaw: Target yaw (rad)
            duration: Length of manuever
            relative: Flag if setpoint is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelGoTo(*pos, yaw, duration, relative)
        self._fullstate_cmd = False

    def notify_setpoint_stop(self):
        """Send a notify setpoint stop cmd to the controller."""
        self.firmware.crtpCommanderHighLevelTellState(self._state)
        self._fullstate_cmd = False

    def _reset_firmware_states(self):
        self._state = self.firmware.state_t()
        self._control = self.firmware.control_t()
        self._setpoint = self.firmware.setpoint_t()
        self._sensor_data = self.firmware.sensorData_t()
        self._tick = 0
        self._pwms[...] = 0

    def _reset_low_pass_filters(self):
        freq = self.params.firmware_freq
        self._acc_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        for i in range(3):
            self.firmware.lpf2pInit(self._acc_lpf[i], freq, self.params.acc_lpf_cutoff)
            self.firmware.lpf2pInit(self._gyro_lpf[i], freq, self.params.gyro_lpf_cutoff)

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
            self.firmware.controllerPidInit()
        else:
            self.firmware.controllerMellingerInit()

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
            ctrl = self.firmware.controllerPid
        else:
            ctrl = self.firmware.controllerMellinger
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
        trst = control.thrust
        thrust = np.array([trst - r + p + y, trst - r - p - y, trst + r - p + y, trst + r + p - y])
        np.clip(thrust, 0, self.params.max_pwm, out=thrust)  # Limit thrust to motor range
        pwms = self._thrust_to_pwm(thrust)
        np.clip(pwms, self.params.min_pwm, self.params.max_pwm, out=pwms)
        self._pwms = pwms

    def _thrust_to_pwm(self, thrust: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert thrust to PWM signal.

        Assumes brushed motors.

        Args:
            thrust: Thrust in Newtons.

        Returns:
            PWM signal.
        """
        thrust = thrust / self.params.max_pwm * 60
        volts = self.params.thrust_curve_a * thrust**2 + self.params.thrust_curve_b * thrust
        percentage = np.minimum(1, volts / self.params.supply_voltage)
        return percentage * self.params.max_pwm

    def _update_sensor_data(
        self, timestamp: float, acc: npt.NDArray[np.float64], gyro: npt.NDArray[np.float64]
    ):
        """Update the onboard sensors with low-pass filtered values.

        Args:
            timestamp: Sensor reading time in microseconds.
            acc: Acceleration values in Gs.
            gyro: Gyro values in deg/s.
        """
        for name, i, val in zip(("x", "y", "z"), range(3), acc):
            setattr(self._sensor_data.acc, name, self.firmware.lpf2pApply(self._acc_lpf[i], val))
        for name, i, val in zip(("x", "y", "z"), range(3), gyro):
            setattr(self._sensor_data.gyro, name, self.firmware.lpf2pApply(self._gyro_lpf[i], val))
        self._sensor_data.interruptTimestamp = timestamp

    def _update_setpoint(self, timestep: float):
        if not self._fullstate_cmd:
            self.firmware.crtpCommanderHighLevelTellState(self._state)
            self.firmware.crtpCommanderHighLevelUpdateTime(timestep)
            self.firmware.crtpCommanderHighLevelGetSetpoint(self._setpoint, self._state)

    def _load_firmware(self) -> ModuleType:
        """Load the firmware module.

        pycffirmware imitates the firmware of the Crazyflie 2.1. Since the module is stateful, we
        load a new instance of the module for each drone object. This allows us to simulate multiple
        drones with different states without interference.
        """
        spec = importlib.util.find_spec("pycffirmware")
        if spec is None:
            raise ImportError("Could not find the pycffirmware module.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


@dataclass
class DroneParams:
    """A collection of physical and inferred parameters of the Crazyflie 2.1.

    The preferred way to create a `DroneParams` object is to read the parameters from the
    corresponding drone URDF.
    """

    mass: float
    arm_len: float
    thrust2weight_ratio: float
    J: npt.NDArray[np.float64]
    kf: float
    km: float
    collision_h: float
    collision_r: float
    collision_z_offset: float
    max_speed_kmh: float
    gnd_eff_coeff: float
    prop_radius: float
    drag_coeff: npt.NDArray[np.float64]
    dw_coeff_1: float
    dw_coeff_2: float
    dw_coeff_3: float
    pwm2rpm_scale: float
    pwm2rpm_const: float
    min_pwm: float
    max_pwm: float
    # Defaults are calculated in __post_init__ according to the other parameters
    min_thrust: float = 0.0
    max_thrust: float = 0.0
    min_rpm: float = 0.0
    max_rpm: float = 0.0
    gnd_eff_min_height_clip: float = 0.0
    J_inv: npt.NDArray[np.float64] = np.zeros((3, 3))
    firmware_freq: int = 500  # Firmware frequency in Hz
    supply_voltage: float = 3.0  # Power supply voltage
    thrust_curve_a: float = -0.0006239  # Thrust curve parameters for brushed motors
    thrust_curve_b: float = 0.088  # Thrust curve parameters for brushed motors
    tumble_threshold: float = -0.5  # Vertical acceleration threshold for tumbling detection
    tumble_duration: int = 30  # Number of consecutive steps before tumbling is detected
    acc_lpf_cutoff: int = 80  # Low-pass filter cutoff freq
    gyro_lpf_cutoff: int = 30  # Low-pass filter cutoff freq

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.J_inv = np.linalg.inv(self.J)
        self.min_rpm = self.pwm2rpm_scale * self.min_pwm + self.pwm2rpm_const
        self.max_rpm = self.pwm2rpm_scale * self.max_pwm + self.pwm2rpm_const
        # TODO: Check if this computation is correct. Some use kf, others 4*kf
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
