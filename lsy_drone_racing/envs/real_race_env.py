from __future__ import annotations

import logging
import multiprocessing as mp
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import cflib
import jax
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from gymnasium import Env
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray

from lsy_drone_racing.envs.utils import gate_passed, load_track
from lsy_drone_racing.ros import ROSConnector
from lsy_drone_racing.ros.ros_utils import check_drone_start_pos, check_race_track

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EnvData:
    """Struct holding the data of all auxiliary variables for the environment."""

    target_gate: NDArray
    gates_visited: NDArray
    obstacles_visited: NDArray
    last_drone_pos: NDArray[np.float32]

    @classmethod
    def create(cls, n_drones: int, n_gates: int, n_obstacles: int) -> EnvData:
        """Create an instance of the EnvData class."""
        return EnvData(
            target_gate=np.zeros(n_drones, dtype=int),
            gates_visited=np.zeros((n_drones, n_gates), dtype=bool),
            obstacles_visited=np.zeros((n_drones, n_obstacles), dtype=bool),
            last_drone_pos=np.zeros((n_drones, 3), dtype=np.float32),
        )


class RealRaceCoreEnv:
    """Deployable version of the multi-agent drone racing environment."""

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        n_drones: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the deployable version of the multi-agent drone racing environment.

        Args:
            n_drones: Number of drones.
            freq: Environment step frequency.
            sensor_range: Sensor range.
        """
        rclpy.init()
        # Static env data
        self.n_drones = n_drones
        self.gates, self.obstacles, self.drones = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array([-3.0, -3.0, 0.0])
        self.pos_limit_high = np.array([3.0, 3.0, 2.5])
        self.sensor_range = sensor_range
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        self.channel = drones[rank]["channel"]
        self.rank = rank
        self.freq = freq
        self.device = jax.devices("cpu")[0]
        assert control_mode in ["state", "attitude"], f"Invalid control mode {control_mode}"
        self.control_mode = control_mode
        self.randomizations = randomizations
        # Dynamic data
        self.data = EnvData.create(
            n_drones=n_drones, n_gates=self.n_gates, n_obstacles=self.n_obstacles
        )
        # Establish drone connection
        self._drone_healthy = mp.Event()
        self._drone_healthy.set()
        self.drone = self._connect_to_drone(
            radio_id=rank, radio_channel=drones[rank]["channel"], drone_id=drones[rank]["id"]
        )

        self._ros_connector = ROSConnector(estimator_names=self.drone_names, timeout=5.0)
        post_fix = "full_state" if control_mode == "attitude" else "state"
        msg_name = f"/{self.drone_name}/" + post_fix
        self.node = rclpy.create_node("RealRaceCoreEnv" + uuid4().hex)
        self._action_pub = self.node.create_publisher(Float64MultiArray, msg_name, 10)
        self._jit()

    def _reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        if options is None or options.get("check_race_track", True):
            check_race_track(self.gates, self.obstacles, self.randomizations)
        if options is None or options.get("check_drone_start_pos", True):
            check_drone_start_pos(self.drones.pos, self.randomizations, self.drone_name)
        # Update the position of gates and obstacles with the real positions measured from Mocap. If
        # disabled, they are equal to the nominal positions defined in the track config.
        if options is None or not options.get("practice_without_track_objects", False):
            # Update the ground truth position and orientation of the gates and obstacles
            tf_names = [f"gate{i}" for i in range(1, self.n_gates + 1)]
            tf_names += [f"obstacle{i}" for i in range(1, self.n_obstacles + 1)]
            ros_connector = ROSConnector(tf_names=tf_names, timeout=5.0)
            for i in range(self.n_gates):
                self.gates.pos[i, ...] = ros_connector.pos[f"gate{i + 1}"]
                self.gates.quat[i, ...] = ros_connector.quat[f"gate{i + 1}"]
            for i in range(self.n_obstacles):
                self.obstacles.pos[i, ...] = ros_connector.pos[f"obstacle{i + 1}"]
            ros_connector.close()

        self._reset_env_data(self.data)
        self._reset_drone()

        if self.control_mode == "attitude":
            # Unlock thrust mode protection by sending a zero thrust command
            self.drone.commander.send_setpoint(0, 0, 0, 0)

        return self.obs(), self.info()

    def _step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment."""
        # Note: We do not send the action to the drone here.
        self.send_action(action)
        obs = self.obs()
        gate_pos = self.gates.pos[self.data.target_gate]
        gate_quat = self.gates.quat[self.data.target_gate]
        drone_pos = obs["pos"]

        with jax.default_device(self.device):  # Ensure gate_passed runs on the CPU
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, (0.45, 0.45)
            )
        self.data.target_gate += np.asarray(passed)
        self.data.target_gate[self.data.target_gate >= self.n_gates] = -1
        self.data.last_drone_pos[...] = drone_pos

        # Send vicon position updates to the drone
        self.drone.extpos.send_extpose(*obs["pos"][self.rank], *obs["quat"][self.rank])

        return obs, self.reward(), self.terminated(), self.truncated(), self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the observation of the environment."""
        # If gates/obstacles are in sensor range use the actual pose, otherwise use the nominal pose
        # The actual pose is measured at the beginning of the episode and is not updated during the
        # episode. If we want to use dynamic gates/obstacles, we need to update the poses here.
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
        drone_pos = np.stack(
            [self._ros_connector.pos[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_quat = np.stack(
            [self._ros_connector.quat[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_vel = np.stack(
            [self._ros_connector.vel[drone] for drone in self.drone_names], dtype=np.float32
        )
        drone_ang_vel = np.stack(
            [self._ros_connector.ang_vel[drone] for drone in self.drone_names], dtype=np.float32
        )
        obs = {
            "pos": drone_pos,
            "quat": drone_quat,
            "vel": drone_vel,
            "ang_vel": drone_ang_vel,
            "target_gate": self.data.target_gate,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }
        return obs

    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            The current sparse reward function will most likely not work directly for training an
            agent. If you want to use reinforcement learning, you will need to define your own
            reward function.

        Returns:
            Reward for the current state.
        """
        return -1.0 * (self.data.target_gate == -1)  # Implicit float conversion

    def terminated(self) -> NDArray:
        """Check if the episode is terminated."""
        terminated = self.data.target_gate == -1
        terminated[self.rank] |= not self._drone_healthy.is_set()
        return terminated

    def truncated(self) -> NDArray:
        """Check if the episode is truncated."""
        return np.zeros(self.n_drones, dtype=bool)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        if self.control_mode == "attitude":
            # Action conversion: We currently expect controllers to send [collective thrust, roll,
            # pitch, yaw] as input with thrust in Newton and angles in radians. The drone expects
            # angles in degrees and thrust in PWMs.
            # TODO: Once we fix this interface in crazyflow, we should also fix it here
            action = (*np.rad2deg(action[1:]), int(thrust2pwm(action[0])))
            self.drone.commander.send_setpoint(*action)
        else:
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            # TODO: We currently limit ourselves to yaw rotation only because the simulation is
            # based on the old crazyswarm full_state command definition. Once the simulation does
            # support the real full_state command, we can remove this limitation and use full
            # quaternions as inputs
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )
            return
        # TODO: Publish command with ros connector
        # self._ros_connector.send_action(self.drone_name, action)

    def _connect_to_drone(self, radio_id: int, radio_channel: int, drone_id: int) -> Crazyflie:
        cflib.crtp.init_drivers()
        uri = f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7" + f"{drone_id:x}".upper()
        import time

        power_switch = PowerSwitch(uri)
        power_switch.stm_power_cycle()
        time.sleep(2)

        drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))

        event = mp.Event()

        def connect_callback(link_uri: str):
            # TODO: Apply settings
            event.set()

        drone.fully_connected.add_callback(connect_callback)
        drone.disconnected.add_callback(lambda _: self._drone_healthy.clear())
        drone.connection_failed.add_callback(
            lambda _, msg: logger.warning(f"Connection failed: {msg}")
        )
        drone.connection_lost.add_callback(lambda _, msg: logger.warning(f"Connection lost: {msg}"))
        drone.open_link(uri)

        logger.info(f"Waiting for drone {drone_id} to connect...")
        event.wait(5.0)
        logger.info(f"Drone {drone_id} connected to {uri}")

        return drone

    def _reset_env_data(self, data: EnvData):
        """Reset the environment data."""
        data.target_gate[...] = 0
        data.gates_visited[...] = False
        data.obstacles_visited[...] = False
        drone_pos = np.stack([self._ros_connector.pos[n] for n in self.drone_names])
        data.last_drone_pos[...] = drone_pos

    def _reset_drone(self):
        self._apply_drone_settings()
        pos = self._ros_connector.pos[self.drone_name]
        # Reset Kalman filter values
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        quat = self._ros_connector.quat[self.drone_name]
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[0]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")

    def _apply_drone_settings(self):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
        self.drone.param.set_value("stabilizer.estimator", 2)
        time.sleep(0.1)  # TODO: Maybe remove
        # enable/disable tumble control. Required 0 for agressive maneuvers
        self.drone.param.set_value("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID; 2:Mellinger
        self.drone.param.set_value("stabilizer.controller", 2)
        # rate: 0, angle: 1
        self.drone.param.set_value("flightmode.stabModeRoll", 1)
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def _return_to_start(self):
        # Enable high-level functions of the drone and disable low-level control access
        logger.warning("Returning to start position")
        self.drone.commander.send_stop_setpoint()
        self.drone.commander.send_notify_setpoint_stop()
        self.drone.param.set_value("commander.enHighLevel", "1")
        self.drone.platform.send_arming_request(True)
        # Fly back to the start position
        RETURN_HEIGHT = 1.75  # m
        BREAKING_DISTANCE = 1.0  # m
        BREAKING_DURATION = 3.0  # s
        RETURN_DURATION = 5.0  # s
        LAND_DURATION = 3.0  # s

        def wait_for_action(dt: float):
            tstart = time.perf_counter()
            # Wait for the action to be completed and send the current position to the drone
            while time.perf_counter() - tstart < dt:
                obs = self.obs()
                self.drone.extpos.send_extpose(*obs["pos"][self.rank], *obs["quat"][self.rank])
                time.sleep(0.05)
                if not self._drone_healthy.is_set():
                    raise RuntimeError("Drone connection lost")

        pos = self._ros_connector.pos[self.drone_name]
        vel = self._ros_connector.vel[self.drone_name]
        break_pos = pos + vel / np.linalg.norm(vel) * BREAKING_DISTANCE
        break_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*break_pos, 0, BREAKING_DURATION)
        wait_for_action(BREAKING_DURATION)
        return_pos = self.drones.pos[self.rank]  # Starting position from the config file
        return_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*return_pos, 0, RETURN_DURATION)
        wait_for_action(RETURN_DURATION)
        return_pos[2] = 0.0
        self.drone.high_level_commander.go_to(*return_pos, 0, LAND_DURATION)
        wait_for_action(LAND_DURATION)

    def _jit(self):
        """JIT compile jax functions.

        We compile all jit-compiled functions at startup to avoid the overhead of compiling them
        at the first call when the drone is already in the air.
        """
        drone_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        gate_quat = np.zeros((self.n_drones, 4), dtype=np.float32)
        with jax.default_device(self.device):
            jax.block_until_ready(
                gate_passed(drone_pos, drone_pos, gate_pos, gate_quat, (0.45, 0.45))
            )

    def close(self):
        """Close the environment."""
        try:  # Check if drone has successfully completed the track and return home
            if self.data.target_gate[self.rank] == -1:
                self._return_to_start()
        finally:  # Kill the drone
            pk = CRTPPacket()
            pk.port = CRTPPort.LOCALIZATION
            pk.channel = Localization.GENERIC_CH
            pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
            self.drone.send_packet(pk)
            self.drone.close_link()
            # Close all ROS connections
            self._ros_connector.close()
            self._action_pub.destroy()
            self.node.destroy_node()


class RealDroneRaceEnv(RealRaceCoreEnv, Env):
    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        super().__init__(
            drones=drones,
            rank=rank,
            n_drones=1,
            freq=freq,
            track=track,
            randomizations=randomizations,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        obs, info = self._reset(seed=seed, options=options)
        return {k: v[0, ...] for k, v in obs.items()}, info

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._step(action)
        return {k: v[0, ...] for k, v in obs.items()}, reward[0], terminated[0], truncated[0], info


class RealMultiDroneRaceEnv(RealRaceCoreEnv, Env):
    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        n_drones: int,
        freq: int,
        track: ConfigDict,
        randomizations: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        super().__init__(
            drones=drones,
            rank=rank,
            n_drones=n_drones,
            freq=freq,
            randomizations=randomizations,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
        )


def thrust2pwm(thrust):
    """Convert thrust to pwm using a quadratic function.

    TODO: Remove in favor of lsy_models
    """
    a_coeff = -1.1264
    b_coeff = 2.2541
    c_coeff = 0.0209
    pwm_max = 65535.0

    pwm = a_coeff * thrust * thrust + b_coeff * thrust + c_coeff
    pwm = np.maximum(pwm, 0.0)
    pwm = np.minimum(pwm, 1.0)
    thrust_pwm = pwm * pwm_max
    return thrust_pwm
