from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
import rospy
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils.import_utils import get_ros_package_path, pycrazyswarm
from lsy_drone_racing.utils.ros_utils import check_drone_start_pos, check_race_track
from lsy_drone_racing.vicon import Vicon

if TYPE_CHECKING:
    from munch import Munch
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingDeployEnv(gymnasium.Env):
    CONTROLLER = "mellinger"

    def __init__(self, config: dict | Munch):
        super().__init__()
        self.config = config
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(13,))
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )
        self.target_gate = 0
        crazyswarm_config_path = (
            get_ros_package_path("crazyswarm", heuristic_search=True) / "launch/crazyflies.yaml"
        )
        # pycrazyswarm expects strings, not Path objects, so we need to convert it first
        swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
        self.cf = swarm.allcfs.crazyflies[0]
        names = [f"gate{g}" for g in range(1, len(config.env.track.gates) + 1)]
        names += [f"obstacle{g}" for g in range(1, len(config.env.track.obstacles) + 1)]
        self.vicon = Vicon(track_names=names, timeout=5)
        self.symbolic = None
        if config.env.symbolic:
            sim = Sim(
                track=config.env.track,
                sim_freq=config.sim.sim_freq,
                ctrl_freq=config.sim.ctrl_freq,
                disturbances=getattr(config.sim, "disturbances", {}),
                randomization=getattr(config.env, "randomization", {}),
                physics=config.sim.physics,
            )
            self.symbolic = sim.symbolic()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment.

        We cannot reset the track in the real world. Instead, we check if the gates, obstacles and
        drone are positioned within tolerances.
        """
        check_race_track(self.config)
        check_drone_start_pos(self.config)
        self.target_gate = 0
        info = self.info
        info["sim.sim_freq"] = self.config.sim.sim_freq
        info["sim.ctrl_freq"] = self.config.sim.ctrl_freq
        info["env.freq"] = self.config.env.freq
        info["sim.drone.mass"] = 0.033  # Crazyflie 2.1 mass in kg
        return self.obs, info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict]:
        tstart = time.perf_counter()
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rate)
        if (dt := time.perf_counter() - tstart) < 1 / self.config.env.freq:
            rospy.sleep(1 / self.config.env.freq - dt)
        return self.obs, -1.0, False, False, self.info

    def close(self):
        self.cf.notifySetpointsStop()
        self.cf.land(0.02, 3.0)

    @property
    def obs(self) -> dict:
        drone = self.vicon.drone_name
        rpy = self.vicon.rpy[drone]
        ang_vel = R.from_euler("xyz", rpy).inv().apply(self.vicon.ang_vel[drone])
        obs = {
            "pos": self.vicon.pos[drone].astype(np.float32),
            "rpy": rpy.astype(np.float32),
            "vel": self.vicon.vel[drone].astype(np.float32),
            "ang_vel": ang_vel.astype(np.float32),
        }
        return obs

    @property
    def info(self) -> dict:
        info = {}
        sensor_range = self.config.env.sensor_range
        n_gates = len(self.config.env.track.gates)
        info["collisions"] = []
        info["target_gate"] = self.target_gate if self.target_gate < n_gates else -1
        info["drone.pos"] = self.vicon.pos[self.vicon.drone_name]
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        drone_pos = self.vicon.pos[self.vicon.drone_name]
        gates_pos = np.array([g.pos for g in self.config.env.track.gates])
        gate_names = [f"gate{g}" for g in range(1, len(gates_pos) + 1)]
        real_gates_pos = np.array([self.vicon.pos[g] for g in gate_names])
        in_range = np.linalg.norm(real_gates_pos - drone_pos, axis=1) < sensor_range
        gates_pos[in_range] = real_gates_pos[in_range]
        gates_rpy = np.array([g.rpy for g in self.config.env.track.gates])
        real_gates_rpy = np.array([self.vicon.rpy[g] for g in gate_names])
        gates_rpy[in_range] = real_gates_rpy[in_range]
        info["gates.pos"] = gates_pos
        info["gates.rpy"] = gates_rpy
        info["gates.in_range"] = in_range

        obstacles_pos = np.array([o.pos for o in self.config.env.track.obstacles])
        obstacle_names = [f"obstacle{g}" for g in range(1, len(obstacles_pos) + 1)]
        real_obstacles_pos = np.array([self.vicon.pos[o] for o in obstacle_names])
        in_range = np.linalg.norm(real_obstacles_pos - drone_pos, axis=1) < sensor_range
        obstacles_pos[in_range] = real_obstacles_pos[in_range]
        info["obstacles.pos"] = obstacles_pos
        info["obstacles.in_range"] = in_range
        info["symbolic.model"] = self.symbolic
        # TODO: Remove check to make sure all keys from the sim env are present
        assert all(
            k in info
            for k in (
                "collisions",
                "target_gate",
                "drone.pos",
                "gates.pos",
                "gates.rpy",
                "gates.in_range",
                "obstacles.pos",
                "obstacles.in_range",
                "symbolic.model",
            )
        )
        return info

    def check_gate_progress(self):
        ...
