from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
import pycrazyswarm
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.drone import Drone
from lsy_drone_racing.sim.symbolic import symbolic
from lsy_drone_racing.utils.import_utils import get_ros_package_path
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
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(13,))
        self.target_gate = 0
        crazyswarm_config_path = get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml"
        # pycrazyswarm expects strings, not Path objects, so we need to convert it first
        swarm = pycrazyswarm.Crazyswarm(str(crazyswarm_config_path))
        self.cf = swarm.allcfs.crazyflies[0]
        self.vicon = Vicon(track_names=[], timeout=5)
        if config.env.symbolic:
            self.symbolic_model = symbolic(Drone(self.CONTROLLER), 1 / config.sim.ctrl_freq)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment.

        We cannot reset the track in the real world. Instead, we check if the gates, obstacles and
        drone are positioned within tolerances.
        """
        check_race_track(self.config)
        check_drone_start_pos(self.config)
        self.target_gate = 0
        return self.obs, 0, False, False, self.info

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict]:
        pos, vel, acc, yaw, rpy_rates = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.cf.cmdFullState(pos, vel, acc, yaw, rpy_rates)
        return self.obs, 0, False, False, {}

    def close(self):
        self.cf.notifySetpointsStop()
        self.cf.land(0.02, 3.0)

    @property
    def obs(self) -> dict:
        drone = self.vicon.drone_name
        obs = {
            "pos": self.vicon.pos[drone],
            "rpy": self.vicon.rpy[drone],
            "vel": self.vicon.vel[drone],
            "ang_vel": R.from_euler("xyz", self.vicon.rpy[drone]).apply(self.vicon.ang_vel[drone]),
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
        real_gates_pos = np.array([self.vicon.pos[g] for g in range(self.config.env.gates)])
        in_range = np.linalg.norm(real_gates_pos - drone_pos, axis=1) < sensor_range
        gates_pos[in_range] = real_gates_pos[in_range]
        gates_rpy = np.array([g.rpy for g in self.config.env.track.gates])
        real_gates_rpy = np.array([self.vicon.rpy[g] for g in range(self.config.env.gates)])
        gates_rpy[in_range] = real_gates_rpy[in_range]
        info["gates.pos"] = gates_pos
        info["gates.rpy"] = gates_rpy
        info["gates.in_range"] = in_range

        obstacles_pos = np.array([o.pos for o in self.config.env.track.obstacles])
        real_obstacles_pos = np.array([self.vicon.pos[o] for o in range(self.config.env.obstacles)])
        in_range = np.linalg.norm(real_obstacles_pos - drone_pos, axis=1) < sensor_range
        obstacles_pos[in_range] = real_obstacles_pos[in_range]
        info["obstacles.pos"] = obstacles_pos
        info["obstacles.in_range"] = in_range

        if self.config.env.symbolic:
            info["symbolic.model"] = self.symbolic_model
        return {}

    def check_gate_progress(self):
        ...
