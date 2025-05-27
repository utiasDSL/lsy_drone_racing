"""Single drone racing environments."""

from __future__ import annotations

from readline import read_init_file
from typing import TYPE_CHECKING, Literal

from matplotlib.pyplot import ticklabel_format
from networkx import generate_random_paths
import numpy as np
from numpy.typing import NDArray
import gymnasium
from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from packaging.version import Version

from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils import draw_line
from lsy_drone_racing.control.attitude_controller import AttitudeController
from lsy_drone_racing.control.easy_controller import TrajectoryController

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict
AutoresetMode = None
if Version(gymnasium.__version__) >= Version("1.1"):
    from gymnasium.vector import AutoresetMode

IMMITATION_LEARNING = True
if IMMITATION_LEARNING:
    from pathlib import Path
    from lsy_drone_racing.utils import load_config

class RLDroneRaceEnv(RaceCoreEnv, Env):
    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(control_mode)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [np.pi]*3)
        self.observation_space = spaces.Box(low=-lim, high=lim, shape=(34,), dtype=np.float32)
        self.autoreset = False

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate = 0
        self.prev_gate_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        self._tick = 0
        # parameters setting
        self.k_gates = 1.0
        self.k_center = 0.3
        self.k_act = 2e-4
        self.k_act_d = 1e-4
        self.k_yaw = 0.02
        self.k_crash = 20
        self.k_success = 25
        self.k_imit = 0.1
        # public variables
        self.obs_env = None
        self.obs_rl = None

    def reset(self, seed=None, options=None):
        self.obs_env, info = self._reset(seed=seed, options=options)
        self.obs_env = {k: np.array(v[0, 0]) for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self._tick = 0
        self.prev_gate = 0
        self.prev_gate_pos = self.obs_env['gates_pos'][0]
        self.prev_drone_pos = self.obs_env['pos']
        self.prev_act = self.act_bias
        self.tick_nearest = 0
        self.obs_rl = self._obs_to_state(self.obs_env, self.act_bias)
        if IMMITATION_LEARNING:
            config = load_config(Path(__file__).parents[2] / "config/level0.toml")
            # self.teacher_controller = AttitudeController(self.obs_env, info, config)
            self.teacher_controller = TrajectoryController(self.obs_env, info, config)
            self.prev_ref_waypoint = self.teacher_controller.get_trajectory_waypoints()[0]
        return self.obs_rl, info

    def step(self, action):
        # if IMMITATION_LEARNING: # test
        #     action = self.teacher_controller.compute_control(self.obs_env, None) - self.act_bias
        #     self.teacher_controller._tick += 1
        self._tick += 1
        action_exec = action + self.act_bias
        self.obs_env, _, terminated, truncated, info = self._step(action_exec)
        self.obs_env = {k: np.array(v[0, 0]) for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.obs_rl = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, action)
        if (terminated or truncated) and self.obs_env['target_gate'] >= 0:
            reward -= self.k_crash
        return self.obs_rl, reward, bool(terminated[0, 0]), bool(truncated[0, 0]), info

    def _obs_to_state(self, obs: dict[str, NDArray], action: Array) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), prev_act(4)]
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        curr_gate = obs['target_gate']
        self.gates_size = [0.4, 0.4] # [width, height]
        self.gate_rot_mat = np.array(R.from_quat(obs['gates_quat'][curr_gate]).as_matrix())
        half_w, half_h = self.gates_size[0] / 2, self.gates_size[1] / 2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners_pos = (self.gate_rot_mat @ corners_local.T).T + obs['gates_pos'][curr_gate]  # shape: (4, 3)
        # draw_line(self, np.stack([gate_corners_pos[0], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        # draw_line(self, np.stack([gate_corners_pos[1], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        # draw_line(self, np.stack([gate_corners_pos[2], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        # draw_line(self, np.stack([gate_corners_pos[3], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        
        self.rel_pos_gate = gate_corners_pos - pos[None, :]  # shape: (4, 3)
        self.rel_pos_gate = self.rel_pos_gate / np.linalg.norm(self.rel_pos_gate, axis=1, keepdims=True) # normalize
        
        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1) , action])
        return state

    def _reward(self, obs, act):
        curr_gate = obs['target_gate']
        gate_pos = obs['gates_pos'][curr_gate]
        drone_pos = obs['pos']
        r = 0.15
        if curr_gate != self.prev_gate: # handle gate switching
            self.prev_gate_pos = gate_pos
            r += self.k_success
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(gate_pos - drone_pos))
        r_center = -self.k_center * np.var(np.linalg.norm(self.rel_pos_gate, axis=1))
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        r_yaw = -self.k_yaw * np.fabs(R.from_quat(obs['quat']).as_euler('zyx', degrees=False)[0])
        if IMMITATION_LEARNING:
            k_imit_p, k_imit_d = 0.3, 1.0
            # tracking the waypoint a bit ahead current position
            waypoints = self.teacher_controller.get_trajectory_waypoints()
            ref_tick, ref_waypoint = self._find_leading_waypoint(waypoints, drone_pos, 0.2, curr_gate)
            # draw_line(self, waypoints, rgba=np.array([1.0, 1.0, 1.0, 0.4]))
            r_imit_p = -k_imit_p * np.linalg.norm(ref_waypoint - drone_pos)
            r_imit_d = k_imit_d * (np.linalg.norm(self.prev_ref_waypoint - self.prev_drone_pos) - np.linalg.norm(ref_waypoint - drone_pos))
            r_imit = self.k_imit * (r_imit_p + r_imit_d)
            draw_line(self, np.stack([ref_waypoint, drone_pos]), rgba=np.array([int(r_imit_d<0), int(r_imit_d>0), 0.0, 1.0]))
            self.prev_ref_waypoint = ref_waypoint
            # # action diff from teacher action (incompatible with state controller)
            self.teacher_controller.set_tick(ref_tick)
            self.teacher_controller.compute_control(obs, None) # only to update trajectory
            # demo_action = self.teacher_controller.compute_control(obs, None) - self.act_bias
            # r_imit = -self.k_imit * np.linalg.norm(demo_action - act)
            r += r_imit
        self.prev_act = act
        self.prev_gate = curr_gate
        self.prev_gate_pos = gate_pos
        self.prev_drone_pos = drone_pos
        r += r_gates + r_center + r_act + r_yaw
        return r
    
    def _find_leading_waypoint(self, waypoints, pos, t_ahead, curr_gate):
        cut_idx = [0, 170, 300, 410, -1]
        # find nearest waypoint
        distances = np.linalg.norm(waypoints[self.tick_nearest:cut_idx[curr_gate+1]] - pos, axis=1)
        draw_line(self, waypoints[self.tick_nearest:cut_idx[curr_gate+1]], rgba=np.array([1.0, 1.0, 1.0, 0.4]))
        self.tick_nearest = np.argmin(distances) + self.tick_nearest
        # find leading waypoint
        tick_leading = min(self.tick_nearest + int(t_ahead * self.freq), len(waypoints) - 1)
        return tick_leading, waypoints[tick_leading]
    

class VecRLDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized single-agent drone racing environment."""

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP if AutoresetMode else None}

    def __init__(
        self,
        num_envs: int,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the vectorized single-agent drone racing environment.

        Args:
            num_envs: Number of worlds in the vectorized environment.
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=num_envs,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.num_envs = num_envs
        self.single_action_space = build_action_space(control_mode)
        self.action_space = batch_space(self.single_action_space, num_envs)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [np.pi]*3)
        self.single_observation_space = spaces.Box(low=-lim, high=lim, shape=(34,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        # parameters setting
        self.k_gates = 1.0
        self.k_act = 2e-4
        self.k_act_d = 1e-4
        self.k_crash = 10

    def reset(self, seed=None, options=None):
        self.obs_env, info = self._reset(seed=seed, options=options)
        self.prev_gate_pos = self.obs_env['gates_pos'][:, 0, :]
        self.prev_drone_pos = self.obs_env['pos'][:, 0, :]
        self.prev_act = np.tile(self.act_bias, (self.num_envs, 1))
        self.obs_rl = self._obs_to_state(self.obs_env, self.prev_act)
        return self.obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        self.obs_env, _, terminated, truncated, info = self._step(action_exec)
        self.obs_env = {k: v[:, 0] for k, v in self.obs_env.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        self.obs_rl = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, action)
        done = (terminated | truncated) & (self.obs_env['target_gate'] >= 0)
        reward -= self.k_crash * done.astype(float)
        return self.obs_rl, reward, terminated, truncated, info

    def _obs_to_state(self, obs, action): # handel vec env obs
        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"]
        ang_vel = obs["ang_vel"]

        state_list = []
        for i in range(self.num_envs):
            curr_gate = obs['target_gate'][i]
            gate_rot_mat = R.from_quat(obs['gates_quat'][i, curr_gate]).as_matrix()
            half_w, half_h = 0.4 / 2, 0.4 / 2
            corners_local = np.array([
                [-half_w, 0.0,  half_h],
                [ half_w, 0.0,  half_h],
                [-half_w, 0.0, -half_h],
                [ half_w, 0.0, -half_h],
            ])
            gate_corners_pos = (gate_rot_mat @ corners_local.T).T + obs['gates_pos'][i, curr_gate]
            rel_pos_gate = (gate_corners_pos - pos[i]) 
            rel_pos_gate = rel_pos_gate / np.linalg.norm(rel_pos_gate, axis=1, keepdims=True)
            rel_pos_gate = rel_pos_gate.reshape(-1)

            rot_mat = R.from_quat(quat[i]).as_matrix().reshape(-1)
            rpy_rates = ang_vel2rpy_rates(ang_vel[i], quat[i])

            state = np.concatenate([pos[i], vel[i], rot_mat, rpy_rates, rel_pos_gate, action[i]])
            state_list.append(state)
        return np.stack(state_list)

    def _reward(self, obs, act):
        reward = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            curr_gate = obs['target_gate'][i]
            gate_pos = obs['gates_pos'][i, curr_gate]
            drone_pos = obs['pos'][i]
            r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos[i] - self.prev_drone_pos[i]) - np.linalg.norm(gate_pos - drone_pos))
            r_act = -self.k_act * np.linalg.norm(act[i]) - self.k_act_d * np.linalg.norm(act[i] - self.prev_act[i])
            reward[i] = r_gates + r_act
            self.prev_act[i] = act[i]
            self.prev_gate_pos[i] = gate_pos
            self.prev_drone_pos[i] = drone_pos
        return reward

class RLDroneHoverEnv(RaceCoreEnv, Env):
    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(control_mode)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [np.pi]*3)
        self.observation_space = spaces.Box(low=-lim, high=lim, shape=(34,), dtype=np.float32)
        self.autoreset = False

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        # parameters setting
        self.dt = 1/50.0
        self.k_pos = 1.0
        self.k_yaw = 0.05
        self.k_gates = 0.1
        self.k_center = 0.5
        self.k_act = 2e-4
        self.k_act_d = 4e-4
        self.k_crash = 10

    def reset(self, seed=None, options=None):
        self.obs_env, info = self._reset(seed=seed, options=options)
        self.obs_env = {k: v[0, 0] for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.hover_position = self.obs_env['pos'] + np.array([-0.2,-1,0.5])
        self.prev_gate_pos = self.hover_position
        self.prev_drone_pos = self.obs_env['pos']
        self.prev_act = self.act_bias
        self.obs_rl = self._obs_to_state(self.obs_env, self.act_bias)
        return self.obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        self.obs_env, _, terminated, truncated, info = self._step(action_exec)
        self.obs_env = {k: v[0, 0] for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.obs_rl = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, action)
        if (terminated or truncated) and self.obs_env['target_gate'] >= 0:
            reward -= self.k_crash
        return self.obs_rl, reward, bool(terminated[0, 0]), bool(truncated[0, 0]), info

    def _obs_to_state(self, obs: dict[str, NDArray], action: Array) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), prev_act(4)]
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        curr_gate = obs['target_gate']
        self.gates_size = [0.4, 0.4] # [width, height]
        self.gate_rot_mat = np.array(R.from_quat(obs['gates_quat'][curr_gate]).as_matrix())
        # # step 1: fake gate corners - replaced with hover goal
        # self.gates_size = [0.4, 0.4] # [width, height]
        # self.gate_rot_mat = np.array([
        #                             [-1, 0, 0],
        #                             [0, -1, 0],
        #                             [0, 0, 1]
        #                         ])
        half_w, half_h = self.gates_size[0] / 2, self.gates_size[1] / 2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners_pos = (self.gate_rot_mat @ corners_local.T).T + obs['gates_pos'][curr_gate]  # shape: (4, 3)
        draw_line(self, np.stack([gate_corners_pos[0], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        draw_line(self, np.stack([gate_corners_pos[1], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        draw_line(self, np.stack([gate_corners_pos[2], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        draw_line(self, np.stack([gate_corners_pos[3], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.5]))
        self.rel_pos_gate = gate_corners_pos - pos[None, :]  # shape: (4, 3)
        self.rel_pos_gate = self.rel_pos_gate / np.linalg.norm(self.rel_pos_gate, axis=1, keepdims=True) # normalize

        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1) , action])
        return state

    def _reward(self, obs, act):
        curr_gate = obs['target_gate']
        curr_gate = min(curr_gate, len(obs['gates_pos']) - 1)
        # gate_pos = self.hover_position # step 1: fake gate
        gate_pos = obs['gates_pos'][curr_gate]
        drone_pos = obs['pos']
        pos_err = np.linalg.norm(drone_pos - gate_pos)
        r_pos = 0.1-self.k_pos * pos_err
        r_gates = self.k_gates * pos_err * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(gate_pos - drone_pos)) / self.dt
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        r_center = -self.k_center * np.var(np.linalg.norm(self.rel_pos_gate, axis=1))
        r_yaw = -self.k_yaw * np.fabs(R.from_quat(obs['quat']).as_euler('zyx', degrees=False)[0])

        self.prev_act = act
        self.prev_gate_pos = gate_pos
        self.prev_drone_pos = drone_pos
        
        return r_pos + r_gates + r_act + r_center + r_yaw


from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=50, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                self.training_env.env_method("render", indices=0)
                # self.training_env.env_method("render", indices=1)
            except Exception as e:
                print(f"Render error: {e}")
        return True