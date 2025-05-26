"""Single drone racing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict
AutoresetMode = None
if Version(gymnasium.__version__) >= Version("1.1"):
    from gymnasium.vector import AutoresetMode


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
        obs, info = self._reset(seed=seed, options=options)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.prev_gate_pos = obs['gates_pos'][0]
        self.prev_drone_pos = obs['pos']
        self.prev_act = self.act_bias
        obs_rl = self._obs_to_state(obs, self.act_bias)
        return obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        obs, _, terminated, truncated, info = self._step(action_exec)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        obs_rl = self._obs_to_state(obs, action)
        reward = self._reward(obs, action)
        if (terminated or truncated) and obs['target_gate'] >= 0:
            reward -= self.k_crash
        return obs_rl, reward, bool(terminated[0, 0]), bool(truncated[0, 0]), info

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
        rel_pos_gate = pos[None, :] - gate_corners_pos  # shape: (4, 3)
        rel_pos_gate = rel_pos_gate / np.linalg.norm(rel_pos_gate, axis=1, keepdims=True) # normalize
        rel_pos_gate = rel_pos_gate.reshape(-1)         # shape: (12,)
        
        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, rel_pos_gate, action])
        return state

    def _reward(self, obs, act):
        curr_gate = obs['target_gate']
        curr_gate = min(curr_gate, len(obs['gates_pos']) - 1)
        gate_pos = obs['gates_pos'][curr_gate]
        drone_pos = obs['pos']
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(gate_pos - drone_pos))
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        self.prev_act = act
        self.prev_gate_pos = gate_pos
        self.prev_drone_pos = drone_pos
        return r_gates + r_act

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
        obs, info = self._reset(seed=seed, options=options)
        self.prev_gate_pos = obs['gates_pos'][:, 0, :]
        self.prev_drone_pos = obs['pos'][:, 0, :]
        self.prev_act = np.tile(self.act_bias, (self.num_envs, 1))
        obs_rl = self._obs_to_state(obs, self.prev_act)
        return obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        obs, _, terminated, truncated, info = self._step(action_exec)
        obs = {k: v[:, 0] for k, v in obs.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        obs_rl = self._obs_to_state(obs, action)
        reward = self._reward(obs, action)
        done = (terminated | truncated) & (obs['target_gate'] >= 0)
        reward -= self.k_crash * done.astype(float)
        return obs_rl, reward, terminated, truncated, info

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
            rel_pos_gate = (pos[i] - gate_corners_pos) 
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
        self.k_gates = 0.1
        self.k_act = 2e-4
        self.k_act_d = 1e-4
        self.k_crash = 10

    def reset(self, seed=None, options=None):
        obs, info = self._reset(seed=seed, options=options)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.hover_position = obs['pos'] + np.array([0,0,0.5])
        self.prev_gate_pos = self.hover_position
        self.prev_drone_pos = obs['pos']
        self.prev_act = self.act_bias
        obs_rl = self._obs_to_state(obs, self.act_bias)
        return obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        obs, _, terminated, truncated, info = self._step(action_exec)
        obs = {k: v[0, 0] for k, v in obs.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        obs_rl = self._obs_to_state(obs, action)
        reward = self._reward(obs, action)
        if (terminated or truncated) and obs['target_gate'] >= 0:
            reward -= self.k_crash
        return obs_rl, reward, bool(terminated[0, 0]), bool(truncated[0, 0]), info

    def _obs_to_state(self, obs: dict[str, NDArray], action: Array) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), prev_act(4)]
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        # fake gate corners - replaced with hover goal
        self.gates_size = [0.4, 0.4] # [width, height]
        self.gate_rot_mat = np.array([
                                    [1, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0]
                                ])
        half_w, half_h = self.gates_size[0] / 2, self.gates_size[1] / 2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners_pos = (self.gate_rot_mat @ corners_local.T).T + self.hover_position # shape: (4, 3)
        rel_pos_gate = gate_corners_pos - pos[None, :]  # shape: (4, 3)
        rel_pos_gate = rel_pos_gate / np.linalg.norm(rel_pos_gate, axis=1, keepdims=True) # normalize
        rel_pos_gate = rel_pos_gate.reshape(-1)         # shape: (12,)
        draw_line(self, gate_corners_pos)
        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, rel_pos_gate, action])
        return state

    def _reward(self, obs, act):
        curr_gate = obs['target_gate']
        curr_gate = min(curr_gate, len(obs['gates_pos']) - 1)
        gate_pos = self.hover_position # fake gate
        drone_pos = obs['pos']
        r_pos = -self.k_pos * np.linalg.norm(drone_pos - gate_pos)
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(gate_pos - drone_pos)) / self.dt
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        self.prev_act = act
        self.prev_gate_pos = gate_pos
        self.prev_drone_pos = drone_pos
        print(r_pos, r_gates, r_act)
        return r_pos + r_gates + r_act


from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=50, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                self.training_env.envs[0].env.render()
            except Exception as e:
                print(f"Render error: {e}")
        return True