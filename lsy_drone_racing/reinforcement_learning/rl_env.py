from posixpath import relpath
import numpy as np
import gymnasium
from gymnasium import spaces
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from jax import Array
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates

from lsy_drone_racing.tools import race_objects

class RLDroneRacingWrapper(gymnasium.Wrapper):
    def __init__(self, env: DroneRaceEnv):
        super().__init__(env)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [np.pi]*3)
        self.observation_space = spaces.Box(low=-lim, high=lim, shape=(34,), dtype=np.float32)
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_gates = obs['gates_pos'].shape[0]
        # gates_rotates = R.from_quat(obs['gates_quat'])
        # self.gates_rot_matrices = np.array(gates_rotates.as_matrix())
        # self.gates_norm = np.array(self.gates_rot_matrices[:,:,1])
        self.prev_gate_pos = obs['gates_pos'][0]
        self.prev_drone_pos = obs['pos']
        self.prev_act = self.act_bias
        obs_rl = self.obs_to_state(obs, self.act_bias)
        return obs_rl, info

    def step(self, action: Array) -> tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action for the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        action_exec = action + self.act_bias # always apply RL output on bias
        obs, _, terminated, truncated, info = self.env.step(action_exec)
        # obs = {k: v[0, 0] for k, v in obs.items()}
        obs_rl = self.obs_to_state(obs, action)
        reward = self.reward(obs, action)
        reward += -self.k_crash * int((terminated or truncated) and int(np.sum(obs['gates_visited'])) < self.num_gates)
        # info = {k: v[0, 0] for k, v in info.items()}
        return obs_rl, reward, terminated, truncated, info

    def obs_to_state(self, obs: dict[str, NDArray], action: Array) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), prev_act(4)]
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        curr_gate = int(np.sum(obs['gates_visited']))
        curr_gate = min(curr_gate, self.num_gates - 1)
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
    
    def reward(self, obs, act):
        curr_gate = int(np.sum(obs['gates_visited']))
        curr_gate = min(curr_gate, self.num_gates - 1)
        gate_pos = obs['gates_pos'][curr_gate]
        drone_pos = obs['pos']
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(gate_pos-drone_pos))
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        self.prev_act = act
        self.prev_gate_pos = gate_pos
        self.prev_drone_pos = drone_pos
        return r_gates + r_act

from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                self.training_env.envs[0].env.render()
            except Exception as e:
                print(f"Render error: {e}")
        return True