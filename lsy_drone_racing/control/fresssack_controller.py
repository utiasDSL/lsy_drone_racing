from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict, Tuple, Set, Union

LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    LOCAL_MODE = True
except:
    LOCAL_MODE = False


import numpy as np
if TYPE_CHECKING:
    from numpy.typing import NDArray

from scipy.interpolate import CubicSpline, BSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


from lsy_drone_racing.control import Controller

from lsy_drone_racing.tools.ext_tools import TransformTool, LinAlgTool
from lsy_drone_racing.tools.race_objects import Gate, Obstacle
from lsy_drone_racing.tools.planners.occupancy_map import OccupancyMap3D




class FresssackController(Controller):
    """Controller base class with predifined functions for further development! """


    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        
        if LOCAL_MODE:
            plt.close('all')
        
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

        # self.init_gates(obs = obs,
        #                  gate_inner_size = [0.4,0.4,0.4,0.4],
        #                  gate_outer_size = [0.6,0.6,0.6,0.6],
        #                  gate_safe_radius = [0.4,0.4,0.4,0.4],
        #                  entry_offset = [0.3,0.7,0.4,0.1],
        #                  exit_offset = [0.8,1.0,0.1,0.2],
        #                  thickness = [0.4, 0.4, 0.1, 0.1])
        # self.init_obstacles(obs = obs,
        #                     obs_safe_radius = [0.2,0.2,0.2,0.2])
        # self.init_states(obs = obs)

        
        
    
    def init_states(self, obs : Dict[str, np.ndarray],):
        self.pos = obs['pos']
        self.vel = obs['vel']
        self.last_pos = self.pos
        self.current_t = 0.0
        self.next_gate = 0

    def init_gates(self, obs : Dict[str, np.ndarray],
                    
                    gate_inner_size : Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    gate_outer_size : Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    gate_safe_radius : Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    entry_offset : Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    exit_offset: Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    thickness: Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,
                    vel_limit : Union[List[np.floating], NDArray[np.floating], np.floating, None] = None) -> None:
        
        self.gates : List[Gate] = []
        self.gates_visited : List[bool] = None
        gates_pos_init  = obs['gates_pos']
        gates_rot_init  = obs['gates_quat']
        gates_num = gates_pos_init.shape[0]


        if np.isscalar(gate_inner_size):
            gate_inner_size = [gate_inner_size for i in range(gates_num)]
        if gate_inner_size is None or len(gate_inner_size) != gates_num:
            gate_inner_size = [0.4 for i in range(gates_num)]
        self.gate_inner_size = gate_inner_size

        if np.isscalar(gate_outer_size):
            gate_outer_size = [gate_outer_size for i in range(gates_num)]
        if gate_outer_size is None or len(gate_outer_size) != gates_num:
            gate_outer_size = [0.8 for i in range(gates_num)]
        self.gate_outer_size = gate_outer_size

        if np.isscalar(gate_safe_radius):
            gate_safe_radius = [gate_safe_radius for i in range(gates_num)]
        if gate_safe_radius is None or len(gate_safe_radius) != gates_num:
            gate_safe_radius = [0.4 for i in range(gates_num)]
        self.gate_safe_radius = gate_safe_radius

        if np.isscalar(entry_offset):
            entry_offset = [entry_offset for i in range(gates_num)]
        elif entry_offset is None or len(entry_offset) != gates_num:
            entry_offset = [0.2 for i in range(gates_num)]
        self.gate_entry_offset = entry_offset

        if np.isscalar(exit_offset):
            exit_offset = [exit_offset for i in range(gates_num)]
        elif exit_offset is None or len(exit_offset) != gates_num:
            exit_offset = [0.2 for i in range(gates_num)]
        self.gate_exit_offset = exit_offset

        if np.isscalar(thickness):
            thickness = [thickness for i in range(gates_num)]
        elif thickness is None or len(thickness) != gates_num:
            thickness = [0.2 for i in range(gates_num)]
        self.gate_thickness = thickness

        if np.isscalar(vel_limit):
            vel_limit = [vel_limit for i in range(gates_num)]
        elif vel_limit is None or len(vel_limit) != gates_num:
            vel_limit = [0.2 for i in range(gates_num)]
        self.gate_vel_limit = vel_limit

        for i in range(gates_num):
            self.gates.append(Gate(pos = gates_pos_init[i],
                                quat = gates_rot_init[i],
                                inner_width = self.gate_inner_size[i],
                                inner_height = self.gate_inner_size[i],
                                outer_width = self.gate_outer_size[i],
                                outer_height = self.gate_outer_size[i],
                                safe_radius = self.gate_safe_radius[i],
                                entry_offset = self.gate_exit_offset[i],
                                exit_offset = self.gate_exit_offset[i],
                                thickness = self.gate_thickness[i],
                                vel_limit = self.gate_vel_limit[i]
                                ))
        self.gates_visited = obs['gates_visited']

    def init_obstacles(self,
                        obs : Dict[str, np.ndarray],
                        obs_safe_radius : Union[List[np.floating], np.floating, NDArray[np.floating], None] = None,) -> None:
        self.obstacles : List[Obstacle] = []
        obs_pos_init = obs['obstacles_pos']
        self.obstacles_visited : List[bool] = None
        obs_num = obs_pos_init.shape[0]
        

        if np.isscalar(obs_safe_radius):
            obs_safe_radius = [obs_safe_radius for i in range(obs_num)]
        elif obs_safe_radius is None or len(obs_safe_radius) != obs_num:
            obs_safe_radius = [0.2 for i in range(obs_num)]
        self.obs_safe_radius = obs_safe_radius

        for i in range(obs_num):
            self.obstacles.append(Obstacle(pos = obs_pos_init[i], safe_radius = self.obs_safe_radius[i]))
        self.obstacles_visited = obs['obstacles_visited']

     
    def draw_drone(fig: figure.Figure, ax: axes.Axes, pos : NDArray[np.floating], color='yellow', label = True) -> Tuple[figure.Figure, axes.Axes]:
        if LOCAL_MODE:
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=50)
            plt.pause(0.001)
        return fig , ax
    
    def draw_drone_vel(fig: figure.Figure, ax: axes.Axes, pos : NDArray[np.floating], vel : NDArray[np.floating], color='purple', label = True) -> Tuple[figure.Figure, axes.Axes]:
        if LOCAL_MODE:
            vel_dir = vel / (np.linalg.norm(vel) + 1e-6)
            fixed_length = 0.4
            vel_dir = vel_dir * fixed_length
            ax.quiver(pos[0], pos[1], pos[2],
                    vel_dir[0], vel_dir[1], vel_dir[2],
                    color=color,
                    length=np.linalg.norm(vel),
                    normalize=False,
                    linewidth=2)
            plt.pause(0.001)
        return fig, ax


    def visualize_trajectory(fig: figure.Figure, ax: axes.Axes, trajectory: CubicSpline, t_start : np.float32, t_end :np.float32, color='red') -> Tuple[figure.Figure, axes.Axes]:
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        t_samples = np.linspace(t_start, t_end, int((t_end - t_start)/ 0.05))
        xyz = trajectory(t_samples)

        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2, label='Trajectory')

        ax.legend()
        plt.pause(0.001)
        return fig, ax

    def gate_entry_exit(gate: Gate, entry_offset: float, exit_offset:float = None):
        if exit_offset is None:
            exit_offset = entry_offset
        entry = gate.pos - gate.norm_vec * entry_offset
        exit  = gate.pos + gate.norm_vec * exit_offset
        return entry, exit
    
    def update_gate_if_needed(self, obs : Dict[str, np.ndarray]) -> Tuple[bool, List[bool]]:
        need_gate_update, update_flags = self.check_gate_change(obs['gates_visited'])
        if need_gate_update:
            for index, flag in enumerate(update_flags):
                if flag:
                    gate_pos = obs['gates_pos'][index]
                    gate_quat = obs['gates_quat'][index]
                    self.update_gates(gate_idx = index, gate_pos=gate_pos, gate_quat = gate_quat )
        return need_gate_update, update_flags
    def update_obstacle_if_needed(self, obs : Dict[str, np.ndarray]) -> Tuple[bool, List[bool]]:
        need_obs_update, update_flags = self.check_obs_change(obs['obstacles_visited'])
        if need_obs_update:
            for index, flag in enumerate(update_flags):
                if flag:
                    obs_pos = obs['obstacles_pos'][index]
                    self.update_obstacles(obst_idx=index, obst_pnt = obs_pos)
        return need_obs_update, update_flags

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        
        if LOCAL_MODE and self._tick % 10 == 0:
            FresssackController.draw_drone(fig = self.fig, ax = self.ax, pos = self.pos)
       
        # need_gate_update, _ = self.update_gate_if_needed(obs = obs)
        # need_obs_update , _ = self.update_obstacle_if_needed(obs = obs)
        # through_gate = self.update_next_gate()
       

        return np.zeros(13)


    # def get_trajectory_tracking_target(self, lookahead: float = 0.3, num_samples: int = 200) -> np.ndarray:

    #     t_min = self.track_t
    #     t_max = self.trajectory.x[-1]
    #     t_samples = np.linspace(t_min, t_max + 1e-5, int((t_max - t_min)/0.05))
    #     if t_samples.shape[0] == 0:
    #         return self.trajectory(t_max)
    #     traj_points = self.trajectory(t_samples)
    #     dists = np.linalg.norm(traj_points - self.pos, axis=1)
    #     idx_closest = np.argmin(dists)

    #     idx_target = min(idx_closest + int(lookahead / self.occ_map_res), len(t_samples) - 1)
    #     t_target = t_samples[idx_target]

    #     self.track_t = t_target
    #     return traj_points[idx_target]


    def step_update(self, obs : Dict[str, np.ndarray]) -> None:
        self.last_pos = self.pos
        self.pos = obs['pos']
        self.vel = obs['vel']
        self.current_t += 1.0 / self._freq
        self._tick += 1


    
    def check_gate_change(self, gates_visited: List[bool])->Tuple[bool, List[bool]]:
        result = []
        result_flag = False
        for idx in range(len(gates_visited)):
            result_flag = result_flag or (gates_visited[idx] ^ self.gates_visited[idx])
            result.append(gates_visited[idx] ^ self.gates_visited[idx])

        self.gates_visited = gates_visited
        return result_flag, result
    
    def check_obs_change(self, obs_visited: List[bool])->Tuple[bool, List[bool]]:
        result = []
        result_flag = False
        for idx in range(len(obs_visited)):
            result_flag = result_flag or (obs_visited[idx] ^ self.obstacles_visited[idx])
            result.append(obs_visited[idx] ^ self.obstacles_visited[idx])
        self.obstacles_visited = obs_visited
        return result_flag, result
    

    def update_gates(self, gate_idx : int, gate_pos : np.array, gate_quat: np.array) -> None:
        self.gates[gate_idx].update(pos = gate_pos, quat = gate_quat)

    def update_obstacles(self, obst_idx : int, obst_pnt : np.array) -> None:
        self.obstacles[obst_idx].pos = obst_pnt

    def update_next_gate(self, distance = 0.5) -> bool:
        if not self.next_gate <= len(self.gates):
            return False
        if np.linalg.norm(self.pos - (self.gates[self.next_gate].pos)) > distance:
            return False
        
        v_1 = self.pos - self.gates[self.next_gate].pos
        v_0 = self.last_pos - self.gates[self.next_gate].pos
        dot_0 = np.dot(v_0, self.gates[self.next_gate].norm_vec)
        dot_1 = np.dot(v_1, self.gates[self.next_gate].norm_vec)
        if(dot_0 * dot_1 < 0):
            self.next_gate += 1
            print('Next Gate:' + str(self.next_gate))
            return True
        else:
            return False
        
    def save_trajectory(trajectory: Union[List[NDArray], NDArray], dt : np.floating, path : str, extend_last_step : float = 0.3) -> None:
        if isinstance(trajectory, list):
            trajectory = np.vstack(trajectory)

        N, D = trajectory.shape
        assert D in [3, 6], f"trajectory must have 3 or 6 columns, got {D}"
        
        t = np.arange(N).reshape(-1, 1) * dt
        data = np.hstack([t, trajectory])  # shape = (N, D+1)
        
        if extend_last_step != 0 and D == 6:
            last_pos = trajectory[-1, 0:3]
            last_vel = trajectory[-1, 3:6]
            num_extend = int(extend_last_step / dt)

            extended = []
            for i in range(1, num_extend + 1):
                t_ext = (N + i - 1) * dt
                pos_ext = last_pos + i * dt * last_vel
                row = np.hstack([[t_ext], pos_ext, last_vel])
                extended.append(row)

            data = np.vstack([data, np.array(extended)])
            
        if D == 3:
            header = 't,x,y,z'
        else:
            header = 't,x,y,z,vx,vy,vz'

        np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6f')
    
    def read_trajectory(path : str) -> Tuple[List[np.floating], List[NDArray], List[NDArray]]:
        with open(path, 'r') as f:
            header = f.readline().strip().split(',')

        data = np.loadtxt(path, delimiter=',', skiprows=1)

        t_idx = header.index('t')
        x_idx = header.index('x')
        y_idx = header.index('y')
        z_idx = header.index('z')

        has_velocity = all(col in header for col in ['vx', 'vy', 'vz'])
        if has_velocity:
            vx_idx = header.index('vx')
            vy_idx = header.index('vy')
            vz_idx = header.index('vz')

        t_list = data[:, t_idx].tolist()
        pos_list = [data[i, [x_idx, y_idx, z_idx]] for i in range(data.shape[0])]
        vel_list = [data[i, [vx_idx, vy_idx, vz_idx]] for i in range(data.shape[0])] if has_velocity else []

        return t_list, pos_list, vel_list