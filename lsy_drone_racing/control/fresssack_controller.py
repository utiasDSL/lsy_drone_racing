from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict, Tuple, Set, Union, Callable, Optional
import pandas as pd
import os
import atexit
LOCAL_MODE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as figure
    import matplotlib.axes as axes
    LOCAL_MODE = True
except:
    LOCAL_MODE = False

ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from rclpy.publisher import MsgType
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Path
    from tf2_ros import TransformBroadcaster
    from transformations import quaternion_from_euler, euler_from_quaternion
    from visualization_msgs.msg import Marker
    ROS_AVAILABLE = True
except:
    ROS_AVAILABLE = False


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





class ControllerROSTx(Node):
    pub : Publisher
    def __init__(self,
                 node_name : str,
                 msg_type : MsgType,
                 topic_name : str,
                 queue_size : np.integer = 10,
                 ):
        super().__init__(node_name = node_name)

        self.pub : Publisher = self.create_publisher(msg_type = msg_type, topic = topic_name, qos_profile = queue_size)
    
    def process_data(self, raw_data)-> MsgType:
        raise NotImplementedError()
    
    def publish(self, raw_data : Union[MsgType, NDArray[np.floating], List[NDArray[np.floating]]]):
        self.pub.publish(msg = self.process_data(raw_data))

class TFTx(Node):
    br : TransformBroadcaster
    child_frame_id : str
    def __init__(self,
                 node_name : str,
                 topic_name : str,
                 ):
        super().__init__(node_name = node_name)
        self.child_frame_id = topic_name
        self.br = TransformBroadcaster(self)
    def publish(self,  raw_data : Dict[str, Union[str, NDArray]]):
        pos = np.array(raw_data['pos'], dtype=float)
        quat = np.array(raw_data.get('quat', [0, 0, 0, 1]), dtype=float)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = str(raw_data.get('frame_id', 'map'))
        t.child_frame_id = str(raw_data.get('child_frame_id', self.child_frame_id))

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.br.sendTransform(t)

class PathTx(ControllerROSTx):
    def __init__(self,
                 node_name : str,
                 topic_name : str,
                 queue_size : np.integer):
        super().__init__(node_name, Path, topic_name, queue_size)

    def process_data(self, raw_data : Dict[str, NDArray])-> MsgType:
        traj = raw_data['traj']
        quat = raw_data.get('quat', [[0,0,0,1] for i in range(len(traj))])
        frame_id = str(raw_data.get('frame_id', 'map'))
        msg = Path()
        stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(frame_id)
        msg.header.stamp = stamp
        for idx, pos in enumerate(traj):
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = str(frame_id)

            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])

            pose_msg.pose.orientation.x = float(quat[idx][0])
            pose_msg.pose.orientation.y = float(quat[idx][1])
            pose_msg.pose.orientation.z = float(quat[idx][2])
            pose_msg.pose.orientation.w = float(quat[idx][3])
            
            msg.poses.append(pose_msg)
        return msg
    

class PoseTx(ControllerROSTx):
    def __init__(self,
                 node_name : str,
                 topic_name : str,
                 queue_size : np.integer):
        super().__init__(node_name, PoseStamped, topic_name, queue_size)

    def process_data(self, raw_data : Dict[str, NDArray])-> MsgType:
        pos = raw_data['pos']
        quat = raw_data.get('quat', [0,0,0,1])

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = str(raw_data.get('frame_id', 'map'))
       
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        return msg

class MeshMarkerTx(ControllerROSTx):
    name : str
    mesh_url : List[str]
    frame_id : str
    def __init__(self,
                 node_name : str,
                 topic_name : str,
                 queue_size : np.integer,
                 frame_id : str,
                 mesh_path : Union[List[str], str]):
        super().__init__(node_name, Marker, topic_name, queue_size)
        self.name = node_name
        self.frame_id = frame_id
        if isinstance(mesh_path, str):
            self.mesh_url = ['file://' + os.path.abspath(mesh_path)]
        else:
            self.mesh_url = ['file://' + os.path.abspath(path) for path in mesh_path]

    def process_data(self, raw_data : Dict[str, NDArray])-> MsgType:
        idx = 0
        if raw_data is not None:
            idx = raw_data.get('idx', 0)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = self.name
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        marker.mesh_resource = self.mesh_url[idx]
        marker.mesh_use_embedded_materials = True

        # Scales
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0

        return marker



class FresssackController(Controller):
    """Controller base class with predifined functions for further development! """
    pos : NDArray[np.floating]
    vel : NDArray[np.floating]
    quat : NDArray[np.floating]
    gates : List[Gate]
    obstacles : List[Obstacle]
    gates_visited : List[bool]
    obstacles_visited : List[bool]

    log_file : bool
    log : Dict[str, Union[List[np.floating], List[np.integer], List[np.NDArray[np.integer]], List[np.NDArray[np.floating]], List[np.NDArray[np.bool]]]]
    data_log_freq : np.floating
    data_log_path : str
    data_log_keys : List[str]
    _data_log_internal_freq : np.integer

    ros_tx : bool
    ego_pose_tx : PoseTx
    ego_TF_tx : TFTx
    drone_marker_tx : MeshMarkerTx
    ros_tx_freq : np.floating
    ros_tx_freq_slow : np.floating
    _ros_tx_internal_freq : np.integer
    _ros_tx_internal_freq_slow : np.integer
    def __init__(self, obs: Dict[str, NDArray[np.floating]],
                info: dict,
                config: dict,
                env = None,
                data_log : dict = None,
                ros_tx_freq : Optional[np.floating] = None, 
                ros_tx_freq_slow : Optional[np.floating] = 10.0):
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
        self.gates = []
        self.obstacles = []
        self.init_states(obs = obs)
        if data_log is not None:
            self.log_file = True
            self.data_log_freq = data_log['freq']
            self.data_log_path = data_log['path']
            self.data_log_keys = data_log['keys']
            self.current_log_frame = None
            self._data_log_internal_freq = max(int(abs(self._freq / self.data_log_freq)), 1)
            self.log = dict()
            for key in list(obs.keys()) + self.data_log_keys:
                self.log[key] = []
            atexit.register(self.write_log)
        else:
            self.log_file = False
        
        if ROS_AVAILABLE and ros_tx_freq is not None:
            self.ros_tx = True
            try:
                rclpy.init()
            except:
                pass
            self.ros_tx_freq = ros_tx_freq
            self.ros_tx_freq_slow = ros_tx_freq_slow
            self._ros_tx_internal_freq = max(int(abs(self._freq / self.ros_tx_freq)), 1)
            self._ros_tx_internal_freq_slow = max(int(abs(self._freq / self.ros_tx_freq_slow)), 1)
            self.ego_pose_tx = PoseTx(
                node_name = 'ego_pose_tx',
                topic_name = 'drone_pose',
                queue_size = 10
            )
            self.ego_TF_tx = TFTx(
                node_name = 'ego_TF_tx',
                topic_name = 'drone'
            )
            current_dir = os.path.dirname(os.path.abspath(__file__))
            drone_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'crazy_flies.dae')
            self.drone_marker_tx = MeshMarkerTx(
                node_name = 'drone_marker_tx',
                topic_name = 'drone_marker',
                mesh_path = os.path.abspath(drone_mesh_file),
                frame_id = 'drone',
                queue_size = 1
            )
            ground_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'ground.dae')

            self.ground_marker_tx = MeshMarkerTx(
                node_name = 'ground_marker_tx',
                topic_name = 'ground_marker',
                mesh_path = os.path.abspath(ground_mesh_file),
                frame_id = 'map',
                queue_size = 1
            )
        else:
            self.ros_tx = False
        
        
    
    def init_states(self, obs : Dict[str, np.ndarray],):
        self.pos = obs['pos']
        self.vel = obs['vel']
        self.quat = obs['quat']
        self.last_pos = self.pos
        self.current_t = np.float64(0.0)
        self.next_gate = 0
        self.gates_visited = obs['gates_visited']
        self.obstacles_visited = obs['obstacles_visited']

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
                                entry_offset = self.gate_entry_offset[i],
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

    def visualize_spline_trajectory(self, fig: figure.Figure, ax: axes.Axes, trajectory: CubicSpline, t_start : np.float32, t_end :np.float32, color='red') -> Tuple[figure.Figure, axes.Axes]:
        if hasattr(self, '_last_traj_plot') and self._last_traj_plot is not None:
            try:
                self._last_traj_plot.remove()
            except:
                pass

        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        t_samples = np.linspace(t_start, t_end, int((t_end - t_start)/ 0.05))
        xyz = trajectory(t_samples)

        self._last_traj_plot = ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2, label='Trajectory')[0]

        ax.legend()
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

    def add_log_frame(self):
        if self.current_log_frame is not None:
            for key, val in self.log.items():
                if key not in self.current_log_frame.keys():
                    val.append(val[-1])
                else:
                    val.append(self.current_log_frame[key])
    def need_log_file(self) -> bool:
        return self.log_file and (self._tick % self._data_log_internal_freq == 0)
    def need_ros_tx(self, slow = False) -> bool:
        return self.ros_tx and (((self._tick % self._ros_tx_internal_freq) == 0) if not slow else ((self._tick % self._ros_tx_internal_freq_slow) == 0))
    
    def step_update(self, obs : Dict[str, np.ndarray]) -> None:
        self.last_pos = self.pos
        self.pos = obs['pos']
        self.quat = obs['quat']
        self.vel = obs['vel']
        self.current_t += 1.0 / self._freq

        if self.need_log_file():
            # Add last log frame
            self.add_log_frame()
            self.current_log_frame = {
                't' : self.current_t
            }
            for key, val in obs.items():
                self.current_log_frame[key] = val

        if self.need_ros_tx(): 
            # self.ego_pose_tx.publish(
            #     raw_data = 
            #     {
            #         'pos' : self.pos,
            #         'quat' : self.quat,
            #         'frame_id' : 'map'
            #     }
            # )
            self.ego_TF_tx.publish(
                raw_data = 
                {
                    'pos' : self.pos,
                    'quat' : self.quat,
                    'frame_id' : 'map'
                }
            )
            self.drone_marker_tx.publish(None)
        if self.need_ros_tx(slow = True):
            self.ground_marker_tx.publish(None)
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

    def update_target_gate(self, obs) -> bool:
        result = self.next_gate != obs['target_gate']
        self.next_gate = obs['target_gate']
        return result

    def update_next_gate(self, distance = 0.5) -> bool:
        # Older version for detecting gate change!
        if not self.next_gate <= len(self.gates):
            return False
        if np.linalg.norm(self.pos - (self.gates[self.next_gate].pos)) > distance:
            return False
        
        v_1 = self.pos - self.gates[self.next_gate].pos
        v_0 = self.last_pos - self.gates[self.next_gate].pos
        dot_0 = np.dot(v_0, self.gates[self.next_gate].norm_vec)
        dot_1 = np.dot(v_1, self.gates[self.next_gate].norm_vec)
        if(dot_0 * dot_1 < 0):
            self.next_gate = self.next_gate + 1 if self.next_gate < len(self.gates) - 1 else self.next_gate
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
    
    def add_log(self, key : str, val) -> bool:
        if key in self.log.keys() and self.current_log_frame is not None:
            self.current_log_frame[key] = val
            return True
        else:
            return False
    
    def write_log(self) -> bool:
        try:
            npz_path = self.data_log_path + ".npz"
            npz_dict = {}

            for key, value in self.log.items():
                arr = np.array(value)
                if arr.dtype == object:
                    try:
                        arr = np.stack(value)
                    except Exception:
                        arr = np.array(value, dtype=object)
                npz_dict[key] = arr

            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez_compressed(npz_path, **npz_dict)

            csv_path = self.data_log_path + ".csv"
            df_data = {}

            for key, value in self.log.items():
                arr = np.array(value)
                if arr.ndim == 1:
                    df_data[key] = arr
                elif arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        df_data[f"{key}_{i}"] = arr[:, i]
                else:
                    continue

            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)

            return True

        except Exception as e:
            print(f"[write_log] Failed to write log: {e}")
            return False
    
    def episode_callback(self):
        self.ego_pose_tx.destroy_node()
        return super().episode_callback()