import numpy as np

from src.utils.config_reader import ConfigReader

class Traj():
    def __init__(self, traj_points: np.ndarray, creation_time: float):
        """"
        traj_points provided as numpy array of shape in format [x,x_dot, x_ddot,y,y_dot, y_ddot,z,z_dot,z_ddot,t]
        """
        self.times = traj_points[:, -2]
        self.belongs_to_segment = traj_points[:, -1]
        self.last_sampled_idx = 0
        positions = []
        velocities = []
        accelerations = []

        for point in traj_points:
            pos = [point[0], point[3], point[6]]
            vel = [point[1], point[4], point[7]]
            acc = [point[2], point[5], point[8]]
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.accelerations = np.array(accelerations)
        self.creation_time = creation_time
        
    def _get_true_t(self, t):
        t_true = t - self.creation_time
        if t_true < 0:
            print(f"Time is before creation time of trajectory, t: {t}, creation_time: {self.creation_time}, t_true: {t_true}")
            exit(1)
        return t_true
    
    def has_ended(self, t):
        t_true = self._get_true_t(t)
        return t_true > self.times[-1]

    
    def get_des_state(self, t):
        t_true = self._get_true_t(t)
        # find the index of the closest time
        idx = np.abs(self.times - t_true).argmin()
        self.last_sampled_idx = idx
        return self.positions[idx], self.velocities[idx], self.accelerations[idx]
    
    def get_last_state(self):
        return self.positions[self.last_sampled_idx], self.velocities[self.last_sampled_idx], self.accelerations[self.last_sampled_idx]
    
    def get_segment_positions(self, segment_id):
        return self.positions[self.belongs_to_segment == segment_id]
    
class Gate:
    def __init__(self, pos: np.ndarray, rot: np.ndarray, type):
        self.pos = pos
        self.rot = rot
        self.gate_type = type
    
    @staticmethod
    def from_nomial_gate_pos_and_type(gate_pos_and_type: np.ndarray):
        pos = np.array(gate_pos_and_type[:3])
        pos[2] = 0
        rot = np.array(gate_pos_and_type[3:6])
        type = int(gate_pos_and_type[6])
        return Gate(pos, rot, type)

    @staticmethod
    def from_within_flight_observation(gate_pose:np.ndarray, gate_type, within_range: bool):
        """
        Important! Durign flight, we receive the center point of the goal as reference coordinates ,this is different,
        than the initial nominal oversvation, where z is the lowest point of the gate. We must therefore subtract the gate height
        """
        pos = np.array(gate_pose[:3])
        if within_range:
            config_reader = ConfigReader.get()
            gate_properties = config_reader.get_gate_properties_by_type(gate_type)
            pos[2] -= gate_properties["height"]
        rot = np.array(gate_pose[3:6])
        return Gate(pos, rot, gate_type)

class Obstacle:
    def __init__(self, pos, rot):
        self.pos = pos
        self.rot = rot

    @staticmethod
    def from_obstacle_pos(obstacle_pos: np.ndarray):
        pos = np.array(obstacle_pos[:3])
        rot = np.array(obstacle_pos[3:6])
        return Obstacle(pos, rot)
