import numpy as np
from src.map.map import Map
from src.path.rtt_star import RRTStar
from src.utils.calc_gate_center import calc_gate_center_and_normal
import polynomial_trajectory as polytraj

from src.utils.types import Gate, Obstacle, Traj

class AdaptiveTrajGenerator():
    def __init__(self, start_point, goal_point, nominal_gate_pos_and_type, nominal_obstacles_pos, gate_types, drone_radius ):
        self.gates = [Gate.from_nomial_gate_pos_and_type(gate_pos_and_type) for gate_pos_and_type in nominal_gate_pos_and_type]
        self.obstacles = [Obstacle.from_obstacle_pos(obstacle_pos) for obstacle_pos in nominal_obstacles_pos]
        self.drone_radius = drone_radius
        self.last_update_time = 0
        self.segments = []
        self.traj = None
        self.start_point = start_point
        self.goal_point = goal_point
        self.gate_types = gate_types
        self.lower_bound = np.array([-1.5, -2, 0])
        self.upper_bound = np.array([1.5, 1, 1])

        self.max_iter = 500
        self.max_extend_length = 1
        self.goal_sample_rate = 0.05

        self.max_vel = 5
        self.max_acc = 3
        self.sample_intervall = 0.2


        # Parse checkpoints
        checkpoints = [start_point]
        for gate in self.gates:
            center, normal = calc_gate_center_and_normal(gate, self.gate_types)
            gate_normal_normalized = normal / np.linalg.norm(normal)
            eps = 0.05
            early_checkpoint = center - (drone_radius + eps) * gate_normal_normalized
            late_checkpoint = center + (drone_radius + eps) * gate_normal_normalized
            checkpoints.append(early_checkpoint)
            checkpoints.append(late_checkpoint)
        checkpoints.append(goal_point)
        self.checkpoints = checkpoints


    
    def pre_compute_traj(self, takeoff_time):

        map = Map(self.lower_bound, self.upper_bound, self.drone_radius)
        map.parse_gates(self.gates)
        map.parse_obstacles(self.obstacles)

        # find segments
        segments = []
        for i, (start_pos, end_pos) in enumerate(zip(self.checkpoints[:-1:2], self.checkpoints[1::2])):
            can_pass_gate = i % 2 == 1
            rrt = RRTStar(start_pos, end_pos, map, can_pass_gates=can_pass_gate, max_iter=self.max_iter, max_extend_length=self.max_extend_length, goal_sample_rate=self.goal_sample_rate)
            segment_path, _ = rrt.plan()
            if len(segment_path) == 0:
                print(f"Failed to generate path for segment {i} from {start_pos} to {end_pos}")
                exit(1)
            segments.append(segment_path)
        
        self.segments = segments
        self._traj_from_segments(time=takeoff_time)

    def _traj_from_segments(self, time):
        merged_path = []
        assert len(self.segments) >= 2, "At least two segments are required"
        for i in range(len(self.segments) - 1):
            merged_path.extend(self.segments[i])
            gate_center, _ = calc_gate_center_and_normal(self.gates[i], self.gate_types)
            merged_path.append(gate_center)
        merged_path.extend(self.segments[-1])

        merged_path = np.array(merged_path)
        print(merged_path)
        insignificance_threshold = 0.01
        purged_path = [merged_path[0]]
        for point in merged_path[1:]:
            if np.linalg.norm(purged_path[-1] - point) > insignificance_threshold:
                purged_path.append(point)
        purged_path = np.array(purged_path)


        traj_points = polytraj.generate_trajectory(purged_path, self.max_vel, self.max_acc, self.sample_intervall)
        self.traj = Traj(traj_points, creation_time=time)

    def update_gate_pos(self, next_gate_pose, next_gate_id, next_gate_within_range, drone_pos, time):
        """
        
        Return True if update necessary due to much divergence, False otherwise
        """

        next_gate_pos = np.array(next_gate_pose[:3])
        next_gate_rot = np.array(next_gate_pose[3:6])
        print(f"Next gate position {next_gate_pos} and rotation {next_gate_rot}")
        ref_gate = self.gates[next_gate_id]
        cur_gate = Gate.from_within_flight_observation(next_gate_pose, ref_gate.gate_type, within_range=next_gate_within_range)

        
        # ignore if position or rotation only changed insignificantly
        pos_insiginificance_threshold = 0.01
        rot_insiginificance_threshold = 0.01
        if np.linalg.norm(ref_gate.pos - cur_gate.pos) < pos_insiginificance_threshold and np.linalg.norm(ref_gate.rot - cur_gate.rot) < rot_insiginificance_threshold:
            return False

        print(f"Updating gate {next_gate_id} from {ref_gate.pos} to {cur_gate.pos} and {ref_gate.rot} to {cur_gate.rot}")
        # update gate
        self.gates[next_gate_id] = cur_gate

        # get new center and normal
        center, normal = calc_gate_center_and_normal(self.gates[next_gate_id], self.gate_types)
        # calculate new pre and post gate checkpoint
        gate_normal_normalized = normal / np.linalg.norm(normal)
        eps = 0.05
        early_checkpoint = center - (self.drone_radius + eps) * gate_normal_normalized
        late_checkpoint = center + (self.drone_radius + eps) * gate_normal_normalized

        # update checkpoints
        self.checkpoints[2*next_gate_id + 1] = early_checkpoint
        self.checkpoints[2*next_gate_id + 2] = late_checkpoint

        # recompute segment of relevance
        segment_id_pre = next_gate_id
        segment_id_post = next_gate_id + 1

        map = Map(self.lower_bound, self.upper_bound, self.drone_radius)
        map.parse_gates(self.gates)
        map.parse_obstacles(self.obstacles)

        can_pass_gate = False
        # calculate pre segment, i.e., from current pos to next gate
        start_pos = drone_pos
        end_pos = early_checkpoint
        rrt = RRTStar(start_pos, end_pos, map, can_pass_gates=can_pass_gate, max_iter=self.max_iter, max_extend_length=self.max_extend_length, goal_sample_rate=self.goal_sample_rate)
        segment_path, _ = rrt.plan()
        if len(segment_path) == 0:
            print(f"Failed to generate path for segment {segment_id_pre} from {start_pos} to {end_pos}")
            exit(1)
        self.segments[segment_id_pre] = segment_path

        # calculate post segment, i.e., from next gate to next next gate
        start_pos = late_checkpoint
        end_pos = self.checkpoints[2*segment_id_post + 1]
        rrt = RRTStar(start_pos, end_pos, map, can_pass_gates=can_pass_gate, max_iter=self.max_iter, max_extend_length=self.max_extend_length, goal_sample_rate=self.goal_sample_rate)
        segment_path, _ = rrt.plan()
        if len(segment_path) == 0:
            print(f"Failed to generate path for segment {segment_id_post} from {start_pos} to {end_pos}")
            exit(1)
        self.segments[segment_id_post] = segment_path
        
        self._traj_from_segments(time=time)
        return True







