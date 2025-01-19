import math
import random
import numpy as np
from scipy.spatial import KDTree

class RRT:
    class Node:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None
            self.cost = 0.0
            self.p = np.array([x, y, z])

    def __init__(self, start, goal, obstacle_list, gates, rand_area, gates_rpy,
                 expand_dis=0.6, path_resolution=0.3, goal_sample_rate=30,
                 max_iter=50000, play_area=None, robot_radius=0.01,
                 gate_width=0.01, gate_height=0.01, gate_depth=0.9):
        self.start = self.Node(start[0], start[1], start[2])
        self.goal = self.Node(goal[0], goal[1], goal[2])
        self.gates = [self.Node(g[0], g[1], g[2]) for g in (gates or [])]
        self.final_goal = self.goal
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.play_area = play_area
        self.gate_width = gate_width
        self.gate_height = gate_height
        self.gate_depth = gate_depth
        self.gates_rpy = gates_rpy
        self.obstacle_kd_tree = KDTree([obs[:3] for obs in obstacle_list]) if obstacle_list else None

    def planning(self):
        full_path = []
        current_start = self.start

        for i, gate in enumerate(self.gates + [self.final_goal]):
            self.goal = gate
            self.node_list = [current_start]

            path_found = False
            for _ in range(self.max_iter):
                rnd_node = self.get_random_node()
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
                nearest_node = self.node_list[nearest_ind]
                new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

                if self.check_collision(new_node.p):
                    near_inds = self.find_near_nodes(new_node)
                    new_node = self.choose_parent(new_node, near_inds)
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

                    dist_to_gate = self.calc_dist_to_goal(new_node.x, new_node.y, new_node.z, gate)

                    if dist_to_gate <= self.expand_dis * 2:
                        rpy = self.gates_rpy[0][i] if i < len(self.gates_rpy[0]) else (0, 0, 0)
                        if self.check_gate_passage(nearest_node.p, new_node.p, gate, rpy):
                            path_segment = self.generate_final_course(len(self.node_list) - 1)
                            path_found = True
                            break

            if not path_found:
                return None

            full_path.extend(path_segment if not full_path else path_segment[1:])

            last_point = np.array(path_segment[-1])
            direction_to_next_gate = np.array([gate.x, gate.y, gate.z]) - last_point
            direction_norm = np.linalg.norm(direction_to_next_gate)

            if direction_norm < 1e-10:
                new_start = last_point
            else:
                direction_to_next_gate = direction_to_next_gate / direction_norm
                new_start = last_point + direction_to_next_gate * (self.expand_dis * 0.5)

            current_start = self.Node(*new_start)

        return full_path

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = self.Node(self.goal.x, self.goal.y, self.goal.z)
        return rnd

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d = np.linalg.norm(to_node.p - from_node.p)

        if d < 1e-10:
            return new_node

        if extend_length > d:
            extend_length = d

        new_node.p = from_node.p + (to_node.p - from_node.p) / d * extend_length
        new_node.x, new_node.y, new_node.z = new_node.p
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length

        return new_node

    def check_collision(self, point):
        clearance = 0.05
        if self.obstacle_kd_tree:
            nearest_dist, nearest_idx = self.obstacle_kd_tree.query(point)
            nearest_obstacle = self.obstacle_list[nearest_idx]
            d = np.linalg.norm(np.array(nearest_obstacle[:3]) - point)
            if d <= nearest_obstacle[3] + self.robot_radius + clearance:
                return False
        return True

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(np.sum((node.p - rnd_node.p)**2) + 1e-10) for node in node_list]
        return dlist.index(min(dlist))

    def calc_dist_to_goal(self, x, y, z, target_node):
        diff = np.array([x, y, z]) - target_node.p
        return max(np.linalg.norm(diff), 1e-10)

    def find_near_nodes(self, new_node):
        r = self.expand_dis * 2.0
        dlist = [np.sum((node.p - new_node.p)**2) for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if self.check_collision(t_node.p):
                costs.append(near_node.cost + np.linalg.norm(near_node.p - t_node.p))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return new_node

        new_node.parent = self.node_list[near_inds[costs.index(min_cost)]]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_cost = np.linalg.norm(near_node.p - new_node.p)
            cost = new_node.cost + edge_cost

            if cost < near_node.cost:
                t_node = self.steer(new_node, near_node)
                if self.check_collision(t_node.p):
                    near_node.parent = new_node
                    near_node.cost = cost

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.p)
            node = node.parent
        path.append(node.p)
        return path[::-1]

    def check_gate_passage(self, point1, point2, gate, rpy):
        gate_pos = np.array([gate.x, gate.y, gate.z])
        R = self.get_rotation_matrix(*rpy)
    
        p1_local = R.T @ (point1 - gate_pos)
        p2_local = R.T @ (point2 - gate_pos)
    
        # Ensure the path crosses the gate plane
        if p1_local[2] * p2_local[2] > 0:
            return False
    
        # Compute intersection with the gate plane
        t = -p1_local[2] / (p2_local[2] - p1_local[2])
        intersection = p1_local + t * (p2_local - p1_local)
    
        # Stricter bounding box check
        safe_width = self.gate_width * 0.5
        safe_height = self.gate_height * 0.5
    
        if abs(intersection[0]) > safe_width / 2 or abs(intersection[1]) > safe_height / 2:
            return False
    
        # Gate clearance check
        clearance_margin = 0.1  # Additional buffer around the gate edges
        if abs(intersection[0]) > (self.gate_width / 2 - clearance_margin) or \
           abs(intersection[1]) > (self.gate_height / 2 - clearance_margin):
            return False
    
        # Better approach angle filtering
        trajectory = p2_local - p1_local
        approach_angle = np.arctan2(np.sqrt(trajectory[0]**2 + trajectory[1]**2), abs(trajectory[2]))
    
        if approach_angle > np.pi / 6:  # Stricter angle constraint
            return False
    
        # Collision-free straight line check through the gate
        num_samples = 10
        for i in range(num_samples):
            sample_point = p1_local + (p2_local - p1_local) * (i / num_samples)
            if not self.check_collision(R @ sample_point + gate_pos):
                return False
    
        return True

    def get_rotation_matrix(self, roll, pitch, yaw):
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        Rx = np.array([[1, 0, 0],
                       [0, cos_r, -sin_r],
                       [0, sin_r, cos_r]])

        Ry = np.array([[cos_p, 0, sin_p],
                       [0, 1, 0],
                       [-sin_p, 0, cos_p]])

        Rz = np.array([[cos_y, -sin_y, 0],
                       [sin_y, cos_y, 0],
                       [0, 0, 1]])

        return Rz @ Ry @ Rx
