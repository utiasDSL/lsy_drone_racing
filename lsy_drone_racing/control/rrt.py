"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z # default z to 0 if not provided
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None
            self.p = np.array([x,y,z])


    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
            self.zmin = float(area[4]) 
            self.zmax = float(area[5]) 


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=50000,
                 gates=None,
                 play_area=None,
                 robot_radius=0.01,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.goal = self.Node(goal[0], goal[1], goal[2])
        self.gates = [self.Node(g[0], g[1], g[2]) for g in (gates or [])]
        self.final_goal = self.start
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation=True):
        """
        RRT path planning through gates and to the final goal.
        """
        full_path = []
        current_start = self.start
        

        for i, gate in enumerate(self.gates + [self.goal]):  # Include gates and final goal + [self.goal]
           

            self.goal = gate
            self.node_list = [current_start]
            # print(f"Position at {current_start.p} current goal {self.goal.p}")
            path_segment = None
            print(f"current starting pos is {current_start.p}, and current goal is {self.goal.p}")
            for j in range(self.max_iter):
                rnd_node = self.get_random_node()
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
                nearest_node = self.node_list[nearest_ind]
                new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

                if self.check_if_outside_play_area(new_node, self.play_area) and \
                self.check_collision(nearest_node.p, new_node.p, self.obstacle_list):
                    self.node_list.append(new_node)

                    if self.calc_dist_to_goal(new_node.x, new_node.y, new_node.z, gate) <= self.expand_dis:
                        if self.check_collision(new_node.p, gate.p, self.obstacle_list):
                                if self.check_collision_with_gate(nearest_node.p, new_node.p, gate, margin=0.1):
                                    gate.parent = new_node
                                    path_segment = self.generate_final_course(len(self.node_list) - 1)
                                    print("Path segment found")
                                    break

            if path_segment is None:  # Handle case where no path to gate is found
                print(f"Cannot find path to gate/goal: {gate.p}")
                return None

            if full_path and np.array_equal(full_path[-1], path_segment[0]):
                full_path.extend(path_segment[1:])  # Skip duplicate start node
            else:
                full_path.extend(path_segment)  # Append the entire segment if no overlap
            

            current_start = self.Node(path_segment[-1][0], path_segment[-1][1], path_segment[-1][2])
            

    # Set the next gate as the goal


            # full_path.append(self.goal.p)
        return np.array(full_path)


    def steer(self, from_node, to_node, extend_length=float(1)):



        dist = np.linalg.norm(from_node.p - to_node.p)

        # print(to_node, from_node.p)

        if dist > extend_length:
            diff = from_node.p - to_node.p
            to_node.p = from_node.p - diff / dist * extend_length
        to_node.parent = from_node

   



        return to_node


    def generate_final_course(self, goal_ind):
        """
        Generate final course (path) by tracing parent nodes.
        """
        path = []
        node = self.node_list[goal_ind]
        while node is not None:  # Check for NoneType
            path.append(node.p)
            node = node.parent
        return path[::-1]  # Reverse the path


    def calc_dist_to_goal(self, x, y, z, target_node):
        dist = np.sqrt(np.sum((np.array([x, y, z]) - target_node.p) ** 2))
        return dist
        # def get_random_node(self):
        #     if random.randint(0, 100) > self.goal_sample_rate:
        #         rnd = self.Node(
        #             random.uniform(self.min_rand, self.max_rand),
        #             random.uniform(self.min_rand, self.max_rand))
        #     else:  # goal point sampling
        #         rnd = self.Node(self.end.x, self.end.y)
        #     return rnd
        
    def get_random_node(self):
        if random.random() > self.goal_sample_rate / 100.0:
            rnd = (
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)
            )
        else:
            rnd = self.goal.p  # goal sampling
        return self.Node(rnd[0], rnd[1], rnd[2])
    

    def check_collision_with_gate(self, near_node, new_node, gate, margin=0.1):
        """
        Check if the path from near_node to new_node passes through the gate's center.
        margin: Defines the size of the "box" around the gate.
        """
        dist_to_gate = np.linalg.norm(gate.p - new_node)  # Distance to the gate center
        return dist_to_gate <= margin  # Ensure it's within the margin of the gate center




    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # print(rnd_node, node_list)
        dlist = [((node.p)[0] - (rnd_node.p)[0])**2 + ((node.p)[1] - (rnd_node.p)[1])**2 + ((node.p)[2] + (rnd_node.p)[2])**2
                 for node in node_list]
        
        min_ind = dlist.index(min(dlist))
        # print(min_ind)

        return min_ind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok
        # NEED TO ADD Z

    # @staticmethod
    def check_collision(self, near_node, new_node, obs):
        """
        Check for collisions along the path from near_node to new_node.

        obs: List of obstacles where each obstacle is (x, y, z, radius).
        """

        dist = np.linalg.norm(near_node - new_node)
        n = max(2, int(dist / 1))  # Ensure at least two points
        points = np.linspace(near_node, new_node, n)

        for p in points:
            for ox, oy, oz, radius in obs:
                # Check if the point p is within the obstacle's radius
                if np.linalg.norm(np.array([ox, oy, oz]) - p) <= radius + self.robot_radius:
                    return False  # Collision detected

        return True  # No collision

    def plan_through_waypoints(self, waypoints):
        all_paths = []  # Store the entire path through all gates
        current_start = self.start  # Initialize the start position
        
        for waypoint in waypoints:
            self.start = current_start  # Set the new start
            self.goal = waypoint        # Set the current goal
            path_segment = self.planning(animation=False)
            
            if path_segment:
                all_paths.extend(path_segment)  # Append the path segment
                current_start = path_segment[-1]  # Update the start for the next segment
            else:
                print(f"Failed to find a path to waypoint {waypoint}")
                break
        
        return all_paths


    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z  # Calculate difference in z direction
        d = math.hypot(math.hypot(dx, dy), dz)  # 3D distance
        theta = math.atan2(dy, dx)  # 2D angle
        phi = math.atan2(dz, math.hypot(dx, dy))  # 3D angle (vertical)
        return d, theta, phi


def main():
    # Define start and goal positions
    start = [0, 0, 0]  # Example start position
    waypoints = np.array([
        [0.45, -1.0, 0.525],
        [1.0, -1.55, 1.0],
        [0.0, 0.5, 0.525],
        [-0.5, -0.5, 1.0]
    ])

    # Initialize RRT with start and first waypoint as goal
    rrt = RRT(
        start=start,
        goal=waypoints[0],  # Set the first waypoint as the initial goal
        rand_area=[-2, 2],  # Define random sampling space
        obstacle_list=[]    # Add your obstacle list here if necessary
    )

    # Plan path through all waypoints
    path = rrt.plan_through_waypoints(waypoints)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!")
        print(path)


if __name__ == '__main__':
    main()