# check whether we have to add to path
import sys
import os
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.path.rrtutils import *
from src.map.map import Map
import matplotlib.pyplot as plt
from src.map.map_utils import Ray

class RRT:
    def __init__(self, start:np.ndarray, goal:np.ndarray, 
                 map: Map,
                 max_extend_length = 0.1,
                 path_resolution = 0.1,
                 goal_sample_rate = 0.05,
                 max_iter = 500 ):
        self.start = Node(start)
        self.goal = Node(goal)
        self.max_extend_length = max_extend_length
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.dim = start.shape[0]
        self.tree = Rtree(self.dim)
        self.map = map
    
    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.tree.add(self.start)
        for i in range(self.max_iter):
            #Generate a random node (rnd_node)
            rnd_node = self.get_random_node()
            #Get nearest node (nearest_node)
            nearest_node = self.tree.nearest(rnd_node)
            #Get new node (new_node) by connecting
            new_node = self.steer(nearest_node,rnd_node)
            #If the path between new_node and the nearest node is not in collision
            ray = Ray(nearest_node.p, new_node.p)
            if not self.map.check_ray_collision(ray):
              self.tree.add(new_node)
              # If the new_node is very close to the goal, connect it
              # directly to the goal and return the final path
              if self.dist(new_node,self.goal) <= self.max_extend_length:
                  ray = Ray(new_node.p, self.goal.p)
                  if not self.map.check_ray_collision(ray):
                      self.goal.parent = new_node
                      return self.final_path()
        # cannot find path
        print("Cannot find path")
        return None
    
    @staticmethod
    def dist(from_node:Node, to_node:Node):
        #euler distance
        return np.linalg.norm(from_node.p - to_node.p)

    def steer(self,from_node, to_node):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        dist = self.dist(from_node, to_node)
        #Rescale the path to the maximum extend_length
        if dist > self.max_extend_length:
            diff = from_node.p - to_node.p
            to_node.p  = from_node.p - diff/dist * self.max_extend_length
        to_node.parent = from_node
        return to_node
    
    def sample(self):
        # Sample random point inside boundaries
        lower = self.map.lower_bound
        upper = self.map.upper_bound

        return lower + np.random.rand(self.dim)*(upper - lower)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            rnd = self.sample()
        else:
            rnd = self.goal.p
        return Node(rnd)
    
    def final_path(self):
        """Compute the final path from the goal node to the start node"""
        path = []
        node = self.goal
        if (node.p == node.parent.p).all(): node = node.parent
        while node.parent:
          path.append(node.p)
          node = node.parent
        path.append(self.start.p)
        return np.array(path[::-1])
    

if __name__ == "__main__":
    map = Map(-4, 4, -4,4, 1.5)
    gates_pos_and_types = np.array([
                                    [0.45, -1.0, 0, 0, 0, 2.35, 1], 
                                    [1.0, -1.55, 0, 0, 0, -0.78, 0], 
                                    [0.0, 0.5, 0, 0, 0, 0, 1], 
                                    [-0.5, -0.5, 0, 0, 0, 3.14/2, 0]
                                    ]
                                    )
    obstacles = [  # x, y, z, r, p, y
      [1.0, -0.5, 0, 0, 0, 0],
      [0.5, -1.5, 0, 0, 0, 0],
      [-0.5, 0, 0, 0, 0, 0],
      [0, 1.0, 0, 0, 0, 0]
    ]

    map.parse_gates(gates_pos_and_types)
    map.parse_obstacles(obstacles)
    #map.easy_plot()

    start = np.array([1, -3, 0])
    goal = np.array([0, 3, 0])

    rrt = RRT(start, goal, map)
    waypoints =  rrt.plan()
    rrt.draw_scene(waypoints)
    