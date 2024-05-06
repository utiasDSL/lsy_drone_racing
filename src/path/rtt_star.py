# check whether we have to add to path
import sys
import os
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.path.rtt import RRT
from math import ceil
from src.path.sampleutils import InformedSampler
from src.map.map import Map
import numpy as np
from src.map.map_utils import Ray

class RRTStar(RRT):
    def __init__(self, start:np.ndarray, goal:np.ndarray, 
                 map: Map,
                 max_extend_length = 0.15,
                 path_resolution = 0.5,
                 goal_sample_rate = 0.05,
                 max_iter = 5000 ,
                 good_enough_abortion_delta = 0.01,
                 ):
        super().__init__(start, goal, map, max_extend_length, path_resolution, goal_sample_rate, max_iter)
        self.final_nodes = []
        self.informed_sampler = InformedSampler(goal, start)
        self.good_enough_abortion_delta = good_enough_abortion_delta

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.start.cost = 0
        self.tree.add(self.start)

        optimal_cost = np.linalg.norm(self.start.p - self.goal.p)

        for i in range(self.max_iter):
            #Generate a random node (rnd_node)
            rnd = self.get_random_node()
            # Get nearest node
            nearest_node = self.tree.nearest(rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd)
            # If path between new_node and nearest node is not in collision
            ray = Ray(nearest_node.p, new_node.p)
            if not self.map.check_ray_collision(ray, can_pass_gate=False):
              #add the node to tree
              self.add(new_node)

              # early abort in case found path is good up to some delta
              if self.goal.parent and (self.goal.cost <= optimal_cost + self.good_enough_abortion_delta):
                print(f"Early aborting at iteration {i}. Optimal cost: {optimal_cost}, found cost: {self.goal.cost}")
                path = self.final_path()
                return path, self.goal.cost
              
        #Return path if it exists
        if not self.goal.parent: path = None
        else: path = self.final_path()
        return path, self.goal.cost

    def add(self,new_node):
        near_nodes = self.near_nodes(new_node)
        # Connect the new node to the best parent in near_inds
        self.choose_parent(new_node,near_nodes)
        #add the new_node to tree
        self.tree.add(new_node)
        # Rewire the nodes in the proximity of new_node if it improves their costs
        self.rewire(new_node,near_nodes)
        #check if it is in close proximity to the goal
        if self.dist(new_node,self.goal) <= self.max_extend_length:
          # Connection between node and goal needs to be collision free
          ray = Ray(new_node.p, self.goal.p)
          if not self.map.check_ray_collision(ray, can_pass_gate=False):
            #add to final nodes if in goal region
            self.final_nodes.append(new_node)
        #set best final node and min_cost
        self.choose_parent(self.goal,self.final_nodes)

    def choose_parent(self, node, parents):
        """Set node.parent to the lowest resulting cost parent in parents and
           node.cost to the corresponding minimal cost
        """
        # Go through all near nodes and evaluate them as potential parent nodes
        for parent in parents:
          #checking whether a connection would result in a collision
          ray = Ray(parent.p, node.p)
          if not self.map.check_ray_collision(ray, can_pass_gate=False):
            #evaluating the cost of the new_node if it had that near node as a parent
            cost = self.new_cost(parent, node)
            #picking the parent resulting in the lowest cost and updating the cost of the new_node to the minimum cost.
            if cost < node.cost:
              node.parent = parent
              node.cost = cost

    def rewire(self, new_node, near_nodes):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        #Go through all near nodes and check whether rewiring them to the new_node is useful
        for node in near_nodes:
          self.choose_parent(node,[new_node])
        self.propagate_cost_to_leaves(new_node)

    def near_nodes(self, node):
        """Find the nodes in close proximity to given node"""
        nnode = self.tree.len + 1
        r = ceil(5.5*np.log(nnode))
        return self.tree.k_nearest(node,r)

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        return from_node.cost + self.dist(from_node, to_node)

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.tree.all():
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def sample(self):
        """Sample random node inside the informed region"""
        lower = self.map.lower_bound
        upper = self.map.upper_bound
        if self.goal.parent:
          rnd = np.inf
          #sample until rnd is inside bounds of the map
          while not self.map.inbounds(rnd):
              # Sample random point inside ellipsoid
              #print(f"Foal cost: {self.goal.cost}")
              rnd = self.informed_sampler.sample(self.goal.cost)
        else:
          # Sample random point inside boundaries
          rnd = lower + np.random.rand(self.dim)*(upper - lower)
        return rnd
    

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

    start = np.array([0.45, -1, 0.525])
    goal = np.array([1, -1.55, 1])

    rrt = RRTStar(start, goal, map)
    waypoints, cost =  rrt.plan()
    map.draw_scene(waypoints)
    
