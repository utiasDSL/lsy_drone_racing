# check whether we have to add to path
import random
import sys
import os

from src.utils.config_reader import ConfigReader
from src.utils.types import Gate, Obstacle
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.map.map_utils import Object, Ray
import matplotlib.pyplot as plt
from rtree import index
from typing import List



class Map:
    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray, drone_radius=0):
        self.config_reader = ConfigReader.get()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.idx = self._create_3d_idx()
        self.drone_radius = drone_radius
        self.obbs = []

    @staticmethod
    def _create_3d_idx():
        p = index.Property()
        p.dimension = 3
        return index.Index(properties=p)
    
    def parse_gates(self, gates: List[Gate]):
    
        for i, gate in enumerate(gates):
            print(f"Gate {i}")
            component = self.config_reader.get_gate_geometry_by_type(gate.gate_type)
            object = Object.transform_urdf_component_into_object(component)
            center = np.array(gate.pos)
            rotation = np.array(gate.rot)
            assert np.allclose(rotation[0], 0) and np.allclose(rotation[1], 0), "Only z-axis rotation supported"

            object.translate(center)
            object.rotate_z(rotation[2])
            for i, obb in enumerate(object.obbs):
                self._add_obb(obb)
                print(f"Obb {i}. Center: {obb.center}")
            

            

    
    def parse_obstacles(self, obstacles: List[Obstacle]):
        for obstacle in obstacles:
            component = self.config_reader.get_obstacle_geometry()
            object = Object.transform_urdf_component_into_object(component)
            
            center = np.array(obstacle.pos)
            rotation = np.array(obstacle.rot)
            assert np.allclose(rotation[0], 0) and np.allclose(rotation[1], 0), "Only z-axis rotation supported"

            object.translate(center)
            object.rotate_z(rotation[2])
            for obb in object.obbs:
                self._add_obb(obb)

    def _add_obb(self, obb):
        # Add the obb to the r-tree by rotating it into world frame and doing worst case estimation
        # Calculate the corners of the OBB based on its center, half_sizes, and rotation
        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * obb.half_sizes  # Scale the unit cube by half sizes
        corners = np.dot(corners, obb.rotation_matrix.T) + obb.center  # Rotate and translate

        # Compute the axis-aligned bounding box (AABB) that encloses the rotated corners
        # is is inflated with the drone size
        min_corner = np.min(corners, axis=0) - self.drone_radius
        max_corner = np.max(corners, axis=0) + self.drone_radius

        # Add the AABB to the r-tree
        new_id = len(self.obbs)
        self.idx.insert(new_id, (*min_corner, *max_corner))
        self.obbs.append(obb)

    def check_path_collision(self, points):
        # random sampling for checking points along path
        sampling_idx = list(range(len(points)))
        random.shuffle(sampling_idx)
        for idx in sampling_idx:
            if self.check_point_collision(points[idx]):
                return True
        return False
    
    
    def check_ray_collision(self, ray: Ray, can_pass_gate):
        # Calculate collision candidates using the r-tree
        ray_min = np.minimum(ray.start, ray.end)
        ray_max = np.maximum(ray.start, ray.end)
        potential_hits = list(self.idx.intersection((*ray_min, *ray_max), objects=True))

        # Check for collision with each candidate
        for item in potential_hits:
            #print(f"Check potential collision with obb: {item.id}")
            obb = self.obbs[item.id]
            if obb.type == "filling" and can_pass_gate:
                continue
            elif obb.check_collision_with_ray(ray, self.drone_radius):
                return True
        return False

    def check_point_collision(self, point: np.ndarray):
        """
        Check whether a single point collides with any object.
        :param point: The point to check.
        :return: True if the point is inside any object, False otherwise.
        """
        
        # Search for objects in the vicinity of the point using the R-tree
        potential_hits = list(self.idx.intersection((*point, *point), objects=True))
        for item in potential_hits:
            obb = self.obbs[item.id]
            if obb.type == "filling":
                continue
            elif obb.check_collision_with_point(point):
                return True
        return False
            


    def create_map_sized_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.lower_bound[0], self.upper_bound[0])
        ax.set_ylim(self.lower_bound[1], self.upper_bound[1])
        ax.set_zlim(self.lower_bound[2], self.upper_bound[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return ax
    
    def add_objects_to_plot(self, ax):
        for obb in self.obbs:
            obb.plot(ax)

    def easy_plot(self):
        ax = self.create_map_sized_figure()
        self.add_objects_to_plot(ax)
        plt.show()

    
    def draw_path(self,ax,path):
        '''draw the path if available'''
        if path is None:
            print("path not available")
        else:
            ax.plot(*np.array(path).T, '-', color = (0.9, 0.2, 0.5, 0.8), zorder = 5)
            # mar path coordinates wit red dots
            ax.scatter(*np.array(path).T, color = 'red', zorder = 5)
    
    def draw_scene(self, path, checkpoints=None):
        ax = self.create_map_sized_figure()
        self.add_objects_to_plot(ax)
        self.draw_path(ax, path)
        if checkpoints is not None:
            for checkpoint in checkpoints:
                ax.scatter(*checkpoint, color='green', zorder=5)
        plt.show()

    
    def inbounds(self,p):
      '''Check if p lies inside map bounds'''
      return (self.lower_bound <= p).all() and (p <= self.upper_bound).all()

if __name__ == "__main__":
    lower_bound = np.array([-2, -2, 0])
    upper_bound = np.array([2, 2, 2])
    map = Map(lower_bound, upper_bound, 0.1)
    # gates_pos_and_types = np.array([[
    #                                 0.45, -1.0, 0, 0, 0, 2.35, 1], 
    #                                 [1.0, -1.55, 0, 0, 0, -0.78, 0], 
    #                                 [0.0, 0.5, 0, 0, 0, 0, 1], 
    #                                 [-0.5, -0.5, 0, 0, 0, 3.14/2, 0]
    #                                 ]
    #                                 )
    # obstacles = [  # x, y, z, r, p, y
    #   [1.0, -0.5, 0, 0, 0, 0],
    #   [0.5, -1.5, 0, 0, 0, 0],
    #   [-0.5, 0, 0, 0, 0, 0],
    #   [0, 1.0, 0, 0, 0, 0]
    # ]

    gates_pos_and_types = np.array([
        [0,0,0,0,0,0,0],
    ])

    obstacles = []

    ray = Ray(np.array([2,2,0]), np.array([1,1,1]))

    map.parse_gates(gates_pos_and_types)
    map.parse_obstacles(obstacles)
    col = map.check_ray_collision(ray)
    print(f"Collision: {col}")
