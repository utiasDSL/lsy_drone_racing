# check whether we have to add to path
import sys
import os
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.map.map_utils import Object, Ray
import matplotlib.pyplot as plt



class Map:
    def __init__(self, min_x, max_x, min_y, max_y, max_z):
        self.lower_bound = np.array([min_x, min_y, 0])
        self.upper_bound = np.array([max_x, max_y, max_z])
        self.objects = []

        self.components = {
    "large_portal": {
        "support": {"position": (0, 0, 0.4), "size": (0.05, 0.05, 0.8)},
        "bottom_bar": {"position": (0, 0, 0.775), "size": (0.5, 0.05, 0.05)},
        "top_bar": {"position": (0, 0, 1.25), "size": (0.5, 0.05, 0.05)},
        "left_bar": {"position": (-0.225, 0, 0.775 + 0.5/2), "size": (0.05, 0.05, 0.5)},
        "right_bar": {"position": (0.225, 0, 0.775 + 0.5/2), "size": (0.05, 0.05, 0.5)}
    },
    "small_portal": {
        "support": {"position": (0, 0, 0.15), "size": (0.05, 0.05, 0.3)},
        "bottom_bar": {"position": (0, 0, 0.3), "size": (0.5, 0.05, 0.05)},
        "top_bar": {"position": (0, 0, 0.75), "size": (0.5, 0.05, 0.05)},
        "left_bar": {"position": (-0.225, 0, 0.3 + 0.5 / 2), "size": (0.05, 0.05, 0.5)},
        "right_bar": {"position": (0.225, 0, 0.3 + 0.5/2), "size": (0.05, 0.05, 0.5)}
    },
    "obstacle": {
        "cylinder": {
            "position": (0, 0, 0.525),
            "size": (0.1, 0.1, 1.05)
        }
    }
}
    
    
    def parse_gates(self, gates_pose_and_types):

        gate_type_id_to_component_name_mapping = {0: "large_portal", 1: "small_portal"}
        
        objects = []
        for gate_pos_and_type in gates_pose_and_types:
            gate_type_id = gate_pos_and_type[6]
            component_name = gate_type_id_to_component_name_mapping[gate_type_id]
            component = self.components[component_name]
            object = Object.transform_urdf_component_into_object(component)
            center = np.array(gate_pos_and_type[0:3])
            rotation = np.array(gate_pos_and_type[3:6])
            assert np.allclose(rotation[0], 0) and np.allclose(rotation[1], 0), "Only z-axis rotation supported"

            object.translate(center)
            object.rotate_z(rotation[2])
            objects.append(object)

        self.objects.extend(objects)
    
    def parse_obstacles(self, obstacles_pose):
        objects = []
        for obstacle_pose in obstacles_pose:
            component = self.components["obstacle"]
            object = Object.transform_urdf_component_into_object(component)
            
            center = np.array(obstacle_pose[0:3])
            rotation = np.array(obstacle_pose[3:6])
            assert np.allclose(rotation[0], 0) and np.allclose(rotation[1], 0), "Only z-axis rotation supported"

            object.translate(center)
            object.rotate_z(rotation[2])
            objects.append(object)
        
        self.objects.extend(objects)
    
    
    def check_ray_collision(self, ray: Ray):
        for obj in self.objects:
            if obj.check_collision_with_ray(ray):
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
        for obj in self.objects:
            obj.plot(ax)

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
    
    def draw_scene(self, path):
        ax = self.create_map_sized_figure()
        self.add_objects_to_plot(ax)
        self.draw_path(ax, path)

        plt.show()

    
    def inbounds(self,p):
      '''Check if p lies inside map bounds'''
      return (self.lower_bound <= p).all() and (p <= self.upper_bound).all()

if __name__ == "__main__":
    map = Map(-4, 4, -4,4, 1.5)
    gates_pos_and_types = np.array([[
                                    0.45, -1.0, 0, 0, 0, 2.35, 1], 
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
    map.easy_plot()
    
